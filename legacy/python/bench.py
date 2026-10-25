#!/usr/bin/env python3
"""YOLOX + TensorRT latency benchmark (TensorRT 10 API, JetPack 6; requires cuda-python, no PyTorch) with per-stage timestamps.

Measured stages (ns, time.perf_counter_ns):
  t_rel   frame release (periodic) or start (b2b)
  t_start start of processing
  t_dec   after JPEG decoding
  t_pre   after pre-processing (letterbox 640x640)
  t_h2d   after host->device copy
  t_inf   after TensorRT inference
  t_d2h   after device->host copy
  t_end   after output decoding + NMS

Modes:
  b2b       each frame starts as soon as the previous one finishes (service latency)
  periodic  frames released every --period-ms (real camera). If processing falls behind so that the
            next frame has already been released, the current frame is dropped (drop=1),
            like a camera that overwrites its buffer. Drops count as deadline misses.

Usage:
  python3 bench.py --engine yolox_s_fp16.engine --frames frames --mode periodic --tag test
"""
import argparse
import csv
import gc
import glob
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import ctypes

try:  # cuda-python >= 12.8 / 13
    from cuda.bindings import runtime as cudart
except ImportError:  # cuda-python 12.x
    from cuda import cudart

INPUT_SIZE = (640, 640)
SCORE_THR = 0.3
NMS_THR = 0.45


# ---------- pre- and post-processing (equivalent to the official YOLOX demo) ----------
def preprocess(img):
    padded = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
    r = min(INPUT_SIZE[0] / img.shape[0], INPUT_SIZE[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                         interpolation=cv2.INTER_LINEAR)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def make_grids(strides=(8, 16, 32)):
    grids, exp = [], []
    for s in strides:
        h, w = INPUT_SIZE[0] // s, INPUT_SIZE[1] // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(g)
        exp.append(np.full((1, g.shape[1], 1), s))
    return np.concatenate(grids, 1).astype(np.float32), np.concatenate(exp, 1).astype(np.float32)


GRIDS, STRIDES = make_grids()


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= thr]
    return keep


def postprocess(out, ratio):
    out = out.reshape(1, -1, out.shape[-1]).astype(np.float32)
    out[..., :2] = (out[..., :2] + GRIDS) * STRIDES
    out[..., 2:4] = np.exp(out[..., 2:4]) * STRIDES
    p = out[0]
    cls_ids = p[:, 5:].argmax(1)
    scores = p[:, 4] * p[np.arange(len(p)), 5 + cls_ids]
    m = scores > SCORE_THR
    if not m.any():
        return 0
    p, scores, cls_ids = p[m], scores[m], cls_ids[m]
    boxes = np.empty((len(p), 4), np.float32)
    boxes[:, 0] = p[:, 0] - p[:, 2] / 2
    boxes[:, 1] = p[:, 1] - p[:, 3] / 2
    boxes[:, 2] = p[:, 0] + p[:, 2] / 2
    boxes[:, 3] = p[:, 1] + p[:, 3] / 2
    boxes /= ratio
    # per-class NMS using a per-class offset
    keep = nms(boxes + cls_ids[:, None] * 4096.0, scores, NMS_THR)
    return len(keep)


# ---------- TensorRT (buffers via cuda-python, no PyTorch) ----------
def ck(ret):
    err = ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA: {cudart.cudaGetErrorString(err)[1]}")
    return ret[1] if len(ret) > 1 else None


H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


class TRTModel:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = ck(cudart.cudaStreamCreate())
        self.dev, self.host, self.nbytes = {}, {}, {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = np.dtype(trt.nptype(self.engine.get_tensor_dtype(n)))
            nbytes = int(np.prod(shape)) * dt.itemsize
            self.nbytes[n] = nbytes
            self.dev[n] = ck(cudart.cudaMalloc(nbytes))
            hptr = ck(cudart.cudaMallocHost(nbytes))  # pinned memory
            buf = (ctypes.c_byte * nbytes).from_address(int(hptr))
            self.host[n] = np.frombuffer(buf, dtype=dt).reshape(shape)
            self.ctx.set_tensor_address(n, int(self.dev[n]))
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp = n
            else:
                self.out = n
        print(f"engine: input {self.inp} {self.host[self.inp].shape} {self.host[self.inp].dtype}, "
              f"output {self.out} {self.host[self.out].shape} {self.host[self.out].dtype}")

    def _sync(self):
        ck(cudart.cudaStreamSynchronize(self.stream))

    def h2d(self, x):
        np.copyto(self.host[self.inp], x.reshape(self.host[self.inp].shape), casting="unsafe")
        ck(cudart.cudaMemcpyAsync(self.dev[self.inp], self.host[self.inp].ctypes.data,
                                  self.nbytes[self.inp], H2D, self.stream))
        self._sync()

    def infer(self):
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        self._sync()

    def d2h(self):
        ck(cudart.cudaMemcpyAsync(self.host[self.out].ctypes.data, self.dev[self.out],
                                  self.nbytes[self.out], D2H, self.stream))
        self._sync()
        return self.host[self.out]


# ---------- loop ----------
def wait_until(t_ns):
    while True:
        rem = t_ns - time.perf_counter_ns()
        if rem <= 0:
            return
        if rem > 2_000_000:
            time.sleep((rem - 1_000_000) / 1e9)


def run_frame(model, jpeg):
    now = time.perf_counter_ns
    t_start = now()
    img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR); t_dec = now()
    x, r = preprocess(img); t_pre = now()
    model.h2d(x); t_h2d = now()
    model.infer(); t_inf = now()
    out = model.d2h(); t_d2h = now()
    n_det = postprocess(out, r); t_end = now()
    return [t_start, t_dec, t_pre, t_h2d, t_inf, t_d2h, t_end, n_det]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--mode", choices=["b2b", "periodic"], default="periodic")
    ap.add_argument("--period-ms", type=float, default=50.0, help="20 fps of the FieldSAFE webcam")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N frames (0 = all)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if args.limit:
        files = files[: args.limit]
    frames = [np.fromfile(f, np.uint8) for f in files]  # in memory: disk I/O kept out of the measurement
    print(f"{len(frames)} frames loaded")

    model = TRTModel(args.engine)
    for i in range(args.warmup):
        run_frame(model, frames[i % len(frames)])

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.csv")
    period = int(args.period_ms * 1e6)
    cols = ["pass", "frame", "drop", "t_rel", "t_start", "t_dec", "t_pre",
            "t_h2d", "t_inf", "t_d2h", "t_end", "n_det"]

    gc.collect()
    gc.disable()  # Python GC kept out of the measurement (stated in the methodology)
    rows = []
    for p in range(args.passes):
        base = time.perf_counter_ns() + 10_000_000
        for k, jpeg in enumerate(frames):
            if args.mode == "periodic":
                rel = base + k * period
                now = time.perf_counter_ns()
                if now < rel:
                    wait_until(rel)
                elif now >= rel + period:
                    rows.append([p, k, 1, rel] + [0] * 7 + [-1])
                    continue
                r = run_frame(model, jpeg)
            else:
                r = run_frame(model, jpeg)
                rel = r[0]
            rows.append([p, k, 0, rel] + r)
        gc.collect()
        print(f"pass {p + 1}/{args.passes} done")
    gc.enable()

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
