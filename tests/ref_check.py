"""Compares post.h and letterbox.h (via test_host) with the pre/post-processing of bench.py."""
import subprocess, sys, numpy as np, cv2, os
BIN, TMP = sys.argv[1], sys.argv[2]
INPUT_SIZE = (640, 640)
def preprocess(img):
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    r = min(640 / img.shape[0], 640 / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r
def make_grids(strides=(8, 16, 32)):
    grids, exp = [], []
    for s in strides:
        h, w = 640 // s, 640 // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(1, -1, 2); grids.append(g); exp.append(np.full((1, g.shape[1], 1), s))
    return np.concatenate(grids, 1).astype(np.float32), np.concatenate(exp, 1).astype(np.float32)
GRIDS, STRIDES = make_grids()
def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T; areas = (x2 - x1) * (y2 - y1); order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9); order = order[1:][ovr <= thr]
    return keep
def postprocess(out, ratio):
    out = out.reshape(1, -1, out.shape[-1]).astype(np.float32)
    out[..., :2] = (out[..., :2] + GRIDS) * STRIDES; out[..., 2:4] = np.exp(out[..., 2:4]) * STRIDES
    p = out[0]; cls_ids = p[:, 5:].argmax(1); scores = p[:, 4] * p[np.arange(len(p)), 5 + cls_ids]
    m = scores > 0.3
    if not m.any(): return 0
    p, scores, cls_ids = p[m], scores[m], cls_ids[m]
    boxes = np.empty((len(p), 4), np.float32)
    boxes[:, 0] = p[:, 0] - p[:, 2] / 2; boxes[:, 1] = p[:, 1] - p[:, 3] / 2
    boxes[:, 2] = p[:, 0] + p[:, 2] / 2; boxes[:, 3] = p[:, 1] + p[:, 3] / 2
    boxes /= ratio
    return len(nms(boxes + cls_ids[:, None] * 4096.0, scores, 0.45))

rng = np.random.default_rng(1)
# 1) post-processing: 200 synthetic outputs with clusters of overlapping boxes
bad = 0
for t in range(200):
    o = np.zeros((8400, 85), np.float32)
    o[:, 0:2] = rng.uniform(-0.5, 0.5, (8400, 2)); o[:, 2:4] = rng.normal(0.5, 0.6, (8400, 2))
    o[:, 4] = rng.beta(0.3, 3, 8400); o[:, 5:] = rng.beta(0.3, 3, (8400, 80))
    hot = rng.choice(8400, rng.integers(5, 400), replace=False)
    o[hot, 4] = rng.uniform(0.5, 1, len(hot)); o[hot, 5 + rng.integers(0, 3, len(hot))] = rng.uniform(0.5, 1, len(hot))
    o.tofile(f"{TMP}/o.bin")
    ref = postprocess(o.copy(), 0.4)
    got = int(subprocess.check_output([BIN, "post", f"{TMP}/o.bin", "8400", "85", "0.4"]))
    bad += ref != got
print(f"post-processing: {200 - bad}/200 cases with identical n_det")

# 2) letterbox: 1600x900 textured image (smoothed noise + edges)
img = cv2.GaussianBlur(rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8), (0, 0), 3)
img[::50] = 255; img[:, ::70] = 0
img.tofile(f"{TMP}/img.raw")
print(subprocess.check_output([BIN, "lb", f"{TMP}/img.raw", "1600", "900", f"{TMP}/lb.raw"]).decode().strip())
ref, r = preprocess(img)
got = np.fromfile(f"{TMP}/lb.raw", np.float32).reshape(3, 640, 640)
d = np.abs(ref - got)
print(f"letterbox: r={r}, max difference {d.max():.0f}, differing pixels {np.mean(d > 0) * 100:.3f}%, "
      f"mean difference {d.mean():.4f} levels")
