#!/usr/bin/env python3
"""Prepares a frame sequence from another dataset for the bench (JPEG, same size, in order).

Usage: python3 scripts/prep_frames.py SOURCE DESTINATION [--quality 90] [--size 1600x900] [--limit 1652]
  SOURCE: folder with .jpg/.png/.bmp of a continuous sequence (alphabetical order = temporal order)
  --size: resizes (optional). Without it, keeps the original resolution.
E.g.: python3 scripts/prep_frames.py /data/rellis/00000/pylon_camera_node data/frames_rellis --quality 90
     FRAMES=data/frames_rellis RES=$PWD/results_rellis ./scripts/11_rates.sh s seq
"""
import argparse
import glob
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--size", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp") for f in glob.glob(os.path.join(a.src, f"*.{ext}")))
    if a.limit:
        files = files[: a.limit]
    os.makedirs(a.dst, exist_ok=True)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    shape = None
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise SystemExit(f"different size in {f}: {img.shape} != {shape}")
        cv2.imwrite(os.path.join(a.dst, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    print(f"{len(files)} frames {shape[1]}x{shape[0]} in {a.dst} (JPEG q={a.quality})")


if __name__ == "__main__":
    main()
