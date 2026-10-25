"""nuScenes frame selection used in the paper (faithful copy of the original script).

Merges sweeps/ and samples/ of CAM_FRONT, splits them into continuous sequences (same log and
interval < 200 ms), sorts them from longest to shortest and copies whole sequences until
<n_frames> is reached. With nuScenes v1.0-mini and n_frames = 1500, this gives 7 sequences, 1652 frames.
Writes <destination>/origin.csv with the original file of each frame.

usage: python3 select_nuscenes.py <destination> <n_frames> <folder1> [folder2 ...]
e.g.:  python3 data_prep/select_nuscenes.py data/frames 1500 \\
         ~/nuscenes/sweeps/CAM_FRONT ~/nuscenes/samples/CAM_FRONT
"""
import os, shutil, sys
# usage: python3 select_nuscenes.py <destination> <n_frames> <folder1> [folder2 ...]
dst, target, srcs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
fs = [(f, d) for d in srcs for f in os.listdir(d) if f.endswith(".jpg")]
key = lambda x: (x[0].split("__")[0], int(x[0].split("__")[-1][:-4]))
fs.sort(key=key)
seqs, cur = [], [fs[0]]
for a, b in zip(fs, fs[1:]):
    (la, ta), (lb, tb) = key(a), key(b)
    if la == lb and 0 < tb - ta < 200_000:
        cur.append(b)
    else:
        seqs.append(cur); cur = [b]
seqs.append(cur)
seqs.sort(key=len, reverse=True)
os.makedirs(dst, exist_ok=True); n = 0
with open(os.path.join(dst, "origin.csv"), "w") as log:
    log.write("idx,original_file\n")
    for s in seqs:
        dts = [key(b)[1] - key(a)[1] for a, b in zip(s, s[1:])]
        for f, d in s:
            shutil.copy(os.path.join(d, f), os.path.join(dst, f"{n:06d}.jpg"))
            log.write(f"{n},{f}\n"); n += 1
        med = sorted(dts)[len(dts)//2] / 1000 if dts else 0
        print(f"sequence of {len(s)} frames, median interval {med:.0f} ms ({s[0][0].split('__')[0]})")
        if n >= target: break
print(f"total: {n} frames in {dst}")
