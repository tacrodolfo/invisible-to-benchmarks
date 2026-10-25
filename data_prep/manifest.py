#!/usr/bin/env python3
"""SHA-256 manifest of the data (frames, ONNX), to prove that a reproduction uses the same files.

  python3 data_prep/manifest.py create data/frames data/frames.sha256
  python3 data_prep/manifest.py verify data/frames data/frames.sha256
The (small) manifest is versioned in git; the data are not.
"""
import hashlib
import os
import sys


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listing(d):
    exts = (".jpg", ".jpeg", ".png", ".onnx")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts) and os.path.isfile(os.path.join(d, f)))


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("create", "verify"):
        sys.exit(__doc__)
    cmd, d, man = sys.argv[1:]
    if cmd == "create":
        with open(man, "w") as m:
            for f in listing(d):
                m.write(f"{digest(os.path.join(d, f))}  {f}\n")
        print(f"{len(listing(d))} files in {man}")
        return
    want = dict(reversed(l.split()) for l in open(man) if l.strip())
    have = set(listing(d))
    miss = sorted(set(want) - have)
    bad = [f for f in sorted(set(want) & have) if digest(os.path.join(d, f)) != want[f]]
    extra = sorted(have - set(want))
    print(f"{len(want)} in the manifest: {len(miss)} missing, {len(bad)} different, {len(extra)} extra")
    sys.exit(1 if miss or bad else 0)


if __name__ == "__main__":
    main()
