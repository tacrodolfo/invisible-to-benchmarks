# data/ (outside git, except this README, the manifest and origin.csv)

| Folder or file | Contents | Source |
|---|---|---|
| `engines/` | `yolox_{s,m,l}_fp16.engine` | built on the device by `setup.sh --install` (`data_prep/build_engines.sh`) |
| `onnx/` | official YOLOX ONNX models (release 0.1.1rc0) | downloaded by `build_engines.sh` |
| `frames/` | 1652 CAM_FRONT frames of nuScenes v1.0-mini (7 sequences) | `NUSCENES=... ./scripts/setup.sh --install` (`data_prep/select_nuscenes.py`) |
| `frames.sha256` | SHA-256 of each frame used in the paper | versioned; setup checks the frames against it |
| `origin.csv` | original nuScenes file of each frame | versioned |
| `py_results/` | CSVs of the earlier Python harness | optional, only with `DATA_SRC`; used by the equivalence check |
