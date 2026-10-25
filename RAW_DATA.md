# edgebench raw data

Raw measurement data for the paper "Invisible to Benchmarks: How Deployment Conditions Limit the
Real-Time Capacity of Edge AI Perception" (T. A. Rodolfo and A. C. S. Beck, IEEE ICIT 2027).
Code, scripts and the derived tables: https://github.com/tacrodolfo/invisible-to-benchmarks (these archives are
attached to its release v1.0.0)

## Archives

| Archive | Folder | Contents |
|---|---|---|
| `edgebench_raw_orin.tgz` | `results_orin/` | Jetson Orin Nano Super, 25 W, single boot: campaign of 16-17 Oct 2026, with the observer experiment (`observer/`) completed on 19 Oct 2026 (locked-clock runs and jtop sustainable-rate runs). 874 bench runs, 57 trtexec runs. |
| `edgebench_raw_nano.tgz` | `results_nano/` | Jetson Nano, MAXN (10 W), single boot: campaign of 16-17 Oct 2026. 425 bench runs, 15 trtexec runs. |
| `edgebench_raw_fan_quiet.tgz` | `results_fan_quiet/` | Orin Nano Super: YOLOX-s factorial at 12 Hz repeated with the factory fan profile (`quiet`). 13 bench runs, 3 trtexec runs. |

## Files of one run (`<tag>.*`)

| File | Contents |
|---|---|
| `.csv` | one row per frame: release, start and per-stage host timestamps (ns, CLOCK_MONOTONIC), drop flag, number of detections, per-stage GPU durations (cudaEvent) |
| `.cond` | experimental condition (`key=value`): experiment, model, power mode, CPU/GPU clock cell, release model, co-runner, monitors, idle states, EMC lock and frequency, fan profile, temperature, boot id |
| `.meta.json` | effective bench arguments, measurement window on both clocks, summary line |
| `.clocks` | governors and frequencies read before the run |
| `.log` | console output of the bench or trtexec |
| `.tegrastats`, `.tegrastats.t0` | tegrastats log (1 s by default) and its real-time start |
| `.probe.csv` | 1 ms sysfs samples (GPU frequency and load, CPU frequency, input power, EMC), only in probe runs |
| `.json` | trtexec `--exportTimes` output (trtexec runs) |
| `<base>.rate` | sustainable-rate search result: highest passing and lowest failing rate and their miss rates |
| `quick/` | short runs of the binary search |
| `platform/` | platform state logged before and after each tier |

## Regenerating the tables

```bash
git clone https://github.com/tacrodolfo/invisible-to-benchmarks.git && cd edgebench
tar xzf edgebench_raw_orin.tgz
python3 analysis/analyze_cpp.py --res results_orin
```
The tables are written to `results_orin/tables/` and match the ones in the repository.

## Notes

- The Orin campaign started with the factory fan profile (`quiet`), which changed to `cool` during
  tier 1; most Orin runs used `cool` (field `fan=` in each `.cond`). `results_fan_quiet/` is the
  control with `quiet`: mean response times differ by at most 0.03 ms and the P99 by at most 0.09 ms.
- The campaigns were run with a build of the scripts whose console messages and platform logs were in
  Portuguese (for example `passada 1/3 concluída` = `pass 1/3 done`, `salvo em` = `saved to`,
  `campanha` = `campaign`, `prioridade` = `tier`, `modo` = `mode`, `n/d` = `n/a`). The files were
  kept exactly as recorded; the measurement code and the data formats are the same as in the
  public repository.
- File paths inside the logs refer to the measurement boards.

## License

CC BY 4.0. The nuScenes frames are not included (CC BY-NC-SA 4.0, https://www.nuscenes.org/).
