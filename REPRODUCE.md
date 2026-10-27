# Reproduction guide

Full chain: data → features → models → manuscript numbers.
Environment: conda (`python` 3.11+ with pandas / numpy / scikit-learn /
lightgbm / torch+GPU for BQN). **LightGBM jobs must run serially**
(deterministic mode conflicts under parallel stacking).

Random seeds are frozen in each script; official predictions are equal-weight
three-seed CDF mixtures (seeds 17 / 42 / 2026 for the model comparison,
seed 42 for the single-model baseline chain).

## Stage A — data pipeline (AWF E1–E3)

| # | Script | Output |
|---|---|---|
| 1 | `scripts/e1_station_qc.py` | usable-station registry (Table 1) |
| 2 | `scripts/build_sensor_registry.py` | sensor metadata, terrain class |
| 3 | `scripts/e1c_label_qc.py` | observation label QC |
| 4 | `scripts/e2_gfs_download.py` | GFS 0.25° point forecasts, 276 days |
| 5 | `scripts/e2b_dedup.py` | download-integrity report |
| 6 | `scripts/e3_pair.py` | `paired_2021.parquet` (station-hour rows) |

External datasets: AntAWS v2 observations (doi:10.48567/key7-ch19);
NOAA GFS open bucket (`noaa-gfs-bdp-pds`, idx+Range partial fetches);
REMA v2 1-km mosaic (SHA256 recorded in the project registry).

## Stage B — diagnostics (E4)

`scripts/e4_diag.py`, `e4b_robust.py`, `e4c_figs.py` → bias structure,
height scenarios, Figures 2–3.

## Stage C — three-model protocol (E10, the manuscript's main numbers)

| # | Script | What it runs |
|---|---|---|
| 1 | `e10_core.py` | scoring library: exact decomposition CRPS, quantiles, grids |
| 2 | `e10_models.py` | BQN / QRF / LightGBM22 wrappers, mixture scoring |
| 3 | `e10_protocol.py` | frozen feature sets, station hashing, splits |
| 4 | `e10_run_T.py` | temporal design: 8-candidate search + official mixtures |
| 5 | `e10_run_S.py` / `e10_run_F.py` | LOO 50 folds / forward 42 stations (presets) |
| 6 | `e10_run_F1.py` | terrain ablation T1/S1/F1 (same presets, 21 features) |
| 7 | `e10_adaptive.py` | support-aware adaptive scoring for extreme BQN folds |
| 8 | `e10_nico_full.py`, `e10_nico_rows.py`, `e10_nico_rescore.py`, `e10_nico_convbound.py` | full scoring of the extreme near-South-Pole folds, precision evidence |
| 9 | `e10_summarize.py` | authoritative per-design summaries (pooled + station-equal) |
| 10 | `e10_audit_v303.py` | cross-design scoring-precision audit |

## Stage D — manuscript

`paper/manuscript.tex` (AMS WAF class, compiled with Tectonic; any pdfLaTeX
toolchain with the AMS v6.1 class works). All numeric tables carry their
source file in the repository delivery manifests.

## Conventions used when quoting numbers

- Denominators: pooled = all station-hour rows; station-equal = unweighted
  station means; the two are reported side by side and can invert rankings.
- Error sign: `err_spd = obs − fcst` in Stage A; forecast-minus-observed in
  diagnostics. Check each table's caption.
- LOO uses all seasons (spatial generalization); temporal designs use fixed
  station sets — cross-design differences are descriptive, never tested.
