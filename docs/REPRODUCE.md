# Reconstructing AWF experiments

The repository preserves several stages of the study. Reproduction requires
the inputs and conventions for the selected stage; running every script in
filename order is not a supported build procedure. Several diagnostic
scripts refit models or replace local output files.

## 1. Inputs and environment

Use a separate working directory and Python environment. Install the packages
in `requirements.txt`; these are unpinned dependencies, not the original
environment lockfile. Installation and end-to-end execution in a fresh
environment have not been validated for this release.

The study uses AntAWS v2 (DOI `10.48567/key7-ch19`), selected GFS 0.25-degree
forecast messages from `noaa-gfs-bdp-pds`, and the REMA v2 1 km mosaic.
Upstream access, filenames and terms must be checked before downloading.
The study's retrieved GFS initialization-date list contains 276 dates in
March-December 2021; do not infer availability for the whole calendar year.

Run from the source-tree root. The expected data layout is:

```text
<AWF_DATA_ROOT>/                 # defaults to ./data
  raw/antaws/                   # source CSVs and geographic files
  raw/gfs/                      # downloaded GRIB subsets
  raw/rema/                     # REMA mosaic and associated metadata
  meta/
    essd_supp.txt               # layout-preserving text of AntAWS Table S1
    gfs_2021_available_days.txt # explicit YYYYMMDD initialization dates
  derived/
    station_qc/                # station screen, registry and observation QC
    paired/                    # point forecasts and paired_2021.parquet
outputs/                       # relative to the current working directory
```

`build_sensor_registry.py` requires the supplementary table as extracted with
layout-preserving PDF text conversion. `AWF_SENSOR_TABLE` can specify another
location. The five explicit manual name matches in that script are retained.
Nominal sensor heights are metadata, not measured station heights.

Bash utilities additionally require `curl`, `grep`, `ps`, `pgrep`, and, for
the detached downloader, `setsid` and `nohup`. Create the `logs` directory
before using `launch_e2_full.sh`. Shell date probes require `python3`.
They may make network requests or start downloads; they are not passive checks.

## 2. Stages and dependencies

| Stage | Scripts | Inputs and purpose |
|---|---|---|
| Observation preparation | `e1_station_qc.py`, `build_sensor_registry.py`, `e1c_label_qc.py` | AntAWS records and Table S1; station screen, registry and label diagnostics |
| Forecast acquisition | `e2_gfs_download.py`, `e2b_dedup.py` | Explicit dates; GFS message selection, station interpolation and duplicate checks |
| Pairing | `e3_pair.py` | Prepared observations and station forecasts; keyed paired table |
| Exploratory diagnostics | `e4_diag.py`, `e4b_robust.py`, `e4c_figs.py` | Paired table and registry; descriptive error structure and nominal-height scenarios |
| Early baselines | `e5*.py` | Paired rows, GFS/REMA terrain inputs; deterministic and three-quantile comparisons |
| Early diagnostics | `e6*.py` | Baseline files; coverage, feature variants, threshold summaries and early figures |
| Intermediate distributions | `e7*.py`, `e8*.py`, `e9_derived.py` | Keyed predictions, terrain tables and score construction; E8 supplies the EMOS reference used by E10 |
| Main configuration record | `e10_protocol.py` | Paired data and registry; split, feature and candidate records |
| Main fits | `e10_run_T.py`, `e10_run_S.py`, `e10_run_F.py`, `e10_run_F1.py` | Matching split records, terrain tables and candidate settings; three-seed predictions |
| Main summaries | `e10_summarize.py` | Matching keyed prediction files; pooled, station-weighted and common-row summaries |
| Numerical diagnostics | `e10_audit_v303.py`, `e10_nico_*.py` | Working coefficients, predictions and reference files; see the execution notes below |
| PIT | `e10_pit_T.py`, `e10_figs_pit.py` | Temporal search results and archived predictions; refit identity check, PIT data and plots |
| Other E10 figures | `e10_figs_bqn.py`, `e10_figs_overview.py` | Architecture description, station registry and external winner tables |

`docs/SCRIPT_INDEX.csv` lists all 65 scripts. Module names such as
`e10_core.py`, `e10_models.py` and `e10_adaptive.py` provide shared functions.
The E7/E8 modules remain in the source tree because later scripts import them
and because they identify the construction of archived intermediate outputs.

The E10 root is `outputs/bqn_lgbm_qrf`. Its `protocol` directory holds the
configuration records; T/T1/S/S1/F/F1 hold the respective design outputs.
The terrain table is read from `outputs/review_v3/terrain_features_v3.csv`.
The temporal EMOS reference is read from
`outputs/review_v3/predictions_v3.parquet`. Matching files are prerequisites;
a directory name or script inventory alone does not supply them.

The separate dataset is an output archive. Its top-level prediction,
summary and diagnostic layout is not a complete copy of this runtime tree.
It must not simply be unpacked into `data` and treated as a ready-to-run
environment. The dataset's own README describes what is included.

## 3. Input-dependent commands

The following examples describe individual entry points, not a verified
one-command reconstruction. Supply the preceding inputs first.

```bash
python -m pip install -r requirements.txt
export AWF_DATA_ROOT=/path/to/prepared/awf-data

# Acquisition for an explicit, verified list of dates
python scripts/e2_gfs_download.py --days-file "$AWF_DATA_ROOT/meta/gfs_2021_available_days.txt" --cycle 0 --leads 0 6 12 24 48 72

# Inspect duplicate handling before modifying a working copy
python scripts/e2b_dedup.py --dry-run
```

Model runners accept the selections implemented in their entry points; read
the corresponding `sys.argv` handling before starting a long job. A model
script can train many folds and should not be used merely to inspect metadata.
Limits and thread settings are explicit in the source. Hardware and dependency
versions may affect timing and numerical results.

## 4. Scoring and sample conventions

- E10 predictions are method-specific equal-weight CDF mixtures over seeds
  17, 42 and 2026. They are not a mixture across the three methods, nor an
  average of the members' quantiles or CRPS values.
- CRPS uses `mean(E|X_c-y|) - sum(E|X_c-X_d|)/(2*m*m)` for m components.
  The pair term varies with each row's conditional distribution even though
  it does not depend directly on the observed y.
- QRF uses its weighted empirical distribution. LightGBM uses its complete
  piecewise-linear quantile distribution. Their discrete/segment formulas
  avoid quadrature-grid error but remain subject to floating-point arithmetic.
  BQN uses numerical inversion and integration. The 1e-4 m/s target is not
  a certified bound over all archived rows.
- Early CRPS3 and equal-weight finite-quantile scores are distinct from full
  CRPS. The E6 dense-quantile list has 22 levels despite legacy names
  containing `20`. E7 and E8 score definitions also differ.
- E3's `err_spd` is observation minus forecast. E4/E5 bias is forecast minus
  observation. Direction statistics require observed speed >= 2.0 m/s and
  a recorded direction. Main wind-speed models retain their scalar rows.
- Paired-row weighting, equal station weighting and common-success-row
  comparisons use different denominators; keep each with its matching
  prediction version. Forward summaries identify stations with at least
  50 test rows separately.
- Date-block bootstrap and early-stage random settings use seed 42 as
  specified in the scripts. Repeat blocks retain their multiplicity in
  the later block-bootstrap implementation.

## 5. Numerical diagnostics and figures

`e10_audit_v303.py`, `e10_nico_full.py`, `e10_nico_rows.py` and
`e10_pit_T.py` refit models. `e10_nico_rescore.py` scores stored coefficients
and updates working prediction files and summaries. `e10_nico_convbound.py`
updates a working diagnostic report and contains fixed historical totals.
Use isolated working copies and preserve the archived predictions
before running these programs.

The Nico refinement series compares numerical resolutions. Neither its
legacy `bound` field names nor sampled reference checks prove a rigorous
all-row error bound. Later reconstruction does not recover every historical
transformation input or establish an isolated causal explanation.

`e10_figs_overview.py` expects one winner table for each spatial design:
`winner_S.csv`, `winner_S1.csv`, `winner_F.csv`, `winner_F1.csv`. Each must
contain `station` and `winner`, with method labels `bqn`, `qrf`, or `lgbm`.
Point `AWF_WINNER_ROOT` at those tables, or use the default
`outputs/manuscript_ams/pit`. The package does not include those tables or
a standalone generator for them; do not invent rankings to fill this gap.
The figure label describes the Nico BQN anomaly without asserting that
missing terrain is its established cause.

If the PIT identity comparison fails, the diagnostic is written to
`outputs/manuscript_ams/pit/pit_identity_FAIL.json` and processing stops.

## 6. File identity

Run IDs and output filenames connect the scripts to the archived data.
Use each output with its matching configuration and scoring convention.
The source files are identified by the `v1.0.2` GitHub tag and
`SHA256SUMS.txt`; the dataset records identify the experimental outputs.
See `docs/LIMITATIONS.md` before reusing the supporting routines.
