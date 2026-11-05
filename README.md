# AWF: Antarctic Wind Forecast Postprocessing

AWF accompanies *Quantile-Based Postprocessing of Antarctic Surface Wind
Forecasts: Probabilistic Skill and Spatial Transfer*. It compares Bernstein
quantile networks (BQN), quantile regression forests (QRF), and LightGBM
quantile models for Antarctic wind-speed forecasts. The study uses temporal,
held-station and forward held-station evaluations, each with a basic predictor
set and a terrain-augmented set. Gaussian EMOS provides the temporal reference.

BQN is implemented in **PyTorch**, QRF uses scikit-learn tree leaves, and the
boosted models use LightGBM. NumPy and SciPy provide numerical utilities.
Station wind observations are used without a height correction in the main
model comparison.

## Version

**Version 1.0.2 — 5 November 2026.**

The repository contains 65 scripts for data preparation, model fitting,
forecast evaluation and figures. The
[GitHub release](https://github.com/vhaoyang/awf/releases/tag/v1.0.2) and
`SHA256SUMS.txt` identify this source version.

## Contents

| Path | Purpose |
|---|---|
| `scripts/` | Acquisition, pairing, diagnostics, model fitting, scoring and figures |
| `docs/REPRODUCE.md` | Stage dependencies, external inputs and reconstruction instructions |
| `docs/SCRIPT_INDEX.csv` | Inventory and role of every included script |
| `docs/LIMITATIONS.md` | Reproduction limits and known issues in supporting routines |
| `requirements.txt` | Python package dependencies, including PyTorch |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | Apache License 2.0 |

## Environment and paths

Use Python 3.10 or later in a separate environment and install the packages
listed in `requirements.txt`. The dependencies are not a lockfile for the
original execution environment. GFS decoding also requires a working
ecCodes installation through the cfgrib stack. Shell utilities use Bash and
common Unix tools; use Linux or WSL for those scripts.

Run Python scripts from the repository root. Set `AWF_DATA_ROOT` to the input
data directory (default: `data`). Intermediate and generated results are
written under `outputs` in the current working directory. BQN selects CUDA
when available unless a device is explicitly supplied; this does not imply
identical numerical results across hardware or library versions.

Two optional settings specify auxiliary input locations:

- `AWF_SENSOR_TABLE`: the extracted AntAWS supplementary station table;
  default `<AWF_DATA_ROOT>/meta/essd_supp.txt`.
- `AWF_WINNER_ROOT`: a directory containing `winner_S.csv`, `winner_S1.csv`,
  `winner_F.csv` and `winner_F1.csv` for the station-comparison map;
  default `outputs/manuscript_ams/pit`.

## Data and experiment structure

This repository contains source code. It does not bundle upstream AntAWS
observations, GFS GRIB files, the REMA mosaic, fitted models, or every
intermediate experiment file. The separate study dataset contains archived
prediction outputs, summaries and diagnostic records. Its reserved identifier
is `10.5281/zenodo.23862371`; record publication and file access must be checked
separately. A reserved identifier is not evidence of accessible files.

The main comparison is the E10 stage. T, S and F denote the temporal,
held-station and forward held-station designs; T1, S1 and F1 add terrain
predictors. E1-E9 contain upstream processing and earlier experiments, some
of which supply inputs to E10. Their scores and figures should not be
substituted for the final comparison.

For each E10 method, seeds 17, 42 and 2026 form an equal-weight **CDF mixture**.
Quantiles, exceedance probabilities and CRPS are derived from that mixture.
The earlier three-quantile score and other equal-weight finite-quantile
scores are different quantities. See the reproduction guide for the scoring
and weighting conventions.

## Reproduction status

Version 1.0.2 has not been validated through a complete end-to-end run.
The dataset contains the archived outputs underlying the reported results.
Reconstructing them requires the matching input files, execution settings
and software environment described in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

Numerical refinement and selected reference checks support the reported
scoring precision; they do not certify an error bound for every prediction.
Nico remains in the study, and the cause of its BQN anomaly is not established
by the archived reconstruction alone. Internal configuration records do not
constitute external preregistration.

## Citation and license

Use `CITATION.cff` and cite the software version actually used. Cite the
separate dataset when using its archived outputs. A manuscript citation or
DOI should be added when available; this package does not assign one.

The source code is distributed under the [Apache License 2.0](LICENSE).
Upstream observations and other third-party inputs retain their own terms.
