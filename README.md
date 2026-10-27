# AWF — Probabilistic Postprocessing of Antarctic Surface Wind Speed

Code and manuscript for a three-model comparison of quantile postprocessing
methods for Antarctic near-surface wind speed, driven by GFS 0.25° forecasts
and 50 automatic weather station observations over 276 days of 2021.

**Models (identical inputs, three random seeds, official prediction = equal-
weight CDF mixture of seeds):**

- **BQN** — Bernstein quantile network (degree-12 nonnegative, nondecreasing
  coefficients; GPU training, early stopping on validation CRPS)
- **QRF** — quantile regression forest (500 trees, native weighted empirical
  distributions)
- **LightGBM** — gradient boosting with a 22-quantile grid and monotone
  rearrangement
- **EMOS** — classical reference

**Evaluation designs (frozen protocol):**

| Design | Task | Train → Test |
|---|---|---|
| T / T1 | temporal | Mar–Aug → Oct–Dec 2021 |
| S / S1 | leave-one-station-out (50 folds) | spatial generalization |
| F / F1 | forward unseen stations (42 stations) | transfer without station history |

Base features (13 columns) for T/S/F; +8 REMA terrain features (21 columns)
for T1/S1/F1. Scoring: exact decomposition CRPS = E|X−y| − ½E|X−X′| with
piecewise-exact integration for discrete components (QRF, LightGBM) and a
support-aware adaptive scheme for extreme-support BQN folds. Paired
date-block bootstrap (2,000 resamples) for in-design uncertainty.

## Key results (pooled CRPS, official seed mixtures)

| Design | BQN | QRF | LightGBM | EMOS |
|---|---|---|---|---|
| T (temporal) | 1.3026 | **1.2438** | 1.2443 | 1.3196 |
| S (LOO, pooled / station-equal) | **1.4956** / 1.5358 | 1.5051 / **1.5249** | 1.5217 / 1.5285 | — |
| F (forward) | 1.7444 | **1.4041** | 1.4386 | — |
| S1 (LOO + terrain, full network†) | 509.52 / 391.83 | 1.4633 / 1.4800 | 1.4698 / 1.4914 | — |
| F1 (forward + terrain, full network†) | 15325.34 / 14885.41 | 1.4146 / 1.4280 | 1.4149 / 1.4358 | — |

† The full-network BQN values in S1/F1 are dominated by a single station near
the South Pole (latitude −89°) whose missing-terrain extrapolation yields
valid but extremely poor predictions under the frozen protocol; the manuscript
reports same-key Nico-excluded sensitivity subsets alongside (BQN 1.5029 in
S1, 1.5351 in F1). Method rankings depend on the transfer scenario, the
evaluation weighting, and the handling of one missing-covariate station — see
the manuscript for the full analysis.

## Repository layout

```
scripts/   full pipeline (Python): data QC, GFS download, pairing,
           diagnostics, three-model protocol (e10_*), scoring, audits
CITATION.cff   citation metadata (GitHub "Cite this repository")
```

The manuscript itself is submitted to *Weather and Forecasting* through the
editorial system and is not part of this repository; a preprint link will be
added here once posted. All numbers quoted above are reproducible from
`scripts/` (see REPRODUCE.md).

## Reproducing

See [REPRODUCE.md](REPRODUCE.md) for the ordered run list (data → features →
models → manuscript numbers), external datasets (AntAWS v2, NOAA GFS bucket,
REMA v2 1-km mosaic with registry SHA256), and the frozen random seeds.
Environment: conda; LightGBM jobs must run serially to preserve determinism.

## Data and code availability

The keyed per-station predictions, per-design summaries, audit records, and
checksum manifests are archived on Zenodo (DOI in the manuscript's data
availability statement). Station observations are redistribution-restricted
and available through AntAWS v2 (doi:10.48567/key7-ch19).

## Citation

```bibtex
@misc{awf2026,
  author = {Yang, Hao and Li, Xiang and Zhang, Lei and Wang, Kai and Wang, Yanni},
  title  = {AWF: Probabilistic postprocessing of Antarctic surface wind speed
            (BQN/QRF/LightGBM comparison)},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). Manuscript text: © the authors.
