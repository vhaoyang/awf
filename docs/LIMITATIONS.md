# Reproduction limits and supporting routines

The source tree preserves the full archived workflow. Completeness of source
files is different from completeness of runtime inputs or validation of a
new environment. Version 1.0.2 has not been executed end to end.

## External inputs and runtime provenance

The original dependency versions, fitted-model checkpoints, all intermediate
arrays, the extracted AntAWS supplementary table and station winner tables
are not all bundled here. A fresh installation of unpinned dependencies is
not guaranteed to reproduce archived values. Record exact environment and
input identities before any independent reconstruction.

The source archive contains some scripts with top-level fitting or scoring
code. Importing them can start work. Read the source before importing a runner.
The Nico reconstruction and rescoring programs can update working output
files. Their backups are not a substitute for preserving the published data.

## Known issues in supporting routines

1. `e6d_fairbase.py::_crps_lognorm` is an unused helper. Its weights have an
   incompatible length relative to the quantile grid, and its expression is
   not the general CRPS identity. No call to the helper was found in the
   included source. It has been retained and labelled for provenance; do not
   use it as a scoring function without a separate correction and validation.
   E10 does not import this module.
2. The early `e6e_densq.py` probability interpolation sorts quantile values
   together with their tau labels. This is not the later fixed-tau monotone
   rearrangement, and crossings can invalidate an interpretation as a
   monotone CDF. Its finite-quantile score has 22 levels; legacy `20` names
   do not change the implemented denominator. The later complete distribution
   in `e8_core.py` is used by the E10 LightGBM implementation.
3. The first constructed-vector loop in `e7_terrain.py` derives its downhill
   direction from the wind vector itself; it is not an independent validation
   of terrain extraction. The separate coordinate-chain comparison is more
   limited than testing the full DEM pipeline. E10 uses the E8 terrain table.
4. GFS spot probes and a binary search over an assumed archive transition
   cannot establish complete temporal coverage. `probe_s3_list.py` is an
   unpaginated inspection helper; its count can be a single response page.
   The paginated monthly listing and explicit date files have a different role.

These issues were identified through static inspection. Their relevance
to a reported result depends on which routines, inputs and outputs were
used; their presence alone does not establish an error in the main results.

## Interpretation of the main results

Numerical convergence checks are diagnostics rather than universal error
bounds. Internal configuration records are not external preregistration and
do not establish the chronology of all data access. The Nico station remains
in the reported comparisons; its anomaly should not be removed or given a
causal explanation merely to simplify the source release.
