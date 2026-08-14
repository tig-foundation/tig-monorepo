# Archived CUR redesign

This document described the former nested-matrix CUR proposal and is retained
only as a historical marker. That proposal has been superseded.

The current design uses eight seed-dependent shared-basis matrices, eight
stratified true-rank/target-rank-ratio pairs, independently perturbed spectra,
random left/right singular-vector re-pairing, and the continuous logarithmic
score.

Use these current sources:

- docs/cur.tex — authoritative mathematical design
- tig-challenges/src/cur_decomposition/OVERVIEW.md — implementation overview
- tig-challenges/src/cur_decomposition/mod.rs — generator and verifier logic
- tig-challenges/src/cur_decomposition_scoring.rs — scoring and aggregation
