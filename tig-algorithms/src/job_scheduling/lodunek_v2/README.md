# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** lodunek_v2
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n=50,s=flow_shop | 5T | `{"aim_target": T}` |
| n=50,s=job_shop | 5T | `{"aim_target": T}` |
| n=50,s=hybrid_flow_shop | 5T | `{"aim_target": T}` |
| n=50,s=fjsp_medium | 5T | `{"aim_target": T}` |
| n=50,s=fjsp_high | 5T | `{"aim_target": T}` |

- `aim_target`: the qualifying quality T of the track, as a fraction. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (101437 on chain is 0.101437).
  Update it when the line moves by more than the stop margin. A stale value costs qualifiers both ways: above
  the line it gives up nonces that would qualify, below it stops nonces short of the line. A missing or
  non-numeric value takes the default, the line expected in rounds 144-145 (from the lines up to block 1401853):

  | Track | flow_shop | job_shop | hybrid_flow_shop | fjsp_medium | fjsp_high |
  |---|---|---|---|---|---|
  | default `aim_target` | 0.0440 | 0.1030 | 0.1020 | 0.1585 | 0.0803 |
  | stop margin | 0.010 | 0.004 | 0.006 | 0.010 | 0.008 |

- No other keys are read. ember_vesuvio reads none either; the track is detected from the instance.

## Changes from ember_vesuvio

ember_vesuvio (c007_a052) with early stopping on every track. A bundle qualifies when its median nonce quality
reaches T, so a nonce stops once it is clearly above T, or once T is out of reach. On flow_shop the search
also stops sooner once a schedule is clearly above T. On hybrid_flow_shop a nonce that ends just under T runs
its search once more from its best schedule. When `aim_target` is sent, a job_shop nonce
that ends just under T reruns its deep and final searches from its best schedule, up to eight times with fresh
tie-breaking, until it reaches T. Otherwise the engines, per-track routing and work ceilings are
ember_vesuvio's, and runs stay deterministic.

## Results

Same fresh nonces for both codes, runs interleaved, official tig-runtime plus tig-verifier CPU per nonce, ember_vesuvio
`null`. Qualifiers per compute = (nonces at or above the line) ratio / compute ratio.

| Track | flow_shop | job_shop | hybrid_flow_shop | fjsp_medium | fjsp_high |
|---|---|---|---|---|---|
| Compared with | ember_vesuvio | ember_vesuvio | ember_vesuvio | ember_vesuvio | ember_vesuvio |
| Lines at block | 1404321 | 1404321 | 1404321 | 1401812 | 1401812 |
| `aim_target` | 0.042954 | 0.099882 | 0.097578 | 0.1560 | 0.0805 |
| Nonces | 48 | 48 | 48 | 48 | 48 |
| At or above the line | 14 / 14 | 20 / 20 | 24 / 23 | 29 / 30 | 25 / 25 |
| Compute ratio | 0.85 | 0.49 | 0.42 | 0.64 | 0.90 |
| Qualifiers per compute | 1.18 | 2.06 | 2.48 | 1.52 | 1.12 |

## References and Acknowledgments

### Academic Papers
- Jackson, *"Scheduling a Production Line to Minimize Maximum Tardiness"*, Research Report 43, Management Science Research Project, UCLA, 1955 - preemptive one-machine bound.
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6 - heads and tails of the one-machine relaxation.
- The references of ember_vesuvio apply to its engines.

### Code References
- ember_vesuvio (c007_a052), FP Labs: every file but `aim.rs`, with the stopping hooks added. Copyright of the ember_vesuvio files remains with FP Labs and its donors (ember_etna, FP Labs; task_tree_h and task_tree_j, NVX; shuttle_bell).
- lodunek (tilk): the stopping in `aim.rs` and `mod.rs`.
- TIG baseline (job_scheduling/baselines/dispatching_rules.rs).
- ember_stromboli (c007_a040) also stops its search once more work does not pay.

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
