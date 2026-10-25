# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** lodunek
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
| n=50,s=fjsp_high | 5T | `null` |

- `aim_target`: the qualifying quality T of the track, as a fraction. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (101437 on chain is 0.101437).
  Update it when the line moves by more than 0.001. A stale value costs qualifiers both ways: above the
  line it gives up nonces that would qualify, below it stops nonces short of the line. A missing or
  non-numeric value takes the default, the line expected in rounds 143-145 (from the lines up to block 1397625):

  | Track | flow_shop | job_shop | hybrid_flow_shop | fjsp_medium | fjsp_high |
  |---|---|---|---|---|---|
  | default `aim_target` | 0.0500 | 0.1025 | 0.1000 | 0.1565 | not used |

- No other keys are read: ember_opt's `track` and etna_highcap_fast's `v1lab` are not needed, the track is
  detected from the instance.

## Changes from ember_etna

ember_etna (c007_a044) with early stopping, the job_shop engine and the 200e9 fjsp_high work cap (ember_etna:
65e9) of etna_highcap_fast (c007_a049) and the hybrid_flow_shop engine of ember_opt (c007_a048). A bundle qualifies when its median nonce quality reaches
T, so a nonce stops once it is clearly above T, or once T is out of reach. fjsp_high runs etna_highcap_fast without stopping.

The track is read from the instance (one machine route shared by all jobs: flow_shop or hybrid_flow_shop;
otherwise job_shop, fjsp_medium or fjsp_high by flexibility). Runs stay deterministic.

## Results

Default settings, same fresh nonces for both codes, runs interleaved, tig-runtime plus tig-verifier CPU per
nonce. Qualifiers per compute = (nonces at or above T) ratio / compute ratio.

| Track | flow_shop | job_shop | hybrid_flow_shop | fjsp_medium | fjsp_high |
|---|---|---|---|---|---|
| Compared with | ember_etna | etna_highcap_fast | ember_opt | ember_etna | etna_highcap_fast |
| Nonces | 200 | 60 | 60 | 48 | - |
| At or above T | 53 / 53 | 27 / 27 | 22 / 22 | 22 / 22 | - |
| Compute ratio | 0.81 | 0.53 | 0.59 | 0.82 | 1.00 |
| Qualifiers per compute | 1.23 | 1.87 | 1.68 | 1.22 | 1.00 |

## References and Acknowledgments

### Academic Papers
- Jackson, *"Scheduling a Production Line to Minimize Maximum Tardiness"*, Research Report 43, Management Science Research Project, UCLA, 1955 - preemptive one-machine bound.
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6 - heads and tails of the one-machine relaxation.
- The references of ember_etna, etna_highcap_fast and ember_opt apply to their engines.

### Code References
- ember_etna (c007_a044), FP Labs. Copyright of the ember_etna files remains with FP Labs.
- etna_highcap_fast (c007_a049), Fred: `job_shop_engine.rs` and the fjsp_high work cap in `mod.rs`. Copyright remains with Fred and its donors (FP Labs, NVX, stinger).
- ember_opt (c007_a048), testing: `hybrid_flow_shop.rs`, `ember_hybrid_engine.rs`, `ember_hfs_construct.rs`, `types.rs`, `preprocess.rs`, `infra_shared.rs`. Copyright remains with testing.
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
