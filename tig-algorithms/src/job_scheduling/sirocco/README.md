# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** sirocco
* **Copyright:** 2026 Fred; donor copyrights remain with FP Labs, NVX, stinger, 0xfe2a and 0x1884
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n=50,s=fjsp_high        | 5T | null |
| n=50,s=fjsp_medium      | 5T | null |
| n=50,s=flow_shop        | 5T | null |
| n=50,s=hybrid_flow_shop | 5T | null |
| n=50,s=job_shop         | 5T | null |

Leave hyperparameters null. The track is detected from the instance, and the code routes each track to one of two engine families, `etna_lexwork_v3`'s (`c007_a055`, the submitter's previous code) or `ember_vesuvio`'s (`c007_a052`, FP Labs), according to the fuel left; every engine is stopped once its saved makespan reaches a lower bound of the instance. The solver is anytime and deterministic: a valid schedule is saved as soon as one exists and replaced only by a better one, so an interrupted run returns the best schedule saved so far. There is no clock, thread, environment variable or filesystem access; the only extern is the fuel counter `__fuel_remaining`, and all pseudo-randomness derives from `challenge.seed`.

**CPU to expect at 5e12 fuel (median per nonce, one core, measured):** fjsp_high ~300 s, job_shop ~140 s (up to ~200 s), fjsp_medium ~200 s (up to ~210 s), hybrid_flow_shop ~70 s (up to ~130 s), flow_shop ~2 s. Every track finishes well below the 5e12 grant. Benchmarkers running several slaves should set `time_before_batch_retry` above the duration of a batch and keep the submission delay short, since a benchmark's active life starts at `proof + 1.2 x submission_delay` and ends 120 blocks after its start.

## Local Benchmark

TIG runtime and verifier 0.0.8, round-139 track settings at 5e12 fuel, null hyperparameters, 48 fresh nonces per track, paired per nonce: mean quality difference with its 95 % Student CI, median CPU time ratio in parentheses, "identical k/n" = the same solution on k nonces. Bold: CI above zero. The binary was built from these files in a fresh copy of the public repository with a one-line `job_scheduling` registry; this README was completed after the benchmark (text only). The fjsp_medium row was measured with this binary on nonces 4500-4547; the other rows on nonces 4400-4447 with the build of the same sources under the previous working name (this binary reproduces its solutions on 8 identity runs across those four tracks, and reproduces `etna_lexwork_v3`'s solutions on 16 fjsp_medium runs below its fuel threshold).

| Track | vs `etna_lexwork_v3` (a055) | vs `ember_vesuvio` (a052, published binary) |
|---|---|---|
| fjsp_high | identical solutions 48/48 (x1.00) | **+2 220 [+1 828 ; +2 613]** (x7.01) |
| fjsp_medium | **+7 343 [+5 032 ; +9 655]** (x1.34) | **+3 362 [+1 961 ; +4 764]** (x1.99), identical 2/48 |
| flow_shop | **+323 [+118 ; +528]** (x0.91), identical 6/48 | identical 48/48 |
| hybrid_flow_shop | +351 [-702 ; +1 404] (x0.60) | **+879 [+392 ; +1 367]** (x1.48), identical 28/48 |
| job_shop | **+1 389 [+717 ; +2 062]** (x0.77) | **+3 547 [+2 696 ; +4 399]** (x5.35) |

## Structure

| File | Role |
|------|------|
| `mod.rs` | Entry point, track detection, routing, lower-bound stops, and `etna_lexwork_v3`'s fjsp pool search (fjsp_high; fjsp_medium at small fuel) |
| `lab.rs` | Settings, thresholds and lower bounds |
| `vesuvio_fjsp_pool.rs` | `ember_vesuvio`'s fjsp pool search (fjsp_medium) |
| `vesuvio_hybrid_engine.rs`, `vesuvio_hfs_construct.rs` | `ember_vesuvio`'s hybrid_flow_shop construction and incremental tabu search |
| `vesuvio_job_shop_engine.rs` | `ember_vesuvio`'s job_shop engine |
| `vesuvio_flow_shop_engine.rs` | `ember_vesuvio`'s flow_shop engine |
| `hybrid_engine.rs`, `hfs_construct.rs`, `types.rs`, `preprocess.rs`, `infra_shared.rs`, `ember_hybrid_engine.rs`, `ember_hfs_construct.rs`, `hybrid_flow_shop.rs` | `etna_lexwork_v3`'s hybrid_flow_shop pipeline (small fuel) |
| `job_shop_engine.rs`, `flow_shop_engine.rs`, `gate_floor.rs`, `t47.rs` | `etna_lexwork_v3`'s other engines, reachable through the study keys only |

## References and Acknowledgments

- `ember_vesuvio` (`c007_a052`) by FP Labs (copyright 2026 FP Labs), published under the TIG Innovator Outbound Game License, UAI null: its fjsp pool search (with the load-balancing assignment of Kacem, Hammadi & Borne, DOI 10.1109/TSMCC.2002.1009117), hybrid_flow_shop construction and incremental tabu, job-shop engine and flow-shop engine are reused as `vesuvio_*.rs`. `ember_vesuvio` itself derives from `ember_etna` (`c007_a044`, FP Labs); portions of its flow and job-shop engines derive from `task_tree_h` (`c007_a032`) and `task_tree_j` (`c007_a036`) by NVX; its `hfs_construct.rs` derives from the hybrid_flow_shop construction phase of `task_tree_j`; its `hybrid_engine.rs` derives from `shuttle_bell` (`c007_a037`); its `gate_floor.rs` replicates the TIG baseline `dispatching_rules`. References it cites: Pearce & Kelly (DOI 10.1145/1187436.1210590), Glover & Laguna (DOI 10.1007/978-1-4615-6089-0), Nowicki & Smutnicki (DOI 10.1287/mnsc.42.6.797), Taillard (DOI 10.1287/ijoc.6.2.108 and 10.1016/0377-2217(90)90090-X), Dueck (DOI 10.1006/jcph.1993.1010), Feo & Resende (DOI 10.1007/BF01096763), Nawaz, Enscore & Ham (DOI 10.1016/0305-0483(83)90088-9), Ruiz & Stützle (DOI 10.1016/j.ejor.2005.12.009), Johnson (DOI 10.1002/nav.3800010110), Palmer (DOI 10.1057/jors.1965.8), Campbell, Dudek & Smith (DOI 10.1287/mnsc.16.10.B630), Carlier (DOI 10.1016/S0377-2217(82)80007-6), Adams, Balas & Zawack (DOI 10.1287/mnsc.34.3.391).
- `ember_etna` (`c007_a044`) and `ember_stromboli` (`c007_a040`) by FP Labs, with components credited in their READMEs to NVX and stinger: the fjsp_high pool search of this code is `etna_lexwork_v3`'s evolution of ember_etna's.
- `etna_highcap` / `etna_highcap_fast` / `etna_lexwork` / `etna_lexwork_v3` (`c007_a046` / `c007_a049` / `c007_a053` / `c007_a055`) by the submitter.
- Hybrid pipeline of `etna_lexwork_v3` from `ember_opt` (`c007_a048`) by 0xfe2a (TIG Innovator Outbound Game License, UAI null), whose constructive portfolio comes from `new_adaptive_js` (`c007_a045`, 0x1884): kept in this package (`types.rs`, `preprocess.rs`, `infra_shared.rs`, `ember_hybrid_engine.rs`, `ember_hfs_construct.rs`, `hybrid_flow_shop.rs`), reachable through the study keys only.
- Lexicographic workload criterion: Gao, Sun & Gen, Computers & Operations Research 35 (2008). Lower bounds: Carlier's one-machine bound with heads and tails (DOI 10.1016/S0377-2217(82)80007-6).

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
