# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** etna_lexwork_v3
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

Leave hyperparameters null. The track is detected from the instance itself; no hyperparameter is needed to route.

## What changes against `etna_lexwork` (`c007_a053`)

`etna_lexwork_v3` keeps the engines of `etna_lexwork` (faster fjsp tabu, lexicographic makespan / total-work criterion with plateau moves on fjsp_high, single-route job shop routing) and adds two things:

1. **On hybrid_flow_shop**, with at least 1e11 fuel left when the hybrid engine starts, the constructive pipeline of `ember_opt` (`c007_a048`, 0xfe2a: the portfolio of `new_adaptive_js`, `c007_a045`, 0x1884, with its local searches and short tabu runs) replaces ember_etna's GRASP construction before ember_etna's graph tabu search (FP Labs), which keeps its 2e11 budget. Below 1e11 the engine of `etna_lexwork` runs unchanged.
2. **Two effort caps are raised to use the fuel benchmarkers actually give.** `etna_lexwork` stops its fjsp_high search at 2e11 fuel and each job-shop tabu search after 300 000 iterations, while the network recipe is a 5e12 fuel budget. In `etna_lexwork_v3`: fjsp_high 4e11, and 1 000 000 iterations per job-shop tabu search. fjsp_medium keeps its 2e11 cap (raising it to 4e11 was measured and rejected: +1 374 [-175 ; +2 923] for twice the CPU) and flow_shop is unchanged. With a smaller fuel budget the caps shrink to `fuel - fuel/64` as before (the solver is anytime).

Why these values: on these tracks the mean quality of these engines grows about linearly with the logarithm of the effort, so the caps, not the algorithms, were the first limit. A cap was raised only where the expected number of qualifying bundles per CPU hour rises against `etna_lexwork` under the network's bundle statistics (40 nonces per bundle, qualification far in the upper tail), not where the mean quality alone is highest.

**CPU cost to expect at 5e12 fuel (median per nonce, one core, measured):** fjsp_high ~300 s, fjsp_medium ~150 s, job_shop ~180 s, hybrid_flow_shop ~110 s, flow_shop ~2 s. Benchmarkers running several slaves should set `time_before_batch_retry` above the duration of a batch, and keep the submission delay short (smaller benchmarks or more cores per benchmark), since a benchmark's active life starts at `proof + 1.2 x submission_delay` and ends 120 blocks after its start.

Files: `mod.rs` (routing, constants, `help()`), `flow_shop_engine.rs`, `gate_floor.rs`, `hfs_construct.rs`, `hybrid_engine.rs`, `job_shop_engine.rs` (one constant: `job_shop_iters`), `lab.rs` and `t47.rs` are `etna_lexwork`'s. `types.rs`, `preprocess.rs`, `infra_shared.rs`, `ember_hybrid_engine.rs` and `ember_hfs_construct.rs` are byte-identical to `ember_opt`'s; `hybrid_flow_shop.rs` differs from it by the `total_floor` parameter. The solver is anytime and deterministic, without clock, thread, environment variable or filesystem access; the only extern is the fuel counter `__fuel_remaining`, and all pseudo-randomness derives from `challenge.seed`. Hyperparameters: leave null; the optional study keys under `"v1lab"` (see `help()`) only lower or raise the same caps for measurement.

## Local Benchmark

TIG runtime and verifier 0.0.8, round-139 track settings at 5e12 fuel (the recipe of the largest benchmarker), null hyperparameters (`ember_opt` and `ember_opt_v2` with `{"track": <track>}`), 48 fresh nonces per track, paired per nonce: mean quality difference with its 95 % Student CI, median CPU time ratio in parentheses. 1 200 runs, 0 invalid, 4/4 reruns identical. The binary was built from these files in a fresh copy of the public repository with a one-line `job_scheduling` registry. The fjsp_high, job_shop, hybrid_flow_shop and flow_shop rows were measured with a build that differs from this one only by the fjsp_medium cap (4e11 instead of 2e11); this binary reproduces its solutions on 16 identity runs. On fjsp_medium this binary gives the solutions of `etna_lexwork` on 45/48 nonces; on the other 3 the published `etna_lexwork` binary stops its search before its 2e11 cap (1.2e11 to 1.9e11 fuel consumed) while this binary uses the full cap, a difference in fuel accounting between builds and not in code, +444 [-132 ; +1 020] on average over the 48 nonces (the same solutions as `etna_lexwork_v2`'s build on 48/48).

| Track | vs `etna_lexwork` (a053) | vs `ember_opt_v2` (a050) | vs `ember_opt` (a048) |
|-------|------|------|------|
| fjsp_high | **+879 [+556 ; +1 203]** (CPU x2.00) | **+1 924 [+1 524 ; +2 325]** (x6.26) | **+4 067 [+3 653 ; +4 482]** (x6.12) |
| fjsp_medium | same code and cap as `etna_lexwork` | +2 572 [+409 ; +4 735] (x0.97) | +2 004 [+769 ; +3 238] (x0.97) |
| flow_shop | identical solutions 48/48 | -42 [-200 ; +115] (x0.94), identical 32/48 | same as a050 |
| hybrid_flow_shop | +1 216 [-42 ; +2 474] (x0.99) | +740 [-517 ; +1 998] (x2.37) | **+1 829 [+1 349 ; +2 309]** (x2.42) |
| job_shop | **+3 604 [+2 525 ; +4 682]** (x2.81) | **+4 913 [+3 829 ; +5 997]** (x3.50) | **+6 278 [+5 127 ; +7 428]** (x6.40) |

Bold: 95 % CI above zero. On fjsp_medium the row against a050 / a048 is `etna_lexwork`'s own result on the same nonces, since this code path is unchanged. On hybrid_flow_shop, `ember_opt` and `ember_opt_v2` reach a slightly lower quality for 2.4x less CPU: a benchmarker who optimises qualifiers per CPU hour on that track alone may prefer them; the raised caps of fjsp_high and job_shop pay for their CPU under the same criterion.

## Credits

- `ember_etna` (`c007_a044`) and `ember_stromboli` (`c007_a040`) by FP Labs, with components credited in their READMEs to NVX and stinger.
- `etna_highcap` / `etna_highcap_fast` / `etna_lexwork` (`c007_a046` / `c007_a049` / `c007_a053`) by the submitter.
- Hybrid pipeline from `ember_opt` (`c007_a048`) by 0xfe2a, published under the TIG Innovator Outbound Game License (UAI null). Its constructive portfolio comes from `new_adaptive_js` (`c007_a045`, 0x1884), and its final tabu is FP Labs' ember_etna engine. `types.rs`, `preprocess.rs`, `infra_shared.rs`, `ember_hybrid_engine.rs` and `ember_hfs_construct.rs` are byte-identical to `ember_opt`'s; `hybrid_flow_shop.rs` differs from it by the `total_floor` parameter.
- Lexicographic workload criterion: Gao, Sun & Gen, Computers & Operations Research 35 (2008).

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
