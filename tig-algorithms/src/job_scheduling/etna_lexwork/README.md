# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** etna_lexwork
* **Copyright:** 2026 Fred; donor copyrights remain with FP Labs, NVX and stinger
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

## What changes against `etna_highcap_fast` (`c007_a049`)

`etna_lexwork` is `etna_highcap_fast` (itself `ember_etna`, `c007_a044`, FP Labs, with a 2e11 fjsp_high work cap and exact job_shop speed-ups) with three changes, all in `mod.rs`; `lab.rs` differs from `c007_a049` by its first comment line only, and the six other `.rs` files are byte-identical.

1. **Faster fjsp tabu search, same decisions (fjsp_high and fjsp_medium).** Initial bounds for the machine-move and machine-swap scans, an operation x machine duration table, critical blocks found by walking the critical operations only, a linear insertion window, reordered swap filters, the move tabu list in an array, and a persistent per-machine gap cache invalidated by the incremental evaluation. At equal iteration counts the search makes the same moves (12/12 identical runs, laboratory replayer); it uses about 36 % fewer instructions per tabu iteration on fjsp_high and 28 % fewer on fjsp_medium, so more iterations fit in the same fuel.
2. **Lexicographic (makespan, total assigned work) search on fjsp_high only.** In the move and swap scans, ties on the makespan estimate are broken by the change of total assigned processing time; plateau moves (estimate equal to the current best move or makespan) are played when they reduce total work; the tabu, pool and global bests are compared lexicographically on (makespan, total work). The criterion is from the literature: Gao, Sun & Gen, *A hybrid genetic and variable neighborhood descent algorithm for flexible job shop scheduling problems*, Computers & Operations Research 35 (2008), lexicographic makespan / total workload / maximal workload.
3. **Job shop routing fix.** A rigid instance (one eligible machine per operation) is sent to the flow shop engine only if every job visits the same machine sequence (single route), otherwise to the job shop engine. `ember_etna` and its descendants split rigid instances on `n_products < 27`, which sent the job shops with fewer than 27 products (about 1 % of job_shop nonces) to the flow shop engine, giving a quality of about -19 800. Every flow_shop instance has a single route and every job_shop instance with at least 27 products has several, so the routing of all other instances is unchanged.

flow_shop and hybrid_flow_shop run `c007_a049` unchanged. The solver is anytime and deterministic, without clock, thread, environment variable or filesystem access; all pseudo-randomness derives from `challenge.seed`. Study settings under `"v1lab"` are those of `c007_a049` (see its README); any other key is ignored.

## Local Benchmark

TIG runtime and verifier 0.0.8, round-136 track settings, null hyperparameters (`ember_opt` with `{"track": <track>}`), 48 fresh nonces per track and fuel, paired per nonce; mean quality difference with 95 % Student CI, median CPU time ratio. Binary built from these files in a fresh copy of the public repository with a one-line `job_scheduling` registry. 3 312 runs, 0 invalid, 8/8 reruns identical.

| Track | Fuel | vs `etna_highcap_fast` (a049) | CPU | vs `ember_etna` (a044) | vs `ember_opt` (a048) |
|-------|------|------------------------------|-----|------------------------|------------------------|
| fjsp_high | 1e10 | +1 110 [+790 ; +1 429] | x1.01 | +1 110 | +875 [+561 ; +1 189] |
| fjsp_high | 1.2e11 | +1 572 [+1 152 ; +1 992] | x1.00 | +2 547 [+2 158 ; +2 936] (CPU x1.84) | +2 217 (CPU x1.85) |
| fjsp_high | 5e12 | +1 143 [+786 ; +1 501] | x1.01 | +2 998 (CPU x3.09) | +2 668 (CPU x3.09) |
| fjsp_medium | 1e10 | +5 801 [+4 351 ; +7 251] | x0.91 | same as vs a049 | +6 166 |
| fjsp_medium | 1.2e11 | +1 591 [+813 ; +2 369] | x0.90 | same as vs a049 | +1 770 |
| fjsp_medium | 5e12 | +1 035 [+459 ; +1 611] | x0.90 | same as vs a049 | +1 143 |
| flow_shop | all | identical solutions 144/144 | x1.00 | identical | +77 [-128 ; +282] |
| hybrid_flow_shop | 1e10 / 1.2e11 / 5e12 | identical solutions 144/144 | x1.00 | identical | +16 215 / **-2 252 [-3 378 ; -1 127]** / +981 (CPU x1.16 / x1.42 / x2.48) |
| job_shop | 1e10 / 1.2e11 / 5e12 | identical solutions 144/144 | x1.00 | -275 / +68 / 0 (a049's job engine) | +2 679 / +1 349 / +1 898 (CPU x1.10 / x2.16 / x2.28) |

Job shop routing fix: on 80 other fresh job_shop nonces (none with fewer than 27 products), identical solutions to a049 on 80/80. On the 11 job_shop nonces with fewer than 27 products (nonce 509 plus 10 found by scanning 900 other nonces), `etna_lexwork` beats a049 and a044 on 11/11 at 1.2e11 and 5e12 (mean +128 121 and +128 510; a049 and a044 return about -18 650 on average) and `ember_opt` on 6/11 (mean +2 801, CI [-613 ; +6 214], not significant).

Where it is not better: on hybrid_flow_shop at 1.2e11 fuel, `ember_opt` is better (+2 252) at 0.7 times the CPU time; elsewhere `ember_opt` is cheaper but lower in quality.

## Credits

Built on `ember_etna` (`c007_a044`) and `ember_stromboli` (`c007_a040`) by FP Labs, with components credited in their READMEs to NVX and stinger; `etna_highcap` / `etna_highcap_fast` (`c007_a046` / `c007_a049`) by the submitter. Lexicographic workload criterion: Gao, Sun & Gen (2008). No code from `ember_opt` (`c007_a048`) is included; it is used only as a benchmark reference.

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
