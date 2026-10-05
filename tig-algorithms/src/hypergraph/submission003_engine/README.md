# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** submission003_engine
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

A speed rebuild of `sigma_freud_opt`: one shared round engine (`round_engine.rs`, per-track kernels generated from one
template) drives every track. The main loop runs as ONE fused grid-barrier kernel launch per round (edge flags + moves +
device-side max-gain/tabu/lottery filter + candidate compaction), with warp-per-hyperedge flags, degree-class warp tasks,
bit-sliced part counters and packed-key host selection. With default hyperparameters the output is per-nonce
IDENTICAL to `sigma_freud_opt` at the benchmarker hyperparameters, at ~3.8x / 3.7x / 3.2x / 3.5x / 2.2x the speed on the
20k / 50k / 100k / 200k / 10k tracks (4 workers, RTX 3090). See PROOF.md for the measurement log.

Optional hyperparameters (all deterministic):
- `runs` (1..256, default 1): best-of-K. Run 0 is the unchanged solver; runs k>=1 start from a ruin-and-recreate of the
  incumbent with per-run seeds/flavors (`run_flavor_mode`, default 2) and `run_refinement_pct` (default 50) of the main-loop
  budget. `runs: 2` gives roughly +4.7k / +6.6k / +6.8k / +3.2k quality on 20k / 50k / 100k / 200k at ~2.8x / 3.0x / 3.6x / 3.2x
  the baseline speed when combined with the tail trims below; on 10k a second run is not worth its time.
- Tail trims for run 0: `run0_ils_iters` (1 recommended), `run0_ils_quick_pct` (5), `run0_polish_pct` (5): ~7-40% faster
  per run at quality-neutral output (defaults keep the baseline tail so that the default output stays identical).
- `fused_mode` (0/1/2): filter geometry; default 0 on 20k, 2 on the larger tracks.
- The benchmarker hyperparameters of each track are baked as defaults; any user value overrides them.

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
