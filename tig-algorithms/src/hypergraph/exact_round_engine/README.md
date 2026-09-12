# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** exact_round_engine
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

An **exact-output** speed rebuild of `sigma_freud_opt`. With default hyperparameters the returned
partition is **per-nonce identical** to `sigma_freud_opt` at the benchmarker hyperparameters on every
track, so quality is unchanged by construction and only the time to produce it changes.

It extends the device-resident round engine of `submission005_exact` (a batch of original main-loop
rounds inside one persistent-kernel launch behind a software grid barrier, with ordered top-quota
selection, capacity-constrained move apply and lottery jump-ahead all moved onto the device exactly)
with a further set of exactness-preserving refinements: exact embedded-incidence descriptors and
warp-grouped event counters, replay-local shared predecessor data replacing global histograms, exact
parallel CTA partial-count folds with register prefix scans, bounded bit-plane adder trees that preserve
the exact counters, one-traversal 10k balancing gains, constant-divisor lottery arithmetic, lane-group
swap evaluation, and exact bit-plane counters for the 10k final polish.

Measured against `sigma_freud_opt`, same pod, same batch, 4 workers, benchmarker hyperparameters:
**10k 10.8x, 20k 10.5x, 50k 14.7x, 100k 16.6x, 200k 17.1x** (RTX 4090). Output verified per-nonce
identical to the baseline on every track and byte-identical to `submission005_exact`. See PROOF.md,
including the note that these kernels are GPU-architecture sensitive.

Optional hyperparameters (all deterministic, all defaulting to baseline behaviour): `runs` (1..256
best-of-K), `run_flavor_mode`, `run_refinement_pct`, `run0_ils_iters` / `run0_ils_quick_pct` /
`run0_polish_pct`, `fused_mode`, and the exact-loop toggles `exact_loop`, `exact_batch_rounds`,
`exact_tail`, `exact_swaps`, `tail_bucket`.

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
