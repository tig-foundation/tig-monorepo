# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_extrem_v7
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

A GPU-accelerated neural network optimizer using a dual-phase consensus approach
with Fisher-aware momentum, spectral learning rate scheduling, and adaptive blend
strategies. Each network depth dispatches to a specialised per-track solver with
its own CUDA kernels; per-track tuned defaults are baked in.

Per-track architecture (dispatch on `num_hidden_layers`):
- **T29 (n_hidden=4)**
- **T30 (n_hidden=7)**
- **T26 (n_hidden=10)**
- **T27 (n_hidden=14)**
- **T28 (n_hidden=18)**

Hyperparameters are optional (pass `null` for per-track tuned defaults) and
override the baked configuration when provided.

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
