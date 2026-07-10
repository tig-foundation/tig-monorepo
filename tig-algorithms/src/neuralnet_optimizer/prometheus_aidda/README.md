# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** prometheus_aidda
* **Copyright:** 2026 AIDDA Swarm
* **Identity of Submitter:** AIDDA Swarm
* **Identity of Creator of Algorithmic Method:** AIDDA Swarm (autonomous multi-agent algorithm-discovery system)
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

### 1. Academic Papers
- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"*, DOI: https://doi.org/10.48550/arXiv.2411.16085
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), DOI: https://doi.org/10.48550/arXiv.1711.05101
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"* (cosine annealing), DOI: https://doi.org/10.48550/arXiv.1608.03983


## Additional Notes

Role-scaled **Cautious AdanW** with a warmup + cosine learning-rate schedule and
validation-plateau damping. Evolved autonomously by the AIDDA swarm
(experiment lineage u42-5 → u42-6/u42-7); best swarm benchmark score
596,186.36 across n_hidden=4/10/18 tracks (seed `test`).

Algorithm outline:

- **Adan moments**: EMAs of the gradient (`m`, β=0.98), of gradient
  differences (`v`, β=0.92), and of the combined second moment
  `n = EMA((g + 0.92·g_diff)²)` (β=0.99); update direction
  `(m + 0.92·v) / (sqrt(n) + eps)`.
- **Cautious weight decay**: decoupled (AdamW-style) decay is applied only on
  coordinates where the Adan direction already moves the parameter toward
  zero.
- **Cautious sign mask**: coordinates whose update opposes the raw gradient
  are damped to `keep_damp`× (default 0.25) instead of zeroed.
- **Role-scaled hyperparameters**: separate LR multipliers / weight decay for
  hidden weights, biases, the output head, and BatchNorm affine parameters;
  BatchNorm running statistics are left untouched.
- **Schedule**: linear warmup (8 epochs) into a cosine anneal
  (lr 3.4e-3 → 2e-5 over 700 epochs), with multiplicative LR damping (×0.82,
  floor 0.35) after 12 epochs without validation improvement.
- `optimizer_query_at_params` is disabled to avoid extra full-model copies
  each batch.

All tuned constants can be overridden via the standard hyperparameters JSON
(see `help()` for the key list); defaults reproduce the tuned behaviour
exactly.


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
