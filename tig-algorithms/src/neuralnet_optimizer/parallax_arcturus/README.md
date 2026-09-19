# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** parallax_arcturus
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

Trains the challenge's MLP by returning per-parameter updates to the harness, maximising the
quality it computes from the checkpoint with the lowest validation loss.

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_hidden=4  | 5T | `{}` |
| n_hidden=7  | 5T | `{}` |
| n_hidden=10 | 5T | `{}` |
| n_hidden=14 | 5T | `{}` |
| n_hidden=18 | 5T | `{}` |

> **The defaults are the intended operating point.** `{}` selects them; every hyperparameter
> listed by `help()` simply overrides its default, so no override is required. Shapes are read
> from `param_sizes` rather than from `track_id`, so the algorithm follows the architecture it
> is handed rather than a track label.
>
> The optimizer is anytime and deterministic. The harness checkpoints on every validation
> record, so a run that ends early still returns its best point. There is no thread, no clock
> and no filesystem access in the control flow, and no randomness: the only state is the
> optimizer's own moments and the tensors the harness provides.
>
> **The recommendation is conservative**: a smaller grant makes the run stop earlier rather
> than fail. Every hyperparameter listed by `help()` overrides its default; values outside a
> parameter's range fall back to that default.


## References and Acknowledgments

### Academic Papers
- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677 — the update rule in `sk_adan_fused`.
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"*, DOI: https://doi.org/10.48550/arXiv.2411.16085 — the cautious mask.
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), DOI: https://doi.org/10.48550/arXiv.1711.05101 — the decoupled decay.
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, DOI: https://doi.org/10.48550/arXiv.1608.03983.
- Sahs et al., *"Shallow Univariate ReLU Networks as Splines"*, DOI: https://doi.org/10.48550/arXiv.2008.01772.
- Ioffe & Szegedy, *"Batch Normalization"*, DOI: https://doi.org/10.48550/arXiv.1502.03167.

### Code References
- **The schedule, the initialisation pass and the per-instance offset are original to this
  submission.** All act only through the update tensors `optimizer_step` returns, on trainable
  parameters; frozen layers and running statistics are left untouched, as the harness requires.
- The fused multi-tensor update and the deferred checkpointing are original to this submission.
- No third-party optimizer is vendored. The frozen dependency set is used as-is.

Hyperparameters are documented by `help()`.


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
