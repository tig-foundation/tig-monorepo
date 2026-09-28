# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** parallax_vega
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_hidden=4  | 5T | `{}` |
| n_hidden=7  | 5T | `{}` |
| n_hidden=10 | 5T | `{}` |
| n_hidden=14 | 5T | `{}` |
| n_hidden=18 | 5T | `{}` |

> **The defaults are the intended operating point.** `{}` selects the settings tuned for
> each depth; every hyperparameter simply overrides them, so no override is required.
> Depth is read from `param_sizes` rather than from `track_id`, so the defaults follow the
> architecture rather than a track label.
>
> The optimizer is anytime and deterministic. A valid solution is checkpointed as soon as
> one exists, and the run stops before the fuel budget can be exhausted mid-kernel. There
> is no thread, no clock and no filesystem access in the control flow, and no randomness:
> the only state is the optimizer's own moments and the tensors the harness provides.
>
> **Actual fuel consumption is far below the recommended grant** — roughly 15 G to 400 G
> depending on the track and the instance, against a 5T budget. The recommendation is
> conservative; a smaller grant makes the run stop earlier rather than fail.


## References and Acknowledgments

### Academic Papers
- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677 — the update rule in `sk_adan_fused`.
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"*, DOI: https://doi.org/10.48550/arXiv.2411.16085 — the cautious mask.
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), DOI: https://doi.org/10.48550/arXiv.1711.05101 — the decoupled decay.
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, DOI: https://doi.org/10.48550/arXiv.1608.03983 — the cosine schedule with warmup.
- Sahs et al., *"Shallow Univariate ReLU Networks as Splines"*, DOI: https://doi.org/10.48550/arXiv.2008.01772 — the breakpoint-density view of a scalar-input ReLU layer.
- Ioffe & Szegedy, *"Batch Normalization"*, DOI: https://doi.org/10.48550/arXiv.1502.03167 — the training/inference asymmetry the output-mean offset rests on.

### Code References
- **The first-step breakpoint placement and the per-instance output-mean offset are original
  to this submission.** Both act only through the update tensors `optimizer_step` returns,
  on trainable parameters; frozen layers and running statistics are left untouched, as the
  harness requires.
- The fused multi-tensor update, the deferred checkpointing and the fuel guard are original
  to this submission.
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
