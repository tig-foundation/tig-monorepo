# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** nebeltrotz
* **Copyright:** 2026 Kernelsmith
* **Identity of Submitter:** Kernelsmith
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

`nebeltrotz` is a training optimizer for the fixed c006 loop (`optimizer_init_state`,
`optimizer_query_at_params`, `optimizer_step`), one configuration for all tracks. Components:

1. **AdamW** with bias correction, decoupled weight decay on the linear weight matrices only, linear
   warmup and cosine decay of the learning rate, per-element gradient clipping, and a higher learning
   rate for biases and batch-norm affines.
2. **Lookahead** (k = 5, alpha = 0.5) on the fast weights.
3. **Weight averaging at validation time:** an exponential moving average of the trajectory
   (decay 0.98) is swapped into the model for the per-epoch validation and early stopping; the
   next epoch continues from the stashed trajectory point via `optimizer_query_at_params`.
4. **Offset probe on the last trainable batch-norm bias:** the output offset along the frozen
   read-out cannot be learned by gradients in the training-mode forward pass. Short, symmetric pulses
   and finite differences of the reported validation loss give a damped Newton step for that
   offset, with drift, noise and convexity guards and a freeze after convergence.
5. Only tensors the harness trains are updated. Frozen layers and batch-norm running statistics
   (identically zero gradients) receive no optimizer step.

Everything is deterministic: repeated runs give identical solutions, runtime signatures and fuel.

Measured on 2026-09-19 with the official `tig-runtime` / `tig-verifier` (instrumented `.so` and PTX
from `build_so` / `build_ptx`, monorepo `a4db8c5`) on an RTX 3060 against the official dc_steer_v8
binary. 10 fresh instances per track, identical settings, hyperparameters `{}`:

| Track (n_hidden) | nebeltrotz avg | dc_steer_v8 avg | higher / lower | time vs. dc_steer_v8 |
|---|---:|---:|---:|---:|
| 4  | 856,953 | 849,129 | 7 / 3  | 1.83x |
| 7  | 856,592 | 842,260 | 10 / 0 | 2.33x |
| 10 | 856,408 | 843,770 | 9 / 1  | 1.88x |
| 14 | 856,352 | 835,047 | 9 / 1  | 3.33x |
| 18 | 849,437 | 821,312 | 10 / 0 | 2.92x |

Maximum fuel 153 billion (3.1 % of the 5e12 budget). Qualities depend on the seeds and are not
directly comparable to network bundle qualities. Other GPU architectures were not tested in this run.

## References and Acknowledgments

- I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
- D. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
- M. Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
- No third-party algorithm code was copied.

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
