# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** dc_vega_v3
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

### 1. Academic Papers
- Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, Shuicheng Yan, *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677
- Ilya Loshchilov, Frank Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, DOI: https://doi.org/10.48550/arXiv.1608.03983

### 2. Code References
- parallax_vega (TIG neuralnet_optimizer submission, public source in the tig-foundation/tig-monorepo algorithm branches) – https://github.com/tig-foundation/tig-monorepo
- ChervovNikita, dc_steer_v6 (TIG neuralnet_optimizer submission; the DC steering stage) – https://github.com/tig-foundation/tig-monorepo

### 3. Other
- Grid-stride kernel loops – NVIDIA CUDA programming practice (each element is visited by exactly one thread iteration; the per-element arithmetic is unchanged).


## Additional Notes

- Single fused Adan-style update launch per optimizer step over all trainable tensors, with DC steering of the last trainable BatchNorm bias.
- Dispatch metadata (pointer planes, per-epoch learning-rate and weight-decay tables, kernel handle) is allocated once per solve; the fused kernel runs as a grid-stride loop over 256-thread blocks.
- The weight of the first 256x256 hidden linear layer is not trained (its update is exactly zero) when the network has 7 or more hidden layers; it is trained on shallower networks. The depth is derived from the parameter layout.
- Recommended hyperparameters: `{}` (all defaults are baked in). Optional keys: `freeze_mask` (bit l freezes the weight of linear layer l; overrides the depth default), `dc_start`, `dc_refresh`, `dc_gain`, `dc_max_abs`, `kink`, `save_eager`, `fuel_reserve`, `lr_max`, `lr_min`, `warmup_epochs`, `t_max`, `wd`, `eps`, `beta1`, `beta2`, `b3`, `cautious`, `plateau_patience`, `plateau_decay`, `plateau_grow`, `plateau_floor`, `head_mult`, `depth_lr`, `bn_w`, `bn_b`, `ab_blend`, `wd_gate`.
- Deterministic: no clock, no randomness, no filesystem access, no cross-block floating-point atomics in the solver kernels.
- Only `updates` entries for trainable tensors are written; frozen parameters are never modified.


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
