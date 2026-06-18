# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** prometheus_aidda_v2
* **Copyright:** 2026 AIDDA Swarm
* **Identity of Submitter:** AIDDA Swarm
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

### 1. Academic Papers
- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"*, DOI: https://doi.org/10.48550/arXiv.2411.16085
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), DOI: https://doi.org/10.48550/arXiv.1711.05101
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, DOI: https://doi.org/10.48550/arXiv.1608.03983

### 2. Code References
- NVX — `neural_extrem_v3` (TIG mainnet, branch `neuralnet_optimizer/neural_extrem_v3`): the n_hidden=4 path (`track_n4.rs`, `helpers.rs`, `kernels_n4.cu`) is its t29 track (spectral-phase LR, dual consensus/Fisher + sign-EF update kernels), used under the TIG Inbound Game License.


## Additional Notes

Per-track dispatcher on `challenge.num_hidden_layers`:

- **n_hidden=4**: spectral-phase LR scheduling with dual consensus/Fisher and sign-EF update kernels.
- **n_hidden=7/10/14/18**: role-scaled Cautious AdanW with warmup/cosine LR
  and validation-plateau damping.

All hyperparameters are optional JSON keys; defaults are the tuned values
listed below. Unknown keys are ignored. The two paths have different key sets.

### Hyperparameters — n_hidden=4 path

| Key | Default | Description |
|-----|---------|-------------|
| `total_steps` | 5500 | LR schedule horizon in optimizer steps (~8 steps/epoch) |
| `warmup_steps` | 16 | linear LR warmup steps |
| `noise_variance` | 0.035 | gradient noise injection variance |
| `spectral_boost` | 1.1 | LR boost applied during the spectral phase |
| `beta1` | 0.90 | first-moment EMA decay |
| `beta2` | 0.96 | second-moment EMA decay |
| `eps` | 1e-8 | denominator epsilon |
| `weight_decay` | 0.015 | decoupled weight decay |
| `bn_layer_boost` | 1.0 | LR multiplier for BatchNorm affine parameters |
| `output_layer_damping` | 0.86 | LR damping on the final layers |
| `threads_per_block` | 256 | CUDA launch: threads per block |
| `blocks_per_sm` | 3 | CUDA launch: grid cap in blocks per SM |

### Hyperparameters — n_hidden=7/10/14/18 path

| Key | Default | Description |
|-----|---------|-------------|
| `lr_max` | 3.4e-3 | peak base learning rate of the cosine schedule |
| `lr_min` | 2e-5 | final learning rate of the cosine schedule |
| `warmup_epochs` | 8 | linear LR warmup epochs |
| `t_max_epochs` | 700 | cosine anneal horizon in epochs |
| `hidden_wd` | 1.6e-3 | weight decay on hidden-layer weights |
| `output_wd` | 1e-5 | weight decay on output-layer weights |
| `plateau_patience` | 12 | epochs without val improvement before LR damping |
| `plateau_decay` | 0.82 | multiplicative LR damping on plateau |
| `min_lr_scale` | 0.35 | floor of the plateau LR scale |
| `hidden_bias_lr_mult` | 1.25 | LR multiplier: hidden biases |
| `output_weight_lr_mult` | 1.75 | LR multiplier: output weights |
| `output_bias_lr_mult` | 2.0 | LR multiplier: output biases |
| `bn_weight_lr_mult` | 0.55 | LR multiplier: BatchNorm weights |
| `bn_bias_lr_mult` | 0.8 | LR multiplier: BatchNorm biases |
| `keep_damp` | 0.25 | cautious-mask damping for sign-disagreeing coordinates |
| `beta_m` | 0.98 | Adan EMA decay of the gradient (first moment) |
| `beta_v` | 0.92 | Adan EMA decay of gradient differences |
| `beta_n` | 0.99 | Adan EMA decay of the combined second moment |
| `mix` | 0.92 | Adan mixing coefficient for gradient differences |


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
