# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_extrem_v6
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

A GPU-accelerated neural network optimizer using a dual-phase consensus approach
with Fisher-aware momentum, spectral learning rate scheduling, and adaptive blend
strategies. Each network depth dispatches to a specialised per-track solver with
its own CUDA kernels.

Per-track architecture (dispatch on `num_hidden_layers`):
- **T29 (n_hidden=4)**
- **T30 (n_hidden=7)**
- **T26 (n_hidden=10)**
- **T27 (n_hidden=14)**
- **T28 (n_hidden=18)**

Hyperparameters (all optional, pass `null` for per-track tuned defaults):
- `total_steps`: total optimizer steps
- `warmup_steps`: warmup period
- `spectral_boost`: spectral LR multiplier
- `noise_variance`: noise floor threshold
- `beta1`: Adam first moment decay
- `beta2`: Adam second moment decay
- `weight_decay`: weight decay coefficient
- `bn_layer_boost`: batch norm layer LR multiplier
- `output_layer_damping`: output layer LR dampening

## References and Acknowledgments

### 1. Academic Papers
- Kingma, Ba, *"Adam: A Method for Stochastic Optimization"*, ICLR 2015
- Loshchilov, Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), ICLR 2019
- Loshchilov, Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, ICLR 2017
- Amari, *"Natural Gradient Works Efficiently in Learning"*, Neural Computation 1998
- Kunstner et al., *"Limitations of the Empirical Fisher Approximation"*, NeurIPS 2019

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
