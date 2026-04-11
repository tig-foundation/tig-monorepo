# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_extrem_v3
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

neural_extrem_v3 is a GPU-accelerated neural network optimizer using a dual-phase consensus approach with Fisher-aware momentum, spectral learning rate scheduling, and adaptive blend strategies.

Key improvement over v2: all 5 tracks now support hyperparameter tuning via JSON (v2 only supported T28).

Per-track architecture with specialized CUDA kernels:
- **T29 (n_hidden=4)**: Fast optimizer with high spectral boost (1.12), low beta1 (0.90)
- **T30 (n_hidden=7)**: Depth-scaled layer LRs, trend-aware trust backoff
- **T26 (n_hidden=10)**: Depth-fractional LR scaling, divergence recovery
- **T27 (n_hidden=14)**: Depth-scaled LRs, divergence factor dampening
- **T28 (n_hidden=18)**: Precision-zone aware, relative update capping

Hyperparameters (all optional, pass `null` for defaults):
- `total_steps`: total optimizer steps (default: per-track 1000-1200)
- `warmup_steps`: warmup period (default: per-track 32-60)
- `spectral_boost`: spectral LR multiplier (default: 1.1-1.12)
- `noise_variance`: noise floor threshold (default: 0.020-0.040)
- `beta1`: Adam first moment decay (default: 0.90-0.92)
- `beta2`: Adam second moment decay (default: 0.995-0.997)
- `weight_decay`: weight decay coefficient (default: 0.0018-0.0025)
- `bn_layer_boost`: batch norm layer LR multiplier (default: 1.35-1.50)
- `output_layer_damping`: output layer LR dampening (default: 0.77-0.86)

## References and Acknowledgments

### 1. Academic Papers
- Kingma, Ba, *"Adam: A Method for Stochastic Optimization"*, ICLR 2015
- Loshchilov, Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), ICLR 2019
- Loshchilov, Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, ICLR 2017
- Amari, *"Natural Gradient Works Efficiently in Learning"*, Neural Computation 1998
- Kunstner et al., *"Limitations of the Empirical Fisher Approximation"*, NeurIPS 2019

### 2. Code References
- neural_extrem_v2 (TIG) - https://github.com/tig-foundation/tig-monorepo

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
