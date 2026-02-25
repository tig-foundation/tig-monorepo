# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_sam
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

**Neural SAM** - Dual-phase consensus optimizer with Fisher normalization for neural network training.

**Key Features:**
- **Dual-Phase CUDA Kernels**: Fast kernel (dual consensus Fisher) for most of training, robust kernel (sign + error feedback + Fisher) for late-phase convergence
- **Adaptive Noise Variance**: Learns proximity to optimal loss with faster tracking for shorter training schedules
- **Spectral Boost Adaptation**: Dynamically adjusts learning rate based on relative improvement trends
- **Precision Zone Detection**: Fine-tunes optimizer parameters when loss enters critical convergence zone
- **Guarded Robust Switch**: Transitions to robust kernel only when loss has converged sufficiently, preventing premature switching on deep networks
- **Adaptive Plateau Escape**: Multi-tier plateau detection with LR pulses scaled to loss proximity
- **Phase Tempo Tracking**: Adjusts weight decay and learning rate based on batches-per-epoch dynamics

**Hyperparameters:**
- `total_steps` (default: 450) - Total optimizer steps, tune per track complexity
- `warmup_steps` (default: 40) - Steps before adaptive mechanisms engage
- `noise_variance` (default: 0.040) - Initial noise floor estimate
- `spectral_boost` (default: 1.1) - Initial spectral learning rate multiplier
- `bn_layer_boost` (default: 1.35) - Batch norm layer learning rate multiplier
- `threads_per_block` (default: 128) - CUDA threads per block
- `blocks_per_sm` (default: 4) - CUDA blocks per SM

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
