# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_scientist_adp
* **Copyright:** 2025 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Here I present my latest neuralnet optimiser - **Neural Scientist Adaptive**

**Key Features:**
- **Adaptive Noise Variance** (0.0-0.05): Learns the achievable loss floor from validation history and triggers conservative updates when approaching it
- **Adaptive Spectral Boost** (0.85-1.5x): Learning rate amplifier that responds to training progress, increases on improvement, decreases on regression
- **Adaptive Beta1** (0.85-0.96): Momentum that adjusts based on training volatility, higher when stable, lower when noisy

**Hyperparameters you can use to tune the algorithm for your specific hardware:**
- `threads_per_block` (default: 128) - Try 64, 128, 256, 512
- `blocks_per_sm` (default: 4) - Try 2-8 for GPU occupancy tuning
- `total_steps` (default: 1000) - Instance-dependent, more complex instance = may require more steps
- `warmup_steps` (default: 96) - Try 50-150 before adaptation kicks in

All other hyperparameters are adaptive or work well at their current defaults.

**Important - this algorithm was predominantly tested on an RTX 5070Ti 12Gb. Please make sure you test on your own specific hardware and tune parameters where necessary to obtain the best results.**

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