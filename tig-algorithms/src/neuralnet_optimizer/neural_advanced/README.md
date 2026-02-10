# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_advanced
* **Copyright:** 2025 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Here I present my latest neuralnet optimiser - **Neural Advanced**

**Key Features:**
- **Adaptive Noise Variance**: Automatically learns when the network is close to its best possible accuracy and slows down to avoid overshooting
- **Adaptive Spectral Boost**: Dynamically adjusts the learning rate - speeds up when making good progress, slows down when struggling
- **Adaptive Beta1**: Adjusts how much the optimizer "remembers" previous updates based on whether training is smooth or chaotic
- **Stability Detection**: Monitors whether gradients are consistent to decide between fast aggressive updates or careful conservative steps
- **Robust Fisher Diagonal**: Estimates the loss landscape curvature while filtering out misleading extreme values
- **Adaptive Plateau Escape**: Detects when training gets stuck and automatically increases learning rate to break free

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