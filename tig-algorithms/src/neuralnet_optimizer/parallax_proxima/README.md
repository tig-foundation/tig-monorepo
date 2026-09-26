# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** parallax_proxima
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

Deterministic GPU optimizer for the challenge's MLP. The `gp_mode` hyperparameter selects the
default method (1) or an alternative optimizer (0), which also serves as a fallback.

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_hidden=4  | 5T | `{}` |
| n_hidden=7  | 5T | `{}` |
| n_hidden=10 | 5T | `{}` |
| n_hidden=14 | 5T | `{}` |
| n_hidden=18 | 5T | `{}` |

> The defaults are the intended operating point: `{}` selects them. `help()` lists every
> hyperparameter with its domain; a value outside its domain falls back to the default. The
> algorithm is deterministic: there is no thread, no clock and no filesystem access, and the only
> pseudo-random draws are seeded by the challenge seed.

## References and Acknowledgments

### Academic Papers
- Solin & Särkkä, *"Hilbert space methods for reduced-rank Gaussian process regression"*, Statistics and Computing 30:419-446 (2020), DOI: https://doi.org/10.1007/s11222-019-09886-w.
- Rasmussen & Williams, *"Gaussian Processes for Machine Learning"*, MIT Press (2006), http://gaussianprocess.org/gpml/.
- Hinton, Vinyals & Dean, *"Distilling the Knowledge in a Neural Network"* (2015), DOI: https://doi.org/10.48550/arXiv.1503.02531.
- Kingma & Ba, *"Adam: A Method for Stochastic Optimization"*, ICLR (2015), DOI: https://doi.org/10.48550/arXiv.1412.6980.
- Ioffe & Szegedy, *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"*, ICML (2015), DOI: https://doi.org/10.48550/arXiv.1502.03167.
- Sahs et al., *"Shallow Univariate ReLU Networks as Splines: Initialization, Loss Surface, Hessian, and Gradient Flow Dynamics"* (2020), DOI: https://doi.org/10.48550/arXiv.2008.01772.
- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"* (2022), DOI: https://doi.org/10.48550/arXiv.2208.06677.
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"* (2024), DOI: https://doi.org/10.48550/arXiv.2411.16085.
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"*, ICLR (2019), DOI: https://doi.org/10.48550/arXiv.1711.05101.
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, ICLR (2017), DOI: https://doi.org/10.48550/arXiv.1608.03983.
- Meterez et al., *"Towards Training Without Depth Limits: Batch Normalization Without Gradient Explosion"*, ICLR (2024), DOI: https://doi.org/10.48550/arXiv.2310.02012.
- K. Jordan, *"Muon: An optimizer for hidden layers in neural networks"* (2024), https://kellerjordan.github.io/posts/muon/.

### Code References
- Original implementation. No third-party code is vendored; the frozen dependency set is used
  as-is, and the network follows the challenge's own layer definitions.

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
