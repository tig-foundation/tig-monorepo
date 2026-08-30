# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_opti
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Solver

`neural_opti` is a depth-dispatched GPU optimizer family for the TIG `neuralnet_optimizer` challenge. `solve_challenge` in `mod.rs` routes on `num_hidden_layers` to a self-contained track engine:

| Hidden layers | Engine | Core update |
|---|---|---|
| 4, 7, 14 | **S3** (p=3) | Nesterov first moment over a cubic power-mean denominator, plus bounded coherent headroom on full 256×256 hidden weights |
| 10 | **S3** (p=2 RMS) | Same Nesterov / headroom skeleton, but second moment is `g²` with a square-root denominator (NAdam-style), plus SAM and LAMB/LARS trust scaling |
| 18 | Fused cautious **Adan** with per-group learning rates and sign-gated weight decay |

Shared S3 machinery (tracks 4 / 7 / 10 / 14):

- Exponential moving averages of the gradient and of a power statistic (`|g|³` on 4/7/14, `g²` on 10).
- Nesterov look-ahead numerator `β m + (1−β) g` (NAdam / Nesterov momentum estimation).
- Decoupled AdamW weight decay on hidden and output weights only.
- Linear warmup then cosine decay to `min_lr_ratio` (SGDR).
- Validation-driven **progress credit** `½(1 + cos(π · stale / horizon))` that scales the headroom gain when val loss has not improved.
- 18-bit round-to-nearest-even mantissa canonicalization on host and device so updates match across GPU architectures.
- Batch-norm running stats are never updated (`GROUP_RUNNING_STAT` → zero delta).

**Bounded coherent headroom** (hidden 256×256 weights): the unit-clipped adaptive step is allowed extra magnitude only when a confidence score is high — directional agreement of `g` and `m` (tracks 4 / 10), three-step sign consensus (track 7), or row-wise cosine coherence of `g` and `m` (track 14). Extra step is `gain · confidence · √|u| · (1 − |u|)`, then re-clipped to 1.

### Track notes

- **4** — AMSGrad monotone cubic envelope `max(envelope, s)`; fused affine row kernel shares a row-mean `|g|³` statistic across each hidden weight row and its bias (Adafactor-style factored second moment) and splits a persistent / transient directional EMA.
- **7** — Optional harmonic-mean mix of the cubic root and a three-step mean-abs consensus scale; specialised BN affine path (cubic scale × RMSProp-style `√EMA(g²)`); Reduce-on-plateau learning-rate restart when val loss is stale for a full horizon.
- **10** — Dual-loss headroom policy (train improving without val improving disables headroom). Weight tensors get a first-order SAM correction using a finite-difference Hessian-vector proxy `(g − g_prev) / ‖g‖`, then a LAMB/LARS trust ratio `‖θ‖ / (‖Δθ‖/lr + ε)` (row-wise on hidden 256×256, tensor-wise elsewhere).
- **14** — Hidden path: AMSGrad envelope, optimistic gradient `g + (g − g_prev)` (OGDA) inside the Nesterov numerator, then a spherical tangent projection that removes the component of the update along `θ`. On a val-loss plateau the engine injects a seed-hashed Rademacher diversification pattern, also projected into the mean-and-radial tangent plane. Non-hidden tensors use a Lion-style `sign` of the Nesterov-predicted momentum, dispatched as a batched descriptor kernel.
- **18** — Single fused kernel `adabelief_fused_kernel_18` implementing Adan (`m`, `v = Δg`, `n = (g + β₂ Δg)²`) with a cautious keep mask (`1` if `dir · g > 0`, else `0.25`) and a sign-aware weight-decay gate. Epoch cosine schedule with warmup, plus ReduceLROnPlateau-style `lr_scale` (`×1.03` on val improvement, `×0.82` after 12 stale epochs, floored at `0.35`).

## Hyperparameters

Defaults live in each track `PROFILE`. JSON overrides (where the track reads them):

**S3 (used on 7 and 10; ignored on 4 and 14):** `s3_total_steps`, `s3_warmup_steps`, `s3_lr`, `s3_beta`, `s3_eps`, `s3_weight_decay`, `s3_min_lr_ratio`, `s3_coherence_gain`, `s3_progress_horizon_epochs`, `s3_arch_mantissa_bits`.

**Track 7 only:** `s3_enable_consensus_denom_mix`, `s3_enable_bn_specialised`, `s3_enable_plateau_lr_restart`.

**Track 18:** `lr_max` (default `2.8e-3`), `t_max_epochs` (default `650`), `ghw_scale` (default `1.0`).

Typical compiled defaults: lr `3e-3`, β `0.95`, ε `1e-8`, weight decay `0.01`, min-lr ratio `0.10`, coherence gain `0.25`, progress horizon `50` epochs, mantissa bits `18`. Step budgets: 3500 / 1200 / 2850 / 1950 on tracks 4 / 7 / 10 / 14.

### Academic papers

Methods below are the ones the kernels and host engines actually implement, not a general optimizer survey.

**First- and second-moment adaptive updates**

- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate *O(1/k²)*. *Soviet Mathematics Doklady*, 27, 372–376.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*. https://arxiv.org/abs/1412.6980
- Dozat, T. (2016). Incorporating Nesterov momentum into Adam. *ICLR Workshop*. https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
- Reddi, S. J., Kale, S., & Kumar, S. (2018). On the convergence of Adam and beyond. *ICLR*. https://arxiv.org/abs/1904.09237
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5 — RMSProp. *COURSERA: Neural Networks for Machine Learning*. (BN affine path on track 7.)
- Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive learning rates with sublinear memory cost. *ICML*. https://arxiv.org/abs/1804.04235 (row-factored cubic statistic on track 4.)

**Decoupled decay, schedules, and layer-wise trust**

- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*. https://arxiv.org/abs/1608.03983
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*. https://arxiv.org/abs/1711.05101
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. https://arxiv.org/abs/1706.02677
- You, Y., Gitman, I., & Ginsburg, B. (2017). Large batch training of convolutional networks. https://arxiv.org/abs/1708.03888 (LARS trust ratio on track 10.)
- You, Y., et al. (2020). Large batch optimization for deep learning: Training BERT in 76 minutes. *ICLR*. https://arxiv.org/abs/1904.00962 (LAMB.)

**Sharpness, optimism, sign updates, Adan, caution**

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. *ICLR*. https://arxiv.org/abs/2010.01412
- Daskalakis, C., Ilyas, A., Syrgkanis, V., & Zeng, H. (2018). Training GANs with optimism. *ICLR*. https://arxiv.org/abs/1711.00141
- Chen, X., et al. (2023). Symbolic discovery of optimization algorithms. https://arxiv.org/abs/2302.06675 (Lion sign-momentum on track 14 non-hidden tensors.)
- Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2022). Adan: Adaptive Nesterov momentum algorithm for faster optimizing deep models. https://arxiv.org/abs/2208.06677 (track 18 fused update; kernel symbol `adabelief_*` is a leftover name.)
- Liang, K., et al. (2024). Cautious optimizers: Improving training with one line of code. https://arxiv.org/abs/2411.16085 (`keep` mask on track 18.)

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
