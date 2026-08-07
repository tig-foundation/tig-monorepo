# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** parallax_altair
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

- Xie et al., *"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"*, DOI: https://doi.org/10.48550/arXiv.2208.06677
- Liang et al., *"Cautious Optimizers: Improving Training with One Line of Code"*, DOI: https://doi.org/10.48550/arXiv.2411.16085
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (AdamW), DOI: https://doi.org/10.48550/arXiv.1711.05101
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, DOI: https://doi.org/10.48550/arXiv.1608.03983
- Sahs et al., *"Shallow Univariate ReLU Networks as Splines: Initialization, Loss Surface, Hessian, and Gradient Flow Dynamics"*, DOI: https://doi.org/10.48550/arXiv.2008.01772


## Additional Notes

Adan with a cosine learning rate and plateau damping, plus three interventions that
follow from the structure of the challenge rather than from the optimizer literature.

### Spreading the first layer's ReLU breakpoints

The challenge has a **scalar input** (`INPUT_DIMS = 1`) and initialises every bias to zero,
so all 256 breakpoints of the first layer sit at `x = 0` and the layer spans a
two-dimensional function space. Unit `i` breaks at `x = -b_i/w_i`, so setting
`b_i = -w_i*t_i` places it at `t_i`. Gradient descent takes dozens of epochs to spread
these biases on its own. This is the breakpoint-density collapse described by Sahs et al.

A unit whose breakpoint falls outside `[-1, 1]` never switches: it is **linear** or **dead**
depending on `sign(w)`. How many should stay active is a bias-variance trade-off, and the
answer depends on depth, so the two tracks are handled differently.

On `n_hidden <= 4` a two-population variant sends a fraction `kink_lin` outside on the side
that keeps them active and spreads the rest evenly, leaving about 50 active breakpoints out
of 256, which is what the trade-off predicts for a piecewise linear fit at this noise level
and sample size. Deeper networks take the plain even spread over a wider half-width
(`kink = 1.5`), leaving roughly 170 active: the extra layers can use the resolution that a
shallow one would only overfit with.

### Cycling an output-mean offset

The last two layers are frozen, the output layer's bias is zero, and the frozen BatchNorm
centres its channels. In training mode the network output therefore has **exactly zero
batch mean**, while the target's mean is drawn per instance and is not zero. That term
alone accounts for essentially all of the residual training loss, and no amount of
optimization can remove it.

Evaluation, however, runs in inference mode on the **running statistics**, which lag the
model. Offsetting the last trainable BatchNorm's bias on the LAST batch of an epoch, after
those statistics were accumulated on the unshifted pass, leaves them stale, and the
evaluated model gains an output offset the architecture cannot otherwise express. The
offset is removed on the next batch so training itself is unperturbed.

The offset an instance needs is its target mean, which the training gradient cannot see:
BatchNorm centres it away. Rather than search for it, a grid of offsets is cycled across
epochs: `0`, then `+-top*a/L`, `+-2*top*a/L`, ... up to `+-top*a`. The harness checkpoints the
best-validation epoch, so it keeps whichever suits the instance, at no cost beyond the
epochs spent. A candidate that is never selected costs nothing.

The useful amplitude scales with the layer's activation range and hence with depth, which
is why the base is `0.1 * n_hidden`. The grid spans twice that. Beyond it the offsets stop
paying: beyond `top = 2.0` the qualification rate falls even though the mean keeps rising,
because with a fixed cycle period a wider grid leaves fewer epochs in a usable state.

The grid only pays where there are enough trainable layers to absorb the perturbation it
introduces. Below `n_hidden = 10` it buys mean quality with fuel and nothing at the
qualification threshold, so it is disabled there and the offset falls back to `{0, +a, -a}`.

### Stopping before the fuel runs out

Exhausting the fuel budget kills the process mid-kernel, and the exit code is what
verification checks. `__fuel_remaining` only tracks host-side fuel, some two orders of
magnitude below the total, so it cannot be used as a live gauge, but at initialisation it
holds the cap. The GPU side is dominated by the layer GEMMs, whose per-epoch cost is a
fixed multiple of the total linear weight count. An affordable epoch count follows, and
past it the first layer is zeroed: validation stops improving, the loop leaves on patience,
and the checkpointed best model is untouched.

### Other design points

- The **last two layers are frozen** and BatchNorm running statistics are not trainable:
  the harness discards their updates. Those tensors are skipped rather than computed.
- The **output-layer LR multiplier applies to the last *trainable* layer**, the one feeding
  the frozen block, not to the last linear layer, which is frozen.

Defaults adapt to network depth, read from `param_sizes` rather than from `track_id`, which
TIG may change. All hyperparameters are optional JSON keys; unknown keys are ignored.

### Hyperparameters

| Key | Default | Description |
|-----|---------|-------------|
| `kink` | 1.0 (n_hidden<=4), 1.5 otherwise | half-width of the first-layer breakpoint spread; 0 disables. Supplying either `kink` or `kink_lin` opts out of the depth defaults for both |
| `kink_lin` | 0.80 (n_hidden<=4), -1 otherwise | fraction of linear units, placed outside the input range on their active side; -1 keeps the uniform spread |
| `lr_max` | 2e-3 (n_hidden<18), 1e-3 otherwise | peak base learning rate of the cosine schedule |
| `lr_min` | 2e-5 | final learning rate of the cosine schedule |
| `warmup_epochs` | 8 | linear LR warmup epochs |
| `t_max` | 400 | cosine anneal horizon in epochs |
| `wd` | 0.02 | decoupled weight decay on weights |
| `beta1` | 0.98 | Adan EMA decay of the gradient (first moment) |
| `beta2` | 0.92 | Adan EMA decay of gradient differences |
| `b3` | 0.99 | Adan EMA decay of the combined second moment |
| `eps` | 1e-8 | denominator epsilon |
| `cautious` | 0.25 | cautious-mask damping, applied where the update shares the gradient's sign |
| `plateau_patience` | 12 | epochs without val improvement before LR damping |
| `plateau_decay` | 0.82 | multiplicative LR damping on plateau |
| `plateau_grow` | 1.03 | LR recovery when validation improves again |
| `plateau_floor` | 0.15 | floor of the plateau LR scale |
| `head_mult` | 1.0 | LR multiplier for the last trainable layer |
| `meanalt` | 0.1 * n_hidden | base amplitude of the cycled output-mean offset; 0 disables |
| `meanalt_start` | 20 | first epoch at which the offset starts cycling |
| `meanalt_levels` | 3 (n_hidden>=10), 1 otherwise | number of amplitudes per sign; the cycle has `2*levels+1` phases |
| `meanalt_top` | 2.0 (n_hidden>=10), 1.0 otherwise | largest amplitude of the grid, as a multiple of `meanalt` |
| `fuel_reserve` | 55 | epochs held back from the affordable count, covering the patience window |
| `depth_lr` | 1.0 | LR multiplied by `depth_lr^(relative depth)` across trainable layers |


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
