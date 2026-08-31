## TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** parallax_imp
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** FP Labs
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

This algorithm is based on **parallax_altair** by **FP Labs** (`parallax_altair`): GPU Adan with cosine learning-rate, first-layer ReLU kink, BatchNorm-bias offset, and fuel-aware stall.

Two optimizer changes are added on top of that base:

1. **Update rollback.** If a coordinate’s new step has the opposite sign to the last applied step, the previous step is undone instead of taking the reversal. This is an oscillation / disagreement restart on the composed update.
2. **Row/column RMS preconditioning.** For weight matrices, the Adan step is scaled by the factored row and column gradient RMS, `sqrt(row_rms · col_rms)`.

On 15-nonce `test_algorithm` vs `parallax_altair`:

| n_hidden| parallax_altair | This work | Gain  |
|---------|----------------:|----------:|------:|
| 4       |            765k |      813k |  +48k |
| 7       |            470k |      633k | +163k |
| 10      |            483k |      718k | +235k |
| 14      |            577k |      724k | +147k |
| 18      |            543k |      686k | +143k |

## Academic papers

- Xie, X., Zhou, P., Li, H., Lin, Z., and Yan, S. *Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models*. IEEE TPAMI, 2024. arXiv: [2208.06677](https://arxiv.org/abs/2208.06677) — base optimizer used by parallax.
- O’Donoghue, B., and Candès, E. *Adaptive Restart for Accelerated Gradient Schemes*. Foundations of Computational Mathematics, 2015. arXiv: [1204.3982](https://arxiv.org/abs/1204.3982) — restart when consecutive steps disagree.
- Shazeer, N., and Stern, M. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*. ICML 2018. arXiv: [1804.04235](https://arxiv.org/abs/1804.04235) — factored row/column second-moment scaling.

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
