# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** engine_block_reheat
* **Copyright:** 2026 SantaClaude
* **Identity of Submitter:** SantaClaude
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Code References
- submission003_engine – https://github.com/tig-foundation/tig-monorepo. This submission modifies that
  code: `round_engine.rs` and `track_10k.rs` are changed, every other file is unchanged.
- sigma_freud_opt – https://github.com/tig-foundation/tig-monorepo

## Additional Notes

engine_block_reheat changes what the best-of-K runs of submission003_engine do after run 0. All changes are host-side;
the CUDA kernels are unchanged.

- Host-side polish of the saved partition (pairwise, part-group, region, multilevel and k-way FM, hyperedge-collapse
  moves), accepted only when the exact connectivity metric is lower.
- Run 1 is a warm reheat of run 0's final partition after a coupled-block ruin, instead of an independent hot run.
- The last run is a vote-guided block reheat: nodes the earlier runs disagree on seed part-group blocks, each block is
  ruined and recreated as a closed k-way sub-problem, then the whole graph is annealed warm to cold.
- 10k track: a part-pair LNS kick in the ILS with an exact 2-way FM repair of the kicked pair.

The code default is `runs = 1`, which leaves the reheat runs off. Recommended hyperparameters:

- n_h_edges=10000: `{"effort": 5, "ils_iterations": 48, "ils_quick_refine": 260, "init_restart_id": 8, "post_ils_polish": 80, "refinement": 20000, "runs": 2, "tabu_tenure": 9}`
- n_h_edges=20000: `{"run0_ils_iters": 1, "run0_ils_quick_pct": 5, "runs": 5}`
- n_h_edges=50000 and 100000: `{"fused_mode": 0, "run0_ils_iters": 1, "run0_ils_quick_pct": 5, "runs": 5}`
- n_h_edges=200000: `{"run0_ils_iters": 1, "run0_ils_quick_pct": 5, "runs": 3}`

With these settings, quality against submission003_engine changed by -0.31% to +1.64% (mean +0.55%) over two instances
per track. Runtime was about 1.6x on 100k and 2.7x on 200k, so check it against your time limit on those tracks.
`help_algorithm` shows the submission003_engine options only; the new options are read in `round_engine.rs`.

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
