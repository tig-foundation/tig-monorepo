# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** task_tree_j_shuttle
* **Copyright:** 2026 fred; donor copyrights remain with NVX, stinger, and FP Labs
* **Identity of Submitter:** fred
* **Identity of Creator of Algorithmic Method:** fred
* **Unique Algorithm Identifier (UAI):** null

## Method

For `hybrid_flow_shop`, this algorithm runs the public `task_tree_j`
construction/refinement pipeline and passes its final best schedule directly
to a warm-started shuttle_bell-family graph-tabu finisher in the same Code
execution. The phase-one schedule remains the incumbent floor and is replaced
only by a verifier-objective improvement. The finisher fuel cap is fixed at
`120,000,000,000`.

For `fjsp_high`, `fjsp_medium`, `job_shop`, and `flow_shop`, execution delegates
unchanged to the public `task_tree_j` implementation.

## Public-Source Attribution

* `track_t47.rs` and `ref_greedy.rs` derive from public `c007_a036 / task_tree_j`
  by NVX, source commit `3299a341237651e974f1b6d22ac6a20c9344d96d`.
* `shuttle_finisher.rs` derives from the public shuttle_bell-family graph-tabu
  implementation published as `c007_a037 / shuttle_bell` by stinger and its
  public `c007_a040 / ember_stromboli` evolution by FP Labs, source commits
  `483701d1f36ff99ef6ebc79770f02a1bf8aa0038` and
  `3fc50755424c7bd1bc7dea458c5a99e6b70eeac3` respectively.
* `mod.rs` provides the new within-execution warm-start composition, track
  isolation, fixed finisher budget, and incumbent-floor handoff.

No opaque or decompiled material is included.

## References

* Nowicki, E. & Smutnicki, C. (1996), *A Fast Taboo Search Algorithm for the
  Job Shop Problem*, Management Science 42(6).
* Mastrolilli, M. & Gambardella, L.M. (2000), *Effective Neighbourhood
  Functions for the Flexible Job Shop Problem*, Journal of Scheduling 3(1).
* Pearce, D.J. & Kelly, P.H.J. (2007), *A Dynamic Topological Sort Algorithm
  for Directed Acyclic Graphs*, Journal of Experimental Algorithmics 11.

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
