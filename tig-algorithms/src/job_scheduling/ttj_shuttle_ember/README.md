# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** ttj_shuttle_ember
* **Copyright:** 2026 Fred; donor copyrights remain with NVX, stinger, and FP Labs
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method

This Code is a universal internal composition for the five public c007 tracks.
For `hybrid_flow_shop`, it preserves the validated `task_tree_j` construction
and refinement pipeline, passes its final best schedule to a warm-started
shuttle_bell-family graph-tabu finisher inside the same execution, and keeps
the phase-one schedule as an immutable incumbent floor. The finisher fuel cap
is fixed at `120,000,000,000`.

For `fjsp_medium`, `fjsp_high`, `job_shop`, and `flow_shop`, it preserves the
public `c007_a040 / ember_stromboli` frontier engines. Those four engines are
public donor behavior and are not presented as new search inventions here.

Leave hyperparameters null. All track routing and settings are internal and
fixed; no hyperparameter JSON is required.

## Public-Source Attribution

* The hybrid phase-one files `track_t47.rs` and `ref_greedy.rs` derive from
  public `c007_a036 / task_tree_j` by NVX, source commit
  `3299a341237651e974f1b6d22ac6a20c9344d96d`.
* `shuttle_finisher.rs` derives from the public `c007_a037 / shuttle_bell`
  graph-tabu family by stinger, source commit
  `483701d1f36ff99ef6ebc79770f02a1bf8aa0038`, and its public
  `c007_a040 / ember_stromboli` evolution by FP Labs, source commit
  `3fc50755424c7bd1bc7dea458c5a99e6b70eeac3`.
* The flat `ember_*.rs` files preserve the public
  `c007_a040 / ember_stromboli` implementation by FP Labs. Their module paths
  are flattened solely for the TIG multi-file uploader.
* `mod.rs` provides track-aware composition: the validated hybrid
  `task_tree_j` to warm graph-tabu pipeline, and unchanged a040 behavior on
  the other four tracks.

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
