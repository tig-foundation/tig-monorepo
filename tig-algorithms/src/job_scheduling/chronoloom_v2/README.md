# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** chronoloom_v2
* **Copyright:** 2026 Danske
* **Identity of Submitter:** Danske
* **Identity of Creator of Algorithmic Method:** Discovered by private Prometheus swarm (model Qwen3-Q8)
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Track-detecting dispatcher for job scheduling: identifies the instance shape
(flow shop, hybrid flow shop, job shop, or one of two flexible job-shop tiers
by flexibility ratio) via `detect_track_simple`, with an optional
hyperparameter override for explicit track selection. Flow shop and job shop
route to a shared engine (`jss_engine`); the flexible job-shop tiers each get
a dedicated tuned solver (`track_t44` for the high-flexibility tier,
`track_t45` for medium), and hybrid flow shop routes to `track_t47`. Every
dispatch path runs behind a save-guard wrapper that only persists a solution
when it is both valid and non-regressing, with a reference-greedy solver
(`ref_greedy`) computing a floor so the final output is never worse than a
plain greedy baseline.

Improves on `chronoloom_v1` in two ways. The reference-greedy floor now
breaks ties between eligible machines by earliest available time
(`sort_by_key(machine_available_time)`) rather than by machine index, which
produces a tighter greedy baseline on the flexible and hybrid tracks. Track
detection was retuned — the high-flexibility cutoff moved from `flex_avg >
5.0` to `> 4.0`, the medium tier now keys on `flex_avg > 2.0` instead of a
uniform-routing test, and the remaining flexible case is guarded against
misclassifying flow-shop instances — and the per-track search budgets were
raised (`fjsp_high` 5,500 → 14,000 iterations, `fjsp_medium` pinned at
10,000, `hybrid_flow_shop` 8,000 → 9,000).

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
