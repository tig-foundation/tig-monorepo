# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_capacity_env
* **Copyright:** 2026 NVX (upstream source); 2026 Athen Labs (new modifications)
* **Identity of Submitter:** Athen Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### Code References

- NVX, `knap_killer_one` (`c003_a149`), complete public source at commit [`25f6564f08976f69830fcdac040558b5bbe4a8d0`](https://github.com/tig-foundation/tig-monorepo/tree/25f6564f08976f69830fcdac040558b5bbe4a8d0/tig-algorithms/src/knapsack/knap_killer_one). This submission incorporates the full five-track solver. Its dispatcher and Tracks 1–4 are unchanged; Track 5 adds capacity-conditioned upper-bound tables to skip exchange scans that cannot improve the incumbent. Original search windows, hyperparameters, tie rules, and stopping behavior are retained.

## Additional Notes

This is an ordinary Code implementation improvement to the cited public solver. It makes no new algorithmic-method or Advance claim. The original source identifies its Creator of Algorithmic Method and UAI as null; this submission retains that status.

The capacity tables apply only under guarded weight, capacity, and nonnegative-interaction conditions. The original scan path remains available when those conditions do not hold. Default hyperparameters and all five public track routes are preserved.

## Tested Performance

On 16 original Track 5 development cases with two repetitions each, the median saved-checkpoint fuel ratio to the rebuilt and official `knap_killer_one` parent was 0.867764 (13.22% lower). Every candidate/parent output pair across the ten all-track smoke cases and 32 Track 5 repetitions matched in ordered items, objective, official quality and regenerated challenge identity. The original 210-call cohort was completed through a separately frozen administrative continuation; all original outcomes, including the administrative timeout, are retained in the review evidence.

Full-process CPU fell by 16.61% and 16.23% in the separately analyzed original and continuation host groups. These are medians of matched, same-host case-fragment ratios; the two host groups are not pooled. Saved-checkpoint fuel is the runtime's last output callback value and excludes subsequent cleanup. Native ARM validation passed all 30 calls on the ten reused smoke cases; it is an architecture check, not an additional independent performance sample.

Against public defaults on those 16 Track 5 cases, quality was higher on 15 cases and lower on one versus `c003_a144`, with more fuel on every case. Versus `c003_a139`, quality was higher on 14 and lower on two, with less fuel on every case. These are measured quality/fuel tradeoffs, not universal leader dominance. No 50% reduction, adoption, reward or Advance eligibility is claimed. Recommended validation settings use the inherited defaults (omit hyperparameters) and the captured 5-trillion fuel cap.

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
