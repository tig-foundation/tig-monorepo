# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** energy_flow_lp
* **Copyright:** 2026 ChervovNikita (V17 parent); 2026 NVX (V11 T51 parent); 2026 Athen Labs (new modifications)
* **Identity of Submitter:** Athen Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null
* **Status:** engineering validation and submission metadata checks passed; ready for the user to submit

## Inherited Notices and Source Lineage

* **2026 ChervovNikita:** titan_v17, c008_a049, source commit `eb31f804d63fb97ec1c9b0265c159d0655c0b86a`. Supplies T49, T50, T52, T53 and all wrapper code outside the T51 dispatch arm.
* **2026 NVX:** titan_v11, c008_a048, source commit `0abad7b6a9dad0f85a459f14751669b54187a47c`. Supplies the complete T51 engine and its exact dispatch/default arm.

These are inherited notices, not the identity of the new submitter. Both original READMEs/license notices and complete parent trees are preserved in the sealed review packet. Athen Labs is the user-provided submitter and declared copyright holder for the new modifications. The inherited creator identity and UAI remain null.

## Contribution and Defaults

The Code change is confined to the LP routine inside V11 T51. Unopened slack columns remain implicit identity columns until their row first leaves the basis. Dense arithmetic covers primal columns, opened slack columns and the RHS; the backing allocation retains the original full row stride. Entering-variable selection preserves original variable-ID order, including slack IDs. Represented floating-point operations, tie handling, pivot limits and the original dense fallback for non-finite cases are retained.

The complete solver retains V11 T51's defaults and V17's other four engines. In particular, T51 inherits `use_lp_only=true` and `seed_sel_mode=2`. Recommended ordinary runs omit `--hyperparameters`; the frozen validation uses the full 5,000,000,000,000 fuel allowance. The rejected T53 feasibility screen is absent. No dependency, runtime, instrumentation or accounting change is included.

The six Rust files are byte-identical to the tested composed candidate. This derivative README is a metadata-only replacement for the inherited V17 README. The Rust `help()` function still prints the inherited `titan_v17` label because its tested bytes are preserved; the declared module and submission name are `energy_flow_lp`.

## Validated Results and Limits

The final composed solver passed all 36 AMD64 development calls, all 26 isolated timing calls and all 36 ARM64 architecture calls. On three fixed T51/multiday inputs with three paired repetitions each, full-process CPU decreased 30.74% and wall time decreased 30.75% versus the composed reference, with exact schedules and quality. These are scoped timing observations, not a population-wide guarantee. Capstone's unchanged engine passed the separate regression tolerance.

All 44 reserved instances passed the complete seven-role comparison: 308 valid first executions, with exact candidate/reference and intended-public-parent schedules and official quality on every instance. Nine calls that never started before an administrative cutoff were completed in a separately sealed window; all original 308 rows and nine additional dispositions are preserved, with no failed execution retried or source, settings, seed, default or selection changes. ARM reused nine exposed development inputs; it supplies architecture validation, not additional independent timing or holdout samples.

On the 16 reserved multiday cases, quality matched public V11 exactly and was higher on 13 cases and lower on three against V17. All public controls had the same per-track threshold-hit counts as the candidate: baseline 3/4, congested 4/4, multiday 10/16, dense 4/4 and capstone 16/16. No increased qualification rate, network adoption or reward is inferred. Capstone retains V17's heavier quality/cost operating point: versus V11 it had higher quality in 10 cases and lower in six, at a median saved-checkpoint fuel ratio of 13.33. Other tracks also have public-control tradeoffs.

The three-case T51 timing cohort used 58.18% less saved-checkpoint fuel by the ratio of summed case medians. The runtime records that counter at its last saved output, so it excludes later work and is distinct from the 30.74% full-process CPU reduction. No universal 50% full-cost or novel-method/Advance claim is made. The accompanying review packet preserves exact source and binary identities, all raw evidence, per-track public comparisons, and independent audits.

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
