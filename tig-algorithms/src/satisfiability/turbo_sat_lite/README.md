# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** turbo_sat_lite
* **Copyright:** 2026 onebooker
* **Identity of Submitter:** onebooker
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Lower-fuel SAT specialist derived from `turbo_sat`.
Key changes: lower random-walk ceiling, lower default fuel budgets,
faster reversion toward base probability, and reduced perturbation pressure
to improve quality-per-fuel on medium-difficulty tracks.

Intended regime:
- medium and less pathological satisfiability tracks
- efficiency frontier optimization rather than maximum late-stage rescue

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
