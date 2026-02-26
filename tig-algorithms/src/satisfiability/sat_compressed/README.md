## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_compressed
* **Copyright:** 2026 JohanE
* **Identity of Submitter:** JohanE
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

In this version of sat_excelled the clause-variable adjacency is stored in compressed sparse row style

### Hyperparameters

- `max_fuel_high` (default: 10,000,000,000) - Fuel for high-density problems (≥422)
- `max_fuel_low` (default: 12,500,000,000) - Fuel for low-density problems (<422)

Higher fuel increases search depth, potentially finding more solutions but with longer runtime.

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
