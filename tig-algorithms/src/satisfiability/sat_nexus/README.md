# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_nexus
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

T1-T3 use tekinon's bitwise probSAT with sp=false, one attempt, 4,194,304
maximum flips and giveup checkpoints (2,097,152,8) and (4,194,304,2).
On failure they run bounded v8 rescue, skipping neon: T1 allocates 4.5B
estimated fuel to cached R5 plus 1.5B to IV-SAT, and T2/T3 pass max_fuel_high=6B.
T4/T5 start directly with tekinon's default reinforced SP plus bitwise SLS;
on failure only, they run the bounded 2B v8 adapter. Every stage result must pass
complete clause verification before saving.

### Code References
- TIG `c001` public submissions (`tekinon` / `c001_a135` by tilk, `sat_tailwalk_v8` / `c001_a137`) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
