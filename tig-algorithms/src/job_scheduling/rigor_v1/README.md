## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** rigor_v1
* **Copyright:** 2026 Brent Beane & Jayanth Ravindran
* **Identity of Submitter:** Brent Beane & Jayanth Ravindran
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

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


Rigor v1  README
Flexible Job Shop Scheduling Solver 

Overview
Rigor v1 is a high-performance, deterministic solver for the Flexible Job Shop Scheduling Problem (FJSP), designed for industrial-scale manufacturing environments

It combines:

Hierarchical constructive heuristics
Critical-path-driven tabu search
Adaptive EET slack control
Multi-phase restart strategy
This hybrid approach delivers SOTA-level performance across job_shop, flexible_job_shop, and hybrid_flow_shop challenge tracks.

Key Features
Feature	Description
Conflict-Aware Construction	Uses hierarchical clustering of job-machine conflicts to guide dispatch priority
Dual Construction Paths	construct_schedule (rule-based) + construct_schedule_hierarchical (topology-aware)
Critical Path Optimization	Tabu search focused on bottleneck operations only
Adaptive Slack (EET+)	Controlled relaxation of earliest eligible time for flexibility exploration
Monotonic Improvement	SolutionSaver ensures only improving solutions are submitted
Full Reproducibility	Seeded via SmallRng, deterministic under fixed input
Rescue Recovery	Extended search phase triggered on stagnation
Compilation & Usage
1. Add to Cargo.toml

Copy
[dependencies]
tig_challenges = "0.1"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"
2. Build

Copy
cargo build --release
3. Integrate with TIG UI
The solve_challenge function is automatically detected by TIG's UI via:


Copy
tig_challenges::job_scheduling::solve_challenge
No configuration required-just drop into the challenge runner.

Hyperparameters
Tunable via JSON input in hyperparameters argument:


Copy
{
  "construction_restarts": 50,
  "construction_top_k": 3,
  "tabu_tenure": 10,
  "max_iterations": 0,
  "max_idle_iterations": 1000,
  "tabu_restarts": 5
}
Parameter	Default	Purpose
construction_restarts	50	Number of randomized constructive restarts
construction_top_k	3	Random selection window in dispatch rules
tabu_tenure	10	Tabu list duration (in iterations)
max_iterations	0	Max tabu steps (0 = unlimited)
max_idle_iterations	1000	Restart if no improvement
tabu_restarts	5	Number of tabu search restarts
For flexible instances, construction_restarts and tabu_restarts are auto-scaled upward.

 Execution Flow
1. Problem Analysis
   ├→ Detect flexibility
   └→ Compute avg_flexibility

2. Tier 1: Construction
   ├→ Deterministic dispatch rules (5 types)
   ├→ Hierarchical clustering priority
   ├→ Randomized restarts (50–100+)
   └→ Eager saving via SolutionSaver

3. Tier 2: Tabu Search
   ├→ Critical path identification
   ├→ Block-based swap & reassignment moves
   ├→ Adaptive neighborhood width
   └→ Perturbation on stagnation

4. Tier 3: Rescue Phase
   └→ Extended construction + tabu if improvement < 2–5%


Diagnostics Output
On execution, prints:


Copy
=== DIAGNOSTICS ===
Flexibility: is_flexible=true, avg_flexibility=3.42
Rule MostWorkRemaining: makespan = 18420
Rule LeastFlexibility (slack=17): makespan = 18150
Hierarchical: makespan = 17980
Random restarts: best=17820, worst=19410
Construction best: 17820
Tabu restart 0: 17820 -> 17460
Tabu restart 1: 17790 -> 17430
Tabu improvement over construction: 2.19% (17820 -> 17430)
Final makespan: 17430
=== END DIAGNOSTICS ===
Useful for tracking progress and tuning.

Validation & Safety
All solutions are validated against:
Machine eligibility
Job precedence
Machine conflicts
Operation count
Built-in evaluate_makespan_check for external verification
SolutionSaver prevents regression
Zero unsafe code
Integration Ready
Designed for real-world deployment:

Compatible with MES/APS systems (Siemens, SAP, Rockwell)
Embeddable as optimization microservice
Supports JSON hyperparameter injection
Outputs standard Solution struct for downstream use
Ideal for integration into platforms like Opcenter, FactoryTalk, or AVEVA.

