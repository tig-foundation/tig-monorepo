// TIG C001 submission: all solver sources are included in this folder.
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::{Challenge, Solution};

mod fuel;
mod track1;
mod track2;
mod track3;
mod track4;
mod track5;

pub fn help() {
    print!("{}", r###"Talos (talos): C001 SAT solver HP
Recommended: null or {} on all five Tracks. Explicit defaults:
T1 n_vars=5000,ratio=4267: {"max_fuel_high":400000000000}
T2 n_vars=7500,ratio=4267: {"max_fuel_high":400000000000}
T3 n_vars=10000,ratio=4267: {"max_fuel_high":125000000000}
T4/T5 n_vars=100000,ratio=4150/4200: null; HP object ignored.
Nonempty HP objects are forwarded without merging the defaults above.

Effective controls (official five Tracks):
max_fuel_high number: T1/T2/T3 effort estimate, not the runtime allowance.
check_interval integer >=1: T1 compatibility and T3; automatic density formula.
stagnation_limit unsigned integer: T1 compatibility, default 3.
max_reinits unsigned integer: T1 compatibility, default 15; 0 disables reinitialization.
sel_variant unsigned integer: T1 compatibility, default 2; 0/1/2/>=3 select control/zero-list/minimum/combined variants.
probe_max_flips unsigned integer: T1 compatibility diagnostic flip-budget override; absent by default.
t4_fuel_b unsigned integer: first T1 explicit complement effort, default 50000000000; does not control T4.

T1 normal portfolio accepts only max_fuel_high, or null/{}.
Adding any other key selects a compatibility route. Only t4_fuel_b selects
the complement plus cached fallback; other explicit keys use the compatibility kernel.
If max_fuel_high is missing from a nonempty object: T1 compatibility/cached
fallback uses 160000000000; T2 uses 250000000000; T3 primary/rescue use
180000000000/125000000000. Include it explicitly to preserve recommended budgets.
check_interval=0 can cause division by zero. Do not encode numbers as strings.
Wrong recognized field types may reset the entire inner parameter structure.

Parsed legacy fields without a tuning effect on official T2/T3:
base_prob, max_prob, perturbation_flips, max_fuel_low.
T2 also ignores check_interval and stagnation_limit; T3 ignores stagnation_limit.
Other donor fields (pf_arms, pf_arm0, sp_*, spr_*, cb_break) are not exposed.

T1 compatibility example (changes route):
{"max_fuel_high":400000000000,"check_interval":100,"stagnation_limit":3,"max_reinits":15,"sel_variant":2,"t4_fuel_b":50000000000}
T1 diagnostic example:
{"max_fuel_high":400000000000,"probe_max_flips":1000000,"sel_variant":0,"t4_fuel_b":1000000}
T3 check-interval example:
{"max_fuel_high":125000000000,"check_interval":100}
See README.md for complete scope, defaults, examples and retained attributions.
"###);
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;
    let m = challenge.clauses.len();
    let defaults = if n <= 10_000 {
        let mut values = Map::new();
        values.insert("max_fuel_high".into(), Value::from(if n <= 7_500 {
            400_000_000_000_u64
        } else {
            125_000_000_000_u64
        }));
        Some(values)
    } else { None };
    let hp = if hyperparameters.as_ref().map_or(true, |v| v.is_empty()) {
        &defaults
    } else { hyperparameters };
    if n <= 5_000 {
        track1::solve_challenge(challenge, save_solution, hp)
    } else if n <= 7_500 {
        track2::solve_challenge(challenge, save_solution, hp)
    } else if n <= 10_000 {
        track3::solve_challenge(challenge, save_solution, hp)
    } else if (m as f64) / (n.max(1) as f64) < 4.175 {
        track4::solve_challenge(challenge, save_solution, hp)
    } else {
        track5::solve_challenge(challenge, save_solution, hp)
    }
}
