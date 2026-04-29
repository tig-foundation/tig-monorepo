use serde_json::{Map, Value};

pub const FULL_MPC_THRESHOLD: usize = 3000;
pub const SHALLOW_MPC_THRESHOLD: usize = 15000;
pub const TERM_LOOK: usize = 12;

// (n_cand_a, horizon_a, n_cand_c, warm_start)
// effort=2 row is bitwise-identical to mpc_v3 (N=5, H=96, N_C=3).
// N values for branch_a follow 4k+1 rule (N=5,9,17) so each effort level's grid is a strict
// superset of effort=2 candidates — guarantees monotone non-regression.
// n_cand_c >= 3 ensures the branch_c [lo,0,hi] path fires, keeping "hold" action available.
// effort=0,1: identical to effort=2 (mpc_v3 baseline quality).
//   N=3 was tried for effort=0 (fast screening) but gave -1.5% BASELINE mean regression because
//   the N=5 midpoints 0.25/0.75 were excluded — no coarser uniform grid avoids this.
//   N=4 was tried for effort=1 but [0,1/3,2/3,1] also excludes 0.25/0.75 (-3.5% BASELINE).
//   Decision: 0-2 are quality aliases; speed reduction must come from outside this algorithm.
const PRESETS: [(usize, usize, usize, bool); 5] = [
    (5,  96,  3, false),  // effort=0: alias for effort=2 (see comment above)
    (5,  96,  3, false),  // effort=1: alias for effort=2 (see comment above)
    (5,  96,  3, false),  // effort=2: balanced, mpc_v3 identical
    (9,  96,  5, false),  // effort=3: deep quality — N=9 ⊇ N=5, CAPSTONE N=5 (default)
    (17, 128, 5, true),   // effort=4: maximum — N=17, warm-start (Direction 3)
];

pub struct Config {
    pub effort: usize,
    pub n_cand_a: usize,
    pub horizon: usize,
    pub n_cand_c: usize,
    pub warm_start: bool,
}

impl Config {
    pub fn initialize(hyperparameters: &Option<Map<String, Value>>) -> Self {
        let effort = hyperparameters
            .as_ref()
            .and_then(|h| h.get("effort"))
            .and_then(|v| v.as_u64())
            .map(|v| v.clamp(0, 4) as usize)
            .unwrap_or(3);
        let (n_cand_a, horizon, n_cand_c, warm_start) = PRESETS[effort];
        Config { effort, n_cand_a, horizon, n_cand_c, warm_start }
    }
}
