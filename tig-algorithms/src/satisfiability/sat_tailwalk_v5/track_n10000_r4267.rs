use super::{imp_v4_track3, Hyperparameters};
use anyhow::Result;
use rand::rngs::SmallRng;
use tig_challenges::satisfiability::*;

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    seed_key: u64,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: Vec<u32>,
    n_cnt: Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>,
    co: &[u32],
    all_three_clauses: bool,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let _ = seed_key;
    imp_v4_track3::solve(
        hp,
        rng,
        nv,
        nc,
        density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        cl,
        co,
        all_three_clauses,
        save_solution,
    )
}
