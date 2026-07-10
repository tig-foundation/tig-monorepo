use super::{target_track_mid, Hyperparameters};
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
    trace_route(hp, seed_key, "default");
    target_track_mid::solve(
        hp,
        rng,
        seed_key,
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

fn trace_route(hp: &Hyperparameters, seed_key: u64, route: &str) {
    if !hp.target_trace_4200.unwrap_or(false) {
        return;
    }
    eprintln!(
        "c001_r4200_route_diag route={} seed_key={} target_max_fuel={} tail_cut_fuel={} init_noise={}",
        route,
        seed_key,
        hp.target_max_fuel
            .map(|fuel| format!("{fuel:.0}"))
            .unwrap_or_else(|| "none".to_string()),
        hp.target_tail_cut_fuel
            .map(|fuel| format!("{fuel:.0}"))
            .unwrap_or_else(|| "none".to_string()),
        hp.init_noise
            .map(|noise| format!("{noise:.6}"))
            .unwrap_or_else(|| "none".to_string())
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4200_trace_flag_is_default_off() {
        assert!(!Hyperparameters::default()
            .target_trace_4200
            .unwrap_or(false));
    }
}
