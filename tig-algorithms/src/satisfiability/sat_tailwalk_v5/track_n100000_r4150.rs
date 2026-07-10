use super::{target_track_low, Hyperparameters};
use anyhow::Result;
use rand::rngs::SmallRng;
use tig_challenges::satisfiability::*;

const R4150_DEFAULT_MAX_FUEL: f64 = 80_000_000_000.0;

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    _seed_key: u64,
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
    let route_hp = r4150_hp(hp);
    target_track_low::solve(
        &route_hp,
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

fn r4150_hp(hp: &Hyperparameters) -> Hyperparameters {
    let mut route_hp = hp.clone();
    if route_hp.target_max_fuel.is_none() {
        route_hp.target_max_fuel = Some(R4150_DEFAULT_MAX_FUEL);
    }
    route_hp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4150_default_hp_caps_low_route_fuel() {
        let hp = r4150_hp(&Hyperparameters::default());

        assert_eq!(hp.target_max_fuel, Some(R4150_DEFAULT_MAX_FUEL));
    }

    #[test]
    fn r4150_default_hp_respects_explicit_fuel() {
        let hp = r4150_hp(&Hyperparameters {
            target_max_fuel: Some(50_000_000_000.0),
            ..Hyperparameters::default()
        });

        assert_eq!(hp.target_max_fuel, Some(50_000_000_000.0));
    }
}
