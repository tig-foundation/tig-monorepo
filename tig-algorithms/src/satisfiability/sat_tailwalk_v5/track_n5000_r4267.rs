use super::{imp_v4_track1, Hyperparameters};
use anyhow::Result;
use rand::rngs::SmallRng;
use tig_challenges::satisfiability::*;

pub(crate) fn default_max_fuel(_hp: &Hyperparameters) -> f64 {
    115_000_000_000.0
}

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
    let route_hp = n5000_route_hp(hp);
    imp_v4_track1::solve(
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

fn n5000_route_hp(hp: &Hyperparameters) -> Hyperparameters {
    let mut route_hp = hp.clone();
    if route_hp.target_max_fuel.is_none() {
        route_hp.target_max_fuel = Some(default_max_fuel(hp));
    }
    route_hp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n5000_default_fuel_matches_r18_probe() {
        assert_eq!(
            default_max_fuel(&Hyperparameters::default()),
            115_000_000_000.0
        );
    }

    #[test]
    fn n5000_route_applies_default_fuel_to_imp_path() {
        let hp = n5000_route_hp(&Hyperparameters::default());

        assert_eq!(hp.target_max_fuel, Some(115_000_000_000.0));
    }

    #[test]
    fn n5000_route_respects_explicit_fuel_override() {
        let hp = n5000_route_hp(&Hyperparameters {
            target_max_fuel: Some(180_000_000_000.0),
            ..Hyperparameters::default()
        });

        assert_eq!(hp.target_max_fuel, Some(180_000_000_000.0));
    }
}
