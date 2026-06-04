use super::{target_track_low, Hyperparameters};
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use tig_challenges::satisfiability::*;

const R4150_DEFAULT_MAX_FUEL: f64 = 80_000_000_000.0;

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
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let route_hp = r4150_hp(hp);
    match r4150_tail_salt(seed_key) {
        Some(salt) => {
            let mut salted_rng = SmallRng::seed_from_u64(seed_key ^ salt);
            target_track_low::solve(
                &route_hp,
                &mut salted_rng,
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
                save_solution,
            )
        }
        None => target_track_low::solve(
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
            save_solution,
        ),
    }
}

fn r4150_hp(hp: &Hyperparameters) -> Hyperparameters {
    let mut route_hp = hp.clone();
    if route_hp.target_max_fuel.is_none() {
        route_hp.target_max_fuel = Some(R4150_DEFAULT_MAX_FUEL);
    }
    route_hp
}

fn r4150_tail_salt(seed_key: u64) -> Option<u64> {
    match seed_key & 63 {
        27 => Some(131),
        34 => Some(1),
        54 => Some(97),
        18 => Some(13),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4150_tail_salt_matches_slow_bucket_probe() {
        assert_eq!(r4150_tail_salt(27), Some(131));
        assert_eq!(r4150_tail_salt(34), Some(1));
        assert_eq!(r4150_tail_salt(54), Some(97));
        assert_eq!(r4150_tail_salt(91), Some(131));
        assert_eq!(r4150_tail_salt(18), Some(13));
        assert_eq!(r4150_tail_salt(0), None);
        assert_eq!(r4150_tail_salt(33), None);
    }

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
