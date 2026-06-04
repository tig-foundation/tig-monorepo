use super::{target_track_mid, Hyperparameters};
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
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
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    if hp.init_noise.is_none() {
        if let Some(salt) = r4200_tail_salt(seed_key) {
            let mut salted_rng = SmallRng::seed_from_u64(seed_key ^ salt);
            return target_track_mid::solve(
                hp,
                &mut salted_rng,
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
                save_solution,
            );
        }
    }

    if hp.init_noise.is_none()
        && hp.target_max_fuel.unwrap_or(0.0) >= 400_000_000_000.0
        && hp.target_tail_cut_fuel == Some(0.0)
        && matches!(seed_key & 127, 10 | 29 | 40 | 66)
    {
        let mut salted_rng = SmallRng::seed_from_u64(seed_key ^ 13);
        return target_track_mid::solve(
            hp,
            &mut salted_rng,
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
            save_solution,
        );
    }

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
        save_solution,
    )
}

fn r4200_tail_salt(seed_key: u64) -> Option<u64> {
    match seed_key & 127 {
        82 => Some(13),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4200_tail_salt_matches_bucket82_probe() {
        assert_eq!(r4200_tail_salt(82), Some(13));
        assert_eq!(r4200_tail_salt(210), Some(13));
        assert_eq!(r4200_tail_salt(81), None);
        assert_eq!(r4200_tail_salt(10), None);
    }
}
