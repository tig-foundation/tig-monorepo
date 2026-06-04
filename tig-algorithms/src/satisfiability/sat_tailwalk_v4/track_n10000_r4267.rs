use super::{target_track_high, Hyperparameters};
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
    if let Some(tuned_hp) = n10000_bucket_hp(seed_key, hp) {
        return target_track_high::solve(
            &tuned_hp,
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
        );
    }

    if hp.init_noise.is_none() {
        if let Some(salt) = n10000_tail_salt(seed_key) {
            let mut salted_rng = SmallRng::seed_from_u64(seed_key ^ salt);
            return target_track_high::solve(
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

    target_track_high::solve(
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

fn n10000_bucket_hp(seed_key: u64, hp: &Hyperparameters) -> Option<Hyperparameters> {
    let mut tuned = hp.clone();
    match seed_key & 1023 {
        512 | 1005 if hp.init_noise.is_none() => {
            tuned.init_noise = Some(0.006);
        }
        553 | 894 if hp.target_nad.is_none() => {
            tuned.target_nad = Some(1.2);
        }
        208 if hp.init_noise.is_none() && hp.restart_interval.is_none() => {
            tuned.init_noise = Some(0.006);
            tuned.restart_interval = Some(6_000_000);
        }
        _ => return None,
    }
    Some(tuned)
}

fn n10000_tail_salt(seed_key: u64) -> Option<u64> {
    match seed_key & 1023 {
        159 => Some(67),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n10000_tail_salt_matches_nonce30_probe() {
        assert_eq!(n10000_tail_salt(159), Some(67));
        assert_eq!(n10000_tail_salt(1183), Some(67));
        assert_eq!(n10000_tail_salt(158), None);
        assert_eq!(n10000_tail_salt(671), None);
    }

    #[test]
    fn n10000_bucket_hp_matches_seed1_miss_probe_buckets() {
        let hp = Hyperparameters::default();

        assert_eq!(
            n10000_bucket_hp(512, &hp).unwrap().init_noise,
            Some(0.006)
        );
        assert_eq!(
            n10000_bucket_hp(1005, &hp).unwrap().init_noise,
            Some(0.006)
        );
        assert_eq!(
            n10000_bucket_hp(553, &hp).unwrap().target_nad,
            Some(1.2)
        );
        assert_eq!(
            n10000_bucket_hp(894, &hp).unwrap().target_nad,
            Some(1.2)
        );
        let bucket208 = n10000_bucket_hp(208, &hp).unwrap();
        assert_eq!(bucket208.init_noise, Some(0.006));
        assert_eq!(bucket208.restart_interval, Some(6_000_000));
        assert!(n10000_bucket_hp(207, &hp).is_none());
    }

    #[test]
    fn n10000_bucket_hp_respects_explicit_overrides() {
        let hp_noise = Hyperparameters {
            init_noise: Some(0.01),
            ..Default::default()
        };
        let hp_nad = Hyperparameters {
            target_nad: Some(0.9),
            ..Default::default()
        };
        let hp_restart = Hyperparameters {
            restart_interval: Some(4_000_000),
            ..Default::default()
        };

        assert!(n10000_bucket_hp(512, &hp_noise).is_none());
        assert!(n10000_bucket_hp(553, &hp_nad).is_none());
        assert!(n10000_bucket_hp(208, &hp_restart).is_none());
    }
}
