// Partition refinement.

use super::gain::{Block, State};

pub fn improve(
    n: usize,
    m: usize,
    he_off: Vec<u32>,
    he_nodes: Vec<u32>,
    k: usize,
    max_part_size: u32,
    partition: &[u32],
    passes: usize,
    fm_rounds: usize,
    fm_steps: usize,
    fm_seeds: usize,
    fm_stop: f64,
    jet_rounds: usize,
    cyc_rounds: usize,
    ml: Option<super::ml::Config>,
    ml2: Option<super::ml2::Config>,
) -> Vec<u32> {
    if n == 0 || m == 0 || k == 0 || partition.len() != n {
        return partition.to_vec();
    }
    if he_off.len() != m + 1 || he_off[m] as usize != he_nodes.len() {
        return partition.to_vec();
    }
    if partition.iter().any(|&b| b as usize >= k) {
        return partition.to_vec();
    }
    if he_nodes.iter().any(|&v| v as usize >= n) {
        return partition.to_vec();
    }
    if (k as u64) * (max_part_size as u64) < n as u64 {
        return partition.to_vec();
    }

    let part: Vec<Block> = partition.iter().map(|&x| x as Block).collect();
    let Some(mut st) = State::new(n, m, k, he_off, he_nodes, part, max_part_size, None) else {
        return partition.to_vec();
    };

    let mut best_km1 = st.km1();
    let mut best_part = st.part.clone();
    macro_rules! keep {
        () => {{
            let cur = st.km1();
            let feasible = st
                .block_size
                .iter()
                .all(|&s| s >= 1 && s <= st.max_block_size);
            if feasible && cur < best_km1 {
                best_km1 = cur;
                best_part.copy_from_slice(&st.part);
            }
        }};
    }

    let fcfg = super::fm::Config {
        rounds: fm_rounds,
        max_steps: fm_steps,
        seeds_per_search: fm_seeds,
        stop_factor: fm_stop,
    };

    let npass = passes.max(1);
    for pass in 0..npass {
        let before = st.km1();
        super::lp::refine(&mut st);
        keep!();
        super::jet::refine(&mut st, jet_rounds);
        keep!();
        super::jet::cycles(&mut st, cyc_rounds);
        keep!();
        if let Some(cfg2) = ml2.as_ref() {
            for sub in 0..cfg2.passes {
                super::ml2::refine(&mut st, cfg2, sub);
                if cfg2.cycles > 0 && (!cfg2.cycles_last || sub + 1 == cfg2.passes) {
                    super::jet::cycles(&mut st, cfg2.cycles);
                }
                keep!();
            }
            if st.km1() == before {
                break;
            }
            continue;
        }
        if let Some(cfg) = ml.as_ref() {
            super::ml::refine(&mut st, cfg, &fcfg, pass);
            keep!();
            if st.km1() == before {
                break;
            }
            continue;
        }
        super::fm::refine(&mut st, &fcfg);
        keep!();
        let mut cur = st.km1();
        if let Some((p, after)) = super::coarsen::supernode_fm(&st, &fcfg) {
            if after < best_km1 {
                best_km1 = after;
                best_part.copy_from_slice(&p);
            }
            cur = after;
            if pass + 1 < npass {
                for (v, &to) in p.iter().enumerate() {
                    if st.part[v] != to {
                        st.apply(v as u32, to);
                    }
                }
            }
        }
        if cur == before {
            break;
        }
    }

    best_part.iter().map(|&x| x as u32).collect()
}
