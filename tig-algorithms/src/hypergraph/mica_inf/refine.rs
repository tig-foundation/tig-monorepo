// Host refinement chain applied to the GPU solver's output.

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
    if k > 64 {
        return partition.to_vec();
    }

    let part: Vec<Block> = partition.iter().map(|&x| x as Block).collect();
    let mut st = State::new(n, m, k, he_off, he_nodes, part, max_part_size, None);

    let mut best_km1 = st.km1();
    let mut best_part = st.part.clone();
    macro_rules! keep {
        () => {{
            let cur = st.km1();
            let feasible = st.block_size.iter().all(|&s| s >= 1 && s <= st.max_block_size);
            if feasible && cur < best_km1 {
                best_km1 = cur;
                best_part.copy_from_slice(&st.part);
            }
        }};
    }

    let fcfg = super::fm::Config { rounds: fm_rounds, max_steps: fm_steps,
        seeds_per_search: fm_seeds, stop_factor: fm_stop };

    let npass = passes.max(1);
    for pass in 0..npass {
        let before = st.km1();
        super::lp::refine(&mut st);
        keep!();
        super::jet::refine(&mut st, jet_rounds);
        keep!();
        super::jet::cycles(&mut st, cyc_rounds);
        keep!();
        super::fm::refine(&mut st, &fcfg);
        keep!();
        let mut cur = st.km1();
        // One level of contraction, then the same FM on the pairs.
        if let Some((p, after)) = super::coarsen::supernode_fm(&st, &fcfg) {
            if after < best_km1 {
                best_km1 = after;
                best_part.copy_from_slice(&p);
            }
            cur = after;
            if pass + 1 < npass {
                let ho = st.he_off.clone();
                let hn = st.he_nodes.clone();
                st = State::new(n, m, k, ho, hn, p, max_part_size, None);
            }
        }
        if cur == before {
            break;
        }
    }

    best_part.iter().map(|&x| x as u32).collect()
}
