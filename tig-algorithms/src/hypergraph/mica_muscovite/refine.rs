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
    tie_fm: u32,
    // hchain: 0 = no host stage, 1 = FM only, 2 = full chain (shipped).
    // fm_rounds: FM depth, set per track by the dispatch table.
    hchain: u32,
    fm_rounds: usize,
    // iso_host: skips host work that cannot change the result.
    iso_host: bool,
    // tie_host: unifies the equal-gain tie-break across host stages.
    tie_host: bool,
    // fm_steps / fm_seeds: length and count of FM's localized searches.
    fm_steps: usize,
    fm_seeds: usize,
    // fm_stop: early-abandon rule for one FM search.
    fm_stop: f64,
    jet_rounds: usize,
    cyc_rounds: usize,
) -> Vec<u32> {
    if hchain == 0 { return partition.to_vec(); }
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

    let lcfg = super::lp::LpConfig { tie_host, ..super::lp::LpConfig::default() };
    let fcfg = super::fm::Config { tie: tie_fm, rounds: fm_rounds,
        max_steps: fm_steps, seeds_per_search: fm_seeds, stop_factor: fm_stop,
        ..super::fm::Config::default() };
    let jcfg = super::jet::Config {
        iso_host,
        tie_host,
        rounds: jet_rounds,
        accept_zero: true,
        unconstrained: true,
        over: 1,
        chain: true,
    };

    for _ in 0..passes.max(1) {
        let before = st.km1();
        if hchain >= 2 {
            super::lp::refine(&mut st, &lcfg);
            keep!();
            super::jet::refine(&mut st, &jcfg);
            keep!();
            super::jet::cycles(&mut st, cyc_rounds, iso_host);
            keep!();
        }
        super::fm::refine(&mut st, &fcfg);
        keep!();
        if st.km1() == before {
            break;
        }
    }

    best_part.iter().map(|&x| x as u32).collect()
}
