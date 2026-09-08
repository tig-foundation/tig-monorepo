// DO NOT CHANGE THESE IMPORTS
use tig_challenges::knapsack::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand::seq::SliceRandom;


pub fn help() {
    println!("Tabu Search with Strategic Oscillation for Quadratic Knapsack");
    println!();
    println!("Uses tabu search that explores infeasible regions via an adaptive penalty");
    println!("factor (alpha) that oscillates the search around the feasibility boundary.");
    println!("Includes greedy initialization, multi-level perturbation on stall, and a");
    println!("post-search polish phase with k-for-k swaps.");
    println!();
    println!("Hyperparameters:");
    println!("  alpha_factor (f64, default 1.0005)        - Multiplier for adaptive penalty adjustment each iteration");
    println!("  tabu_tenure_scale (f64, default 1.0)      - Scales tabu tenure relative to sqrt(n)");
    println!("  stall_limit_numerator (usize, default 5000000) - Numerator for stall limit (divided by n)");
    println!("  stall_limit_min (usize, default 5000)     - Minimum stall limit before perturbation");
    println!("  max_iters (usize, default 1000)           - Maximum tabu-search iterations");
    println!("  perturb_small_frac (f64, default 20.0)    - Divisor for small perturbation count (n / frac)");
    println!("  perturb_medium_frac (f64, default 15.0)   - Divisor for medium perturbation count (n / frac)");
    println!("  p_small_perturb (u32, default 50)         - Probability (0-100) of small perturbation on stall");
    println!("  p_medium_perturb (u32, default 85)        - Cumulative probability threshold for medium perturbation");
    println!("  max_swap_k (usize, default 2)             - Max swap order in polish phase (1 = 1-for-1 only, 2 = up to 2-for-2)");
    println!("  greedy_noise_low (f64, default 0.5)       - Lower bound of noise multiplier for restart greedy");
    println!("  greedy_noise_high (f64, default 1.5)      - Upper bound of noise multiplier for restart greedy");
}


#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Hyperparameters {
    pub alpha_factor: f64,
    pub tabu_tenure_scale: f64,
    pub stall_limit_numerator: usize,
    pub stall_limit_min: usize,
    pub max_iters: usize,
    pub perturb_small_frac: f64,
    pub perturb_medium_frac: f64,
    pub p_small_perturb: u32,
    pub p_medium_perturb: u32,
    pub max_swap_k: usize,
    pub greedy_noise_low: f64,
    pub greedy_noise_high: f64,
}

impl Hyperparameters {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            alpha_factor: 1.00025,
            tabu_tenure_scale: 1.2,
            stall_limit_numerator: 5_000_000,
            stall_limit_min: 5_000,
            max_iters: 1_000,
            perturb_small_frac: 20.0,
            perturb_medium_frac: 15.0,
            p_small_perturb: 50,
            p_medium_perturb: 85,
            max_swap_k: 2,
            greedy_noise_low: 0.5,
            greedy_noise_high: 1.5,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("alpha_factor").and_then(|v| v.as_f64()) { p.alpha_factor = v; }
            if let Some(v) = m.get("tabu_tenure_scale").and_then(|v| v.as_f64()) { p.tabu_tenure_scale = v; }
            if let Some(v) = m.get("stall_limit_numerator").and_then(|v| v.as_u64()) { p.stall_limit_numerator = v as usize; }
            if let Some(v) = m.get("stall_limit_min").and_then(|v| v.as_u64()) { p.stall_limit_min = v as usize; }
            if let Some(v) = m.get("max_iters").and_then(|v| v.as_u64()) { p.max_iters = v as usize; }
            if let Some(v) = m.get("perturb_small_frac").and_then(|v| v.as_f64()) { p.perturb_small_frac = v; }
            if let Some(v) = m.get("perturb_medium_frac").and_then(|v| v.as_f64()) { p.perturb_medium_frac = v; }
            if let Some(v) = m.get("p_small_perturb").and_then(|v| v.as_u64()) { p.p_small_perturb = v as u32; }
            if let Some(v) = m.get("p_medium_perturb").and_then(|v| v.as_u64()) { p.p_medium_perturb = v as u32; }
            if let Some(v) = m.get("max_swap_k").and_then(|v| v.as_u64()) { p.max_swap_k = v as usize; }
            if let Some(v) = m.get("greedy_noise_low").and_then(|v| v.as_f64()) { p.greedy_noise_low = v; }
            if let Some(v) = m.get("greedy_noise_high").and_then(|v| v.as_f64()) { p.greedy_noise_high = v; }
        }
        p
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // EVOLVE-BLOCK-START

    // APPROACH: Tabu Search with Strategic Oscillation
    //
    // This fundamentally different approach uses a Tabu Search algorithm that is allowed
    // to explore infeasible regions (Strategic Oscillation). It dynamically penalizes
    // constraint violations using an adaptive penalty factor `alpha`.
    //
    // 1. The neighborhood consists of all possible single-item flips (Add or Remove).
    // 2. `alpha` is adjusted every iteration: increased if infeasible, decreased if feasible.
    //    This forces the search to naturally oscillate around the feasibility boundary,
    //    implicitly exploring complex swaps (Add-then-Remove or Remove-then-Add) without
    //    the O(n^2) cost of explicit swap evaluation.
    // 3. A Tabu list prevents reversing recent moves to escape local optima.
    // 4. Strong perturbations are applied if the best feasible solution stalls.
    // 5. Uses a flat adjacency list for maximum cache locality and speed.

    // ---- Phase: adjacency-list construction ----
    #[inline(never)]
    fn build_adjacency_list(
        n: usize,
        interaction_values: &[Vec<i32>],
    ) -> (Vec<usize>, Vec<(usize, i64)>) {
        // Build symmetric sparse adjacency lists (CSR) using a two-pass method:
        // 1) Count degrees by scanning only the upper triangle.
        // 2) Prefix-sum to get offsets; allocate exact capacity.
        // 3) Fill edges in a second scan without per-node Vec allocations.
        let mut deg = vec![0usize; n];
        for i in 0..n {
            let row = &interaction_values[i];
            for j in (i + 1)..n {
                let w = row[j];
                if w > 0 {
                    deg[i] += 1;
                    deg[j] += 1;
                }
            }
        }

        let mut adj_offsets = vec![0usize; n + 1];
        for i in 0..n {
            adj_offsets[i + 1] = adj_offsets[i] + deg[i];
        }
        let m = adj_offsets[n];
        let mut adj_edges = vec![(0usize, 0i64); m];
        let mut next_pos = adj_offsets[..n].to_vec();

        for i in 0..n {
            let row = &interaction_values[i];
            for j in (i + 1)..n {
                let w = row[j];
                if w > 0 {
                    let w64 = w as i64;
                    let p_i = next_pos[i];
                    adj_edges[p_i] = (j, w64);
                    next_pos[i] += 1;

                    let p_j = next_pos[j];
                    adj_edges[p_j] = (i, w64);
                    next_pos[j] += 1;
                }
            }
        }
        (adj_offsets, adj_edges)
    }

    // ---- Phase: greedy initialisation ----
    #[inline(never)]
    fn greedy_init(
        n: usize,
        values: &[u32],
        weights: &[i64],
        max_weight_i64: i64,
        adj_offsets: &[usize],
        adj_edges: &[(usize, i64)],
        state: &mut SolutionState,
    ) -> (Vec<f64>, Vec<usize>) {
        let mut potential = vec![0.0; n];
        for i in 0..n {
            let mut pot = values[i] as f64;
            let start = adj_offsets[i];
            let end = adj_offsets[i + 1];
            for k in start..end {
                pot += adj_edges[k].1 as f64;
            }
            potential[i] = pot / weights[i] as f64;
        }

        let mut candidates: Vec<usize> = (0..n).collect();
        candidates.sort_unstable_by(|&a, &b| potential[b].partial_cmp(&potential[a]).unwrap());

        for &i in &candidates {
            if state.weight + weights[i] <= max_weight_i64 {
                flip(i, state, adj_offsets, adj_edges, weights);
            }
        }

        (potential, candidates)
    }

    // ---- Phase: perturbation logic ----
    #[inline(never)]
    fn perturb(
        state: &mut SolutionState,
        best_feasible_selected: &[bool],
        rng: &mut SmallRng,
        n: usize,
        perturb_small_count: usize,
        perturb_medium_count: usize,
        p_small_perturb: u32,
        p_medium_perturb: u32,
        greedy_noise_low: f64,
        greedy_noise_high: f64,
        adj_offsets: &[usize],
        adj_edges: &[(usize, i64)],
        weights_i64: &[i64],
        potential: &[f64],
        candidates: &mut Vec<usize>,
        max_weight_i64: i64,
    ) -> f64 {
        let r = rng.gen_range(0..100);
        if r < p_small_perturb {
            // Revert to best and small perturb
            for i in 0..n {
                if state.selected[i] != best_feasible_selected[i] {
                    flip(i, state, adj_offsets, adj_edges, weights_i64);
                }
            }
            for _ in 0..perturb_small_count {
                let i = rng.gen_range(0..n);
                flip(i, state, adj_offsets, adj_edges, weights_i64);
            }
        } else if r < p_medium_perturb {
            // Revert to best and balanced medium perturb
            for i in 0..n {
                if state.selected[i] != best_feasible_selected[i] {
                    flip(i, state, adj_offsets, adj_edges, weights_i64);
                }
            }
            let mut sel = Vec::new();
            let mut unsel = Vec::new();
            for i in 0..n {
                if state.selected[i] { sel.push(i); } else { unsel.push(i); }
            }
            sel.shuffle(rng);
            unsel.shuffle(rng);
            for &i in sel.iter().take(perturb_medium_count) {
                flip(i, state, adj_offsets, adj_edges, weights_i64);
            }
            for &i in unsel.iter().take(perturb_medium_count) {
                flip(i, state, adj_offsets, adj_edges, weights_i64);
            }
        } else {
            // Restart
            for i in 0..n {
                if state.selected[i] {
                    flip(i, state, adj_offsets, adj_edges, weights_i64);
                }
            }
            let mut rand_pot = vec![0.0; n];
            for i in 0..n {
                rand_pot[i] = potential[i] * rng.gen_range(greedy_noise_low..greedy_noise_high);
            }
            candidates.sort_unstable_by(|&a, &b| {
                rand_pot[b].partial_cmp(&rand_pot[a]).unwrap()
            });
            for &i in candidates.iter() {
                if state.weight + weights_i64[i] <= max_weight_i64 {
                    flip(i, state, adj_offsets, adj_edges, weights_i64);
                }
            }
        }

        if state.score > 0 && max_weight_i64 > 0 {
            (state.score as f64) / (max_weight_i64 as f64)
        } else {
            1.0
        }
    }

    // ---- Phase: main tabu search loop ----
    #[inline(never)]
    fn tabu_search(
        n: usize,
        max_weight_i64: i64,
        weights_i64: &[i64],
        adj_offsets: &[usize],
        adj_edges: &[(usize, i64)],
        hp: &Hyperparameters,
        rng: &mut SmallRng,
        state: &mut SolutionState,
        initial_alpha: f64,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        potential: &[f64],
        candidates: &mut Vec<usize>,
    ) -> (i64, Vec<bool>) {
        let mut best_feasible_score = state.score;
        let mut best_feasible_selected = state.selected.clone();

        let items: Vec<usize> = best_feasible_selected.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect();
        let _ = save_solution(&Solution { items });

        let mut alpha = initial_alpha;
        let alpha_factor = hp.alpha_factor;
        let t_min = ((n as f64).sqrt() * hp.tabu_tenure_scale) as usize;
        let t_max = t_min + ((n as f64).sqrt() * hp.tabu_tenure_scale) as usize / 2;
        let stall_limit = (hp.stall_limit_numerator / n).max(hp.stall_limit_min);

        let perturb_small_count = (n as f64 / hp.perturb_small_frac).max(2.0) as usize;
        let perturb_medium_count = (n as f64 / hp.perturb_medium_frac).max(2.0) as usize;

        let mut tabu_until = vec![0_usize; n];
        let mut last_improve = 0;
        let mut iter = 0;

        loop {
            iter += 1;

            let mut best_flip = usize::MAX;
            let mut best_fitness = f64::NEG_INFINITY;
            let mut best_tabu_flip = usize::MAX;
            let mut best_tabu_fitness = f64::NEG_INFINITY;

            let current_weight = state.weight;
            let current_score = state.score;

            for i in 0..n {
                let sign = state.selected_sign[i];
                let new_weight = current_weight + weights_i64[i] * sign;
                let new_score = current_score + state.contrib[i] * sign;

                let fitness = new_score as f64 - new_weight as f64 * alpha;

                if iter >= tabu_until[i] || (new_weight <= max_weight_i64 && new_score > best_feasible_score) {
                    if fitness > best_fitness {
                        best_fitness = fitness;
                        best_flip = i;
                    }
                } else {
                    if fitness > best_tabu_fitness {
                        best_tabu_fitness = fitness;
                        best_tabu_flip = i;
                    }
                }
            }

            let chosen_flip = if best_flip != usize::MAX {
                best_flip
            } else {
                best_tabu_flip
            };

            flip(chosen_flip, state, adj_offsets, adj_edges, weights_i64);
            tabu_until[chosen_flip] = iter + rng.gen_range(t_min..=t_max);

            if state.weight <= max_weight_i64 && state.score > best_feasible_score {
                best_feasible_score = state.score;
                best_feasible_selected.clone_from(&state.selected);
                last_improve = iter;

                let items: Vec<usize> = best_feasible_selected.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect();
                let _ = save_solution(&Solution { items });
            }

            if state.weight > max_weight_i64 {
                alpha *= alpha_factor;
            } else {
                alpha /= alpha_factor;
            }
            alpha = alpha.clamp(0.0001, 1e9);

            if iter - last_improve > stall_limit {
                alpha = perturb(
                    state,
                    &best_feasible_selected,
                    rng,
                    n,
                    perturb_small_count,
                    perturb_medium_count,
                    hp.p_small_perturb,
                    hp.p_medium_perturb,
                    hp.greedy_noise_low,
                    hp.greedy_noise_high,
                    adj_offsets,
                    adj_edges,
                    weights_i64,
                    potential,
                    candidates,
                    max_weight_i64,
                );
                tabu_until.fill(0);
                last_improve = iter;

                if state.weight <= max_weight_i64 && state.score > best_feasible_score {
                    best_feasible_score = state.score;
                    best_feasible_selected.clone_from(&state.selected);
                    let items: Vec<usize> = best_feasible_selected.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect();
                    let _ = save_solution(&Solution { items });
                }
            }

            if iter >= hp.max_iters {
                break;
            }
        }

        (best_feasible_score, best_feasible_selected)
    }

    // ---- Phase: polish / post-processing ----
    #[inline(never)]
    fn polish(
        n: usize,
        interaction_values: &[Vec<i32>],
        adj_offsets: &[usize],
        adj_edges: &[(usize, i64)],
        weights_i64: &[i64],
        max_weight_i64: i64,
        max_swap_k: usize,
        values: &[u32],
        initial_selected: &[bool],
    ) -> Vec<usize> {
        // Fast path for large instances: avoid expensive swap-based search.
        // Strategy: start from given selection, drop nonpositive contributors, then greedily add positive-gain items that fit.
        if n >= 3000 {
            let mut final_state = SolutionState::new(n, values);
            for (i, &sel) in initial_selected.iter().enumerate() {
                if sel {
                    flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                }
            }
            for i in 0..n {
                if final_state.selected[i] && final_state.contrib[i] <= 0 {
                    flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                }
            }
            loop {
                let mut best_i = usize::MAX;
                let mut best_score = 0.0;
                for i in 0..n {
                    if !final_state.selected[i] && final_state.weight + weights_i64[i] <= max_weight_i64 && final_state.contrib[i] > 0 {
                        let score = final_state.contrib[i] as f64 / weights_i64[i] as f64;
                        if score > best_score {
                            best_score = score;
                            best_i = i;
                        }
                    }
                }
                if best_i == usize::MAX { break; }
                flip(best_i, &mut final_state, adj_offsets, adj_edges, weights_i64);
            }
            return final_state.selected.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect();
        }

        let mut final_state = SolutionState::new(n, values);
        for (i, &sel) in initial_selected.iter().enumerate() {
            if sel {
                flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
            }
        }

        let mut improved = true;
        while improved {
            improved = false;

            for i in 0..n {
                if final_state.selected[i] && final_state.contrib[i] <= 0 {
                    flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                    improved = true;
                }
            }

            loop {
                let mut best_i = usize::MAX;
                let mut best_score = 0.0;
                for i in 0..n {
                    if !final_state.selected[i] && final_state.weight + weights_i64[i] <= max_weight_i64 && final_state.contrib[i] > 0 {
                        let score = final_state.contrib[i] as f64 / weights_i64[i] as f64;
                        if score > best_score {
                            best_score = score;
                            best_i = i;
                        }
                    }
                }
                if best_i == usize::MAX { break; }
                flip(best_i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                improved = true;
            }

            let mut sel = Vec::with_capacity(n);
            let mut unsel = Vec::with_capacity(n);
            for i in 0..n {
                if final_state.selected[i] { sel.push(i); } else { unsel.push(i); }
            }
            sel.sort_unstable_by_key(|&i| final_state.contrib[i]);
            unsel.sort_unstable_by_key(|&j| std::cmp::Reverse(final_state.contrib[j]));

            if sel.is_empty() || unsel.is_empty() { break; }

            // 1-for-1 swaps (always attempted)
            'outer_1_1: for &i in &sel {
                if final_state.contrib[unsel[0]] - final_state.contrib[i] < 0 {
                    break;
                }
                let row_i = &interaction_values[i];
                for &j in &unsel {
                    let diff = final_state.contrib[j] - final_state.contrib[i];
                    if diff < 0 { break; }
                    let w_diff = weights_i64[j] - weights_i64[i];
                    if final_state.weight + w_diff <= max_weight_i64 {
                        let inter = row_i[j] as i64;
                        let delta = diff - inter;
                        if delta > 0 || (delta == 0 && w_diff < 0) {
                            flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                            flip(j, &mut final_state, adj_offsets, adj_edges, weights_i64);
                            improved = true;
                            break 'outer_1_1;
                        }
                    }
                }
            }

            // Higher-order swaps gated by max_swap_k
            if !improved && max_swap_k >= 2
            {
                let max_unsel_contrib = final_state.contrib[unsel[0]];

                // 1-for-2 swaps
                'outer_1_2: for &i in &sel {
                    if -final_state.contrib[i] + 2 * max_unsel_contrib + 100 < 0 {
                        break;
                    }
                    let row_i = &interaction_values[i];
                    for idx_j1 in 0..unsel.len() {
                        let j1 = unsel[idx_j1];
                        if -final_state.contrib[i] + 2 * final_state.contrib[j1] + 100 < 0 {
                            break;
                        }
                        let base_diff = -final_state.contrib[i] + final_state.contrib[j1];
                        let row_j1 = &interaction_values[j1];

                        for idx_j2 in (idx_j1 + 1)..unsel.len() {
                            let j2 = unsel[idx_j2];
                            if base_diff + final_state.contrib[j2] + 100 < 0 {
                                break;
                            }
                            let w_diff = weights_i64[j1] + weights_i64[j2] - weights_i64[i];
                            if final_state.weight + w_diff <= max_weight_i64 {
                                let inter_i_j1 = row_i[j1] as i64;
                                let inter_i_j2 = row_i[j2] as i64;
                                let inter_j1_j2 = row_j1[j2] as i64;

                                let delta = base_diff + final_state.contrib[j2]
                                          + inter_j1_j2 - inter_i_j1 - inter_i_j2;
                                if delta > 0 || (delta == 0 && w_diff < 0) {
                                    flip(i, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                    flip(j1, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                    flip(j2, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                    improved = true;
                                    break 'outer_1_2;
                                }
                            }
                        }
                    }
                }

                // 2-for-1 swaps
                if !improved {
                    'outer_2_1: for idx_i1 in 0..sel.len() {
                        let i1 = sel[idx_i1];
                        if -2 * final_state.contrib[i1] + max_unsel_contrib + 100 < 0 {
                            break;
                        }
                        let row_i1 = &interaction_values[i1];
                        for idx_i2 in (idx_i1 + 1)..sel.len() {
                            let i2 = sel[idx_i2];
                            if -final_state.contrib[i1] - final_state.contrib[i2] + max_unsel_contrib + 100 < 0 {
                                break;
                            }
                            let row_i2 = &interaction_values[i2];
                            let inter_i1_i2 = row_i1[i2] as i64;
                            let base_diff = -final_state.contrib[i1] - final_state.contrib[i2] + inter_i1_i2;

                            for &j in &unsel {
                                if final_state.contrib[j] + base_diff < 0 {
                                    break;
                                }
                                let w_diff = weights_i64[j] - weights_i64[i1] - weights_i64[i2];
                                if final_state.weight + w_diff <= max_weight_i64 {
                                    let inter_i1_j = row_i1[j] as i64;
                                    let inter_i2_j = row_i2[j] as i64;

                                    let delta = final_state.contrib[j] + base_diff - inter_i1_j - inter_i2_j;

                                    if delta > 0 || (delta == 0 && w_diff < 0) {
                                        flip(i1, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                        flip(i2, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                        flip(j, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                        improved = true;
                                        break 'outer_2_1;
                                    }
                                }
                            }
                        }
                    }
                }

                // 2-for-2 swaps
                if !improved {
                    'outer_2_2: for idx_i1 in 0..sel.len() {
                        let i1 = sel[idx_i1];
                        if -2 * final_state.contrib[i1] + 2 * max_unsel_contrib + 100 < 0 {
                            break;
                        }
                        let row_i1 = &interaction_values[i1];
                        for idx_i2 in (idx_i1 + 1)..sel.len() {
                            let i2 = sel[idx_i2];
                            if -final_state.contrib[i1] - final_state.contrib[i2] + 2 * max_unsel_contrib + 100 < 0 {
                                break;
                            }
                            let row_i2 = &interaction_values[i2];
                            let inter_i1_i2 = row_i1[i2] as i64;
                            let base_diff_i = -final_state.contrib[i1] - final_state.contrib[i2] + inter_i1_i2;

                            for idx_j1 in 0..unsel.len() {
                                let j1 = unsel[idx_j1];
                                if base_diff_i + 2 * final_state.contrib[j1] + 100 < 0 {
                                    break;
                                }
                                let base_diff_j1 = base_diff_i + final_state.contrib[j1];
                                let row_j1 = &interaction_values[j1];

                                for idx_j2 in (idx_j1 + 1)..unsel.len() {
                                    let j2 = unsel[idx_j2];
                                    if base_diff_j1 + final_state.contrib[j2] + 100 < 0 {
                                        break;
                                    }

                                    let w_diff = weights_i64[j1] + weights_i64[j2] - weights_i64[i1] - weights_i64[i2];
                                    if final_state.weight + w_diff <= max_weight_i64 {
                                        let inter_i1_j1 = row_i1[j1] as i64;
                                        let inter_i1_j2 = row_i1[j2] as i64;
                                        let inter_i2_j1 = row_i2[j1] as i64;
                                        let inter_i2_j2 = row_i2[j2] as i64;
                                        let inter_j1_j2 = row_j1[j2] as i64;

                                        let delta = base_diff_j1 + final_state.contrib[j2]
                                                  + inter_j1_j2
                                                  - inter_i1_j1 - inter_i1_j2 - inter_i2_j1 - inter_i2_j2;

                                        if delta > 0 || (delta == 0 && w_diff < 0) {
                                            flip(i1, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                            flip(i2, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                            flip(j1, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                            flip(j2, &mut final_state, adj_offsets, adj_edges, weights_i64);
                                            improved = true;
                                            break 'outer_2_2;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        final_state.selected.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect()
    }

    struct SolutionState {
        selected: Vec<bool>,
        selected_sign: Vec<i64>,
        contrib: Vec<i64>,
        weight: i64,
        score: i64,
    }

    impl SolutionState {
        fn new(n: usize, values: &[u32]) -> Self {
            Self {
                selected: vec![false; n],
                selected_sign: vec![1; n],
                contrib: values.iter().map(|&v| v as i64).collect(),
                weight: 0,
                score: 0,
            }
        }
    }

    #[inline]
    fn flip(
        i: usize,
        state: &mut SolutionState,
        adj_offsets: &[usize],
        adj_edges: &[(usize, i64)],
        weights: &[i64],
    ) {
        state.selected[i] = !state.selected[i];
        state.selected_sign[i] = -state.selected_sign[i];
        let sign = -state.selected_sign[i]; // 1 if selected, -1 if removed
        state.weight += weights[i] * sign;

        let start = adj_offsets[i];
        let end = adj_offsets[i + 1];

        if sign == 1 {
            state.score += state.contrib[i];
            for k in start..end {
                let (j, w) = adj_edges[k];
                state.contrib[j] += w;
            }
        } else {
            for k in start..end {
                let (j, w) = adj_edges[k];
                state.contrib[j] -= w;
            }
            state.score -= state.contrib[i];
        }
    }

    let hyperparameters = Hyperparameters::initialize(hyperparameters);

    let n = challenge.values.len();
    let max_weight = challenge.max_weight;
    let weights = &challenge.weights;
    let mut rng = SmallRng::from_seed(challenge.seed);

    // Build flat adjacency list for fast interaction lookups
    let (adj_offsets, adj_edges) = build_adjacency_list(n, &challenge.interaction_values);

    let weights_i64: Vec<i64> = weights.iter().map(|&w| w as i64).collect();
    let max_weight_i64 = max_weight as i64;
    let mut state = SolutionState::new(n, &challenge.values);

    // Greedy initialization based on potential density
    let (potential, mut candidates) = greedy_init(
        n,
        &challenge.values,
        &weights_i64,
        max_weight_i64,
        &adj_offsets,
        &adj_edges,
        &mut state,
    );

    let initial_alpha = if state.score > 0 && max_weight > 0 {
        (state.score as f64) / (max_weight as f64)
    } else {
        1.0
    };

    let (_best_feasible_score, best_feasible_selected) = tabu_search(
        n,
        max_weight_i64,
        &weights_i64,
        &adj_offsets,
        &adj_edges,
        &hyperparameters,
        &mut rng,
        &mut state,
        initial_alpha,
        save_solution,
        &potential,
        &mut candidates,
    );

    let selected = polish(
        n,
        &challenge.interaction_values,
        &adj_offsets,
        &adj_edges,
        &weights_i64,
        max_weight_i64,
        hyperparameters.max_swap_k,
        &challenge.values,
        &best_feasible_selected,
    );
    // EVOLVE-BLOCK-END

    save_solution(&Solution { items: selected })?;
    Ok(())
}
