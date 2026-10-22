use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

/// Deterministic 64-bit Xorshift PRNG
#[derive(Clone, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x8a5cd789635d2dff } else { seed };
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn gen_range(&mut self, min: u64, max: u64) -> u64 {
        debug_assert!(max >= min);
        let span = max - min + 1;
        min + (self.next_u64() % span)
    }
}

pub const DEFAULT_DEGREE_WEIGHT_NUM: i64 = 1164;
pub const DEFAULT_POWER_W: f64 = 0.80;
pub const DEFAULT_TABU_BASE_TENURE: usize = 5;
pub const DEFAULT_RUIN_RATE: f64 = 0.045;
pub const DEFAULT_MAX_KICK_ROUNDS: usize = 8;

#[inline]
pub fn select_scale_aware_elite_k(num_items: usize) -> usize {
    if num_items <= 600 {
        22
    } else {
        28
    }
}

#[inline]
pub fn select_scale_aware_fuel_budget(num_items: usize) -> u64 {
    if num_items <= 600 {
        1_200_000
    } else {
        5_000_000
    }
}

#[inline]
pub fn fold_seed_u64(seed: &[u8; 32]) -> u64 {
    let w0 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(seed[8..16].try_into().unwrap());
    let w2 = u64::from_le_bytes(seed[16..24].try_into().unwrap());
    let w3 = u64::from_le_bytes(seed[24..32].try_into().unwrap());
    w0 ^ w1.rotate_left(13) ^ w2.rotate_left(29) ^ w3.rotate_left(47)
}

/// Optional hyperparameters for stage_b2_multi_swap knapsack solver
#[derive(Clone, Debug, Default)]
pub struct Hyperparameters {
    pub degree_weight_num: Option<i64>,
    pub power_w: Option<f64>,
    pub tabu_base_tenure: Option<usize>,
    pub ruin_rate: Option<f64>,
    pub max_kick_rounds: Option<usize>,
    pub fuel_budget: Option<u64>,
    pub elite_k: Option<usize>,
}

impl Hyperparameters {
    pub fn from_map(map: &Map<String, Value>) -> Result<Self> {
        let mut hp = Self::default();
        for (k, v) in map.iter() {
            match k.as_str() {
                "degree_weight_num" => {
                    let val = v.as_i64().ok_or_else(|| anyhow!("Invalid degree_weight_num: expected integer"))?;
                    if !(0..=1_000_000).contains(&val) {
                        return Err(anyhow!("Invalid degree_weight_num: must be in 0..=1_000_000"));
                    }
                    hp.degree_weight_num = Some(val);
                }
                "power_w" => {
                    let val = v.as_f64().ok_or_else(|| anyhow!("Invalid power_w: expected f64"))?;
                    if !val.is_finite() || val <= 0.0 || val > 10.0 {
                        return Err(anyhow!("Invalid power_w: must be finite and in (0.0, 10.0]"));
                    }
                    hp.power_w = Some(val);
                }
                "tabu_base_tenure" => {
                    let val = v.as_u64().ok_or_else(|| anyhow!("Invalid tabu_base_tenure: expected u64"))?;
                    if val > 10_000 {
                        return Err(anyhow!("Invalid tabu_base_tenure: must be <= 10_000"));
                    }
                    hp.tabu_base_tenure = Some(val as usize);
                }
                "ruin_rate" => {
                    let val = v.as_f64().ok_or_else(|| anyhow!("Invalid ruin_rate: expected f64"))?;
                    if !val.is_finite() || val < 0.0 || val > 1.0 {
                        return Err(anyhow!("Invalid ruin_rate: must be in [0.0, 1.0]"));
                    }
                    hp.ruin_rate = Some(val);
                }
                "max_kick_rounds" => {
                    let val = v.as_u64().ok_or_else(|| anyhow!("Invalid max_kick_rounds: expected u64"))?;
                    if val > 1024 {
                        return Err(anyhow!("Invalid max_kick_rounds: must be <= 1024"));
                    }
                    hp.max_kick_rounds = Some(val as usize);
                }
                "fuel_budget" => {
                    let val = v.as_u64().ok_or_else(|| anyhow!("Invalid fuel_budget: expected u64"))?;
                    hp.fuel_budget = Some(val);
                }
                "elite_k" => {
                    let val = v.as_u64().ok_or_else(|| anyhow!("Invalid elite_k: expected u64"))?;
                    if val > 10_000 {
                        return Err(anyhow!("Invalid elite_k: must be <= 10_000"));
                    }
                    hp.elite_k = Some(val as usize);
                }
                other => {
                    return Err(anyhow!("Unknown hyperparameter: {}", other));
                }
            }
        }
        Ok(hp)
    }
}

/// Prints concise summary of the 4-stage solver and its scale-aware defaults
pub fn help() {
    println!("Stage B2 Multi-Swap Solver for TIG Knapsack Challenge");
    println!("Stages: A (Warm Start) -> B (1-for-1 Tabu Search) -> C (Plateau Ruin/Recreate Kick) -> B2 (Monotonic Elite-K Multi-Swap Polish)");
    println!("Defaults: Scale-aware elite_k (22 for N<=600, 28 for N>600), fuel_budget (1.2M for N<=600, 5.0M for N>600)");
    println!("Parameters: power_w=0.80, ruin_rate=0.045, degree_weight_num=1164, tabu_base_tenure=5, max_kick_rounds=8");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    if n == 0 || n > 20_000 {
        return Err(anyhow!("Invalid challenge num_items: {}", n));
    }
    if challenge.weights.len() != n || challenge.values.len() != n || challenge.interaction_values.len() != n {
        return Err(anyhow!("Dimension mismatch in challenge arrays"));
    }
    let total_weight_sum: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if total_weight_sum > u32::MAX as u64 {
        return Err(anyhow!("Total item weight sum ({}) overflows u32::MAX", total_weight_sum));
    }
    for (i, row) in challenge.interaction_values.iter().enumerate() {
        if row.len() != n {
            return Err(anyhow!("interaction_values row {} length {} != {}", i, row.len(), n));
        }
    }

    let parsed_hp = match hyperparameters {
        Some(map) => Some(Hyperparameters::from_map(map)?),
        None => None,
    };

    let elite_k = parsed_hp
        .as_ref()
        .and_then(|h| h.elite_k)
        .unwrap_or_else(|| select_scale_aware_elite_k(n));

    let min_stage_a_fuel = ((2 * n * n) + (2 * n)) as u64;
    let fuel_budget = match parsed_hp.as_ref().and_then(|h| h.fuel_budget) {
        Some(fb) => {
            if fb < min_stage_a_fuel {
                return Err(anyhow!(
                    "fuel_budget ({}) must be >= Stage A initialization cost 2*n^2 + 2*n ({})",
                    fb,
                    min_stage_a_fuel
                ));
            }
            fb
        }
        None => select_scale_aware_fuel_budget(n).max(min_stage_a_fuel),
    };

    let degree_weight_num = parsed_hp.as_ref().and_then(|h| h.degree_weight_num).unwrap_or(DEFAULT_DEGREE_WEIGHT_NUM);
    let power_w = parsed_hp.as_ref().and_then(|h| h.power_w).unwrap_or(DEFAULT_POWER_W);
    let tabu_base_tenure = parsed_hp.as_ref().and_then(|h| h.tabu_base_tenure).unwrap_or(DEFAULT_TABU_BASE_TENURE);
    let ruin_rate = parsed_hp.as_ref().and_then(|h| h.ruin_rate).unwrap_or(DEFAULT_RUIN_RATE);
    let max_kick_rounds = parsed_hp.as_ref().and_then(|h| h.max_kick_rounds).unwrap_or(DEFAULT_MAX_KICK_ROUNDS);

    let seed_u64 = fold_seed_u64(&challenge.seed);

    let mut q_flat = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let v = challenge.interaction_values[i][j] as i64;
                q_flat[i * n + j] = v;
            }
        }
    }

    let mut fuel_ops: u64 = 0;
    let mut rng = Xorshift64::new(seed_u64);
    let mut kick_rng = Xorshift64::new(seed_u64 ^ 0x9E37_79B9_7F4A_7C15);

    let mut sum_q = vec![0i64; n];
    for i in 0..n {
        let mut s: i64 = 0;
        let row_offset = i * n;
        for j in 0..n {
            s += q_flat[row_offset + j];
        }
        sum_q[i] = s;
    }
    fuel_ops += (n * n) as u64;

    let mut scores: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let num = ((challenge.values[i] as i64) * 1024 + degree_weight_num * sum_q[i]) as f64;
            let den = (challenge.weights[i] as f64).powf(power_w).max(1.0);
            (i, num / den)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    fuel_ops += n as u64;

    let mut selected = vec![0u8; n];
    let mut current_weight: u32 = 0;
    for &(idx, _) in &scores {
        fuel_ops += 1;
        if current_weight + challenge.weights[idx] <= challenge.max_weight {
            selected[idx] = 1;
            current_weight += challenge.weights[idx];
        }
    }

    let sel_indices: Vec<usize> = (0..n).filter(|&i| selected[i] == 1).collect();
    let mut delta = vec![0i64; n];
    let mut current_obj: i64 = 0;
    for k in 0..n {
        delta[k] = challenge.values[k] as i64;
        let row_offset = k * n;
        for &j in &sel_indices {
            if j != k {
                delta[k] += q_flat[row_offset + j];
            }
        }
        if selected[k] == 1 {
            current_obj += challenge.values[k] as i64;
            for &j in &sel_indices {
                if j > k {
                    current_obj += q_flat[row_offset + j];
                }
            }
        }
    }
    fuel_ops += (n * n) as u64;

    let mut best_selected = selected.clone();
    let mut best_obj = current_obj;
    let mut best_weight = current_weight;
    let mut best_delta = delta.clone();

    let mut tabu_until = vec![0u64; n];
    let mut iteration_step: u64 = 0;
    let mut kick_rounds: usize = 0;
    let mut polish_phase = false;
    let mut plateau_basins: Vec<(i64, u32, Vec<u8>, Vec<i64>)> = Vec::with_capacity(max_kick_rounds);
    let swap_scan_limit = fuel_budget.saturating_sub(n as u64);

    while fuel_ops < swap_scan_limit && (kick_rounds < max_kick_rounds || polish_phase) {
        let mut improved = false;

        if fuel_ops + (2 * n as u64) <= fuel_budget {
            let mut best_add: Option<(usize, i64)> = None;
            let mut best_add_gain = 0i64;
            for k in 0..n {
                fuel_ops += 1;
                if selected[k] == 0 && current_weight + challenge.weights[k] <= challenge.max_weight {
                    let gain = delta[k];
                    if gain > best_add_gain {
                        best_add_gain = gain;
                        best_add = Some((k, gain));
                    }
                }
            }

            if let Some((add_idx, gain)) = best_add {
                selected[add_idx] = 1;
                current_weight += challenge.weights[add_idx];
                current_obj += gain;

                let row_offset = add_idx * n;
                for k in 0..n {
                    if k != add_idx {
                        delta[k] += q_flat[row_offset + k];
                    }
                }
                fuel_ops += n as u64;

                if current_obj > best_obj {
                    best_obj = current_obj;
                    best_selected = selected.clone();
                    best_weight = current_weight;
                    best_delta = delta.clone();
                }
                improved = true;
            }
        }

        let mut best_swap: Option<(usize, usize, i64, u32)> = None;
        let mut best_swap_gain = 0i64;
        iteration_step += 1;

        'outer_multi_1to1: for a in 0..n {
            if selected[a] == 0 {
                continue;
            }
            let w_a = challenge.weights[a];
            let a_offset = a * n;
            for b in 0..n {
                if selected[b] == 1 || delta[b] - delta[a] <= best_swap_gain {
                    continue;
                }
                if fuel_ops >= swap_scan_limit {
                    break 'outer_multi_1to1;
                }
                fuel_ops += 1;
                let w_b = challenge.weights[b];
                let new_weight = current_weight - w_a + w_b;
                if new_weight <= challenge.max_weight {
                    let q_ab = q_flat[a_offset + b];
                    let gain = delta[b] - delta[a] - q_ab;
                    let is_tabu = tabu_until[a] > iteration_step || tabu_until[b] > iteration_step;
                    let aspiration = current_obj + gain > best_obj;

                    if gain > best_swap_gain && (!is_tabu || aspiration) {
                        best_swap_gain = gain;
                        best_swap = Some((a, b, gain, new_weight));
                    }
                }
            }
        }

        if let Some((a, b, gain, new_weight)) = best_swap {
            if gain > 0 {
                selected[a] = 0;
                selected[b] = 1;
                current_weight = new_weight;
                current_obj += gain;

                let b_offset = b * n;
                let a_offset = a * n;
                for k in 0..n {
                    let q_kb = q_flat[b_offset + k];
                    let q_ka = q_flat[a_offset + k];
                    delta[k] += q_kb - q_ka;
                }
                fuel_ops += n as u64;

                let tenure = tabu_base_tenure as u64 + (rng.gen_range(0, 3));
                tabu_until[a] = iteration_step.saturating_add(tenure);
                tabu_until[b] = iteration_step.saturating_add(tenure);

                if current_obj > best_obj {
                    best_obj = current_obj;
                    best_selected = selected.clone();
                    best_weight = current_weight;
                    best_delta = delta.clone();
                }
                improved = true;
            }
        }

        if !improved && !polish_phase {
            if current_obj < best_obj {
                plateau_basins.push((current_obj, current_weight, selected.clone(), delta.clone()));
            }
            if kick_rounds + 1 >= max_kick_rounds {
                polish_phase = true;
                kick_rounds = max_kick_rounds;
                plateau_basins.sort_by_key(|b| b.0);
                selected = best_selected.clone();
                current_weight = best_weight;
                current_obj = best_obj;
                delta = best_delta.clone();
                tabu_until.fill(0);
            }
        }

        if !improved && polish_phase {
            if fuel_ops + (2 * n as u64) > fuel_budget {
                break;
            }
            let slack = (challenge.max_weight as i64) - (current_weight as i64);

            let mut unselected_cands: Vec<(usize, f64)> = Vec::with_capacity(n);
            let mut selected_cands: Vec<(usize, f64)> = Vec::with_capacity(n);

            for i in 0..n {
                let w = challenge.weights[i] as f64;
                let d = delta[i] as f64;
                let density = d / w;
                if selected[i] == 0 {
                    unselected_cands.push((i, density));
                } else {
                    selected_cands.push((i, density));
                }
            }
            fuel_ops += n as u64;

            unselected_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
            selected_cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

            let k_limit = elite_k.min(unselected_cands.len());
            let m_limit = elite_k.min(selected_cands.len());

            let elite_unselected: Vec<usize> = unselected_cands[..k_limit].iter().map(|&(idx, _)| idx).collect();
            let elite_selected: Vec<usize> = selected_cands[..m_limit].iter().map(|&(idx, _)| idx).collect();

            enum MultiSwap {
                OneToTwo { a: usize, b: usize, c: usize, gain: i64, new_wt: u32 },
                TwoToOne { a1: usize, a2: usize, b: usize, gain: i64, new_wt: u32 },
            }

            let mut best_multi: Option<MultiSwap> = None;
            let mut best_multi_gain: i64 = 0;

            'outer_1to2: for &a in &elite_selected {
                let w_a = challenge.weights[a] as i64;
                let max_bc_weight = slack + w_a;
                let a_offset = a * n;
                for i_b in 0..elite_unselected.len() {
                    let b = elite_unselected[i_b];
                    let w_b = challenge.weights[b] as i64;
                    if w_b >= max_bc_weight {
                        continue;
                    }
                    let b_offset = b * n;
                    for i_c in (i_b + 1)..elite_unselected.len() {
                        let c = elite_unselected[i_c];
                        if fuel_ops >= swap_scan_limit {
                            break 'outer_1to2;
                        }
                        fuel_ops += 1;
                        let w_c = challenge.weights[c] as i64;
                        if w_b + w_c <= max_bc_weight {
                            let gain = delta[b] + delta[c] - delta[a]
                                - q_flat[a_offset + b] - q_flat[a_offset + c] + q_flat[b_offset + c];
                            if gain > best_multi_gain {
                                best_multi_gain = gain;
                                let new_wt = (current_weight as i64 - w_a + w_b + w_c) as u32;
                                best_multi = Some(MultiSwap::OneToTwo { a, b, c, gain, new_wt });
                            }
                        }
                    }
                }
            }

            'outer_2to1: for i_a1 in 0..elite_selected.len() {
                let a1 = elite_selected[i_a1];
                let w_a1 = challenge.weights[a1] as i64;
                let a1_offset = a1 * n;
                for i_a2 in (i_a1 + 1)..elite_selected.len() {
                    let a2 = elite_selected[i_a2];
                    let w_a2 = challenge.weights[a2] as i64;
                    let max_b_weight = slack + w_a1 + w_a2;
                    let _a2_offset = a2 * n;
                    for &b in &elite_unselected {
                        if fuel_ops >= swap_scan_limit {
                            break 'outer_2to1;
                        }
                        fuel_ops += 1;
                        let w_b = challenge.weights[b] as i64;
                        if w_b <= max_b_weight {
                            let b_offset = b * n;
                            let gain = delta[b] - delta[a1] - delta[a2]
                                + q_flat[a1_offset + a2] - q_flat[b_offset + a1] - q_flat[b_offset + a2];
                            if gain > best_multi_gain {
                                best_multi_gain = gain;
                                let new_wt = (current_weight as i64 - w_a1 - w_a2 + w_b) as u32;
                                best_multi = Some(MultiSwap::TwoToOne { a1, a2, b, gain, new_wt });
                            }
                        }
                    }
                }
            }

            if let Some(op) = best_multi {
                match op {
                    MultiSwap::OneToTwo { a, b, c, gain, new_wt } => {
                        selected[a] = 0;
                        selected[b] = 1;
                        selected[c] = 1;
                        current_weight = new_wt;
                        current_obj += gain;

                        let b_offset = b * n;
                        let c_offset = c * n;
                        let a_offset = a * n;
                        for k in 0..n {
                            delta[k] += q_flat[b_offset + k] + q_flat[c_offset + k] - q_flat[a_offset + k];
                        }
                        fuel_ops += n as u64;

                        let tenure = tabu_base_tenure as u64 + (rng.gen_range(0, 3));
                        tabu_until[a] = iteration_step.saturating_add(tenure);
                        tabu_until[b] = iteration_step.saturating_add(tenure);
                        tabu_until[c] = iteration_step.saturating_add(tenure);

                        improved = true;
                    }
                    MultiSwap::TwoToOne { a1, a2, b, gain, new_wt } => {
                        selected[a1] = 0;
                        selected[a2] = 0;
                        selected[b] = 1;
                        current_weight = new_wt;
                        current_obj += gain;

                        let b_offset = b * n;
                        let a1_offset = a1 * n;
                        let a2_offset = a2 * n;
                        for k in 0..n {
                            delta[k] += q_flat[b_offset + k] - q_flat[a1_offset + k] - q_flat[a2_offset + k];
                        }
                        fuel_ops += n as u64;

                        let tenure = tabu_base_tenure as u64 + (rng.gen_range(0, 3));
                        tabu_until[a1] = iteration_step.saturating_add(tenure);
                        tabu_until[a2] = iteration_step.saturating_add(tenure);
                        tabu_until[b] = iteration_step.saturating_add(tenure);

                        improved = true;
                    }
                }

                if fuel_ops + (2 * n as u64) <= fuel_budget {
                    let mut best_fill: Option<(usize, i64)> = None;
                    let mut best_fill_gain = 0i64;
                    for k in 0..n {
                        fuel_ops += 1;
                        if selected[k] == 0 && current_weight + challenge.weights[k] <= challenge.max_weight {
                            let fill_gain = delta[k];
                            if fill_gain > best_fill_gain {
                                best_fill_gain = fill_gain;
                                best_fill = Some((k, fill_gain));
                            }
                        }
                    }
                    if let Some((fill_idx, fill_gain)) = best_fill {
                        selected[fill_idx] = 1;
                        current_weight += challenge.weights[fill_idx];
                        current_obj += fill_gain;
                        let fill_offset = fill_idx * n;
                        for k in 0..n {
                            if k != fill_idx {
                                delta[k] += q_flat[fill_offset + k];
                            }
                        }
                        fuel_ops += n as u64;
                    }
                }

                if current_obj > best_obj {
                    best_obj = current_obj;
                    best_selected = selected.clone();
                    best_weight = current_weight;
                    best_delta = delta.clone();
                }
            } else if let Some((p_obj, p_wt, p_sel, p_del)) = plateau_basins.pop() {
                selected = p_sel;
                current_weight = p_wt;
                current_obj = p_obj;
                delta = p_del;
                tabu_until.fill(0);
                improved = true;
            } else {
                break;
            }
        }

        if !improved && !polish_phase {
            kick_rounds += 1;
            if kick_rounds >= max_kick_rounds || fuel_ops >= swap_scan_limit {
                break;
            }

            let selected_indices: Vec<usize> = (0..n).filter(|&i| selected[i] == 1).collect();
            let num_to_drop = ((selected_indices.len() as f64) * ruin_rate).ceil() as usize;
            for _ in 0..num_to_drop {
                if selected_indices.is_empty() || fuel_ops + (n as u64) > fuel_budget {
                    break;
                }
                let idx_in_sel = kick_rng.gen_range(0, selected_indices.len() as u64 - 1) as usize;
                let drop_item = selected_indices[idx_in_sel];
                if selected[drop_item] == 1 {
                    selected[drop_item] = 0;
                    current_weight -= challenge.weights[drop_item];
                    current_obj -= delta[drop_item];
                    let drop_offset = drop_item * n;
                    for k in 0..n {
                        if k != drop_item {
                            delta[k] -= q_flat[drop_offset + k];
                        }
                    }
                    fuel_ops += n as u64;
                }
            }

            let mut candidates: Vec<usize> = (0..n).filter(|&i| selected[i] == 0).collect();
            for i in (1..candidates.len()).rev() {
                let j = kick_rng.gen_range(0, i as u64) as usize;
                candidates.swap(i, j);
            }

            for &cand in &candidates {
                if fuel_ops >= fuel_budget {
                    break;
                }
                fuel_ops += 1;
                if current_weight + challenge.weights[cand] <= challenge.max_weight {
                    if fuel_ops + (n as u64) > fuel_budget {
                        break;
                    }
                    selected[cand] = 1;
                    current_weight += challenge.weights[cand];
                    current_obj += delta[cand];
                    let cand_offset = cand * n;
                    for k in 0..n {
                        if k != cand {
                            delta[k] += q_flat[cand_offset + k];
                        }
                    }
                    fuel_ops += n as u64;
                }
            }

            if current_obj > best_obj {
                best_obj = current_obj;
                best_selected = selected.clone();
                best_weight = current_weight;
                best_delta = delta.clone();
            }
        }
    }

    if best_weight > challenge.max_weight {
        return Err(anyhow!("Solution weight {} exceeds max_weight {}", best_weight, challenge.max_weight));
    }

    let solution = Solution {
        items: (0..n).filter(|&i| best_selected[i] == 1).collect(),
    };
    save_solution(&solution)?;

    Ok(())
}
