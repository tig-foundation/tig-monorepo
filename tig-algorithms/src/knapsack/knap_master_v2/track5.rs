use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Reverse;
use tig_challenges::knapsack::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub window_k: Option<usize>,
    pub ils_rounds: Option<usize>,
    pub core_half_dp: Option<usize>,
}

#[derive(Clone, Copy)]
struct Rng { state: u64 }
impl Rng {
    fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 { s = 1; }
        Self { state: s }
    }
    #[inline] fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }
    #[inline] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    #[inline] fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    #[inline] fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() % bound as u64) as usize
    }
}

struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    solution_hash: u64,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            solution_hash: 0,
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
        }
    }

    #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            let mut k = 0usize;
            while k + 8 <= n {
                let cp = contrib_ptr.add(k);
                let rp = row_ptr.add(k);
                *cp.add(0) = (*cp.add(0)).wrapping_add(*rp.add(0));
                *cp.add(1) = (*cp.add(1)).wrapping_add(*rp.add(1));
                *cp.add(2) = (*cp.add(2)).wrapping_add(*rp.add(2));
                *cp.add(3) = (*cp.add(3)).wrapping_add(*rp.add(3));
                *cp.add(4) = (*cp.add(4)).wrapping_add(*rp.add(4));
                *cp.add(5) = (*cp.add(5)).wrapping_add(*rp.add(5));
                *cp.add(6) = (*cp.add(6)).wrapping_add(*rp.add(6));
                *cp.add(7) = (*cp.add(7)).wrapping_add(*rp.add(7));
                k += 8;
            }
            while k < n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_add(*row_ptr.add(k));
                k += 1;
            }
        }
        self.solution_hash ^= zobrist_item(i);
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            let mut k = 0usize;
            while k + 8 <= n {
                let cp = contrib_ptr.add(k);
                let rp = row_ptr.add(k);
                *cp.add(0) = (*cp.add(0)).wrapping_sub(*rp.add(0));
                *cp.add(1) = (*cp.add(1)).wrapping_sub(*rp.add(1));
                *cp.add(2) = (*cp.add(2)).wrapping_sub(*rp.add(2));
                *cp.add(3) = (*cp.add(3)).wrapping_sub(*rp.add(3));
                *cp.add(4) = (*cp.add(4)).wrapping_sub(*rp.add(4));
                *cp.add(5) = (*cp.add(5)).wrapping_sub(*rp.add(5));
                *cp.add(6) = (*cp.add(6)).wrapping_sub(*rp.add(6));
                *cp.add(7) = (*cp.add(7)).wrapping_sub(*rp.add(7));
                k += 8;
            }
            while k < n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_sub(*row_ptr.add(k));
                k += 1;
            }
        }
        self.solution_hash ^= zobrist_item(j);
        self.selected_bit[j] = false;
    }

    #[inline(always)]
    fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    fn clone_solution(&self) -> SolState {
        SolState {
            bits: self.selected_bit.clone(),
            contrib: self.contrib.clone(),
            value: self.total_value,
            weight: self.total_weight,
            solution_hash: self.solution_hash,
        }
    }

    fn restore_solution(&mut self, sol: &SolState) {
        self.selected_bit.clone_from(&sol.bits);
        self.contrib.clone_from(&sol.contrib);
        self.total_value = sol.value;
        self.total_weight = sol.weight;
        self.solution_hash = sol.solution_hash;
    }
}

#[derive(Clone)]
struct SolState {
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
    solution_hash: u64,
}

#[inline(always)]
fn zobrist_item(i: usize) -> u64 {
    let mut h: u64 = 0x517CC1B727220A95;
    h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
    h
}

struct TabuTable {
    slots: [(u64, u16); 512],
}

impl TabuTable {
    fn new() -> Self {
        Self { slots: [(0, 0); 512] }
    }

    #[inline(always)]
    fn start(h: u64) -> usize {
        let mut x = h;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        (x as usize) & 511
    }

    #[inline(always)]
    fn contains(&self, h: u64) -> bool {
        let mut p = Self::start(h);
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 { return false; }
            if cnt != u16::MAX && key == h { return true; }
            p = (p + 1) & 511;
        }
    }

    #[inline(always)]
    fn insert(&mut self, h: u64) {
        let mut p = Self::start(h);
        let mut first_tomb = usize::MAX;
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 {
                let q = if first_tomb == usize::MAX { p } else { first_tomb };
                self.slots[q] = (h, 1);
                return;
            }
            if cnt == u16::MAX {
                if first_tomb == usize::MAX { first_tomb = p; }
            } else if key == h {
                self.slots[p].1 = cnt + 1;
                return;
            }
            p = (p + 1) & 511;
        }
    }

    #[inline(always)]
    fn remove(&mut self, h: u64) {
        let mut p = Self::start(h);
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 { return; }
            if cnt != u16::MAX && key == h {
                if cnt > 1 { self.slots[p].1 = cnt - 1; }
                else { self.slots[p].1 = u16::MAX; }
                return;
            }
            p = (p + 1) & 511;
        }
    }
}

fn set_all_selected(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let mut tv: i64 = 0;
    let mut tw: u32 = 0;
    let mut zhash: u64 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;
        state.contrib[i] = state.ch.values[i] as i32 + total_interactions[i] as i32;
        tv += state.ch.values[i] as i64;
        tw += state.ch.weights[i];
        zhash ^= zobrist_item(i);
    }
    tv += total_interactions.iter().sum::<i64>() / 2;
    state.total_value = tv;
    state.total_weight = tw;
    state.solution_hash = zhash;
}

fn build_greedy_density_from_all(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    set_all_selected(state, total_interactions);
    while state.total_weight > cap {
        let mut worst = 0;
        let mut worst_s = i64::MAX;
        for i in 0..n {
            if state.selected_bit[i] {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w;
                if s < worst_s { worst_s = s; worst = i; }
            }
        }
        state.remove_item(worst);
    }
    for _ in 0..2 {
        let mut by_density: Vec<usize> = (0..n).collect();
        let contrib = &state.contrib;
        let weights = &state.ch.weights;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = weights[a] as i64;
            let wb = weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });
        let mut target = vec![false; n];
        let mut rem = cap;
        for &i in &by_density {
            if state.ch.weights[i] <= rem { target[i] = true; rem -= state.ch.weights[i]; }
        }
        let mut to_rm = Vec::new();
        let mut to_add = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !target[i] { to_rm.push(i); }
            if !state.selected_bit[i] && target[i] { to_add.push(i); }
        }
        if to_rm.is_empty() && to_add.is_empty() { break; }
        for &r in &to_rm { state.remove_item(r); }
        for &a in &to_add { state.add_item(a); }
    }
}

fn build_greedy_value(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(state.ch.values[i]));
    for &i in &order {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn build_greedy_hub(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| (i, total_interactions[i])).collect();
    hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &hub_scores {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn build_greedy_synergy_weight(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let nm1 = (n as i64 - 1).max(1);
    let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
        let avg_syn = total_interactions[i] / nm1;
        let w = (state.ch.weights[i] as i64).max(1);
        (i, (state.ch.values[i] as i64 + avg_syn) * 100 / w)
    }).collect();
    scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &scores {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    match mode {
        2 => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let s = state.contrib[i] as i64;
                if s <= 0 { continue; }
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
        3 => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w + (state.ch.weights[i] as i64) * 3;
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
        4 => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            let mut second_i: Option<usize> = None;
            let mut second_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w + (rng.next_u32() & 0x1F) as i64;
                if s > best_s {
                    second_s = best_s;
                    second_i = best_i;
                    best_s = s;
                    best_i = Some(i);
                } else if s > second_s {
                    second_s = s;
                    second_i = Some(i);
                }
            }
            let pick = if second_i.is_some() {
                if (rng.next_u32() & 3) == 0 { second_i } else { best_i }
            } else {
                best_i
            };
            if let Some(i) = pick { state.add_item(i); } else { break; }
        },
        5.. => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            let mut second_i: Option<usize> = None;
            let mut second_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w + (rng.next_u32() & 0x7F) as i64;
                if s > best_s {
                    second_s = best_s;
                    second_i = best_i;
                    best_s = s;
                    best_i = Some(i);
                } else if s > second_s {
                    second_s = s;
                    second_i = Some(i);
                }
            }
            let pick = if second_i.is_some() {
                if (rng.next_u32() & 1) == 0 { second_i } else { best_i }
            } else {
                best_i
            };
            if let Some(i) = pick { state.add_item(i); } else { break; }
        },
        _ => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w;
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
    }
}

fn build_synergy_supernodes(state: &mut State, total_interactions: &[i64], rng: &mut Rng) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight as u64;
    let mut seeds: Vec<(usize, i64)> = Vec::with_capacity(24);
    for i in 0..n {
        let score = total_interactions[i];
        let pos = seeds.iter().position(|&(_, prior)| score > prior)
            .unwrap_or(seeds.len());
        if pos < 24 {
            seeds.insert(pos, (i, score));
            if seeds.len() > 24 { seeds.pop(); }
        }
    }

    let mut assigned = vec![false; n];
    let mut components: Vec<(Vec<usize>, u64, i64)> = Vec::new();
    for &(seed, _) in &seeds {
        if assigned[seed] || state.ch.weights[seed] as u64 > cap { continue; }
        let mut neighbors: Vec<(usize, i32)> = Vec::with_capacity(4);
        for item in 0..n {
            if item == seed || assigned[item] { continue; }
            let interaction = state.ch.interaction_values[seed][item];
            if interaction <= 0 { continue; }
            let pos = neighbors.iter().position(|&(_, prior)| interaction > prior)
                .unwrap_or(neighbors.len());
            if pos < 4 {
                neighbors.insert(pos, (item, interaction));
                if neighbors.len() > 4 { neighbors.pop(); }
            }
        }
        if neighbors.is_empty() { continue; }

        let mut items = vec![seed];
        let mut weight = state.ch.weights[seed] as u64;
        for &(item, _) in &neighbors {
            let item_weight = state.ch.weights[item] as u64;
            if weight + item_weight <= cap {
                items.push(item);
                weight += item_weight;
            }
        }
        if items.len() < 2 { continue; }

        let mut internal_value = 0i64;
        for ai in 0..items.len() {
            let item = items[ai];
            internal_value += state.ch.values[item] as i64;
            for bi in 0..ai {
                internal_value += state.ch.interaction_values[item][items[bi]] as i64;
            }
        }
        for &item in &items { assigned[item] = true; }
        if internal_value > 0 {
            components.push((items, weight, internal_value));
        }
    }

    components.sort_unstable_by(|a, b| {
        ((b.2 as i128) * (a.1 as i128)).cmp(&((a.2 as i128) * (b.1 as i128)))
    });
    for (items, weight, _) in components {
        if state.total_weight as u64 + weight > cap { continue; }
        let mut delta = 0i64;
        for ai in 0..items.len() {
            let item = items[ai];
            delta += state.contrib[item] as i64;
            for bi in 0..ai {
                delta += state.ch.interaction_values[item][items[bi]] as i64;
            }
        }
        if delta > 0 {
            for item in items { state.add_item(item); }
        }
    }
    construct_forward_incremental(state, 4, rng);
}

fn build_hub_pair_kth(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut pairs: Vec<(i32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            if state.ch.weights[i] + state.ch.weights[j] <= cap {
                pairs.push((state.ch.interaction_values[i][j], i, j));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
    let mut used = Vec::new();
    let mut count = 0;
    for &(_, pi, pj) in &pairs {
        if used.contains(&pi) || used.contains(&pj) { continue; }
        if count == k {
            state.add_item(pi);
            state.add_item(pj);
            break;
        }
        used.push(pi);
        used.push(pj);
        count += 1;
    }
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = 0;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let c = state.contrib[i] as i64;
            if c <= 0 { continue; }
            let w = (state.ch.weights[i] as i64).max(1);
            let s = (c * 1000) / w;
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

fn dp_refinement_hp(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let weights = &state.ch.weights;

    let mut by_density: Vec<usize> = (0..n).collect();
    by_density.sort_unstable_by(|&a, &b| {
        let na = contrib[a] as i64;
        let nb = contrib[b] as i64;
        let wa = weights[a] as i64;
        let wb = weights[b] as i64;
        (na * wb).cmp(&(nb * wa)).reverse()
    });
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    let mut rem = cap;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = weights[i];
        if w <= rem { rem -= w; idx_last_inserted = idx; }
        else if idx_first_rejected == n { idx_first_rejected = idx; }
    }

    let left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    let locked: Vec<usize> = by_density[..left].to_vec();
    let core: Vec<usize> = by_density[left..right].to_vec();

    let used_locked: u64 = locked.iter().map(|&i| weights[i] as u64).sum();
    let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
    let myk = core.len();
    if myk == 0 || rem_cap == 0 { return; }

    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for &it in &core {
        let wt = weights[it] as usize;
        total_core_weight += wt;
        if contrib[it] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap { all_pos_fit = false; }
        }
    }

    let target_sel = if all_pos_fit {
        let mut sel: Vec<usize> = locked.clone();
        for &it in &core { if contrib[it] > 0 { sel.push(it); } }
        sel.sort_unstable();
        sel
    } else {
        let myw = rem_cap.min(total_core_weight);
        let dp_size = myw + 1;
        let choose_size = myk * dp_size;
        if state.dp_cache.len() < dp_size { state.dp_cache.resize(dp_size, i64::MIN / 4); }
        if state.choose_cache.len() < choose_size { state.choose_cache.resize(choose_size, 0); }
        let init_val = i64::MIN / 4;
        for v in &mut state.dp_cache[..dp_size] { *v = init_val; }
        state.dp_cache[0] = 0;
        state.choose_cache[..choose_size].fill(0);

        let mut w_hi: usize = 0;
        for (t, &it) in core.iter().enumerate() {
            let wt = weights[it] as usize;
            if wt > myw { continue; }
            let val = contrib[it] as i64;
            let new_hi = (w_hi + wt).min(myw);
            for w in (wt..=new_hi).rev() {
                let cand = state.dp_cache[w - wt] + val;
                if cand > state.dp_cache[w] {
                    state.dp_cache[w] = cand;
                    state.choose_cache[t * dp_size + w] = 1;
                }
            }
            w_hi = new_hi;
        }

        let mut sel: Vec<usize> = locked.clone();
        let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = weights[it] as usize;
            if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                sel.push(it);
                w_star -= wt;
            }
        }
        sel.sort_unstable();
        sel
    };

    let mut j = 0;
    let m = target_sel.len();
    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target { j += 1; }
        if state.selected_bit[i] && !in_target {
            state.remove_item(i);
        }
    }
    for &i in &target_sel {
        if !state.selected_bit[i] {
            state.add_item(i);
        }
    }
}

fn apply_best_add(state: &mut State, unselected: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best_i: Option<usize> = None;
    let mut best_d: i32 = 0;
    for &i in unselected {
        if state.ch.weights[i] > slack { continue; }
        let d = state.contrib[i];
        if d > best_d { best_d = d; best_i = Some(i); }
    }
    if let Some(i) = best_i { state.add_item(i); true } else { false }
}

fn apply_best_swap_1_1(state: &mut State, selected: &[usize], unselected: &[usize]) -> bool {
    let slack = state.slack();
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in selected {
        let w_rm = state.ch.weights[rm];
        let max_w = w_rm + slack;
        for &cand in unselected {
            let wc = state.ch.weights[cand];
            if wc > max_w { continue; }
            let delta = state.contrib[cand] - state.contrib[rm]
                - state.ch.interaction_values[cand][rm];
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
}

fn apply_pair_add(state: &mut State, unselected: &[usize]) -> bool {
    let slack = state.slack();
    if slack < 2 { return false; }
    let m = unselected.len();

    let mut best_delta: i64 = 0;
    let mut best_pair: Option<(usize, usize)> = None;
    for ai in 0..m {
        let a = unselected[ai];
        let wa = state.ch.weights[a];
        if wa >= slack { continue; }
        let ca = state.contrib[a] as i64;
        for bi in (ai+1)..m {
            let b = unselected[bi];
            let wb = state.ch.weights[b];
            if wb >= slack || wa + wb > slack { continue; }
            let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
            if delta > best_delta {
                best_delta = delta;
                best_pair = Some((a, b));
            }
        }
    }
    if let Some((a, b)) = best_pair {
        state.add_item(a);
        state.add_item(b);
        true
    } else { false }
}

fn apply_chain_move(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &rm in &sel {
        let w_rm = state.ch.weights[rm] as i64;
        let c_rm = state.contrib[rm] as i64;
        let budget = state.slack() as i64 + w_rm;

        for ui in 0..unsel.len() {
            let a1 = unsel[ui];
            let w_a1 = state.ch.weights[a1] as i64;
            if w_a1 >= budget { continue; }
            let c_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;

            for uj in (ui+1)..unsel.len() {
                let a2 = unsel[uj];
                let w_a2 = state.ch.weights[a2] as i64;
                if w_a1 + w_a2 > budget { continue; }

                let c_a2 = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                let syn = state.ch.interaction_values[a1][a2] as i64;
                let delta = c_a1 + c_a2 + syn - c_rm;

                if delta > best_delta {
                    let new_w = state.total_weight as i64 - w_rm + w_a1 + w_a2;
                    if new_w <= cap as i64 {
                        best_delta = delta;
                        best_move = Some((rm, a1, a2));
                    }
                }
            }
        }
    }

    if let Some((rm, a1, a2)) = best_move {
        state.remove_item(rm);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else { false }
}

fn apply_reverse_chain(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &add in &unsel {
        let w_add = state.ch.weights[add] as i64;
        let c_add = state.contrib[add] as i64;

        for si in 0..sel.len() {
            let r1 = sel[si];
            let w_r1 = state.ch.weights[r1] as i64;
            let c_r1 = state.contrib[r1] as i64;
            let c_add_r1 = state.ch.interaction_values[add][r1] as i64;

            for sj in (si+1)..sel.len() {
                let r2 = sel[sj];
                let w_r2 = state.ch.weights[r2] as i64;
                let freed = w_r1 + w_r2;
                let new_w = state.total_weight as i64 - freed + w_add;
                if new_w > cap as i64 || new_w < 0 { continue; }

                let c_r2 = state.contrib[r2] as i64;
                let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                let c_add_r2 = state.ch.interaction_values[add][r2] as i64;

                let lost = c_r1 + c_r2 - syn_r1_r2;
                let gained = c_add - c_add_r1 - c_add_r2;
                let delta = gained - lost;

                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some((r1, r2, add));
                }
            }
        }
    }

    if let Some((r1, r2, add)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(add);
        true
    } else { false }
}

fn apply_swap_2_2_bounded(state: &mut State, k: usize) -> bool {
    let n = state.ch.num_items;
    let mut sel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    sel_ranked.sort_unstable_by_key(|&(_, c)| c);
    sel_ranked.truncate(k);

    let mut unsel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
    unsel_ranked.truncate(k);

    let cap = state.ch.max_weight;
    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None;

    for si in 0..sel_ranked.len() {
        let r1 = sel_ranked[si].0;
        let w_r1 = state.ch.weights[r1] as i64;
        let c_r1 = state.contrib[r1] as i64;
        for sj in (si+1)..sel_ranked.len() {
            let r2 = sel_ranked[sj].0;
            let w_r2 = state.ch.weights[r2] as i64;
            let c_r2 = state.contrib[r2] as i64;
            let freed_weight = w_r1 + w_r2;
            let removed_syn = state.ch.interaction_values[r1][r2] as i64;
            let lost = c_r1 + c_r2 - removed_syn;
            let budget = state.slack() as i64 + freed_weight;

            for ui in 0..unsel_ranked.len() {
                let a1 = unsel_ranked[ui].0;
                let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 > budget { continue; }
                let c_a1 = state.contrib[a1] as i64
                    - state.ch.interaction_values[a1][r1] as i64
                    - state.ch.interaction_values[a1][r2] as i64;
                for uj in (ui+1)..unsel_ranked.len() {
                    let a2 = unsel_ranked[uj].0;
                    let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget { continue; }
                    let c_a2 = state.contrib[a2] as i64
                        - state.ch.interaction_values[a2][r1] as i64
                        - state.ch.interaction_values[a2][r2] as i64;
                    let added_syn = state.ch.interaction_values[a1][a2] as i64;
                    let delta = c_a1 + c_a2 + added_syn - lost;
                    if delta > best_delta {
                        let new_weight = state.total_weight as i64 - freed_weight + w_a1 + w_a2;
                        if new_weight <= cap as i64 {
                            best_delta = delta;
                            best_move = Some((r1, r2, a1, a2));
                        }
                    }
                }
            }
        }
    }
    if let Some((r1, r2, a1, a2)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else { false }
}

fn local_search_vnd_fast(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    let mut unselected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..80 {
        selected_buf.clear();
        unselected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] { selected_buf.push(i); } else { unselected_buf.push(i); }
        }
        if apply_best_add(state, &unselected_buf) { continue; }
        if apply_best_swap_1_1(state, &selected_buf, &unselected_buf) { continue; }
        break;
    }
}

fn ils_vnd(state: &mut State) {
    local_search_vnd_fast(state);
}

fn local_search_vnd_heavy(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    let mut unselected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..297 {
        selected_buf.clear();
        unselected_buf.clear();
        for i in 0..n {
            if state.selected_bit[i] { selected_buf.push(i); } else { unselected_buf.push(i); }
        }
        if apply_best_add(state, &unselected_buf) { continue; }
        if apply_best_swap_1_1(state, &selected_buf, &unselected_buf) { continue; }
        if apply_pair_add(state, &unselected_buf) { continue; }
        if apply_swap_2_2_bounded(state, 25) { continue; }
        if apply_chain_move(state) { continue; }
        if apply_reverse_chain(state) { continue; }
        break;
    }
}

fn crossover_frequency(population: &[SolState], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let pop_size = population.len();
    let mut freq = vec![0usize; n];
    for sol in population {
        for i in 0..n { if sol.bits[i] { freq[i] += 1; } }
    }
    let threshold = (pop_size * 3) / 4;
    let mut child_bits = vec![false; n];
    let mut child_weight: u32 = 0;
    let mut consensus: Vec<usize> = Vec::new();
    let mut exploratory: Vec<usize> = Vec::new();
    for i in 0..n {
        if freq[i] > threshold { consensus.push(i); }
        else if freq[i] > 0 { exploratory.push(i); }
    }
    for &i in &consensus {
        if child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
        }
    }
    for &i in &exploratory {
        if rng.next_u32() % 2 == 0 && child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
        }
    }
    child_bits
}

fn crossover_uniform(sol_a: &SolState, sol_b: &SolState, ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    for i in 0..n {
        if sol_a.bits[i] && sol_b.bits[i] {
            if weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    for i in 0..n {
        if bits[i] { continue; }
        if sol_a.bits[i] || sol_b.bits[i] {
            if rng.next_u32() % 2 == 0 && weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    bits
}

fn crossover_linkage(state: &mut State, sol_a: &SolState, sol_b: &SolState) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut shared: Vec<usize> = Vec::new();
    let mut inherited: Vec<(usize, i64)> = Vec::new();

    for i in 0..n {
        if sol_a.bits[i] && sol_b.bits[i] {
            shared.push(i);
        } else if sol_a.bits[i] || sol_b.bits[i] {
            let weight = (state.ch.weights[i] as i64).max(1);
            inherited.push((i, state.ch.values[i] as i64 * 1000 / weight));
        }
    }

    shared.sort_unstable_by(|&a, &b| {
        let va = state.ch.values[a] as i64;
        let vb = state.ch.values[b] as i64;
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        (vb * wa).cmp(&(va * wb))
    });
    for i in shared {
        if state.ch.weights[i] <= state.slack() {
            state.add_item(i);
        }
    }

    inherited.sort_unstable_by_key(|&(_, score)| Reverse(score));
    inherited.truncate(56);

    let mut edges: Vec<(usize, usize, i64)> = Vec::with_capacity(40);
    for ai in 0..inherited.len() {
        let a = inherited[ai].0;
        for bi in (ai + 1)..inherited.len() {
            let b = inherited[bi].0;
            let together_a = sol_a.bits[a] && sol_a.bits[b];
            let together_b = sol_b.bits[a] && sol_b.bits[b];
            if !together_a && !together_b {
                continue;
            }
            let interaction = state.ch.interaction_values[a][b] as i64;
            if interaction <= 0 {
                continue;
            }
            let weight = (state.ch.weights[a] as i64 + state.ch.weights[b] as i64).max(1);
            let score = (state.ch.values[a] as i64 + state.ch.values[b] as i64 + interaction) * 1000 / weight;
            let pos = edges.iter().position(|&(_, _, prior)| score > prior).unwrap_or(edges.len());
            if pos < 40 {
                edges.insert(pos, (a, b, score));
                if edges.len() > 40 {
                    edges.pop();
                }
            }
        }
    }

    for (a, b, _) in edges {
        if state.selected_bit[a] || state.selected_bit[b] {
            continue;
        }
        let pair_weight = state.ch.weights[a] as u64 + state.ch.weights[b] as u64;
        if pair_weight <= state.slack() as u64 {
            state.add_item(a);
            state.add_item(b);
        }
    }

    loop {
        let slack = state.slack();
        let mut best: Option<(usize, i64)> = None;
        for i in 0..n {
            if state.selected_bit[i] || state.ch.weights[i] > slack {
                continue;
            }
            let weight = (state.ch.weights[i] as i64).max(1);
            let score = state.contrib[i] as i64 * 1000 / weight;
            if state.contrib[i] > 0 && best.map_or(true, |(_, prior)| score > prior) {
                best = Some((i, score));
            }
        }
        if let Some((i, _)) = best {
            state.add_item(i);
        } else {
            break;
        }
    }

    let _ = cap;
}

fn crossover_residual_modules(
    state: &mut State,
    backbone: &SolState,
    donor: &SolState,
) {
    let n = state.ch.num_items;
    let target_weight = backbone.weight as u64 * 82 / 100;
    let mut shared: Vec<usize> = Vec::new();
    let mut private: Vec<usize> = Vec::new();
    for item in 0..n {
        if backbone.bits[item] {
            if donor.bits[item] {
                shared.push(item);
            } else {
                private.push(item);
            }
        }
    }
    let density_cmp = |a: &usize, b: &usize| {
        let wa = (state.ch.weights[*a] as i64).max(1);
        let wb = (state.ch.weights[*b] as i64).max(1);
        (backbone.contrib[*b] as i64 * wa).cmp(&(backbone.contrib[*a] as i64 * wb))
    };
    shared.sort_unstable_by(density_cmp);
    private.sort_unstable_by(density_cmp);

    let mut protected = vec![false; n];
    for item in shared {
        state.add_item(item);
        protected[item] = true;
    }
    for item in private {
        if state.total_weight as u64 >= target_weight {
            break;
        }
        state.add_item(item);
        protected[item] = true;
    }

    let mut donor_items: Vec<usize> = (0..n)
        .filter(|&item| donor.bits[item] && !backbone.bits[item])
        .collect();
    donor_items.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        (donor.contrib[b] as i64 * wa).cmp(&(donor.contrib[a] as i64 * wb))
    });
    donor_items.truncate(48);

    let mut modules: Vec<(Vec<usize>, i64)> = Vec::with_capacity(96);
    for &item in &donor_items {
        let weight = (state.ch.weights[item] as i64).max(1);
        modules.push((vec![item], donor.contrib[item] as i64 * 1000 / weight));
    }
    for ai in 0..donor_items.len() {
        let a = donor_items[ai];
        for bi in (ai + 1)..donor_items.len() {
            let b = donor_items[bi];
            let interaction = state.ch.interaction_values[a][b] as i64;
            if interaction <= 0 {
                continue;
            }
            let weight = (state.ch.weights[a] as i64 + state.ch.weights[b] as i64).max(1);
            let score = (donor.contrib[a] as i64 + donor.contrib[b] as i64 + interaction) * 1000 / weight;
            modules.push((vec![a, b], score));
        }
    }
    modules.sort_unstable_by_key(|(_, score)| Reverse(*score));
    modules.truncate(56);

    let mut weak: Vec<usize> = (0..n).filter(|&item| protected[item]).collect();
    weak.sort_unstable_by_key(|&item| state.contrib[item]);
    weak.truncate(12);

    for _ in 0..48 {
        let mut best: Option<(usize, i64, Option<usize>)> = None;
        for (module_idx, (items, _)) in modules.iter().enumerate() {
            if items.iter().any(|&item| state.selected_bit[item]) {
                continue;
            }
            let module_weight: u64 = items.iter().map(|&item| state.ch.weights[item] as u64).sum();
            let mut gain = 0i64;
            for &item in items {
                gain += state.contrib[item] as i64;
            }
            if items.len() == 2 {
                gain += state.ch.interaction_values[items[0]][items[1]] as i64;
            }
            if state.total_weight as u64 + module_weight <= state.ch.max_weight as u64 {
                if gain > 0 && best.map_or(true, |(_, prior, _)| gain > prior) {
                    best = Some((module_idx, gain, None));
                }
                continue;
            }
            for &removed in &weak {
                if !state.selected_bit[removed]
                    || state.total_weight as u64 + module_weight
                        > state.ch.max_weight as u64 + state.ch.weights[removed] as u64
                {
                    continue;
                }
                let mut replacement_gain = gain - state.contrib[removed] as i64;
                for &item in items {
                    replacement_gain -= state.ch.interaction_values[item][removed] as i64;
                }
                if replacement_gain > 0
                    && best.map_or(true, |(_, prior, _)| replacement_gain > prior)
                {
                    best = Some((module_idx, replacement_gain, Some(removed)));
                }
            }
        }
        let Some((module_idx, _, removed)) = best else { break; };
        if let Some(item) = removed {
            state.remove_item(item);
        }
        let items = modules[module_idx].0.clone();
        for item in items {
            state.add_item(item);
        }
    }
}

fn set_state_from_bits(state: &mut State, bits: &[bool]) {
    let n = state.ch.num_items;
    let ch = state.ch;
    state.selected_bit.clone_from_slice(bits);
    for i in 0..n {
        state.contrib[i] = ch.values[i] as i32;
    }
    state.total_value = 0;
    state.total_weight = 0;
    let mut zhash: u64 = 0;
    for i in 0..n {
        if bits[i] {
            state.total_value += state.contrib[i] as i64;
            state.total_weight += ch.weights[i];
            zhash ^= zobrist_item(i);
            let row_ptr = unsafe { ch.interaction_values.get_unchecked(i).as_ptr() };
            let contrib_ptr = state.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(*row_ptr.add(k));
                }
            }
        }
    }
    state.solution_hash = zhash;
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(window_k.min(n));
    let mut worst_used: Vec<usize> = Vec::with_capacity(window_k.min(n));
    for _ in 0..80 {
        unused_r.clear();
        used_r.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            if state.selected_bit[i] { used_r.push((i, c, w)); } else { unused_r.push((i, c, w)); }
        }
        let ku = window_k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| {
                ((b.1 as i128) * (a.2 as i128)).cmp(&((a.1 as i128) * (b.2 as i128)))
            });
        }
        let ks = window_k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| {
                ((a.1 as i128) * (b.2 as i128)).cmp(&((b.1 as i128) * (a.2 as i128)))
            });
        }
        for x in &unused_r[..ku] { best_unused.push(x.0); }
        for x in &used_r[..ks] { worst_used.push(x.0); }
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack { continue; }
                        let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
            }
        }

        if !improved { break; }
    }
}

fn local_search_vnd_windowed_deep(state: &mut State, window_k: usize) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(window_k.min(n));
    let mut worst_used: Vec<usize> = Vec::with_capacity(window_k.min(n));
    for _ in 0..120 {
        unused_r.clear();
        used_r.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            if state.selected_bit[i] { used_r.push((i, c, w)); } else { unused_r.push((i, c, w)); }
        }
        let ku = window_k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| {
                ((b.1 as i128) * (a.2 as i128)).cmp(&((a.1 as i128) * (b.2 as i128)))
            });
        }
        let ks = window_k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| {
                ((a.1 as i128) * (b.2 as i128)).cmp(&((b.1 as i128) * (a.2 as i128)))
            });
        }
        for x in &unused_r[..ku] { best_unused.push(x.0); }
        for x in &used_r[..ks] { worst_used.push(x.0); }
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack { continue; }
                        let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ks = worst_used.len().min(80);
            let ku = best_unused.len().min(80);
            for si in 0..ks {
                let rm = worst_used[si];
                let c_rm = state.contrib[rm] as i64;
                let w_rm = state.ch.weights[rm];
                let budget = state.slack() + w_rm;
                for ai in 0..ku {
                    let a1 = best_unused[ai];
                    let wa1 = state.ch.weights[a1];
                    if wa1 >= budget { continue; }
                    let ca1_eff = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;
                    for bi in (ai+1)..ku {
                        let a2 = best_unused[bi];
                        let wa2 = state.ch.weights[a2];
                        if wa1 + wa2 > budget { continue; }
                        let ca2_eff = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                        let syn = state.ch.interaction_values[a1][a2] as i64;
                        let delta = ca1_eff + ca2_eff + syn - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
                        }
                    }
                }
            }
            if let Some((rm, a1, a2)) = bm {
                state.remove_item(rm); state.add_item(a1); state.add_item(a2);
                continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ku = best_unused.len().min(80);
            let ks = worst_used.len().min(80);
            for ui in 0..ku {
                let add = best_unused[ui];
                let c_add = state.contrib[add] as i64;
                let w_add = state.ch.weights[add];
                for si in 0..ks {
                    let r1 = worst_used[si];
                    let wr1 = state.ch.weights[r1];
                    let cr1 = state.contrib[r1] as i64;
                    let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
                    for sj in (si+1)..ks {
                        let r2 = worst_used[sj];
                        let wr2 = state.ch.weights[r2];
                        let new_w = state.total_weight + w_add - wr1 - wr2;
                        if new_w > cap { continue; }
                        let cr2 = state.contrib[r2] as i64;
                        let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                        let c_add_r2 = state.ch.interaction_values[add][r2] as i64;
                        let lost = cr1 + cr2 - syn_r1_r2;
                        let gained = c_add - c_add_r1 - c_add_r2;
                        let delta = gained - lost;
                        if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                    }
                }
            }
            if let Some((r1, r2, add)) = bm {
                state.remove_item(r1); state.remove_item(r2); state.add_item(add);
                continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize, usize)> = None;
            let ks = worst_used.len().min(25);
            let ku = best_unused.len().min(25);
            for si in 0..ks {
                let r1 = worst_used[si];
                let wr1 = state.ch.weights[r1];
                let cr1 = state.contrib[r1] as i64;
                for sj in (si+1)..ks {
                    let r2 = worst_used[sj];
                    let wr2 = state.ch.weights[r2];
                    let cr2 = state.contrib[r2] as i64;
                    let syn_rm = state.ch.interaction_values[r1][r2] as i64;
                    let lost = cr1 + cr2 - syn_rm;
                    let budget = state.slack() + wr1 + wr2;
                    for ui in 0..ku {
                        let a1 = best_unused[ui];
                        let wa1 = state.ch.weights[a1];
                        if wa1 >= budget { continue; }
                        let ca1_eff = state.contrib[a1] as i64
                            - state.ch.interaction_values[a1][r1] as i64
                            - state.ch.interaction_values[a1][r2] as i64;
                        for uj in (ui+1)..ku {
                            let a2 = best_unused[uj];
                            let wa2 = state.ch.weights[a2];
                            if wa1 + wa2 > budget { continue; }
                            let ca2_eff = state.contrib[a2] as i64
                                - state.ch.interaction_values[a2][r1] as i64
                                - state.ch.interaction_values[a2][r2] as i64;
                            let syn_add = state.ch.interaction_values[a1][a2] as i64;
                            let delta = ca1_eff + ca2_eff + syn_add - lost;
                            if delta > bd && state.total_weight + wa1 + wa2 <= cap + wr1 + wr2 {
                                bd = delta; bm = Some((r1, r2, a1, a2));
                            }
                        }
                    }
                }
            }
            if let Some((r1, r2, a1, a2)) = bm {
                state.remove_item(r1); state.remove_item(r2);
                state.add_item(a1); state.add_item(a2);
                continue;
            }
        }

        if !improved { break; }
    }
}

fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams, total_interactions: &[i64]) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }
    let mut removal_candidates: Vec<(usize, i64)>;

    match strategy {
        0 => {
            removal_candidates = selected.iter().map(|&i| (i, state.contrib[i] as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        },
        1 => {
            removal_candidates = selected.iter().map(|&i| (i, -(state.ch.weights[i] as i64))).collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        },
        2 => {
            removal_candidates = selected.iter().map(|&i| {
                let syn = state.contrib[i] as i64 - state.ch.values[i] as i64;
                (i, syn)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        3 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (state.contrib[i] as i64 * 1000) / w)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        4 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let density = (state.contrib[i] as i64 * 100) / w;
                (i, state.ch.weights[i] as i64 - density)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        5 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (state.contrib[i] as i64 * 10000) / (w * w))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        6 => {
            removal_candidates = selected.iter().map(|&i| (i, rng.next_u32() as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        7 => {
            removal_candidates = selected.iter().map(|&i| {
                let anti = 2 * state.contrib[i] as i64 - total_interactions[i];
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (anti * 1000) / w)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        8 => {
            removal_candidates = selected.iter().map(|&i| {
                let c = state.contrib[i] as i64;
                let potential = total_interactions[i];
                (i, c * 100 - potential)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        _ => {
            removal_candidates = selected.iter().map(|&i| (i, -(state.contrib[i] as i64))).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }

    let base_remove = (selected.len() / hp.perturb_base_frac).max(2);
    let adaptive_mult = 1 + (stall_count / 2);
    let n_remove = (base_remove * adaptive_mult).min(strength).min(selected.len() * 2 / hp.perturb_max_frac);
    for j in 0..n_remove {
        if j < removal_candidates.len() {
            state.remove_item(removal_candidates[j].0);
        }
    }
}

fn greedy_reconstruct(state: &mut State, strategy: usize, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

    match strategy % 6 {
        0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
        1 => candidates.sort_unstable_by(|&a, &b| {
            state.ch.weights[a].cmp(&state.ch.weights[b])
                .then(state.contrib[b].cmp(&state.contrib[a]))
        }),
        2 => candidates.sort_unstable_by_key(|&i| {
            -(total_interactions[i] + state.contrib[i] as i64 / 10)
        }),
        3 => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            -((state.contrib[i] as i64 * 100) / w)
        }),
        4 => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            let c = state.contrib[i] as i64;
            let anti = 2 * c - total_interactions[i];
            -((anti * 100) / w + c / 5)
        }),
        _ => candidates.sort_unstable_by_key(|&i| {
            let c = state.contrib[i] as i64;
            let potential = total_interactions[i] / (n as i64).max(1);
            -(c + potential * 3)
        }),
    }

    for &i in &candidates {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

struct TopNeighbors {
    friends: Vec<Vec<usize>>,
}
impl TopNeighbors {
    fn new(ch: &Challenge, k: usize) -> Self {
        let n = ch.num_items;
        let mut min_weight = u64::MAX;
        let mut max_weight = 0u64;
        for &weight in &ch.weights {
            let weight = (weight as u64).max(1);
            min_weight = min_weight.min(weight);
            max_weight = max_weight.max(weight);
        }
        let weight_aware = k >= 2 && min_weight != u64::MAX
            && max_weight > min_weight.saturating_mul(3) / 2;
        let abs_slots = if weight_aware { (k + 1) / 2 } else { k };
        let mut friends = vec![Vec::with_capacity(k); n];

        for i in 0..n {
            let mut absolute: Vec<(usize, i32, u64)> = Vec::with_capacity(abs_slots);
            let mut density: Vec<(usize, i32, u64)> = Vec::with_capacity(k);
            let wi = (ch.weights[i] as u64).max(1);

            for j in 0..n {
                if i == j { continue; }
                let interaction = ch.interaction_values[i][j];
                if interaction <= 0 { continue; }

                let combined_weight = wi + (ch.weights[j] as u64).max(1);
                let abs_pos = absolute.iter().position(|&(_, value, _)| value < interaction)
                    .unwrap_or(absolute.len());
                if abs_pos < abs_slots {
                    absolute.insert(abs_pos, (j, interaction, combined_weight));
                    if absolute.len() > abs_slots { absolute.pop(); }
                }

                if weight_aware {
                    let density_pos = density.iter().position(|&(_, value, weight)| {
                        (interaction as u128) * (weight as u128)
                            > (value as u128) * (combined_weight as u128)
                    }).unwrap_or(density.len());
                    if density_pos < k {
                        density.insert(density_pos, (j, interaction, combined_weight));
                        if density.len() > k { density.pop(); }
                    }
                }
            }

            let mut neighbors = Vec::with_capacity(k);
            for (j, _, _) in absolute {
                neighbors.push(j);
            }
            if weight_aware {
                for (j, _, _) in density {
                    if neighbors.len() >= k { break; }
                    if !neighbors.contains(&j) {
                        neighbors.push(j);
                    }
                }
            }
            friends[i] = neighbors;
        }
        Self { friends }
    }
}

fn local_search_vnd_tsn(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = if n > 3000 { 200 } else { 300 };
    let mut unused_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    for _ in 0..80 {
        unused_buf.clear();
        used_buf.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
        }
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        {
            let slack_i = state.slack() as i32;
            if slack_i >= 2 {
                let mut bd = 0i64;
                let mut bp = None;
                for &a1 in &best_unused {
                    let ca1 = state.contrib[a1] as i64;
                    if ca1 <= 0 && bd > 0 { break; }
                    let wa1 = state.ch.weights[a1] as i32;
                    if wa1 >= slack_i { continue; }
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                            let delta = ca1 + state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
                        }
                    }
                }
                if let Some((a1, a2)) = bp { state.add_item(a1); state.add_item(a2); continue; }
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            for &rm in &worst_used {
                let c_rm = state.contrib[rm] as i64;
                let w_rm = state.ch.weights[rm];
                let budget = state.slack() + w_rm;
                for &a1 in &best_unused {
                    let ca1 = state.contrib[a1] as i64;
                    let wa1 = state.ch.weights[a1];
                    if wa1 >= budget { continue; }
                    let ca1_eff = ca1 - state.ch.interaction_values[a1][rm] as i64;
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        let wa2 = state.ch.weights[a2];
                        if wa1 + wa2 > budget { continue; }
                        let delta = ca1_eff + state.contrib[a2] as i64
                            - state.ch.interaction_values[a2][rm] as i64
                            + state.ch.interaction_values[a1][a2] as i64 - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
                        }
                    }
                }
            }
            if let Some((rm, a1, a2)) = bm {
                state.remove_item(rm); state.add_item(a1); state.add_item(a2); continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            for &add in &best_unused {
                let c_add = state.contrib[add] as i64;
                let w_add = state.ch.weights[add];
                if c_add <= 0 && bd > 0 { break; }
                for &r1 in &worst_used {
                    let cr1 = state.contrib[r1] as i64;
                    let wr1 = state.ch.weights[r1];
                    let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
                    for &r2 in &tsn.friends[r1] {
                        if !state.selected_bit[r2] || r1 == r2 { continue; }
                        let wr2 = state.ch.weights[r2];
                        if state.total_weight + w_add <= cap + wr1 + wr2 {
                            let delta = c_add - c_add_r1
                                - state.ch.interaction_values[add][r2] as i64
                                - cr1 - state.contrib[r2] as i64
                                + state.ch.interaction_values[r1][r2] as i64;
                            if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                        }
                    }
                }
            }
            if let Some((r1, r2, add)) = bm {
                state.remove_item(r1); state.remove_item(r2); state.add_item(add); continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            let ks = worst_used.len().min(30);
            let ku = best_unused.len().min(30);
            for i in 0..ks {
                let r1 = worst_used[i];
                let cr1 = state.contrib[r1] as i64;
                let wr1 = state.ch.weights[r1];
                for &r2 in &tsn.friends[r1] {
                    if !state.selected_bit[r2] || r1 == r2 { continue; }
                    let cr2 = state.contrib[r2] as i64;
                    let wr2 = state.ch.weights[r2];
                    let lost = cr1 + cr2 - state.ch.interaction_values[r1][r2] as i64;
                    let budget = state.slack() + wr1 + wr2;
                    for u in 0..ku {
                        let a1 = best_unused[u];
                        let wa1 = state.ch.weights[a1];
                        if wa1 >= budget { continue; }
                        let ca1_eff = state.contrib[a1] as i64
                            - state.ch.interaction_values[a1][r1] as i64
                            - state.ch.interaction_values[a1][r2] as i64;
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            let wa2 = state.ch.weights[a2];
                            if wa1 + wa2 <= budget {
                                let gained = ca1_eff + state.contrib[a2] as i64
                                    - state.ch.interaction_values[a2][r1] as i64
                                    - state.ch.interaction_values[a2][r2] as i64
                                    + state.ch.interaction_values[a1][a2] as i64;
                                if gained - lost > bd && state.total_weight + wa1 + wa2 <= cap + wr1 + wr2 {
                                    bd = gained - lost; bm = Some((r1, r2, a1, a2));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((r1, r2, a1, a2)) = bm {
                state.remove_item(r1); state.remove_item(r2);
                state.add_item(a1); state.add_item(a2); continue;
            }
        }

        break;
    }
}

fn cluster_bomb_perturb(state: &mut State, tsn: &TopNeighbors, rng: &mut Rng, strength: usize) {
    let sel = state.selected_items();
    if sel.is_empty() { return; }
    let target = state.total_weight / (strength as u32).max(2);
    let mut freed = 0u32;
    let root = sel[rng.next_usize(sel.len())];
    state.remove_item(root);
    freed += state.ch.weights[root];
    for &f in &tsn.friends[root] {
        if state.selected_bit[f] {
            state.remove_item(f);
            freed += state.ch.weights[f];
            if freed >= target { break; }
        }
    }
    let slack = state.slack();
    if slack > 0 {
        let unsel: Vec<usize> = (0..state.ch.num_items)
            .filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= slack)
            .collect();
        if !unsel.is_empty() { state.add_item(unsel[rng.next_usize(unsel.len())]); }
    }
}

struct Hparams {
    n_random_starts: usize,
    n_crossover_gen: usize,
    ils_rounds: usize,
    ils_restart_interval: usize,
    perturb_base_frac: usize,
    perturb_max_frac: usize,
    n_full_restarts: usize,
    use_hub_pair: bool,
    use_heavy_polish: bool,
    window_k: usize,
    core_half_dp: usize,
}

impl Hparams {
    fn defaults() -> Self {
        Self {
            n_random_starts: 5, n_crossover_gen: 3, ils_rounds: 47,
            ils_restart_interval: 11, perturb_base_frac: 6,
            perturb_max_frac: 4, n_full_restarts: 13, use_hub_pair: false,
            use_heavy_polish: false, window_k: 179, core_half_dp: 41,
        }
    }

    fn from_map(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self::defaults();
        if let Some(m) = h {
            if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64()) { p.n_random_starts = v as usize; }
            if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64()) { p.n_crossover_gen = v as usize; }
            if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) { p.ils_rounds = v as usize; }
            if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) { p.ils_restart_interval = v as usize; }
            if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()) { p.perturb_base_frac = v as usize; }
            if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) { p.perturb_max_frac = v as usize; }
            if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) { p.n_full_restarts = v as usize; }
            if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
            if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) { p.core_half_dp = v as usize; }
        }
        p
    }
}

fn local_search_vnd_tsn_light(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = 189;
    let mut unused_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    for _ in 0..40 {
        unused_buf.clear();
        used_buf.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
        }
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        {
            let slack_i = state.slack() as i32;
            if slack_i >= 2 {
                let mut bd = 0i64;
                let mut bp = None;
                for &a1 in &best_unused {
                    let ca1 = state.contrib[a1] as i64;
                    if ca1 <= 0 && bd > 0 { break; }
                    let wa1 = state.ch.weights[a1] as i32;
                    if wa1 >= slack_i { continue; }
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                            let delta = ca1 + state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
                        }
                    }
                }
                if let Some((a1, a2)) = bp { state.add_item(a1); state.add_item(a2); continue; }
            }
        }

        break;
    }
}

fn vnd_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t);
    } else if hp.window_k < state.ch.num_items {
        local_search_vnd_windowed(state, hp.window_k);
    } else {
        ils_vnd(state);
    }
}

fn polish_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t);
    } else if hp.use_heavy_polish {
        local_search_vnd_heavy(state);
    } else {
        vnd_v2(state, hp, None);
    }
}

struct ReactivePerturbSchedule {
    rewards: [i64; 8],
    uses: [usize; 8],
    cursor: usize,
    polish_stalls: usize,
}

fn basin_signature(state: &State) -> u64 {
    let n = state.ch.num_items;
    let mut selected_count = 0usize;
    let mut sample: Vec<(i64, usize)> = Vec::with_capacity(24);
    for i in 0..n {
        if !state.selected_bit[i] {
            continue;
        }
        selected_count += 1;
        let weight = (state.ch.weights[i] as i64).max(1);
        let score = state.contrib[i] as i64 * 1000 / weight;
        let pos = sample.iter().position(|&(prior, _)| score > prior)
            .unwrap_or(sample.len());
        if pos < 24 {
            sample.insert(pos, (score, i));
            if sample.len() > 24 {
                sample.pop();
            }
        }
    }

    let mut edges: Vec<(i32, usize, usize)> = Vec::with_capacity(8);
    for ai in 0..sample.len() {
        let a = sample[ai].1;
        for bi in (ai + 1)..sample.len() {
            let b = sample[bi].1;
            let interaction = state.ch.interaction_values[a][b];
            if interaction <= 0 {
                continue;
            }
            let pos = edges.iter().position(|&(prior, _, _)| interaction > prior)
                .unwrap_or(edges.len());
            if pos < 8 {
                edges.insert(pos, (interaction, a.min(b), a.max(b)));
                if edges.len() > 8 {
                    edges.pop();
                }
            }
        }
    }

    let capacity_band = if state.ch.max_weight > 0 {
        (state.total_weight as u64 * 16 / state.ch.max_weight as u64).min(16)
    } else {
        0
    };
    let mut signature = (selected_count as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ capacity_band.wrapping_mul(0xBF58476D1CE4E5B9);
    for (interaction, a, b) in edges {
        let edge = ((a as u64) << 32) ^ b as u64
            ^ (interaction as u32 as u64).wrapping_mul(0x94D049BB133111EB);
        signature ^= edge.rotate_left(17);
        signature = signature.rotate_left(11).wrapping_mul(0x9E3779B185EBCA87);
    }
    signature
}

impl ReactivePerturbSchedule {
    fn new() -> Self {
        Self {
            rewards: [0; 8],
            uses: [0; 8],
            cursor: 0,
            polish_stalls: 0,
        }
    }

    fn choose(&mut self) -> usize {
        for offset in 0..8 {
            let strategy = (self.cursor + offset) % 8;
            if self.uses[strategy] == 0 {
                self.uses[strategy] = 1;
                self.cursor = (strategy + 1) % 8;
                return strategy;
            }
        }

        let mut best = self.cursor;
        for strategy in 0..8 {
            let lhs = self.rewards[strategy] as i128 * self.uses[best] as i128;
            let rhs = self.rewards[best] as i128 * self.uses[strategy] as i128;
            if lhs > rhs || (lhs == rhs && strategy == self.cursor) {
                best = strategy;
            }
        }
        self.uses[best] += 1;
        self.cursor = (best + 1) % 8;
        best
    }

    fn note_perturb(&mut self, strategy: usize, before: i64, after: i64, repeated: bool) {
        let gain = (after - before).max(0);
        self.rewards[strategy] = self.rewards[strategy].saturating_mul(3) / 4 + gain;
        if repeated {
            self.rewards[strategy] = self.rewards[strategy].saturating_mul(3) / 4;
        }
    }

    fn note_polish(&mut self, before: i64, after: i64) {
        if after > before {
            self.polish_stalls = 0;
        } else {
            self.polish_stalls += 1;
        }
    }
}

fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64]) -> (Solution, i64) {
    run_one_instance_seeded(challenge, hp, rng_offset, shared_tsn, total_interactions, None)
}

fn run_one_instance_seeded(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64], seed_sol: Option<&Solution>) -> (Solution, i64) {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..rng_offset * 100 { rng.next_u32(); }
    let ch = hp.core_half_dp;

    let tsn_opt: Option<TopNeighbors> = if shared_tsn.is_none() && n > 1200 {
        Some(TopNeighbors::new(challenge, 12))
    } else { None };
    let tsn_ref = shared_tsn.or(tsn_opt.as_ref());

    let mut population: Vec<SolState> = Vec::with_capacity(16);

    if let Some(seed) = seed_sol {
        let mut st = State::new_empty(challenge);
        for &i in &seed.items { st.add_item(i); }
        let sel = st.selected_items();
        let n_remove = (sel.len() / 8).max(2).min(sel.len());
        let mut scored: Vec<(usize, i64)> = sel.iter().map(|&i| {
            let w = (challenge.weights[i] as i64).max(1);
            (i, (st.contrib[i] as i64 * 1000) / w)
        }).collect();
        scored.sort_unstable_by_key(|&(_, s)| s);
        for j in 0..n_remove { st.remove_item(scored[j].0); }
        greedy_reconstruct(&mut st, rng_offset % 6, total_interactions);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    let n_greedy = 4;
    let n_rand = if seed_sol.is_some() { hp.n_random_starts.saturating_sub(1) } else { hp.n_random_starts };
    for variant in 0..n_greedy {
        let mut st = State::new_empty(challenge);
        match variant {
            0 => build_greedy_density_from_all(&mut st, total_interactions),
            1 => build_greedy_value(&mut st),
            2 => build_greedy_synergy_weight(&mut st, total_interactions),
            3 => build_synergy_supernodes(&mut st, total_interactions, &mut rng),
            _ => build_greedy_hub(&mut st, total_interactions),
        }
        dp_refinement_hp(&mut st, ch);
        polish_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    for mode in 4..(4 + n_rand) {
        let mut st = State::new_empty(challenge);
        let m = if mode < 6 { mode } else { mode - 2 };
        construct_forward_incremental(&mut st, m, &mut rng);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    if hp.use_hub_pair {
        for k in 0..4 {
            let mut st = State::new_empty(challenge);
            build_hub_pair_kth(&mut st, k);
            dp_refinement_hp(&mut st, ch);
            vnd_v2(&mut st, hp, tsn_ref);
            population.push(st.clone_solution());
        }
    }

    population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
    population.truncate(8);

    let mut state = State::new_empty(challenge);
    let mut xo_stall = 0usize;
    for gen in 0..hp.n_crossover_gen {
        let best_before = population[0].value;
        let child_bits = crossover_frequency(&population, challenge, &mut rng);
        set_state_from_bits(&mut state, &child_bits);
        dp_refinement_hp(&mut state, ch);
        vnd_v2(&mut state, hp, tsn_ref);
        population.push(state.clone_solution());

        if population.len() >= 2 {
            let a = gen % population.len().min(4);
            let b = (gen + 1) % population.len().min(4);
            if a != b {
                let child_bits = crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                set_state_from_bits(&mut state, &child_bits);
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());

                let parent_a = population[a].clone();
                let parent_b = population[b].clone();
                state = State::new_empty(challenge);
                if gen % 2 == 0 {
                    crossover_residual_modules(&mut state, &parent_a, &parent_b);
                } else {
                    crossover_residual_modules(&mut state, &parent_b, &parent_a);
                }
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());

                state = State::new_empty(challenge);
                crossover_linkage(&mut state, &parent_a, &parent_b);
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());
            }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
        if population[0].value <= best_before {
            xo_stall += 1;
            if xo_stall >= 3 { break; }
        } else {
            xo_stall = 0;
        }
    }

    state.restore_solution(&population[0]);
    let mut best_val = state.total_value;
    let mut best_sel: Vec<usize> = state.selected_items();
    let mut best_bit = state.selected_bit.clone();

    let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
    tabu_hashes.push(state.solution_hash);
    let mut tabu_membership = TabuTable::new();
    tabu_membership.insert(state.solution_hash);
    let mut basin_signatures = [0u64; 32];
    basin_signatures[0] = basin_signature(&state);
    let mut basin_len = 1usize;

    let mut stall_count = 0;
    let max_stall = (hp.ils_rounds / 3).min(7);
    let use_light_vnd = tsn_ref.is_some();
    let mut reactive_perturb = ReactivePerturbSchedule::new();
    let mut cnt = hp.ils_rounds;
    for round in 0..cnt {
        if stall_count >= max_stall { break; }
        let snap = state.clone_solution();

        dp_refinement_hp(&mut state, ch);
        if use_light_vnd {
            local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
        } else {
            vnd_v2(&mut state, hp, tsn_ref);
        }

        reactive_perturb.note_polish(snap.value, state.total_value);
        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel = state.selected_items();
            best_bit.clone_from(&state.selected_bit);
            stall_count = 0;
        }

        if state.total_value <= snap.value {
            state.restore_solution(&snap);
            stall_count += 1;

            if hp.ils_restart_interval > 0 && stall_count > 0 && stall_count % hp.ils_restart_interval == 0 {
                let pi = (stall_count / hp.ils_restart_interval) % population.len();
                state.restore_solution(&population[pi]);
            }

            let use_bomb = tsn_ref.is_some() && round % 3 == 0;
            if use_bomb {
                let str_v = if stall_count > 15 { 3 } else { 5 };
                cluster_bomb_perturb(&mut state, tsn_ref.unwrap(), &mut rng, str_v);
            } else {
                let strategy = reactive_perturb.choose();
                let strength = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp, total_interactions);
            }
            let strategy = reactive_perturb.cursor.wrapping_add(7) % 8;
            greedy_reconstruct(&mut state, strategy % 6, total_interactions);

            let candidate_hash = state.solution_hash;
            let candidate_signature = basin_signature(&state);
            let aspirational = state.total_value > best_val;
            let mut common_with_best = 0usize;
            let mut candidate_count = 0usize;
            for i in 0..n {
                if state.selected_bit[i] {
                    candidate_count += 1;
                    if best_bit[i] {
                        common_with_best += 1;
                    }
                }
            }
            let novel_from_best = candidate_count > 0
                && common_with_best.saturating_mul(8) < candidate_count.saturating_mul(7);
            let repeated_basin = basin_signatures[..basin_len].contains(&candidate_signature);
            if !aspirational
                && ((tabu_membership.contains(candidate_hash) && !novel_from_best) || repeated_basin) {
                let alternate_strength = 7 + round / 3;
                perturb_by_strategy(
                    &mut state,
                    alternate_strength,
                    stall_count + 2,
                    (round + 5) % 9,
                    &mut rng,
                    hp,
                    total_interactions,
                );
                greedy_reconstruct(&mut state, (round + 3) % 6, total_interactions);
            }

            if use_light_vnd {
                local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
            } else {
                vnd_v2(&mut state, hp, tsn_ref);
            }

            let h = state.solution_hash;
            if tabu_membership.contains(h) {
                if let Some(t) = tsn_ref {
                    cluster_bomb_perturb(&mut state, t, &mut rng, 2);
                } else {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp, total_interactions);
                }
                greedy_reconstruct(&mut state, 0, total_interactions);
                if use_light_vnd {
                    local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
                } else {
                    vnd_v2(&mut state, hp, tsn_ref);
                }
            }
            let h2 = state.solution_hash;
            let signature2 = basin_signature(&state);
            reactive_perturb.note_perturb(
                strategy,
                snap.value,
                state.total_value,
                h2 == snap.solution_hash,
            );
            if basin_len < basin_signatures.len() {
                basin_signatures[basin_len] = signature2;
                basin_len += 1;
            } else {
                basin_signatures[round % basin_signatures.len()] = signature2;
            }
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h2);
                tabu_membership.insert(h2);
            } else {
                let pos = round % 128;
                tabu_membership.remove(tabu_hashes[pos]);
                tabu_hashes[pos] = h2;
                tabu_membership.insert(h2);
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                best_bit.clone_from(&state.selected_bit);
                stall_count = 0;
                cnt += 2;
            }
        } else {
            stall_count = 0;
            let h = state.solution_hash;
            let signature = basin_signature(&state);
            if basin_len < basin_signatures.len() {
                basin_signatures[basin_len] = signature;
                basin_len += 1;
            } else {
                basin_signatures[round % basin_signatures.len()] = signature;
            }
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h);
                tabu_membership.insert(h);
            }
        }
    }

    let mut final_state = State::new_empty(challenge);
    for &i in &best_sel { final_state.add_item(i); }

    if let Some(t) = tsn_ref {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_tsn(&mut final_state, t);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before { break; }
        }
    } else if hp.use_heavy_polish {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_heavy(&mut final_state);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before { break; }
        }
    } else {
        loop {
            let before = final_state.clone_solution();
            local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            if reactive_perturb.polish_stalls < 3 {
                dp_refinement_hp(&mut final_state, ch);
                local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            }
            reactive_perturb.note_polish(before.value, final_state.total_value);
            if final_state.total_value <= before.value {
                final_state.restore_solution(&before);
                break;
            }
        }
    }

    if final_state.total_value > best_val {
        let v = final_state.total_value;
        (Solution { items: final_state.selected_items() }, v)
    } else {
        (Solution { items: best_sel }, best_val)
    }
}

fn eval_solution(ch: &Challenge, sol: &Solution) -> i64 {
    let mut val: i64 = 0;
    for &i in &sol.items {
        val += ch.values[i] as i64;
        for &j in &sol.items {
            if j > i { val += ch.interaction_values[i][j] as i64; }
        }
    }
    val
}

fn path_relink(challenge: &Challenge, sol_a: &Solution, sol_b: &Solution, hp: &Hparams) -> Solution {
    let n = challenge.num_items;
    let mut in_a = vec![false; n];
    let mut in_b = vec![false; n];
    for &i in &sol_a.items { in_a[i] = true; }
    for &i in &sol_b.items { in_b[i] = true; }

    let mut state = State::new_empty(challenge);
    for &i in &sol_a.items { state.add_item(i); }

    let mut to_add: Vec<usize> = (0..n).filter(|&i| in_b[i] && !in_a[i]).collect();
    let mut to_remove: Vec<usize> = (0..n).filter(|&i| in_a[i] && !in_b[i]).collect();

    let mut best_val = state.total_value;
    let mut best_bits = state.selected_bit.clone();
    let cap = challenge.max_weight;
    let total_moves = to_add.len() + to_remove.len();
    let checkpoint_interval = (total_moves / 4).max(3);
    let mut move_count = 0usize;

    while !to_add.is_empty() || !to_remove.is_empty() {
        let mut best_delta = i64::MIN;
        let mut best_action: Option<(bool, usize)> = None;

        for (idx, &item) in to_add.iter().enumerate() {
            if state.total_weight + challenge.weights[item] <= cap {
                let delta = state.contrib[item] as i64;
                if delta > best_delta { best_delta = delta; best_action = Some((true, idx)); }
            }
        }
        for (idx, &item) in to_remove.iter().enumerate() {
            let delta = -(state.contrib[item] as i64);
            if delta > best_delta { best_delta = delta; best_action = Some((false, idx)); }
        }

        match best_action {
            Some((true, idx)) => {
                let item = to_add[idx];
                state.add_item(item);
                to_add.swap_remove(idx);
            }
            Some((false, idx)) => {
                let item = to_remove[idx];
                state.remove_item(item);
                to_remove.swap_remove(idx);
            }
            None => break,
        }
        move_count += 1;

        if state.total_weight <= cap && state.total_value > best_val {
            best_val = state.total_value;
            best_bits = state.selected_bit.clone();
        }

        if move_count % checkpoint_interval == 0 && state.total_weight <= cap {
            let mut tmp = State::new_empty(challenge);
            for i in 0..n { if state.selected_bit[i] { tmp.add_item(i); } }
            local_search_vnd_fast(&mut tmp);
            dp_refinement_hp(&mut tmp, hp.core_half_dp);
            if tmp.total_value > best_val {
                best_val = tmp.total_value;
                best_bits = tmp.selected_bit.clone();
            }
        }
    }

    let mut final_state = State::new_empty(challenge);
    for i in 0..n { if best_bits[i] { final_state.add_item(i); } }
    loop {
        let v_before = final_state.total_value;
        local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
        dp_refinement_hp(&mut final_state, hp.core_half_dp);
        if final_state.total_value <= v_before { break; }
    }
    Solution { items: final_state.selected_items() }
}

fn build_frequency_biased(state: &mut State, freq: &[f64], rng: &mut Rng) {
    let n = state.ch.num_items;
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: f64 = f64::MIN;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let c = state.contrib[i] as f64;
            if c <= 0.0 { continue; }
            let w = (state.ch.weights[i] as f64).max(1.0);
            let s = (c / w) * (1.0 + freq[i] * 2.0)
                + (rng.next_u32() & 0x3F) as f64 * 0.01;
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

fn crossover_elite_frequency(elite: &[(Solution, i64)], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let mut freq = vec![0.0f64; n];
    let total = elite.len() as f64;
    for (sol, _) in elite {
        for &i in &sol.items { freq[i] += 1.0; }
    }
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    let mut order: Vec<(usize, f64)> = (0..n).map(|i| {
        let p = freq[i] / total;
        let w = ch.weights[i] as f64;
        (i, p * 1000.0 + (ch.values[i] as f64) / w.max(1.0))
    }).collect();
    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for entry in &order {
        let i = entry.0;
        let p = freq[i] / total;
        let threshold = if p > 0.8 { 0.1 } else if p > 0.5 { 0.4 } else { 0.7 };
        if rng.next_f64() > threshold && weight + ch.weights[i] <= ch.max_weight {
            bits[i] = true;
            weight += ch.weights[i];
        }
    }
    bits
}

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let n = challenge.num_items;
        let hp = Hparams::from_map(hyperparameters);
        let n_restarts = hp.n_full_restarts.max(1);

        let total_interactions: Vec<i64> = {
            let mut sums = vec![0i64; n];
            for i in 0..n {
                let row = unsafe { challenge.interaction_values.get_unchecked(i) };
                let mut si: i64 = 0;
                for j in 0..n { si += row[j] as i64; }
                sums[i] = si;
            }
            sums
        };

        if n > 1200 {
            let tsn = TopNeighbors::new(challenge, 12);
            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            let mut restart_stall = 0usize;
            for restart in 0..n_restarts {
                let (sol, val) = if restart >= 2 && restart % 2 == 0 && best_sol.is_some() {
                    run_one_instance_seeded(challenge, &hp, restart, Some(&tsn), &total_interactions, best_sol.as_ref())
                } else {
                    run_one_instance(challenge, &hp, restart, Some(&tsn), &total_interactions)
                };
                if val > best_quality {
                    best_quality = val;
                    best_sol = Some(sol);
                    restart_stall = 0;
                } else {
                    restart_stall += 1;
                    if restart_stall >= 5 { break; }
                }
            }
            return Ok(best_sol);
        }

        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..9999 { rng.next_u32(); }

        let ch_dp = hp.core_half_dp;
        let mut elite: Vec<(Solution, i64)> = Vec::new();

        let n_phase1 = n_restarts.min(6).max(2);
        let mut restart_stall_1k = 0usize;
        for restart in 0..n_phase1 {
            let (sol, val) = run_one_instance(challenge, &hp, restart, None, &total_interactions);
            let prev_best = elite.first().map(|e| e.1).unwrap_or(i64::MIN);
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            if val <= prev_best {
                restart_stall_1k += 1;
                if restart_stall_1k >= 3 && restart >= 2 { break; }
            } else {
                restart_stall_1k = 0;
            }
        }

        let mut freq = vec![0.0f64; n];
        let top_k = elite.len().min(4);
        for (sol, _) in &elite[..top_k] {
            for &i in &sol.items { freq[i] += 1.0 / top_k as f64; }
        }

        let n_phase3 = 4;
        for restart in 0..n_phase3 {
            let mut state = State::new_empty(challenge);
            if restart % 3 == 0 {
                let bits = crossover_elite_frequency(&elite[..top_k.min(elite.len())], challenge, &mut rng);
                set_state_from_bits(&mut state, &bits);
            } else {
                build_frequency_biased(&mut state, &freq, &mut rng);
            }
            dp_refinement_hp(&mut state, ch_dp);
            local_search_vnd_windowed(&mut state, hp.window_k);

            let mut best_snap = state.clone_solution();
            let mut stall = 0usize;
            for round in 0..19 {
                if stall >= 7 { break; }
                let snap = state.clone_solution();
                let strategy = round % 8;
                let strength = 4 + round / 3;
                perturb_by_strategy(&mut state, strength, stall, strategy, &mut rng, &hp, &total_interactions);
                greedy_reconstruct(&mut state, strategy % 10, &total_interactions);
                local_search_vnd_windowed(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, ch_dp);
                if state.total_value > best_snap.value {
                    best_snap = state.clone_solution();
                    stall = 0;
                } else {
                    state.restore_solution(&snap);
                    stall += 1;
                }
            }
            state.restore_solution(&best_snap);
            let val = state.total_value;
            elite.push((Solution { items: state.selected_items() }, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(8);

            freq.fill(0.0);
            let tk = elite.len().min(4);
            for (sol, _) in &elite[..tk] {
                for &i in &sol.items { freq[i] += 1.0 / tk as f64; }
            }
        }

        let n_relink = elite.len().min(3);
        for i in 0..n_relink {
            for j in (i+1)..n_relink {
                let sol_ab = path_relink(challenge, &elite[i].0, &elite[j].0, &hp);
                let val_ab = eval_solution(challenge, &sol_ab);
                elite.push((sol_ab, val_ab));
                let sol_ba = path_relink(challenge, &elite[j].0, &elite[i].0, &hp);
                let val_ba = eval_solution(challenge, &sol_ba);
                elite.push((sol_ba, val_ba));
            }
        }
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));

        {
            let (sol, val) = run_one_instance_seeded(challenge, &hp, 99, None, &total_interactions, Some(&elite[0].0));
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(8);
        }

        let deep_dp = 150;
        let mut best_val = i64::MIN;
        let mut best_sel = Vec::new();

        for idx in 0..elite.len().min(2) {
            let mut state = State::new_empty(challenge);
            for &i in &elite[idx].0.items { state.add_item(i); }
            loop {
                let v_before = state.total_value;
                local_search_vnd_heavy(&mut state);
                dp_refinement_hp(&mut state, deep_dp);
                if state.total_value <= v_before { break; }
            }
            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
            }
        }

        let mut state = State::new_empty(challenge);
        for &i in &best_sel { state.add_item(i); }

        for lns_round in 0..7 {
            let sel = state.selected_items();
            let pct = 20 + (lns_round % 4) * 8;
            let n_remove = sel.len() * pct / 100;
            let mut candidates: Vec<(usize, i64)> = sel.iter().map(|&i| {
                let score = match lns_round % 10 {
                    0 => state.contrib[i] as i64,
                    1 => -(state.ch.weights[i] as i64),
                    2 => state.contrib[i] as i64 - state.ch.values[i] as i64,
                    3 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 1000) / w },
                    4 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 10000) / (w * w) },
                    5 => -(state.contrib[i] as i64),
                    6 => rng.next_u32() as i64,
                    7 => {
                        let anti = 2 * state.contrib[i] as i64 - total_interactions[i];
                        let w = (state.ch.weights[i] as i64).max(1);
                        (anti * 1000) / w
                    },
                    8 => {
                        state.contrib[i] as i64 * 100 - total_interactions[i]
                    },
                    _ => {
                        if rng.next_u32() % 2 == 0 { rng.next_u32() as i64 }
                        else { state.contrib[i] as i64 }
                    },
                };
                (i, score)
            }).collect();
            candidates.sort_unstable_by_key(|&(_, s)| s);
            for j in 0..n_remove.min(candidates.len()) {
                state.remove_item(candidates[j].0);
            }

            match lns_round % 5 {
                0 => greedy_reconstruct(&mut state, 0, &total_interactions),
                1 => greedy_reconstruct(&mut state, 3, &total_interactions),
                2 => {
                    let mut cands: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
                    cands.sort_unstable_by(|&a, &b| {
                        let sa = state.contrib[a] as f64 / (state.ch.weights[a] as f64).max(1.0)
                            + freq[a] * 50.0;
                        let sb = state.contrib[b] as f64 / (state.ch.weights[b] as f64).max(1.0)
                            + freq[b] * 50.0;
                        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for &i in &cands {
                        if state.total_weight + state.ch.weights[i] <= challenge.max_weight {
                            state.add_item(i);
                        }
                    }
                },
                _ => greedy_reconstruct(&mut state, 2, &total_interactions),
            }

            loop {
                let v_before = state.total_value;
                local_search_vnd_windowed_deep(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, deep_dp);
                if state.total_value <= v_before { break; }
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
            }
            let mut rst = State::new_empty(challenge);
            for &i in &best_sel { rst.add_item(i); }
            state.restore_solution(&rst.clone_solution());
        }

        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        let tk = elite.len().min(6);
        let mut item_freq = vec![0usize; n];
        for (sol, _) in &elite[..tk] {
            for &i in &sol.items { item_freq[i] += 1; }
        }
        let always_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == tk).collect();
        let disputed: Vec<usize> = (0..n).filter(|&i| item_freq[i] > 0 && item_freq[i] < tk).collect();

        if disputed.len() > 0 && disputed.len() <= 199 {
            let mut state = State::new_empty(challenge);
            for &i in &always_in {
                if state.total_weight + challenge.weights[i] <= challenge.max_weight {
                    state.add_item(i);
                }
            }
            let fixed_weight = state.total_weight;
            let rem_cap = (challenge.max_weight - fixed_weight) as usize;

            if rem_cap > 0 {
                let dk = disputed.len();
                let mut total_disp_weight: usize = 0;
                for &it in &disputed { total_disp_weight += challenge.weights[it] as usize; }
                let myw = rem_cap.min(total_disp_weight);
                let dp_size = myw + 1;

                if dp_size <= 1_999_999 {
                    let mut dp = vec![i64::MIN / 4; dp_size];
                    let mut choose = vec![0u8; dk * dp_size];
                    dp[0] = 0;
                    let mut w_hi: usize = 0;

                    for (t, &it) in disputed.iter().enumerate() {
                        let wt = challenge.weights[it] as usize;
                        if wt > myw { continue; }
                        let val = state.contrib[it] as i64;
                        let new_hi = (w_hi + wt).min(myw);
                        for w in (wt..=new_hi).rev() {
                            let cand = dp[w - wt] + val;
                            if cand > dp[w] {
                                dp[w] = cand;
                                choose[t * dp_size + w] = 1;
                            }
                        }
                        w_hi = new_hi;
                    }

                    let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
                    let mut dp_selected = Vec::new();
                    for t in (0..dk).rev() {
                        let it = disputed[t];
                        let wt = challenge.weights[it] as usize;
                        if wt <= w_star && choose[t * dp_size + w_star] == 1 {
                            dp_selected.push(it);
                            w_star -= wt;
                        }
                    }

                    for &i in &dp_selected { state.add_item(i); }

                    loop {
                        let v_before = state.total_value;
                        local_search_vnd_heavy(&mut state);
                        dp_refinement_hp(&mut state, deep_dp);
                        if state.total_value <= v_before { break; }
                    }

                    if state.total_value > best_val {
                        best_sel = state.selected_items();
                    }
                }
            }
        }

        Ok(Some(Solution { items: best_sel }))
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!("The future of Tig is bright!!!");
}

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hp: &Option<serde_json::Map<String, serde_json::Value>>,
) -> anyhow::Result<()> {
    solve_challenge(challenge, save, hp)
}