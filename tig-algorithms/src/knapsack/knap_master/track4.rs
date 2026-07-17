use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

#[allow(dead_code, unused_imports, clippy::all)]
mod inner_four {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
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

    #[inline(always)]
    fn zobrist_key(i: usize) -> u64 {
        let mut h: u64 = 0x517CC1B727220A95;
        h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9)
    }

    struct State<'a> {
        ch: &'a Challenge,
        selected_bit: Vec<bool>,
        contrib: Vec<i32>,
        total_value: i64,
        total_weight: u32,
        hash: u64,
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
                hash: 0,
                dp_cache: Vec::new(),
                choose_cache: Vec::new(),
            }
        }

        #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

        #[inline(always)]
        fn add_item(&mut self, i: usize) {
            self.total_value += self.contrib[i] as i64;
            self.total_weight += self.ch.weights[i];
            self.hash ^= zobrist_key(i);
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(*row_ptr.add(k));
                }
            }
            self.selected_bit[i] = true;
        }

        #[inline(always)]
        fn remove_item(&mut self, j: usize) {
            self.total_value -= self.contrib[j] as i64;
            self.total_weight -= self.ch.weights[j];
            self.hash ^= zobrist_key(j);
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_sub(*row_ptr.add(k));
                }
            }
            self.selected_bit[j] = false;
        }

        #[inline(always)]
        fn replace_item(&mut self, rm: usize, cand: usize) {
            let row_rm = unsafe { self.ch.interaction_values.get_unchecked(rm) };
            let removed = self.contrib[rm] as i64;
            let added = self.contrib[cand].wrapping_sub(row_rm[cand]) as i64;
            self.total_value -= removed;
            self.total_value += added;
            self.total_weight -= self.ch.weights[rm];
            self.total_weight += self.ch.weights[cand];
            self.hash ^= zobrist_key(rm);
            self.hash ^= zobrist_key(cand);
            let n = self.ch.num_items;
            let row_rm_ptr = row_rm.as_ptr();
            let row_add_ptr = unsafe { self.ch.interaction_values.get_unchecked(cand).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_sub(*row_rm_ptr.add(k))
                        .wrapping_add(*row_add_ptr.add(k));
                }
            }
            self.selected_bit[rm] = false;
            self.selected_bit[cand] = true;
        }

        #[inline(always)]
        fn add_pair(&mut self, a: usize, b: usize) {
            let row_a = unsafe { self.ch.interaction_values.get_unchecked(a) };
            let gain_a = self.contrib[a] as i64;
            let gain_b = self.contrib[b].wrapping_add(row_a[b]) as i64;
            self.total_value += gain_a;
            self.total_value += gain_b;
            self.total_weight += self.ch.weights[a];
            self.total_weight += self.ch.weights[b];
            self.hash ^= zobrist_key(a);
            self.hash ^= zobrist_key(b);
            let n = self.ch.num_items;
            let row_a_ptr = row_a.as_ptr();
            let row_b_ptr = unsafe { self.ch.interaction_values.get_unchecked(b).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_add(*row_a_ptr.add(k))
                        .wrapping_add(*row_b_ptr.add(k));
                }
            }
            self.selected_bit[a] = true;
            self.selected_bit[b] = true;
        }

        #[inline(always)]
        fn replace_one_with_two(&mut self, rm: usize, a1: usize, a2: usize) {
            let row_rm = unsafe { self.ch.interaction_values.get_unchecked(rm) };
            let row_a1 = unsafe { self.ch.interaction_values.get_unchecked(a1) };
            let removed = self.contrib[rm] as i64;
            let gain_a1 = self.contrib[a1].wrapping_sub(row_rm[a1]) as i64;
            let gain_a2 = self.contrib[a2]
                .wrapping_sub(row_rm[a2])
                .wrapping_add(row_a1[a2]) as i64;
            self.total_value -= removed;
            self.total_value += gain_a1;
            self.total_value += gain_a2;
            self.total_weight -= self.ch.weights[rm];
            self.total_weight += self.ch.weights[a1];
            self.total_weight += self.ch.weights[a2];
            self.hash ^= zobrist_key(rm);
            self.hash ^= zobrist_key(a1);
            self.hash ^= zobrist_key(a2);
            let n = self.ch.num_items;
            let row_rm_ptr = row_rm.as_ptr();
            let row_a1_ptr = row_a1.as_ptr();
            let row_a2_ptr = unsafe { self.ch.interaction_values.get_unchecked(a2).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_sub(*row_rm_ptr.add(k))
                        .wrapping_add(*row_a1_ptr.add(k))
                        .wrapping_add(*row_a2_ptr.add(k));
                }
            }
            self.selected_bit[rm] = false;
            self.selected_bit[a1] = true;
            self.selected_bit[a2] = true;
        }

        #[inline(always)]
        fn replace_two_with_one(&mut self, r1: usize, r2: usize, add: usize) {
            let row_r1 = unsafe { self.ch.interaction_values.get_unchecked(r1) };
            let row_r2 = unsafe { self.ch.interaction_values.get_unchecked(r2) };
            let removed_r1 = self.contrib[r1] as i64;
            let removed_r2 = self.contrib[r2].wrapping_sub(row_r1[r2]) as i64;
            let gain = self.contrib[add]
                .wrapping_sub(row_r1[add])
                .wrapping_sub(row_r2[add]) as i64;
            self.total_value -= removed_r1;
            self.total_value -= removed_r2;
            self.total_value += gain;
            self.total_weight -= self.ch.weights[r1];
            self.total_weight -= self.ch.weights[r2];
            self.total_weight += self.ch.weights[add];
            self.hash ^= zobrist_key(r1);
            self.hash ^= zobrist_key(r2);
            self.hash ^= zobrist_key(add);
            let n = self.ch.num_items;
            let row_r1_ptr = row_r1.as_ptr();
            let row_r2_ptr = row_r2.as_ptr();
            let row_add_ptr = unsafe { self.ch.interaction_values.get_unchecked(add).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_sub(*row_r1_ptr.add(k))
                        .wrapping_sub(*row_r2_ptr.add(k))
                        .wrapping_add(*row_add_ptr.add(k));
                }
            }
            self.selected_bit[r1] = false;
            self.selected_bit[r2] = false;
            self.selected_bit[add] = true;
        }

        #[inline(always)]
        fn replace_two_with_two(&mut self, r1: usize, r2: usize, a1: usize, a2: usize) {
            let row_r1 = unsafe { self.ch.interaction_values.get_unchecked(r1) };
            let row_r2 = unsafe { self.ch.interaction_values.get_unchecked(r2) };
            let row_a1 = unsafe { self.ch.interaction_values.get_unchecked(a1) };
            let removed_r1 = self.contrib[r1] as i64;
            let removed_r2 = self.contrib[r2].wrapping_sub(row_r1[r2]) as i64;
            let gain_a1 = self.contrib[a1]
                .wrapping_sub(row_r1[a1])
                .wrapping_sub(row_r2[a1]) as i64;
            let gain_a2 = self.contrib[a2]
                .wrapping_sub(row_r1[a2])
                .wrapping_sub(row_r2[a2])
                .wrapping_add(row_a1[a2]) as i64;
            self.total_value -= removed_r1;
            self.total_value -= removed_r2;
            self.total_value += gain_a1;
            self.total_value += gain_a2;
            self.total_weight -= self.ch.weights[r1];
            self.total_weight -= self.ch.weights[r2];
            self.total_weight += self.ch.weights[a1];
            self.total_weight += self.ch.weights[a2];
            self.hash ^= zobrist_key(r1);
            self.hash ^= zobrist_key(r2);
            self.hash ^= zobrist_key(a1);
            self.hash ^= zobrist_key(a2);
            let n = self.ch.num_items;
            let row_r1_ptr = row_r1.as_ptr();
            let row_r2_ptr = row_r2.as_ptr();
            let row_a1_ptr = row_a1.as_ptr();
            let row_a2_ptr = unsafe { self.ch.interaction_values.get_unchecked(a2).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_sub(*row_r1_ptr.add(k))
                        .wrapping_sub(*row_r2_ptr.add(k))
                        .wrapping_add(*row_a1_ptr.add(k))
                        .wrapping_add(*row_a2_ptr.add(k));
                }
            }
            self.selected_bit[r1] = false;
            self.selected_bit[r2] = false;
            self.selected_bit[a1] = true;
            self.selected_bit[a2] = true;
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
                hash: self.hash,
            }
        }

        fn restore_solution(&mut self, sol: &SolState) {
            self.selected_bit.clone_from(&sol.bits);
            self.contrib.clone_from(&sol.contrib);
            self.total_value = sol.value;
            self.total_weight = sol.weight;
            self.hash = sol.hash;
        }
    }

    #[derive(Clone)]
    struct SolState {
        bits: Vec<bool>,
        contrib: Vec<i32>,
        value: i64,
        weight: u32,
        hash: u64,
    }

    #[derive(Clone, Copy)]
    struct ExactDivider {
        divisor: u64,
        reciprocal: u64,
    }

    impl ExactDivider {
        #[inline(always)]
        fn new(divisor: u32) -> Self {
            let divisor = divisor.max(1) as u64;
            let reciprocal = if divisor == 1 {
                0
            } else {
                u64::MAX / divisor + 1
            };
            Self { divisor, reciprocal }
        }

        #[inline(always)]
        fn divide(&self, numerator: u64) -> u64 {
            if self.divisor == 1 {
                return numerator;
            }
            let estimate =
                ((numerator as u128 * self.reciprocal as u128) >> 64) as u64;
            estimate - (((estimate as u128 * self.divisor as u128)
                > numerator as u128) as u64)
        }
    }

    thread_local! {
        static FORWARD_DIVIDERS: std::cell::RefCell<Option<Vec<ExactDivider>>> =
            std::cell::RefCell::new(None);
    }

    struct InteractionNeighborCache {
        queries: Vec<u8>,
        rows: Vec<Option<Vec<usize>>>,
    }

    thread_local! {
        static INTERACTION_NEIGHBOR_CACHE: std::cell::RefCell<Option<InteractionNeighborCache>> =
            std::cell::RefCell::new(None);
    }

    fn touch_interaction_order(cache: &mut InteractionNeighborCache, ch: &Challenge, anchor: usize) {
        cache.queries[anchor] = cache.queries[anchor].saturating_add(1);
        if cache.rows[anchor].is_none() && cache.queries[anchor] >= 8 {
            let row = &ch.interaction_values[anchor];
            let mut order: Vec<usize> = (0..ch.num_items).collect();
            order.sort_unstable_by(|&a, &b| {
                row[b].cmp(&row[a]).then(a.cmp(&b))
            });
            cache.rows[anchor] = Some(order);
        }
    }

    fn collect_cached_synergy_neighbors(
        state: &State,
        anchor: usize,
        present: &[bool],
    ) -> Option<Vec<(usize, i32)>> {
        INTERACTION_NEIGHBOR_CACHE.with(|cell| {
            let mut cached = cell.borrow_mut();
            let cache = cached.as_mut()?;
            touch_interaction_order(cache, state.ch, anchor);
            let order = cache.rows[anchor].as_ref()?;
            let row = &state.ch.interaction_values[anchor];
            let mut neighbors = Vec::with_capacity(4);
            for &item in order {
                let synergy = row[item];
                if synergy <= 0 { break; }
                if state.selected_bit[item] || present[item] { continue; }
                neighbors.push((item, synergy));
                if neighbors.len() == 4 { break; }
            }
            Some(neighbors)
        })
    }

    fn collect_cached_ejection_supplemental(
        state: &State,
        anchors: &[usize],
        need: usize,
    ) -> Option<Vec<(usize, i32)>> {
        INTERACTION_NEIGHBOR_CACHE.with(|cell| {
            let mut cached = cell.borrow_mut();
            let cache = cached.as_mut()?;
            for &anchor in anchors {
                touch_interaction_order(cache, state.ch, anchor);
            }
            if anchors.iter().any(|&anchor| cache.rows[anchor].is_none()) {
                return None;
            }

            let mut positions = vec![0usize; anchors.len()];
            let mut result: Vec<(usize, i32)> = Vec::with_capacity(need);
            while result.len() < need {
                let mut best: Option<(usize, usize, i32)> = None;
                for (anchor_pos, &anchor) in anchors.iter().enumerate() {
                    let order = cache.rows[anchor].as_ref().unwrap();
                    let row = &state.ch.interaction_values[anchor];
                    let mut pos = positions[anchor_pos];
                    while pos < order.len() {
                        let item = order[pos];
                        let synergy = row[item];
                        if synergy <= 0 {
                            pos = order.len();
                            break;
                        }
                        if state.selected_bit[item]
                            || anchors.contains(&item)
                            || result.iter().any(|&(seen, _)| seen == item)
                        {
                            pos += 1;
                            continue;
                        }
                        break;
                    }
                    positions[anchor_pos] = pos;
                    if pos == order.len() { continue; }
                    let item = order[pos];
                    let synergy = row[item];
                    if best.map_or(true, |(_, best_item, best_synergy)| {
                        synergy > best_synergy
                            || (synergy == best_synergy && item < best_item)
                    }) {
                        best = Some((anchor_pos, item, synergy));
                    }
                }

                if let Some((anchor_pos, item, synergy)) = best {
                    positions[anchor_pos] += 1;
                    result.push((item, synergy));
                } else {
                    break;
                }
            }
            Some(result)
        })
    }

    fn build_greedy_density(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        for i in 0..n { state.add_item(i); }
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

    fn build_greedy_hub(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| {
            let s: i64 = state.ch.interaction_values[i].iter().map(|&v| v as i64).sum();
            (i, s)
        }).collect();
        hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        for &(i, _) in &hub_scores {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    fn build_greedy_synergy_weight(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
            let avg_syn: i64 = if n > 1 {
                state.ch.interaction_values[i].iter().map(|&v| v as i64).sum::<i64>() / (n as i64 - 1)
            } else { 0 };
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
        FORWARD_DIVIDERS.with(|cache| {
            let cached = cache.borrow();
            let dividers = cached.as_ref().unwrap();
            loop {
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
                    let mut s = match mode {
                        2 => c,
                        3 => dividers[i].divide((c as u64) * 1000) as i64
                            + (state.ch.weights[i] as i64) * 3,
                        _ => dividers[i].divide((c as u64) * 1000) as i64,
                    };
                    if mode >= 4 {
                        let mask = if mode >= 5 { 0x7F } else { 0x1F };
                        s += (rng.next_u32() & mask) as i64;
                    }
                    if s > best_s {
                        second_s = best_s; second_i = best_i;
                        best_s = s; best_i = Some(i);
                    } else if s > second_s {
                        second_s = s; second_i = Some(i);
                    }
                }
                let pick = if mode >= 4 && second_i.is_some() {
                    let m = if mode >= 5 { 1 } else { 3 };
                    if (rng.next_u32() & m) == 0 { second_i } else { best_i }
                } else { best_i };
                if let Some(i) = pick { state.add_item(i); } else { break; }
            }
        });
    }

    fn build_anchor_neighborhood_seed(state: &mut State, anchor: usize) {
        if state.ch.weights[anchor] > state.ch.max_weight { return; }
        state.add_item(anchor);
        let n = state.ch.num_items;

        loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best: Option<(usize, i64)> = None;
            for item in 0..n {
                if state.selected_bit[item] || state.ch.weights[item] > slack { continue; }
                let marginal = state.contrib[item] as i64;
                if marginal <= 0 { continue; }
                let weight = (state.ch.weights[item] as i64).max(1);
                let affinity = (state.ch.interaction_values[anchor][item] as i64).max(0);
                let score = (marginal * 1000 + affinity * 350) / weight;
                if best.map_or(true, |(_, best_score)| score > best_score) {
                    best = Some((item, score));
                }
            }
            if let Some((item, _)) = best {
                state.add_item(item);
            } else {
                break;
            }
        }
    }

    fn select_interaction_anchors(challenge: &Challenge, limit: usize) -> Vec<usize> {
        if limit == 0 || challenge.num_items == 0 { return Vec::new(); }
        let n = challenge.num_items;
        let mut candidates: Vec<usize> = (0..n)
            .filter(|&item| challenge.weights[item] <= challenge.max_weight)
            .collect();
        if candidates.is_empty() { return candidates; }

        if n > 500 {
            candidates.sort_unstable_by(|&a, &b| {
                let va = challenge.values[a] as i64;
                let vb = challenge.values[b] as i64;
                let wa = (challenge.weights[a] as i64).max(1);
                let wb = (challenge.weights[b] as i64).max(1);
                (va * wb).cmp(&(vb * wa)).reverse()
            });
            candidates.truncate(40);
        }

        let mut scored: Vec<(usize, i64)> = Vec::with_capacity(candidates.len());
        for item in candidates {
            let positive_row_sum: i64 = challenge.interaction_values[item]
                .iter()
                .map(|&interaction| (interaction as i64).max(0))
                .sum();
            let weight = (challenge.weights[item] as i64).max(1);
            let score = (challenge.values[item] as i64 * 1000
                + positive_row_sum * 350 / n.max(1) as i64)
                / weight;
            scored.push((item, score));
        }
        scored.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
        scored.into_iter().take(limit).map(|(item, _)| item).collect()
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
                state.add_pair(pi, pj);
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
        let mut reachability: Vec<u64> = Vec::new();

        for _iter in 0..3 {
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
            let core = &by_density[left..right];

            let mut target = vec![0u8; n];
            let mut used_locked: u64 = 0;
            for &it in &by_density[..left] {
                used_locked += weights[it] as u64;
                target[it] = 1;
            }
            let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
            let myk = core.len();
            if myk == 0 || rem_cap == 0 { break; }

            let mut total_core_weight: usize = 0;
            let mut total_pos_weight: usize = 0;
            let mut all_pos_fit = true;
            for &it in core {
                let wt = weights[it] as usize;
                total_core_weight += wt;
                if contrib[it] > 0 {
                    total_pos_weight += wt;
                    if total_pos_weight > rem_cap { all_pos_fit = false; }
                }
            }

            if all_pos_fit {
                for &it in core {
                    if contrib[it] > 0 { target[it] = 1; }
                }
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

                let reach_words = (dp_size + 63) / 64;
                if reachability.len() < reach_words {
                    reachability.resize(reach_words, 0);
                }
                reachability[..reach_words].fill(0);
                reachability[0] = 1;
                let mut reachable_count = 1usize;
                let mut sparse = true;
                let mut w_hi: usize = 0;
                for (t, &it) in core.iter().enumerate() {
                    let wt = weights[it] as usize;
                    if wt > myw { continue; }
                    let val = contrib[it] as i64;
                    let new_hi = (w_hi + wt).min(myw);
                    if sparse && reachable_count.saturating_mul(4) >= dp_size {
                        sparse = false;
                    }
                    if sparse {
                        let max_source = w_hi.min(myw - wt);
                        let last_word = max_source >> 6;
                        for word_idx in (0..=last_word).rev() {
                            let mut word = reachability[word_idx];
                            if word_idx == last_word {
                                let keep_bits = (max_source & 63) + 1;
                                if keep_bits < 64 {
                                    word &= (1u64 << keep_bits) - 1;
                                }
                            }
                            while word != 0 {
                                let bit = 63 - word.leading_zeros() as usize;
                                let source = (word_idx << 6) + bit;
                                let w = source + wt;
                                let cand = state.dp_cache[source] + val;
                                if cand > state.dp_cache[w] {
                                    state.dp_cache[w] = cand;
                                    state.choose_cache[t * dp_size + w] = 1;
                                }
                                let dest_word = w >> 6;
                                let dest_mask = 1u64 << (w & 63);
                                if reachability[dest_word] & dest_mask == 0 {
                                    reachability[dest_word] |= dest_mask;
                                    reachable_count += 1;
                                }
                                word &= !(1u64 << bit);
                            }
                        }
                    } else {
                        for w in (wt..=new_hi).rev() {
                            let cand = state.dp_cache[w - wt] + val;
                            if cand > state.dp_cache[w] {
                                state.dp_cache[w] = cand;
                                state.choose_cache[t * dp_size + w] = 1;
                            }
                        }
                    }
                    w_hi = new_hi;
                }

                let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
                for t in (0..myk).rev() {
                    let it = core[t];
                    let wt = weights[it] as usize;
                    if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                        target[it] = 1;
                        w_star -= wt;
                    }
                }
            }

            let before = state.clone_solution();
            let mut changed = false;
            for i in 0..n {
                if state.selected_bit[i] && target[i] == 0 {
                    state.remove_item(i);
                    changed = true;
                }
            }
            for i in 0..n {
                if target[i] != 0 && !state.selected_bit[i] {
                    state.add_item(i);
                    changed = true;
                }
            }
            if !changed { break; }
            if state.total_value <= before.value {
                state.restore_solution(&before);
                break;
            }
        }
    }

    fn apply_best_add(state: &mut State) -> bool {
        let slack = state.slack();
        if slack == 0 { return false; }
        let n = state.ch.num_items;
        let mut best_i: Option<usize> = None;
        let mut best_d: i32 = 0;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let d = state.contrib[i];
            if d > best_d { best_d = d; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); true } else { false }
    }

    fn apply_best_swap_1_1(state: &mut State, selected: &[usize]) -> bool {
        let n = state.ch.num_items;
        let slack = state.slack();
        let mut best: Option<(usize, usize, i32)> = None;
        for &rm in selected {
            let w_rm = state.ch.weights[rm];
            let max_w = w_rm + slack;
            for cand in 0..n {
                if state.selected_bit[cand] { continue; }
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

    fn apply_pair_add(state: &mut State) -> bool {
        let slack = state.slack();
        if slack < 2 { return false; }
        let n = state.ch.num_items;
        let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= slack).collect();
        let m = unsel.len();
        if m < 2 { return false; }

        let mut best_delta: i64 = 0;
        let mut best_pair: Option<(usize, usize)> = None;
        for ai in 0..m {
            let a = unsel[ai];
            let wa = state.ch.weights[a];
            let ca = state.contrib[a] as i64;
            for bi in (ai+1)..m {
                let b = unsel[bi];
                if wa + state.ch.weights[b] > slack { continue; }
                let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                if delta > best_delta {
                    best_delta = delta;
                    best_pair = Some((a, b));
                }
            }
        }
        if let Some((a, b)) = best_pair {
            state.add_pair(a, b);
            true
        } else { false }
    }

    fn build_ejection_windows(state: &State, best_unused: &[usize], worst_used: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut unused: Vec<usize> = best_unused.iter().take(24).copied().collect();
        let used: Vec<usize> = worst_used.iter().take(24).copied().collect();

        if !unused.is_empty() && unused.len() < 32 {
            let need = 32 - unused.len();
            if let Some(supplemental) =
                collect_cached_ejection_supplemental(state, &unused, need)
            {
                for (i, _) in supplemental {
                    unused.push(i);
                }
            } else {
                let mut supplemental: Vec<(usize, i32)> = Vec::new();
                for i in 0..state.ch.num_items {
                    if state.selected_bit[i] || unused.contains(&i) { continue; }
                    let mut best_syn = 0i32;
                    for &anchor in &unused {
                        let syn = state.ch.interaction_values[i][anchor];
                        if syn > best_syn { best_syn = syn; }
                    }
                    if best_syn > 0 { supplemental.push((i, best_syn)); }
                }
                supplemental.sort_unstable_by_key(|&(_, syn)| std::cmp::Reverse(syn));
                for (i, _) in supplemental.into_iter().take(need) {
                    unused.push(i);
                }
            }
        }

        (unused, used)
    }

    fn apply_chain_move(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        if unused.len() < 2 || used.is_empty() { return false; }

        let mut best_delta = 0i64;
        let mut best_move: Option<(usize, usize, usize)> = None;
        let slack = state.slack() as i64;

        for &rm in used {
            let w_rm = state.ch.weights[rm] as i64;
            let c_rm = state.contrib[rm] as i64;
            let budget = slack + w_rm;

            for (ai, &a1) in unused.iter().enumerate() {
                let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 > budget { continue; }
                let gain_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;

                for &a2 in unused.iter().skip(ai + 1) {
                    let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget { continue; }
                    let delta = gain_a1
                        + state.contrib[a2] as i64
                        - state.ch.interaction_values[a2][rm] as i64
                        + state.ch.interaction_values[a1][a2] as i64
                        - c_rm;
                    if delta > best_delta {
                        best_delta = delta;
                        best_move = Some((rm, a1, a2));
                    }
                }
            }
        }

        if let Some((rm, a1, a2)) = best_move {
            state.replace_one_with_two(rm, a1, a2);
            true
        } else { false }
    }

    fn apply_reverse_chain(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        if unused.is_empty() || used.len() < 2 { return false; }

        let mut best_delta = 0i64;
        let mut best_move: Option<(usize, usize, usize)> = None;
        let slack = state.slack() as i64;

        for &add in unused {
            let w_add = state.ch.weights[add] as i64;
            let c_add = state.contrib[add] as i64;

            for (ri, &r1) in used.iter().enumerate() {
                let w_r1 = state.ch.weights[r1] as i64;
                let c_r1 = state.contrib[r1] as i64;
                for &r2 in used.iter().skip(ri + 1) {
                    let w_r2 = state.ch.weights[r2] as i64;
                    if w_add > slack + w_r1 + w_r2 { continue; }
                    let delta = c_add
                        - state.ch.interaction_values[add][r1] as i64
                        - state.ch.interaction_values[add][r2] as i64
                        - c_r1
                        - state.contrib[r2] as i64
                        + state.ch.interaction_values[r1][r2] as i64;
                    if delta > best_delta {
                        best_delta = delta;
                        best_move = Some((r1, r2, add));
                    }
                }
            }
        }

        if let Some((r1, r2, add)) = best_move {
            state.replace_two_with_one(r1, r2, add);
            true
        } else { false }
    }

    fn capacity_repair_search_dfs(
        removal_count: usize,
        target_count: usize,
        pos: usize,
        chosen: &mut [u8; 5],
        chosen_len: usize,
        freed: u32,
        loss: i64,
        removal_weights: &[u32; 18],
        removal_values: &[i64; 18],
        removal_pairs: &[[i64; 18]; 18],
        target_penalties: &[[i64; 18]; 8],
        target_values: &[i64; 8],
        needs: &[u32; 8],
        current_penalties: &mut [i64; 8],
        best_deltas: &mut [i64; 8],
        best_memberships: &mut [[u8; 5]; 8],
        best_lengths: &mut [u8; 8],
    ) {
        if chosen_len != 0 {
            for target_pos in 0..target_count {
                if freed < needs[target_pos] { continue; }
                let delta = target_values[target_pos] - current_penalties[target_pos] - loss;
                if delta > 0 && delta > best_deltas[target_pos] {
                    best_deltas[target_pos] = delta;
                    best_memberships[target_pos] = *chosen;
                    best_lengths[target_pos] = chosen_len as u8;
                }
            }
        }
        if pos >= removal_count || chosen_len >= 5 { return; }

        for idx in pos..removal_count {
            let mut pair_correction = 0i64;
            for prior_pos in 0..chosen_len {
                pair_correction += removal_pairs[idx][chosen[prior_pos] as usize];
            }
            chosen[chosen_len] = idx as u8;
            for target_pos in 0..target_count {
                current_penalties[target_pos] += target_penalties[target_pos][idx];
            }
            capacity_repair_search_dfs(
                removal_count,
                target_count,
                idx + 1,
                chosen,
                chosen_len + 1,
                freed.saturating_add(removal_weights[idx]),
                loss + removal_values[idx] - pair_correction,
                removal_weights,
                removal_values,
                removal_pairs,
                target_penalties,
                target_values,
                needs,
                current_penalties,
                best_deltas,
                best_memberships,
                best_lengths,
            );
            for target_pos in 0..target_count {
                current_penalties[target_pos] -= target_penalties[target_pos][idx];
            }
        }
    }

    fn apply_capacity_repair(state: &mut State, best_unused: &[usize], worst_used: &[usize]) -> bool {
        if best_unused.is_empty() || worst_used.len() < 2 || state.slack() >= state.ch.max_weight / 5 {
            return false;
        }

        let mut targets: Vec<usize> = best_unused.iter()
            .copied()
            .filter(|&i| state.ch.weights[i] > state.slack())
            .collect();
        targets.sort_unstable_by_key(|&i| std::cmp::Reverse(state.contrib[i]));
        targets.truncate(8);
        if targets.is_empty() { return false; }

        let removals: Vec<usize> = worst_used.iter().take(18).copied().collect();
        let removal_count = removals.len();
        let target_count = targets.len();
        let mut removal_weights = [0u32; 18];
        let mut removal_values = [0i64; 18];
        let mut removal_pairs = [[0i64; 18]; 18];
        for idx in 0..removal_count {
            let rm = removals[idx];
            removal_weights[idx] = state.ch.weights[rm];
            removal_values[idx] = state.contrib[rm] as i64;
            let row = &state.ch.interaction_values[rm];
            for prior in 0..idx {
                removal_pairs[idx][prior] = row[removals[prior]] as i64;
            }
        }

        let slack = state.slack();
        let mut needs = [0u32; 8];
        let mut target_values = [0i64; 8];
        let mut target_penalties = [[0i64; 18]; 8];
        for target_pos in 0..target_count {
            let add = targets[target_pos];
            needs[target_pos] = state.ch.weights[add] - slack;
            target_values[target_pos] = state.contrib[add] as i64;
            let row = &state.ch.interaction_values[add];
            for idx in 0..removal_count {
                target_penalties[target_pos][idx] = row[removals[idx]] as i64;
            }
        }

        let mut chosen = [0u8; 5];
        let mut current_penalties = [0i64; 8];
        let mut best_deltas = [0i64; 8];
        let mut best_memberships = [[0u8; 5]; 8];
        let mut best_lengths = [0u8; 8];
        capacity_repair_search_dfs(
            removal_count,
            target_count,
            0,
            &mut chosen,
            0,
            0,
            0,
            &removal_weights,
            &removal_values,
            &removal_pairs,
            &target_penalties,
            &target_values,
            &needs,
            &mut current_penalties,
            &mut best_deltas,
            &mut best_memberships,
            &mut best_lengths,
        );

        let mut best_move: Option<(i64, usize)> = None;
        for target_pos in 0..target_count {
            let delta = best_deltas[target_pos];
            if delta > 0 && best_move.as_ref().map_or(true, |(best_delta, _)| delta > *best_delta) {
                best_move = Some((delta, target_pos));
            }
        }

        if let Some((_, target_pos)) = best_move {
            let add = targets[target_pos];
            let before = state.clone_solution();
            let membership = best_memberships[target_pos];
            let membership_len = best_lengths[target_pos] as usize;
            for member_pos in 0..membership_len {
                state.remove_item(removals[membership[member_pos] as usize]);
            }
            if state.ch.weights[add] <= state.slack() {
                state.add_item(add);
                if state.total_value > before.value {
                    return true;
                }
            }
            state.restore_solution(&before);
        }
        false
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
            state.replace_two_with_two(r1, r2, a1, a2);
            true
        } else { false }
    }

    fn local_search_vnd_fast_certified(state: &mut State) -> bool {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..79 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            return true;
        }
        false
    }

    fn local_search_vnd_fast(state: &mut State) {
        let _ = local_search_vnd_fast_certified(state);
    }

    fn local_search_vnd_medium_certified(state: &mut State, k: usize) -> bool {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..119 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            if apply_pair_add(state) { continue; }
            if apply_swap_2_2_bounded(state, k) { continue; }
            return true;
        }
        false
    }

    fn local_search_vnd_medium(state: &mut State, k: usize) {
        let _ = local_search_vnd_medium_certified(state, k);
    }

    fn ils_vnd_certified(state: &mut State, hp: &Hparams) -> bool {
        match hp.ils_vnd_level {
            0 => local_search_vnd_fast_certified(state),
            1 => local_search_vnd_medium_certified(state, hp.bounded_2_2_k),
            _ => local_search_vnd_heavy_certified(state),
        }
    }

    fn ils_vnd(state: &mut State, hp: &Hparams) {
        let _ = ils_vnd_certified(state, hp);
    }

    fn local_search_vnd_heavy_certified(state: &mut State) -> bool {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..299 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            if apply_pair_add(state) { continue; }
            if apply_swap_2_2_bounded(state, 25) { continue; }
            let (best_unused, worst_used) = build_windows(state, 32);
            let (chain_unused, chain_used) = build_ejection_windows(state, &best_unused, &worst_used);
            if apply_chain_move(state, &chain_unused, &chain_used) { continue; }
            if apply_reverse_chain(state, &chain_unused, &chain_used) { continue; }
            return true;
        }
        false
    }

    fn local_search_vnd_heavy(state: &mut State) {
        let _ = local_search_vnd_heavy_certified(state);
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

    fn set_state_from_bits(state: &mut State, bits: &[bool]) {
        let n = state.ch.num_items;
        for i in (0..n).rev() {
            if state.selected_bit[i] && !bits[i] { state.remove_item(i); }
        }
        for i in 0..n {
            if bits[i] && !state.selected_bit[i] { state.add_item(i); }
        }
    }

    fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
        let n = state.ch.num_items;
        let mut unused_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        let mut used_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let r = state.contrib[i] as f64 / (state.ch.weights[i] as f64).max(1.0);
            if state.selected_bit[i] { used_r.push((i, r)); } else { unused_r.push((i, r)); }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        (unused_r[..ku].iter().map(|x| x.0).collect(), used_r[..ks].iter().map(|x| x.0).collect())
    }

    fn augment_synergy_neighbors(state: &State, best_unused: &[usize]) -> Vec<usize> {
        let mut augmented: Vec<usize> = best_unused.iter().copied().collect();
        augmented.sort_unstable_by(|&a, &b| {
            let ca = state.contrib[a] as i64;
            let cb = state.contrib[b] as i64;
            let wa = (state.ch.weights[a] as i64).max(1);
            let wb = (state.ch.weights[b] as i64).max(1);
            (ca * wb).cmp(&(cb * wa)).reverse()
        });
        augmented.truncate(32);

        let mut present = vec![false; state.ch.num_items];
        for &item in &augmented {
            present[item] = true;
        }
        let anchors: Vec<usize> = augmented.iter().take(8).copied().collect();
        for anchor in anchors {
            if augmented.len() >= 56 { break; }
            let neighbors = if let Some(neighbors) =
                collect_cached_synergy_neighbors(state, anchor, &present)
            {
                neighbors
            } else {
                let row = &state.ch.interaction_values[anchor];
                let mut neighbors: Vec<(usize, i32)> = Vec::with_capacity(4);
                for item in 0..state.ch.num_items {
                    if state.selected_bit[item] || present[item] { continue; }
                    let synergy = row[item];
                    if synergy <= 0 { continue; }
                    if neighbors.len() < 4 {
                        neighbors.push((item, synergy));
                        neighbors.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
                    } else if synergy > neighbors[3].1 {
                        neighbors[3] = (item, synergy);
                        neighbors.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
                    }
                }
                neighbors
            };
            for (item, _) in neighbors {
                if augmented.len() >= 56 { break; }
                present[item] = true;
                augmented.push(item);
            }
        }
        augmented
    }

    fn screen_compound_candidates(state: &State, best_unused: &[usize], worst_used: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut source: Vec<usize> = best_unused.iter().copied().collect();
        source.sort_unstable_by(|&a, &b| {
            let ca = state.contrib[a] as i64;
            let cb = state.contrib[b] as i64;
            let wa = (state.ch.weights[a] as i64).max(1);
            let wb = (state.ch.weights[b] as i64).max(1);
            (ca * wb).cmp(&(cb * wa)).reverse()
        });
        source.truncate(48);

        let removals: Vec<usize> = worst_used.iter().take(24).copied().collect();
        let mut scored: Vec<(usize, i64)> = Vec::with_capacity(source.len());
        for &i in &source {
            let mut first = 0i64;
            let mut second = 0i64;
            let mut third = 0i64;
            for &j in &source {
                if i == j { continue; }
                let syn = state.ch.interaction_values[i][j] as i64;
                if syn > first {
                    third = second;
                    second = first;
                    first = syn;
                } else if syn > second {
                    third = second;
                    second = syn;
                } else if syn > third {
                    third = syn;
                }
            }
            let removal_penalty: i64 = removals.iter()
                .map(|&r| (state.ch.interaction_values[i][r] as i64).max(0))
                .sum();
            scored.push((i, state.contrib[i] as i64 + first + second + third - removal_penalty / 2));
        }
        scored.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
        scored.truncate(40);

        (scored.into_iter().map(|(i, _)| i).collect(), removals)
    }

    fn apply_swap_2_2_windowed(state: &mut State, best_unused: &[usize], worst_used: &[usize]) -> bool {
        let unused = &best_unused[..best_unused.len().min(24)];
        let used = &worst_used[..worst_used.len().min(24)];
        if unused.len() < 2 || used.len() < 2 { return false; }

        let m = unused.len();
        let weights = &state.ch.weights;
        let interactions = &state.ch.interaction_values;
        let mut add_weights = [0i64; 24];
        let mut add_contrib = [0i64; 24];
        let mut add_synergy = [[0i64; 24]; 24];
        let mut removal_penalty = [[0i32; 24]; 24];
        for ai in 0..m {
            let a = unused[ai];
            add_weights[ai] = weights[a] as i64;
            add_contrib[ai] = state.contrib[a] as i64;
            let row = &interactions[a];
            for (ri, &r) in used.iter().enumerate() {
                removal_penalty[ai][ri] = row[r];
            }
            for bi in (ai + 1)..m {
                add_synergy[ai][bi] = row[unused[bi]] as i64;
            }
        }

        let mut best_delta = 0i64;
        let mut best_move: Option<(usize, usize, usize, usize)> = None;
        let slack = state.slack() as i64;
        let mut gains = [0i64; 24];

        for (ri, &r1) in used.iter().enumerate() {
            let w_r1 = weights[r1] as i64;
            let c_r1 = state.contrib[r1] as i64;
            for (rj, &r2) in used.iter().enumerate().skip(ri + 1) {
                let w_r2 = weights[r2] as i64;
                let c_r2 = state.contrib[r2] as i64;
                let removed_syn = interactions[r1][r2] as i64;
                let lost = c_r1 + c_r2 - removed_syn;
                let budget = slack + w_r1 + w_r2;

                for ai in 0..m {
                    gains[ai] = add_contrib[ai]
                        - removal_penalty[ai][ri] as i64
                        - removal_penalty[ai][rj] as i64;
                }

                for ai in 0..m {
                    let w_a1 = add_weights[ai];
                    if w_a1 > budget { continue; }
                    let gain_a1 = gains[ai];

                    for bi in (ai + 1)..m {
                        if w_a1 + add_weights[bi] > budget { continue; }
                        let delta = gain_a1 + gains[bi]
                            + add_synergy[ai][bi]
                            - lost;
                        if delta > best_delta {
                            best_delta = delta;
                            best_move = Some((r1, r2, unused[ai], unused[bi]));
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

    fn local_search_vnd_windowed_certified(state: &mut State, window_k: usize) -> bool {
        for _ in 0..79 {
            let (best_unused, worst_used) = build_windows(state, window_k);
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

            let augmented_unused = augment_synergy_neighbors(state, &best_unused);
            let (compound_unused, compound_used) = screen_compound_candidates(state, &augmented_unused, &worst_used);

            let slack = state.slack();
            if slack >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..compound_unused.len() {
                    let a = compound_unused[ai];
                    let wa = state.ch.weights[a];
                    if wa > slack { continue; }
                    let ca = state.contrib[a] as i64;
                    for bi in (ai+1)..compound_unused.len() {
                        let b = compound_unused[bi];
                        let wb = state.ch.weights[b];
                        if wb > slack || wa + wb > slack { continue; }
                        let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_pair(a, b); continue; }
            }

            if apply_swap_2_2_windowed(state, &compound_unused, &compound_used) { continue; }

            let (chain_unused, chain_used) = build_ejection_windows(state, &compound_unused, &compound_used);
            if apply_chain_move(state, &chain_unused, &chain_used) { continue; }
            if apply_reverse_chain(state, &chain_unused, &chain_used) { continue; }
            if apply_capacity_repair(state, &compound_unused, &compound_used) { continue; }

            {
                let mut bs: Option<(usize, usize, i32)> = None;
                for &rm in &worst_used {
                    let w_rm = state.ch.weights[rm] as i64;
                    for &c in &best_unused {
                        let w_c = state.ch.weights[c] as i64;
                        if w_c >= w_rm { continue; }
                        let dw = (w_rm - w_c) as usize;
                        if dw == 0 || dw > 4 { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }

            if state.slack() > 0 {
                let mut bs: Option<(usize, usize, f64)> = None;
                for &rm in &worst_used {
                    let w_rm = state.ch.weights[rm] as i64;
                    for &c in &best_unused {
                        let w_c = state.ch.weights[c] as i64;
                        if w_c <= w_rm { continue; }
                        let dw = w_c - w_rm;
                        if dw as usize > 4 || state.slack() < dw as u32 { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                        if d > 0 {
                            let r = d as f64 / dw as f64;
                            if bs.map_or(true, |(_, _, br)| r > br) { bs = Some((c, rm, r)); }
                        }
                    }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }

            return true;
        }
        false
    }

    fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
        let _ = local_search_vnd_windowed_certified(state, window_k);
    }

    fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams) {
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

    fn greedy_reconstruct(state: &mut State, strategy: usize) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

        match strategy % 4 {
            0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
            1 => candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a].cmp(&state.ch.weights[b])
                    .then(state.contrib[b].cmp(&state.contrib[a]))
            }),
            2 => {
                let limit = n.min(100);
                let mut keys = vec![0i64; n];
                for &i in &candidates {
                    let syn: i64 = state.ch.interaction_values[i].iter()
                        .take(limit).map(|&v| v as i64).sum();
                    keys[i] = -(syn + state.contrib[i] as i64 / 10);
                }
                candidates.sort_unstable_by_key(|&i| keys[i]);
            },
            _ => candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                -((state.contrib[i] as i64 * 100) / w)
            }),
        }

        for &i in &candidates {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    struct Hparams {
        n_random_starts: usize,
        n_crossover_gen: usize,
        ils_rounds: usize,
        ils_restart_interval: usize,
        perturb_base_frac: usize,
        perturb_max_frac: usize,
        ils_vnd_level: usize,
        bounded_2_2_k: usize,
        n_full_restarts: usize,
        use_hub_pair: bool,
        use_heavy_polish: bool,
        window_k: usize,
        core_half_dp: usize,
    }

    impl Hparams {
        fn defaults() -> Self {
            Self{n_random_starts: 5, n_crossover_gen: 13, ils_rounds: 113,
            ils_restart_interval: 13, perturb_base_frac: 8,
            perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 0,
            n_full_restarts: 4, use_hub_pair: false,
            use_heavy_polish: false, window_k: 213, core_half_dp: 53,}
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
                if let Some(v) = m.get("ils_vnd_level").and_then(|v| v.as_u64()) { p.ils_vnd_level = v as usize; }
                if let Some(v) = m.get("bounded_2_2_k").and_then(|v| v.as_u64()) { p.bounded_2_2_k = v as usize; }
                if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) { p.n_full_restarts = v as usize; }
                if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
                if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) { p.core_half_dp = v as usize; }
            }
            p
        }
    }

    fn vnd_dispatch_certified(state: &mut State, hp: &Hparams) -> bool {
        if hp.window_k < state.ch.num_items {
            local_search_vnd_windowed_certified(state, hp.window_k)
        } else {
            ils_vnd_certified(state, hp)
        }
    }

    fn vnd_dispatch(state: &mut State, hp: &Hparams) {
        let _ = vnd_dispatch_certified(state, hp);
    }

    thread_local! {
        static DETERMINISTIC_SEED_GROUPS: std::cell::RefCell<Option<(Vec<SolState>, Vec<SolState>)>> =
            std::cell::RefCell::new(None);
    }

    fn build_deterministic_seed_groups(challenge: &Challenge, hp: &Hparams) -> (Vec<SolState>, Vec<SolState>) {
        let ch = hp.core_half_dp;
        let mut leading = Vec::with_capacity(3);

        let n_greedy = 3;
        for variant in 0..n_greedy {
            let mut st = State::new_empty(challenge);
            match variant {
                0 => build_greedy_density(&mut st),
                1 => build_greedy_value(&mut st),
                2 => build_greedy_synergy_weight(&mut st),
                _ => build_greedy_hub(&mut st),
            }
            dp_refinement_hp(&mut st, ch);
            if hp.use_heavy_polish { local_search_vnd_heavy(&mut st); }
            else { vnd_dispatch(&mut st, hp); }
            leading.push(st.clone_solution());
        }

        let anchor_limit = if challenge.num_items <= 500 { 3 } else { 1 };
        let mut trailing = Vec::with_capacity(anchor_limit + if hp.use_hub_pair { 4 } else { 0 });
        for anchor in select_interaction_anchors(challenge, anchor_limit) {
            let mut st = State::new_empty(challenge);
            build_anchor_neighborhood_seed(&mut st, anchor);
            dp_refinement_hp(&mut st, ch);
            vnd_dispatch(&mut st, hp);
            trailing.push(st.clone_solution());
        }

        if hp.use_hub_pair {
            for k in 0..4 {
                let mut st = State::new_empty(challenge);
                build_hub_pair_kth(&mut st, k);
                dp_refinement_hp(&mut st, ch);
                vnd_dispatch(&mut st, hp);
                trailing.push(st.clone_solution());
            }
        }

        (leading, trailing)
    }

    fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize) -> (Solution, i64) {
        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..rng_offset * 100 { rng.next_u32(); }
        let ch = hp.core_half_dp;

        let mut population: Vec<SolState> = Vec::with_capacity(16);
        DETERMINISTIC_SEED_GROUPS.with(|cache| {
            let cached = cache.borrow();
            let groups = cached.as_ref().unwrap();
            population.extend(groups.0.iter().cloned());
        });

        for mode in 4..(4 + hp.n_random_starts) {
            let mut st = State::new_empty(challenge);
            let m = if mode < 6 { mode } else { mode - 2 };
            construct_forward_incremental(&mut st, m, &mut rng);
            dp_refinement_hp(&mut st, ch);
            vnd_dispatch(&mut st, hp);
            population.push(st.clone_solution());
        }

        DETERMINISTIC_SEED_GROUPS.with(|cache| {
            let cached = cache.borrow();
            let groups = cached.as_ref().unwrap();
            population.extend(groups.1.iter().cloned());
        });

        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);

        let mut state = State::new_empty(challenge);
        for gen in 0..hp.n_crossover_gen {
            let ranked_len = population.len();
            let child_bits = crossover_frequency(&population, challenge, &mut rng);
            set_state_from_bits(&mut state, &child_bits);
            dp_refinement_hp(&mut state, ch);
            vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());

            if population.len() >= 2 {
                let a = gen % population.len().min(4);
                let b = (gen + 1) % population.len().min(4);
                if a != b {
                    let child_bits = crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                    set_state_from_bits(&mut state, &child_bits);
                    dp_refinement_hp(&mut state, ch);
                    vnd_dispatch(&mut state, hp);
                    population.push(state.clone_solution());
                }
            }

            let mut tied = false;
            for i in 1..ranked_len {
                if population[i - 1].value == population[i].value {
                    tied = true;
                    break;
                }
            }
            if !tied {
                'ties: for i in ranked_len..population.len() {
                    for j in 0..i {
                        if population[i].value == population[j].value {
                            tied = true;
                            break 'ties;
                        }
                    }
                }
            }

            if tied {
                population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
            } else {
                let second_child = if population.len() > ranked_len + 1 {
                    population.pop()
                } else {
                    None
                };
                let first_child = population.pop().unwrap();
                let insert_ranked = |population: &mut Vec<SolState>, child: SolState| {
                    let mut lo = 0usize;
                    let mut hi = population.len();
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if population[mid].value > child.value {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    population.insert(lo, child);
                };
                insert_ranked(&mut population, first_child);
                if let Some(child) = second_child {
                    insert_ranked(&mut population, child);
                }
            }
            population.truncate(8);
        }

        state.restore_solution(&population[0]);
        let mut best_val = state.total_value;
        let mut best_state = state.clone_solution();
        let mut vnd_fixed_point = false;

        let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
        tabu_hashes.push(state.hash);

        let mut stall_count = 0;
        let mut cnt = hp.ils_rounds;

        for round in 0..cnt {
            let snap = state.clone_solution();

            let value_before_dp = state.total_value;
            dp_refinement_hp(&mut state, ch);
            if !vnd_fixed_point || state.total_value != value_before_dp {
                vnd_fixed_point = vnd_dispatch_certified(&mut state, hp);
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_state.bits.clone_from(&state.selected_bit);
                best_state.contrib.clone_from(&state.contrib);
                best_state.value = state.total_value;
                best_state.weight = state.total_weight;
                best_state.hash = state.hash;
                stall_count = 0;
            }

            if state.total_value <= snap.value {
                state.restore_solution(&snap);
                stall_count += 1;

                if hp.ils_restart_interval > 0 && stall_count > 0 && stall_count % hp.ils_restart_interval == 0 {
                    let pi = (stall_count / hp.ils_restart_interval) % population.len();
                    state.restore_solution(&population[pi]);
                }

                let strategy = round % 8;
                let strength = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp);
                greedy_reconstruct(&mut state, strategy);
                vnd_fixed_point = vnd_dispatch_certified(&mut state, hp);

                let h = state.hash;
                if tabu_hashes.contains(&h) {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp);
                    greedy_reconstruct(&mut state, 0);
                    vnd_fixed_point = vnd_dispatch_certified(&mut state, hp);
                }
                let h2 = state.hash;
                if tabu_hashes.len() < 128 { tabu_hashes.push(h2); }
                else { tabu_hashes[round % 128] = h2; }

                if state.total_value > best_val {
                    best_val = state.total_value;
                    best_state.bits.clone_from(&state.selected_bit);
                    best_state.contrib.clone_from(&state.contrib);
                    best_state.value = state.total_value;
                    best_state.weight = state.total_weight;
                    best_state.hash = state.hash;
                    stall_count = 0;
                    cnt+= 3;
                }
            } else {
                stall_count = 0;
                let h = state.hash;
                if tabu_hashes.len() < 128 { tabu_hashes.push(h); }
            }
        }

        let mut final_state = State::new_empty(challenge);
        final_state.restore_solution(&best_state);

        if hp.use_heavy_polish {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_heavy(&mut final_state);
                dp_refinement_hp(&mut final_state, ch);
                if final_state.total_value <= v_before { break; }
            }
        } else {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_windowed(&mut final_state, hp.window_k);
                dp_refinement_hp(&mut final_state, ch);
                if final_state.total_value <= v_before { break; }
            }
        }

        if final_state.total_value > best_val {
            (Solution { items: final_state.selected_items() }, final_state.total_value)
        } else {
            let items = (0..challenge.num_items)
                .filter(|&i| best_state.bits[i])
                .collect();
            (Solution { items }, best_val)
        }
    }

    pub struct Solver;

    impl Solver {
        pub fn solve(
            challenge: &Challenge,
            _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
            hyperparameters: &Option<Map<String, Value>>,
        ) -> Result<Option<Solution>> {
            let hp = Hparams::from_map(hyperparameters);
            let n_restarts = hp.n_full_restarts.max(1);
            FORWARD_DIVIDERS.with(|cache| {
                *cache.borrow_mut() = Some(challenge.weights.iter()
                    .map(|&weight| ExactDivider::new(weight))
                    .collect());
            });
            INTERACTION_NEIGHBOR_CACHE.with(|cache| {
                *cache.borrow_mut() = Some(InteractionNeighborCache {
                    queries: vec![0; challenge.num_items],
                    rows: vec![None; challenge.num_items],
                });
            });
            let deterministic_seed_groups = build_deterministic_seed_groups(challenge, &hp);
            DETERMINISTIC_SEED_GROUPS.with(|cache| {
                *cache.borrow_mut() = Some(deterministic_seed_groups);
            });
            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            for restart in 0..n_restarts {
                let (sol, val) = run_one_instance(challenge, &hp, restart);
                if val > best_quality {
                    best_quality = val;
                    best_sol = Some(sol);
                }
            }
            DETERMINISTIC_SEED_GROUPS.with(|cache| {
                *cache.borrow_mut() = None;
            });

            {
                let p9_k: usize = 35;
                let mut rng_p9 = {
                    let mut r = Rng::from_seed(&challenge.seed);
                    for _ in 0..7919 { r.next_u32(); }
                    r
                };
                for _pass in 0..3usize {
                    let current_items: Vec<usize> = match &best_sol {
                        Some(s) => s.items.clone(),
                        None => break,
                    };
                    let mut st = State::new_empty(challenge);
                    for &i in &current_items { st.add_item(i); }

                    let mut sel_asc: Vec<(usize, i32)> = current_items.iter()
                        .map(|&i| (i, st.contrib[i]))
                        .collect();
                    sel_asc.sort_unstable_by_key(|&(_, c)| c);
                    for &(i, _) in sel_asc.iter().take(p9_k) {
                        st.remove_item(i);
                    }

                    let n_items = challenge.num_items;
                    let mut complement: Vec<usize> = (0..n_items)
                        .filter(|&i| !st.selected_bit[i])
                        .collect();
                    complement.sort_unstable_by_key(|&i| std::cmp::Reverse(st.contrib[i]));
                    let top_len = complement.len().min(2 * p9_k);
                    for i in 0..top_len {
                        let j = i + rng_p9.next_usize(top_len - i);
                        complement.swap(i, j);
                    }
                    let mut added = 0usize;
                    for ci in 0..top_len {
                        if added >= p9_k { break; }
                        let idx = complement[ci];
                        if st.total_weight + challenge.weights[idx] <= challenge.max_weight {
                            st.add_item(idx);
                            added += 1;
                        }
                    }

                    local_search_vnd_windowed(&mut st, hp.window_k);
                    dp_refinement_hp(&mut st, hp.core_half_dp);

                    if st.total_value > best_quality {
                        best_quality = st.total_value;
                        best_sol = Some(Solution { items: st.selected_items() });
                    }
                }
            }

            INTERACTION_NEIGHBOR_CACHE.with(|cache| {
                *cache.borrow_mut() = None;
            });
            FORWARD_DIVIDERS.with(|cache| {
                *cache.borrow_mut() = None;
            });
            Ok(best_sol)
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

    #[allow(dead_code)]
    pub fn help() {
        println!("Okay");
    }
}

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    inner_four::solve_challenge(challenge, save, hp)
}