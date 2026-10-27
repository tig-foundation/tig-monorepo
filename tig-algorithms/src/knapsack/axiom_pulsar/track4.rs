use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

#[allow(dead_code, unused_imports, clippy::all)]
mod inner_four {

    #[inline(always)]
    fn dw(x: i64, w: i64) -> i64 {
        x / w
    }
    use anyhow::Result;
    // use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use tig_challenges::knapsack::*;

    // #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        pub window_k: Option<usize>,
        pub ils_rounds: Option<usize>,
        pub core_half_dp: Option<usize>,
    }

    #[derive(Clone, Copy)]
    struct Rng {
        state: u64,
    }
    impl Rng {
        fn from_seed(seed: &[u8; 32]) -> Self {
            let mut s: u64 = 0x9E3779B97F4A7C15;
            for (i, &b) in seed.iter().enumerate() {
                s ^= (b as u64) << ((i & 7) * 8);
                s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
            }
            if s == 0 {
                s = 1;
            }
            Self { state: s }
        }
        #[inline]
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            self.state = x;
            x
        }
        #[inline]
        fn next_u32(&mut self) -> u32 {
            (self.next_u64() >> 32) as u32
        }
        #[inline]
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        #[inline]
        fn next_usize(&mut self, bound: usize) -> usize {
            if bound == 0 {
                return 0;
            }
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
        positive_weights: bool,
        max_item_weight: u32,
        selected_bit: Vec<bool>,
        contrib: Vec<i32>,
        total_value: i64,
        total_weight: u32,
        hash: u64,
        dp_cache: Vec<i64>,
        choose_cache: Vec<u8>,
        windows: std::cell::RefCell<super::exact_windows::Windows<'a>>,
        ids_buf: Vec<usize>,
        keys_buf: Vec<i64>,
        dens_buf: Vec<usize>,
        target_buf: Vec<u8>,
        dp_rows_buf: Vec<i32>,
        diff_rm_buf: Vec<usize>,
        diff_add_buf: Vec<usize>,
        rm_buf: Vec<(usize, i64)>,
        present_buf: std::cell::RefCell<Vec<bool>>,
        forbid_buf: std::cell::RefCell<Vec<bool>>,
    }

    impl<'a> State<'a> {
        fn new_empty(ch: &'a Challenge) -> Self {
            let n = ch.num_items;
            let mut contrib = vec![0i32; n];
            for i in 0..n {
                contrib[i] = ch.values[i] as i32;
            }
            Self {
                ch,
                positive_weights: ch.weights.iter().all(|&w| w > 0),
                max_item_weight: ch.weights.iter().copied().max().unwrap_or(0),
                selected_bit: vec![false; n],
                contrib,
                total_value: 0,
                total_weight: 0,
                hash: 0,
                dp_cache: Vec::new(),
                windows: std::cell::RefCell::new(super::exact_windows::Windows::new(
                    &ch.weights,
                )),
                choose_cache: Vec::new(),
                ids_buf: Vec::new(),
                keys_buf: Vec::new(),
                dens_buf: Vec::new(),
                target_buf: Vec::new(),
                dp_rows_buf: Vec::new(),
                diff_rm_buf: Vec::new(),
                diff_add_buf: Vec::new(),
                rm_buf: Vec::with_capacity(n),
                present_buf: std::cell::RefCell::new(Vec::new()),
                forbid_buf: std::cell::RefCell::new(Vec::new()),
            }
        }

        #[inline(always)]
        fn slack(&self) -> u32 {
            self.ch.max_weight - self.total_weight
        }

        #[inline(always)]
        fn add_item(&mut self, i: usize) {
            self.total_value += self.contrib[i] as i64;
            self.total_weight += self.ch.weights[i];
            self.hash ^= zobrist_key(i);
            let n = self.ch.num_items;
            let row = unsafe { &self.ch.interaction_values.get_unchecked(i)[..n] };
            for (ck, &rv) in self.contrib[..n].iter_mut().zip(row.iter()) {
                *ck = ck.wrapping_add(rv);
            }
            self.selected_bit[i] = true;
        }

        #[inline(always)]
        fn remove_item(&mut self, j: usize) {
            self.total_value -= self.contrib[j] as i64;
            self.total_weight -= self.ch.weights[j];
            self.hash ^= zobrist_key(j);
            let n = self.ch.num_items;
            let row = unsafe { &self.ch.interaction_values.get_unchecked(j)[..n] };
            for (ck, &rv) in self.contrib[..n].iter_mut().zip(row.iter()) {
                *ck = ck.wrapping_sub(rv);
            }
            self.selected_bit[j] = false;
        }

        #[inline(always)]
        fn replace_item(&mut self, rm: usize, cand: usize) {
            let n = self.ch.num_items;
            let row_rm = unsafe { &self.ch.interaction_values.get_unchecked(rm)[..n] };
            let removed = self.contrib[rm] as i64;
            let added = self.contrib[cand].wrapping_sub(row_rm[cand]) as i64;
            self.total_value -= removed;
            self.total_value += added;
            self.total_weight -= self.ch.weights[rm];
            self.total_weight += self.ch.weights[cand];
            self.hash ^= zobrist_key(rm);
            self.hash ^= zobrist_key(cand);
            let row_add = unsafe { &self.ch.interaction_values.get_unchecked(cand)[..n] };
            for ((ck, &r_rm), &r_add) in self.contrib[..n].iter_mut().zip(row_rm.iter()).zip(row_add.iter()) {
                *ck = ck.wrapping_sub(r_rm).wrapping_add(r_add);
            }
            self.selected_bit[rm] = false;
            self.selected_bit[cand] = true;
        }

        #[inline(always)]
        fn add_pair(&mut self, a: usize, b: usize) {
            let n = self.ch.num_items;
            let row_a = unsafe { &self.ch.interaction_values.get_unchecked(a)[..n] };
            let gain_a = self.contrib[a] as i64;
            let gain_b = self.contrib[b].wrapping_add(row_a[b]) as i64;
            self.total_value += gain_a;
            self.total_value += gain_b;
            self.total_weight += self.ch.weights[a];
            self.total_weight += self.ch.weights[b];
            self.hash ^= zobrist_key(a);
            self.hash ^= zobrist_key(b);
            let row_b = unsafe { &self.ch.interaction_values.get_unchecked(b)[..n] };
            for ((ck, &ra), &rb) in self.contrib[..n].iter_mut().zip(row_a.iter()).zip(row_b.iter()) {
                *ck = ck.wrapping_add(ra).wrapping_add(rb);
            }
            self.selected_bit[a] = true;
            self.selected_bit[b] = true;
        }

        #[inline(always)]
        fn replace_one_with_two(&mut self, rm: usize, a1: usize, a2: usize) {
            let n = self.ch.num_items;
            let row_rm = unsafe { &self.ch.interaction_values.get_unchecked(rm)[..n] };
            let row_a1 = unsafe { &self.ch.interaction_values.get_unchecked(a1)[..n] };
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
            let row_a2 = unsafe { &self.ch.interaction_values.get_unchecked(a2)[..n] };
            for (((ck, &r_rm), &r_a1), &r_a2) in self.contrib[..n]
                .iter_mut()
                .zip(row_rm.iter())
                .zip(row_a1.iter())
                .zip(row_a2.iter())
            {
                *ck = ck.wrapping_sub(r_rm).wrapping_add(r_a1).wrapping_add(r_a2);
            }
            self.selected_bit[rm] = false;
            self.selected_bit[a1] = true;
            self.selected_bit[a2] = true;
        }

        #[inline(always)]
        fn replace_two_with_one(&mut self, r1: usize, r2: usize, add: usize) {
            let n = self.ch.num_items;
            let row_r1 = unsafe { &self.ch.interaction_values.get_unchecked(r1)[..n] };
            let row_r2 = unsafe { &self.ch.interaction_values.get_unchecked(r2)[..n] };
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
            let row_add = unsafe { &self.ch.interaction_values.get_unchecked(add)[..n] };
            for (((ck, &rr1), &rr2), &radd) in self.contrib[..n]
                .iter_mut()
                .zip(row_r1.iter())
                .zip(row_r2.iter())
                .zip(row_add.iter())
            {
                *ck = ck.wrapping_sub(rr1).wrapping_sub(rr2).wrapping_add(radd);
            }
            self.selected_bit[r1] = false;
            self.selected_bit[r2] = false;
            self.selected_bit[add] = true;
        }

        #[inline(always)]
        fn replace_two_with_two(&mut self, r1: usize, r2: usize, a1: usize, a2: usize) {
            let n = self.ch.num_items;
            let row_r1 = unsafe { &self.ch.interaction_values.get_unchecked(r1)[..n] };
            let row_r2 = unsafe { &self.ch.interaction_values.get_unchecked(r2)[..n] };
            let row_a1 = unsafe { &self.ch.interaction_values.get_unchecked(a1)[..n] };
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
            let row_a2 = unsafe { &self.ch.interaction_values.get_unchecked(a2)[..n] };
            for ((((ck, &rr1), &rr2), &ra1), &ra2) in self.contrib[..n]
                .iter_mut()
                .zip(row_r1.iter())
                .zip(row_r2.iter())
                .zip(row_a1.iter())
                .zip(row_a2.iter())
            {
                *ck = ck
                    .wrapping_sub(rr1)
                    .wrapping_sub(rr2)
                    .wrapping_add(ra1)
                    .wrapping_add(ra2);
            }
            self.selected_bit[r1] = false;
            self.selected_bit[r2] = false;
            self.selected_bit[a1] = true;
            self.selected_bit[a2] = true;
        }

        fn apply_items(&mut self, ids: &[usize], add: bool) {
            self.total_value = self
                .total_value
                .wrapping_add(super::exact_updates::apply(
                    &mut self.contrib,
                    &self.ch.interaction_values,
                    ids,
                    add,
                ));
            for &i in ids {
                if add {
                    self.total_weight += self.ch.weights[i];
                } else {
                    self.total_weight -= self.ch.weights[i];
                }
                self.selected_bit[i] = add;
                self.hash ^= zobrist_key(i);
            }
        }
        fn selected_items(&self) -> Vec<usize> {
            if fast4(32) {
                let n = self.ch.num_items;
                let mut out: Vec<usize> = Vec::with_capacity(n + 1);
                unsafe {
                    let op = out.as_mut_ptr();
                    let sp = self.selected_bit.as_ptr();
                    let mut k = 0usize;
                    let mut i = 0usize;
                    macro_rules! one { ($i:expr) => {{ op.add(k).write($i); k += (*sp.add($i)) as usize; }}; }
                    while i + 8 <= n {
                        one!(i); one!(i + 1); one!(i + 2); one!(i + 3);
                        one!(i + 4); one!(i + 5); one!(i + 6); one!(i + 7);
                        i += 8;
                    }
                    while i < n { one!(i); i += 1; }
                    out.set_len(k);
                }
                return out;
            }
            (0..self.ch.num_items)
                .filter(|&i| self.selected_bit[i])
                .collect()
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
            Self {
                divisor,
                reciprocal,
            }
        }

        #[inline(always)]
        fn divide(&self, numerator: u64) -> u64 {
            if self.divisor == 1 {
                return numerator;
            }
            let estimate = ((numerator as u128 * self.reciprocal as u128) >> 64) as u64;
            estimate - (((estimate as u128 * self.divisor as u128) > numerator as u128) as u64)
        }
    }

    thread_local! {
        static FORWARD_DIVIDERS: std::cell::RefCell<Option<Vec<ExactDivider>>> =
            std::cell::RefCell::new(None);
    }

    static FASTM4: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[inline(always)]
    fn fast4(bit: u64) -> bool { FASTM4.load(std::sync::atomic::Ordering::Relaxed) & bit != 0 }

    #[cfg(test)]
    mod wide_mask_tests {
        use super::*;

        #[test]
        fn fastmask_preserves_high_override_bits() {
            let mask = (1u64 << 60) | 63;
            let mut hp = Map::new();
            hp.insert("fastmask".into(), Value::Number(serde_json::Number::from(mask as i64)));
            assert_eq!(Hparams::from_map(&Some(hp)).fastmask, mask);
        }
    }
    thread_local! {
        static RAND_MEMBER4: std::cell::RefCell<Option<SolState>> = std::cell::RefCell::new(None);
    }
    thread_local! { static STATIC_RECONSTRUCT_SYNERGY: std::cell::RefCell<Vec<i64>> = std::cell::RefCell::new(Vec::new()); }

    struct InteractionNeighborCache {
        queries: Vec<u8>,
        rows: Vec<Option<Vec<usize>>>,
    }

    thread_local! {
        static INTERACTION_NEIGHBOR_CACHE: std::cell::RefCell<Option<InteractionNeighborCache>> =
            std::cell::RefCell::new(None);
    }

    thread_local! {
        static EJC_THRESH: std::cell::Cell<u8> = std::cell::Cell::new(1);
    }

    fn touch_interaction_order(
        cache: &mut InteractionNeighborCache,
        ch: &Challenge,
        anchor: usize,
    ) {
        if cache.rows[anchor].is_some() {
            return;
        }
        cache.queries[anchor] = cache.queries[anchor].saturating_add(1);
        let __thr = EJC_THRESH.with(|c| c.get());
        if cache.queries[anchor] >= __thr {
            let row = &ch.interaction_values[anchor];
            let mut order;
            if row.iter().copied().max().unwrap_or(i32::MIN) <= 1000 {
                let mut counts = [0usize; 1001];
                unsafe {
                    let ptr = row.as_ptr();
                    macro_rules! count {
                        ($i:expr) => {{
                            let v = *ptr.add($i);
                            if v > 0 {
                                *counts.get_unchecked_mut(v as usize) += 1;
                            }
                        }};
                    }
                    let mut i = 0;
                    let full = row.len() / 8 * 8;
                    while i < full {
                        count!(i);
                        count!(i + 1);
                        count!(i + 2);
                        count!(i + 3);
                        count!(i + 4);
                        count!(i + 5);
                        count!(i + 6);
                        count!(i + 7);
                        i += 8;
                    }
                    while i < row.len() {
                        count!(i);
                        i += 1;
                    }
                }
                let mut total = 0usize;
                for v in (1..=1000).rev() {
                    let count = counts[v];
                    counts[v] = total;
                    total += count;
                }
                order = vec![0usize; total];
                unsafe {
                    let ptr = row.as_ptr();
                    let out = order.as_mut_ptr();
                    macro_rules! emit {
                        ($i:expr) => {{
                            let id = $i;
                            let v = *ptr.add(id);
                            if v > 0 {
                                let p = counts.get_unchecked_mut(v as usize);
                                *out.add(*p) = id;
                                *p += 1;
                            }
                        }};
                    }
                    let mut i = 0;
                    let full = row.len() / 8 * 8;
                    while i < full {
                        emit!(i);
                        emit!(i + 1);
                        emit!(i + 2);
                        emit!(i + 3);
                        emit!(i + 4);
                        emit!(i + 5);
                        emit!(i + 6);
                        emit!(i + 7);
                        i += 8;
                    }
                    while i < row.len() {
                        emit!(i);
                        i += 1;
                    }
                }
            } else {
                order = (0..ch.num_items).collect();
                order.sort_unstable_by(|&a, &b| row[b].cmp(&row[a]).then(a.cmp(&b)));
            }
            cache.rows[anchor] = Some(order);
        }
    }

    fn collect_cached_synergy_neighbors_buf(
        state: &State,
        anchor: usize,
        present: &[bool],
        buf: &mut [(usize, i32); 4],
    ) -> Option<usize> {
        INTERACTION_NEIGHBOR_CACHE.with(|cell| {
            let mut cached = cell.borrow_mut();
            let cache = cached.as_mut()?;
            touch_interaction_order(cache, state.ch, anchor);
            let order = cache.rows[anchor].as_ref()?;
            let row = &state.ch.interaction_values[anchor];
            let mut count = 0;
            for &item in order {
                let synergy = row[item];
                if synergy <= 0 {
                    break;
                }
                if state.selected_bit[item] || present[item] {
                    continue;
                }
                buf[count] = (item, synergy);
                count += 1;
                if count == 4 {
                    break;
                }
            }
            Some(count)
        })
    }

    fn collect_cached_ejection_supplemental_into(
        state: &State,
        anchors: &[usize],
        need: usize,
        unused: &mut Vec<usize>,
    ) -> bool {
        INTERACTION_NEIGHBOR_CACHE.with(|cell| {
            let mut cached = cell.borrow_mut();
            let cache = match cached.as_mut() {
                Some(c) => c,
                None => return false,
            };
            for &anchor in anchors {
                touch_interaction_order(cache, state.ch, anchor);
            }
            if anchors.iter().any(|&anchor| cache.rows[anchor].is_none()) {
                return false;
            }

            let mut forbidden_cell = state.forbid_buf.borrow_mut();
            if forbidden_cell.len() != state.selected_bit.len() {
                forbidden_cell.clear();
                forbidden_cell.extend_from_slice(&state.selected_bit);
            } else {
                forbidden_cell.copy_from_slice(&state.selected_bit);
            }
            let forbidden: &mut Vec<bool> = &mut forbidden_cell;
            for &anchor in anchors {
                forbidden[anchor] = true;
            }
            let mut positions = [0usize; 32];
            let target_len = unused.len() + need;
            while unused.len() < target_len {
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
                        if forbidden[item] {
                            pos += 1;
                            continue;
                        }
                        break;
                    }
                    positions[anchor_pos] = pos;
                    if pos == order.len() {
                        continue;
                    }
                    let item = order[pos];
                    let synergy = row[item];
                    if best.map_or(true, |(_, best_item, best_synergy)| {
                        synergy > best_synergy || (synergy == best_synergy && item < best_item)
                    }) {
                        best = Some((anchor_pos, item, synergy));
                    }
                }

                if let Some((anchor_pos, item, _)) = best {
                    positions[anchor_pos] += 1;
                    forbidden[item] = true;
                    unused.push(item);
                } else {
                    break;
                }
            }
            true
        })
    }

    fn build_greedy_density(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        state.apply_items(&(0..n).collect::<Vec<_>>(), true);
        if let Some(mut queue) =
            super::exact_density::Density::new(&state.ch.weights, &state.selected_bit)
        {
            while state.total_weight > cap {
                let worst = queue.pop(&state.contrib, 10, false).unwrap();
                state.remove_item(worst);
            }
        } else {
            while state.total_weight > cap {
                let mut worst = 0;
                let mut worst_s = i64::MAX;
                for i in 0..n {
                    unsafe {
                        if *state.selected_bit.get_unchecked(i) {
                            let c = *state.contrib.get_unchecked(i) as i64;
                            let w = (*state.ch.weights.get_unchecked(i) as i64).max(1);
                            let s = dw(c * 1000, w);
                            if s < worst_s {
                                worst_s = s;
                                worst = i;
                            }
                        }
                    }
                }
                state.remove_item(worst);
            }
        }
        for _ in 0..2 {
            let mut by_density: Vec<usize> = (0..n).collect();
            let contrib = &state.contrib;
            let weights = &state.ch.weights;
            const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
            let mut dkey = vec![0i64; n];
            for i in 0..n {
                dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)];
            }
            by_density.sort_unstable_by(|&a, &b| unsafe {
                dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
            });
            let mut target = vec![false; n];
            let mut rem = cap;
            for &i in &by_density {
                if state.ch.weights[i] <= rem {
                    target[i] = true;
                    rem -= state.ch.weights[i];
                }
            }
            let mut to_rm = Vec::new();
            let mut to_add = Vec::new();
            for i in 0..n {
                if state.selected_bit[i] && !target[i] {
                    to_rm.push(i);
                }
                if !state.selected_bit[i] && target[i] {
                    to_add.push(i);
                }
            }
            if to_rm.is_empty() && to_add.is_empty() {
                break;
            }
            state.apply_items(&to_rm, false);
            state.apply_items(&to_add, true);
        }
    }

    fn build_greedy_value(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(state.ch.values[i]));
        let mut to_add = Vec::with_capacity(n);
        let mut w = state.total_weight;
        for &i in &order {
            if w + state.ch.weights[i] <= cap {
                w += state.ch.weights[i];
                to_add.push(i);
            }
        }
        state.apply_items(&to_add, true);
    }

    fn build_greedy_hub(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut hub_scores: Vec<(usize, i64)> = (0..n)
            .map(|i| {
                let s: i64 = state.ch.interaction_values[i]
                    .iter()
                    .map(|&v| v as i64)
                    .sum();
                (i, s)
            })
            .collect();
        hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        let mut to_add = Vec::with_capacity(n);
        let mut w = state.total_weight;
        for &(i, _) in &hub_scores {
            if w + state.ch.weights[i] <= cap {
                w += state.ch.weights[i];
                to_add.push(i);
            }
        }
        state.apply_items(&to_add, true);
    }

    fn build_greedy_synergy_weight(state: &mut State) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut scores: Vec<(usize, i64)> = (0..n)
            .map(|i| {
                let avg_syn: i64 = if n > 1 {
                    state.ch.interaction_values[i]
                        .iter()
                        .map(|&v| v as i64)
                        .sum::<i64>()
                        / (n as i64 - 1)
                } else {
                    0
                };
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (state.ch.values[i] as i64 + avg_syn) * 100 / w)
            })
            .collect();
        scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        let mut to_add = Vec::with_capacity(n);
        let mut w = state.total_weight;
        for &(i, _) in &scores {
            if w + state.ch.weights[i] <= cap {
                w += state.ch.weights[i];
                to_add.push(i);
            }
        }
        state.apply_items(&to_add, true);
    }

    fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
        let n = state.ch.num_items;
        FORWARD_DIVIDERS.with(|cache| {
            let cached = cache.borrow();
            let dividers = cached.as_ref().unwrap();
            loop {
                let slack = state.slack();
                if slack == 0 {
                    break;
                }
                let mut best_i: Option<usize> = None;
                let mut best_s: i64 = i64::MIN;
                let mut second_i: Option<usize> = None;
                let mut second_s: i64 = i64::MIN;
                for i in 0..n {
                    if state.selected_bit[i] {
                        continue;
                    }
                    if state.ch.weights[i] > slack {
                        continue;
                    }
                    let c = state.contrib[i] as i64;
                    if c <= 0 {
                        continue;
                    }
                    let mut s = match mode {
                        2 => c,
                        3 => {
                            dividers[i].divide((c as u64) * 1000) as i64
                                + (state.ch.weights[i] as i64) * 3
                        }
                        _ => dividers[i].divide((c as u64) * 1000) as i64,
                    };
                    if mode >= 4 {
                        let mask = if mode >= 5 { 0x7F } else { 0x1F };
                        s += (rng.next_u32() & mask) as i64;
                    }
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
                let pick = if mode >= 4 && second_i.is_some() {
                    let m = if mode >= 5 { 1 } else { 3 };
                    if (rng.next_u32() & m) == 0 {
                        second_i
                    } else {
                        best_i
                    }
                } else {
                    best_i
                };
                if let Some(i) = pick {
                    state.add_item(i);
                } else {
                    break;
                }
            }
        });
    }

    fn build_anchor_neighborhood_seed(state: &mut State, anchor: usize) {
        if state.ch.weights[anchor] > state.ch.max_weight {
            return;
        }
        state.add_item(anchor);
        let n = state.ch.num_items;

        if n <= u16::MAX as usize {
            let include: Vec<bool> = state.selected_bit.iter().map(|&b| !b).collect();
            if let Some(mut queue) =
                super::exact_density::Density::new(&state.ch.weights, &include)
            {
                while state.slack() > 0 {
                    let Some(item) = queue.pop_affinity(
                        &state.contrib,
                        &state.ch.interaction_values[anchor],
                        state.slack(),
                    ) else {
                        break;
                    };
                    state.add_item(item);
                }
                return;
            }
        }

        loop {
            let slack = state.slack();
            if slack == 0 {
                break;
            }
            let mut best: Option<(usize, i64)> = None;
            for item in 0..n {
                if state.selected_bit[item] || state.ch.weights[item] > slack {
                    continue;
                }
                let marginal = state.contrib[item] as i64;
                if marginal <= 0 {
                    continue;
                }
                let weight = (state.ch.weights[item] as i64).max(1);
                let affinity = (unsafe { *state.ch.interaction_values.get_unchecked(anchor).get_unchecked(item) }  as i64).max(0);
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
        if limit == 0 || challenge.num_items == 0 {
            return Vec::new();
        }
        let n = challenge.num_items;
        let mut candidates: Vec<usize> = (0..n)
            .filter(|&item| challenge.weights[item] <= challenge.max_weight)
            .collect();
        if candidates.is_empty() {
            return candidates;
        }

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
        scored
            .into_iter()
            .take(limit)
            .map(|(item, _)| item)
            .collect()
    }

    fn hub_pairs_sorted(ch: &Challenge) -> Vec<(i32, usize, usize)> {
        let n = ch.num_items;
        let cap = ch.max_weight;
        const MAX_PAIRS: usize = 32;
        let mut top: Vec<(i32, usize, usize)> = Vec::with_capacity(MAX_PAIRS + 1);
        let mut cutoff = i32::MIN;
        for i in 0..n {
            let wi = ch.weights[i];
            if wi >= cap {
                continue;
            }
            let max_wj = cap - wi;
            let row = unsafe { ch.interaction_values.get_unchecked(i) };
            for j in (i + 1)..n {
                let s = unsafe { *row.get_unchecked(j) };
                if s <= cutoff {
                    continue;
                }
                if unsafe { *ch.weights.get_unchecked(j) } <= max_wj {
                    let pos = top.partition_point(|&(old_s, oi, oj)| {
                        old_s > s || (old_s == s && (oi, oj) < (i, j))
                    });
                    if pos < MAX_PAIRS {
                        top.insert(pos, (s, i, j));
                        if top.len() > MAX_PAIRS {
                            top.pop();
                        }
                        if top.len() == MAX_PAIRS {
                            cutoff = top[MAX_PAIRS - 1].0;
                        }
                    }
                }
            }
        }
        top
    }

    fn build_hub_pair_kth(state: &mut State, k: usize, pairs: &[(i32, usize, usize)]) {
        let n = state.ch.num_items;
        let mut used = Vec::new();
        let mut count = 0;
        for &(_, pi, pj) in pairs {
            if used.contains(&pi) || used.contains(&pj) {
                continue;
            }
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
            if slack == 0 {
                break;
            }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = 0;
            for i in 0..n {
                if state.selected_bit[i] {
                    continue;
                }
                if state.ch.weights[i] > slack {
                    continue;
                }
                let c = state.contrib[i] as i64;
                if c <= 0 {
                    continue;
                }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = dw(c * 1000, w);
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i {
                state.add_item(i);
            } else {
                break;
            }
        }
    }

    thread_local! { static VND_PATHS:std::cell::RefCell<super::exact_paths::Paths>=std::cell::RefCell::new(super::exact_paths::Paths::new()); }
    #[derive(Clone, Copy)]
    struct DpMemoKey {
        parameter: usize,
        hash: u64,
        value: i64,
        weight: u32,
    }

    thread_local! {
        static DP_MEMO: std::cell::RefCell<Vec<Option<(DpMemoKey, SolState)>>> =
            std::cell::RefCell::new(Vec::new());
    }

    fn dp_refinement_hp(state: &mut State, core_half: usize) {
        let slot = super::exact_memo::address(&state.selected_bit, core_half);
        let hit = DP_MEMO.with(|cell| {
            let cache = cell.borrow();
            if let Some(Some((key, output))) = cache.get(slot) {
                if key.parameter == core_half
                    && key.hash == state.hash
                    && key.value == state.total_value
                    && key.weight == state.total_weight
                {
                    state.restore_solution(output);
                    return true;
                }
            }
            false
        });
        if hit {
            return;
        }
        let in_key = DpMemoKey {
            parameter: core_half,
            hash: state.hash,
            value: state.total_value,
            weight: state.total_weight,
        };
        dp_refinement_uncached(state, core_half);
        let output = state.clone_solution();
        DP_MEMO.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.is_empty() {
                cache.resize_with(512, || None);
            }
            cache[slot] = Some((in_key, output));
        });
    }

    thread_local! {
        static DP_STEP_MEMO: std::cell::RefCell<Vec<Option<(DpMemoKey, SolState, bool)>>> = std::cell::RefCell::new(Vec::new());
    }

    fn dp_refinement_uncached(state: &mut State, core_half: usize) {
        for _ in 0..3 {
            if !dp_refinement_step_cached(state, core_half) {
                break;
            }
        }
    }

    fn dp_refinement_step_cached(state: &mut State, core_half: usize) -> bool {
        let slot = super::exact_memo::address(&state.selected_bit, core_half);
        let hit = DP_STEP_MEMO.with(|cell| {
            let cache = cell.borrow();
            if let Some(Some((key, output, again))) = cache.get(slot) {
                if key.parameter == core_half
                    && key.hash == state.hash
                    && key.value == state.total_value
                    && key.weight == state.total_weight
                {
                    state.restore_solution(output);
                    return Some(*again);
                }
            }
            None
        });
        if let Some(again) = hit {
            return again;
        }
        let in_key = DpMemoKey {
            parameter: core_half,
            hash: state.hash,
            value: state.total_value,
            weight: state.total_weight,
        };
        let again = dp_refinement_step_uncached(state, core_half);
        let output = state.clone_solution();
        DP_STEP_MEMO.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.is_empty() {
                cache.resize_with(512, || None);
            }
            cache[slot] = Some((in_key, output, again));
        });
        again
    }

    fn dp_refinement_step_uncached(state: &mut State, core_half: usize) -> bool {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut reachability: Vec<u64> = Vec::new();

        let contrib = &state.contrib;
        let weights = &state.ch.weights;

        {
            let buf = &mut state.dens_buf;
            if buf.len() != n {
                buf.clear();
                buf.reserve(n);
                for i in 0..n {
                    buf.push(i);
                }
            } else {
                for (q, v) in buf.iter_mut().enumerate() {
                    *v = q;
                }
            }
        }
        let by_density: &mut Vec<usize> = &mut state.dens_buf;
        const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        if usize::BITS == 64 && n <= u16::MAX as usize {
            super::exact_sort::pack_density(by_density, &contrib[..n], &weights[..n]);
            if state.positive_weights
                && state.max_item_weight <= 10
                && core_half < n
                && state.total_weight == cap
            {
                let certified = super::exact_dp::strict_density_solution(
                    by_density,
                    &state.selected_bit,
                    weights,
                    cap,
                );
                #[cfg(feature = "probe")]
                eprintln!("TIG_DP_CERTIFICATE_T4={}", certified as u8);
                if certified {
                    return false;
                }
            }
            if state.positive_weights && super::exact_sort::compatible() {
                super::exact_sort::capacity_core(
                    by_density,
                    |a, b| ((a as i64) >> 16) > ((b as i64) >> 16),
                    |key| weights[key & 0xffff],
                    cap,
                    core_half,
                    state.max_item_weight,
                );
            } else {
                by_density.sort_unstable_by(|&a, &b| ((b as i64) >> 16).cmp(&((a as i64) >> 16)));
            }
            for item in by_density.iter_mut() {
                *item &= 0xffff;
            }
        } else {
            let mut dkey = vec![0i64; n];
            for i in 0..n {
                dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)];
            }
            by_density.sort_unstable_by(|&a, &b| unsafe {
                dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
            });
        }
        let mut idx_last_inserted = 0usize;
        let mut idx_first_rejected = n;
        let mut rem = cap;
        let mut start_idx = 0usize;
        if fast4(4) {
            let bp = by_density.as_ptr();
            let wp = weights.as_ptr();
            let mut t = 0usize;
            while t + 8 <= n {
                let s8 = unsafe {
                    *wp.add(*bp.add(t)) + *wp.add(*bp.add(t + 1)) + *wp.add(*bp.add(t + 2))
                        + *wp.add(*bp.add(t + 3)) + *wp.add(*bp.add(t + 4)) + *wp.add(*bp.add(t + 5))
                        + *wp.add(*bp.add(t + 6)) + *wp.add(*bp.add(t + 7))
                };
                if s8 >= rem { break; }
                rem -= s8;
                idx_last_inserted = t + 7;
                t += 8;
            }
            start_idx = t;
        }
        for (idx, &i) in by_density.iter().enumerate().skip(start_idx) {
            let w = weights[i];
            if w <= rem {
                rem -= w;
                idx_last_inserted = idx;
                if state.positive_weights && rem == 0 {
                    if idx_first_rejected == n {
                        idx_first_rejected = (idx + 1).min(n);
                    }
                    break;
                }
            } else if idx_first_rejected == n {
                idx_first_rejected = idx;
            }
        }

        let left = idx_first_rejected.saturating_sub(core_half + 1);
        let right = (idx_last_inserted + core_half + 1).min(n);
        let core = &by_density[left..right];

        if state.target_buf.len() != n {
            state.target_buf.clear();
            state.target_buf.resize(n, 0);
        } else {
            state.target_buf.fill(0);
        }
        let target: &mut Vec<u8> = &mut state.target_buf;
        let mut used_locked: u64 = 0;
        for &it in &by_density[..left] {
            used_locked += weights[it] as u64;
            target[it] = 1;
        }
        let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
        let myk = core.len();

        if myk == 0 || rem_cap == 0 {
            return false;
        }

        let mut total_core_weight: usize = 0;
        let mut total_pos_weight: usize = 0;
        let mut all_pos_fit = true;
        for &it in core {
            let wt = weights[it] as usize;
            total_core_weight += wt;
            if contrib[it] > 0 {
                total_pos_weight += wt;
                if total_pos_weight > rem_cap {
                    all_pos_fit = false;
                }
            }
        }

        if all_pos_fit {
            for &it in core {
                if contrib[it] > 0 {
                    target[it] = 1;
                }
            }
        } else {
            let myw = rem_cap.min(total_core_weight);
            let dp_size = myw + 1;
            let mut w_star = if let Some(w) = super::exact_dp::fill(
                core,
                weights,
                contrib,
                myw,
                &mut state.choose_cache,
                &mut state.dp_rows_buf,
            ) {
                w
            } else {
                let choose_size = myk * dp_size;
                if state.dp_cache.len() < dp_size {
                    state.dp_cache.resize(dp_size, i64::MIN / 4);
                }
                if state.choose_cache.len() < choose_size {
                    state.choose_cache.resize(choose_size, 0);
                }
                let init_val = i64::MIN / 4;
                for v in &mut state.dp_cache[..dp_size] {
                    *v = init_val;
                }
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
                    if wt > myw {
                        continue;
                    }
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

                (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0)
            };
            for t in (0..myk).rev() {
                let it = core[t];
                let wt = weights[it] as usize;
                if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                    target[it] = 1;
                    w_star -= wt;
                }
            }
        }

        let mut removed = core::mem::take(&mut state.diff_rm_buf);
        let mut added = core::mem::take(&mut state.diff_add_buf);
        super::exact_updates::differences(
            &state.selected_bit,
            &state.target_buf,
            &mut removed,
            &mut added,
        );
        if removed.is_empty() && added.is_empty() {
            state.diff_rm_buf = removed;
            state.diff_add_buf = added;
            return false;
        }
        let delta = super::exact_updates::value_delta(
            &state.contrib,
            &state.ch.interaction_values,
            &removed,
            &added,
        );
        if delta <= 0 {
            state.diff_rm_buf = removed;
            state.diff_add_buf = added;
            return false;
        }
        state.apply_items(&removed, false);
        state.apply_items(&added, true);
        state.diff_rm_buf = removed;
        state.diff_add_buf = added;

        true
    }

    fn apply_best_add(state: &mut State) -> bool {
        let slack = state.slack();
        if slack == 0 {
            return false;
        }
        let n = state.ch.num_items;
        let mut best_i: Option<usize> = None;
        let mut best_d: i32 = 0;
        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            if state.ch.weights[i] > slack {
                continue;
            }
            let d = state.contrib[i];
            if d > best_d {
                best_d = d;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            state.add_item(i);
            true
        } else {
            false
        }
    }

    fn apply_best_swap_1_1(state: &mut State, selected: &[usize]) -> bool {
        let n = state.ch.num_items;
        let slack = state.slack();
        let mut best_delta: i32 = 0;
        let mut best_cand = 0usize;
        let mut best_rm = 0usize;
        for &rm in selected {
            let w_rm = state.ch.weights[rm];
            let max_w = w_rm + slack;
            let c_rm = state.contrib[rm];
            let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
            for cand in 0..n {
                if state.selected_bit[cand] {
                    continue;
                }
                let wc = state.ch.weights[cand];
                if wc > max_w {
                    continue;
                }
                let delta = state.contrib[cand] - c_rm
                    - unsafe { *row_rm.get_unchecked(cand) };
                if delta > best_delta {
                    best_delta = delta;
                    best_cand = cand;
                    best_rm = rm;
                }
            }
        }
        if best_delta > 0 {
            state.replace_item(best_rm, best_cand);
            true
        } else {
            false
        }
    }

    fn apply_pair_add(state: &mut State) -> bool {
        let slack = state.slack();
        if slack < 2 {
            return false;
        }
        let n = state.ch.num_items;
        let mut unsel = core::mem::take(&mut state.ids_buf);
        unsel.clear();
        for i in 0..n {
            if !state.selected_bit[i] && state.ch.weights[i] <= slack {
                unsel.push(i);
            }
        }
        let m = unsel.len();
        if m < 2 {
            state.ids_buf = unsel;
            return false;
        }

        let mut best_delta: i64 = 0;
        let mut best_a = 0usize;
        let mut best_b = 0usize;
        for ai in 0..m {
            let a = unsel[ai];
            let wa = state.ch.weights[a];
            let slack_rem = slack - wa;
            let ca = state.contrib[a] as i64;
            let row_a = unsafe { state.ch.interaction_values.get_unchecked(a) };
            for bi in (ai + 1)..m {
                let b = unsel[bi];
                if state.ch.weights[b] > slack_rem {
                    continue;
                }
                let delta = ca
                    + state.contrib[b] as i64
                    + unsafe { *row_a.get_unchecked(b) } as i64;
                if delta > best_delta {
                    best_delta = delta;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        state.ids_buf = unsel;
        if best_delta > 0 {
            state.add_pair(best_a, best_b);
            true
        } else {
            false
        }
    }

    fn build_ejection_windows_into(
        state: &State,
        best_unused: &[usize],
        worst_used: &[usize],
        unused: &mut Vec<usize>,
        used: &mut Vec<usize>,
    ) {
        unused.clear();
        unused.extend(best_unused.iter().take(24).copied());
        used.clear();
        used.extend(worst_used.iter().take(24).copied());

        if !unused.is_empty() && unused.len() < 32 {
            let n_anchors = unused.len();
            let mut anchors = [0usize; 32];
            anchors[..n_anchors].copy_from_slice(&unused);
            let need = 32 - n_anchors;
            if !collect_cached_ejection_supplemental_into(state, &anchors[..n_anchors], need, unused) {
                let nn = state.ch.num_items;
                let mut best_syn_all = [0i32; 5000];
                let bsa = &mut best_syn_all[..nn];
                for &anchor in &anchors[..n_anchors] {
                    let row = &state.ch.interaction_values[anchor];
                    for i in 0..nn {
                        let v = unsafe { *row.get_unchecked(i) };
                        let b = unsafe { bsa.get_unchecked_mut(i) };
                        if v > *b {
                            *b = v;
                        }
                    }
                }
                let mut present_cell = state.present_buf.borrow_mut();
                if present_cell.len() != nn {
                    present_cell.clear();
                    present_cell.resize(nn, false);
                }
                let in_unused: &mut Vec<bool> = &mut present_cell;
                for &a in &anchors[..n_anchors] {
                    in_unused[a] = true;
                }
                let mut supplemental: Vec<(usize, i32)> = Vec::with_capacity(nn);
                for i in 0..nn {
                    if state.selected_bit[i] || in_unused[i] {
                        continue;
                    }
                    let best_syn = bsa[i];
                    if best_syn > 0 {
                        supplemental.push((i, best_syn));
                    }
                }
                for &a in &anchors[..n_anchors] {
                    in_unused[a] = false;
                }
                drop(present_cell);
                supplemental.sort_unstable_by_key(|&(_, syn)| std::cmp::Reverse(syn));
                for (i, _) in supplemental.into_iter().take(need) {
                    unused.push(i);
                }
            }
        }
    }

    fn build_ejection_windows(
        state: &State,
        best_unused: &[usize],
        worst_used: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut unused: Vec<usize> = Vec::with_capacity(32);
        let mut used: Vec<usize> = Vec::with_capacity(32);
        build_ejection_windows_into(state, best_unused, worst_used, &mut unused, &mut used);
        (unused, used)
    }

    fn apply_chain_move(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        use super::exact_compound::{best, groups};
        let w = |i: usize| state.ch.weights[i] as i64;
        let c = |i: usize| state.contrib[i] as i64;
        let q = |i: usize, j: usize| (unsafe { *state.ch.interaction_values.get_unchecked(i).get_unchecked(j) })  as i64;
        let adds = groups(unused, 2, w, c, q, true);
        let removes = groups(used, 1, w, c, q, false);
        if let Some((r, a)) = best(&adds, &removes, state.slack() as i64, false, false, q) {
            state.replace_one_with_two(r.ids[0], a.ids[0], a.ids[1]);
            true
        } else {
            false
        }
    }

    fn apply_chain_move_scan(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        if unused.len() < 2 || used.is_empty() {
            return false;
        }

        let mut best_delta = 0i64;
        let mut best_move: Option<(usize, usize, usize)> = None;
        let slack = state.slack() as i64;

        for &rm in used {
            let w_rm = state.ch.weights[rm] as i64;
            let c_rm = state.contrib[rm] as i64;
            let budget = slack + w_rm;

            for (ai, &a1) in unused.iter().enumerate() {
                let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 > budget {
                    continue;
                }
                let gain_a1 = state.contrib[a1] as i64
                    - unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(rm) }  as i64;

                for &a2 in unused.iter().skip(ai + 1) {
                    let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget {
                        continue;
                    }
                    let delta = gain_a1 + state.contrib[a2] as i64
                        - unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(rm) }  as i64
                        + unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }  as i64
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
        } else {
            false
        }
    }

    fn apply_reverse_chain(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        use super::exact_compound::{best, groups};
        let w = |i: usize| state.ch.weights[i] as i64;
        let c = |i: usize| state.contrib[i] as i64;
        let q = |i: usize, j: usize| (unsafe { *state.ch.interaction_values.get_unchecked(i).get_unchecked(j) })  as i64;
        for (p, &r1) in used.iter().enumerate() {
            for &r2 in &used[p + 1..] {
                if c(r2) < q(r1, r2) {
                    return apply_reverse_chain_scan(state, unused, used);
                }
            }
        }
        let adds = groups(unused, 1, w, c, q, true);
        let removes = groups(used, 2, w, c, q, false);
        if let Some((r, a)) = best(&adds, &removes, state.slack() as i64, true, false, q) {
            state.replace_two_with_one(r.ids[0], r.ids[1], a.ids[0]);
            true
        } else {
            false
        }
    }

    fn apply_reverse_chain_scan(state: &mut State, unused: &[usize], used: &[usize]) -> bool {
        if unused.is_empty() || used.len() < 2 {
            return false;
        }

        let mut best_delta = 0i64;
        let mut best_move: Option<(usize, usize, usize)> = None;
        let slack = state.slack() as i64;

        let min_c_used = used
            .iter()
            .map(|&r| state.contrib[r] as i64)
            .min()
            .unwrap_or(i64::MAX);
        for &add in unused {
            let w_add = state.ch.weights[add] as i64;
            let c_add = state.contrib[add] as i64;
            if c_add - min_c_used <= best_delta {
                continue;
            }

            for (ri, &r1) in used.iter().enumerate() {
                let w_r1 = state.ch.weights[r1] as i64;
                let c_r1 = state.contrib[r1] as i64;
                if c_add - c_r1 <= best_delta {
                    continue;
                }
                for &r2 in used.iter().skip(ri + 1) {
                    let w_r2 = state.ch.weights[r2] as i64;
                    if w_add > slack + w_r1 + w_r2 {
                        continue;
                    }
                    let delta = c_add
                        - unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r1) }  as i64
                        - unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r2) }  as i64
                        - c_r1
                        - state.contrib[r2] as i64
                        + unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }  as i64;
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
        } else {
            false
        }
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
                if freed < needs[target_pos] {
                    continue;
                }
                let delta = target_values[target_pos] - current_penalties[target_pos] - loss;
                if delta > 0 && delta > best_deltas[target_pos] {
                    best_deltas[target_pos] = delta;
                    best_memberships[target_pos] = *chosen;
                    best_lengths[target_pos] = chosen_len as u8;
                }
            }
        }
        if pos >= removal_count || chosen_len >= 5 {
            return;
        }

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

    fn capacity_repair_search_pruned(
        removal_count: usize,
        target_count: usize,
        pos: usize,
        mut active_targets: u8,
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
        for target_pos in 0..target_count {
            if active_targets & (1 << target_pos) == 0 {
                continue;
            }
            let delta = target_values[target_pos] - current_penalties[target_pos] - loss;
            if chosen_len != 0 && freed >= needs[target_pos] {
                if delta > 0 && delta > best_deltas[target_pos] {
                    best_deltas[target_pos] = delta;
                    best_memberships[target_pos] = *chosen;
                    best_lengths[target_pos] = chosen_len as u8;
                }
                active_targets &= !(1 << target_pos);
            } else if delta <= best_deltas[target_pos] {
                active_targets &= !(1 << target_pos);
            }
        }
        if active_targets == 0 {
            return;
        }
        if pos >= removal_count || chosen_len >= 5 {
            return;
        }

        for idx in pos..removal_count {
            let mut pair_correction = 0i64;
            for prior_pos in 0..chosen_len {
                pair_correction += removal_pairs[idx][chosen[prior_pos] as usize];
            }
            chosen[chosen_len] = idx as u8;
            for target_pos in 0..target_count {
                current_penalties[target_pos] += target_penalties[target_pos][idx];
            }
            capacity_repair_search_pruned(
                removal_count,
                target_count,
                idx + 1,
                active_targets,
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

    fn capacity_lower_bounds(
        count: usize,
        target_count: usize,
        weights: &[u32; 18],
        values: &[i64; 18],
        pairs: &[[i64; 18]; 18],
        penalties: &[[i64; 18]; 8],
    ) -> Vec<[[i64; 11]; 19]> {
        const INF: i64 = i64::MAX / 4;
        let mut lower = vec![[[INF; 11]; 19]; target_count];
        let mut unavoidable = [0i64; 18];
        for i in 0..count {
            let mut largest = [0i64; 4];
            for j in 0..i {
                let value = pairs[i][j];
                largest[3] = largest[3].max(value.min(largest[2]));
                largest[2] = largest[2].max(value.min(largest[1]));
                largest[1] = largest[1].max(value.min(largest[0]));
                largest[0] = largest[0].max(value);
            }
            let corrections: i64 = largest.iter().sum();
            unavoidable[i] = values[i] - corrections;
        }
        for t in 0..target_count {
            lower[t][count][0] = 0;
            for i in (0..count).rev() {
                lower[t][i][0] = 0;
                let cost = unavoidable[i] + penalties[t][i];
                for need in 1..=10usize {
                    let rest = need.saturating_sub(weights[i] as usize);
                    lower[t][i][need] = lower[t][i + 1][need].min(cost + lower[t][i + 1][rest]);
                }
            }
        }
        lower
    }

    fn capacity_repair_search_bounded(
        removal_count: usize,
        target_count: usize,
        pos: usize,
        mut active_targets: u8,
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
        lower: &[[[i64; 11]; 19]],
        current_penalties: &mut [i64; 8],
        best_deltas: &mut [i64; 8],
        best_memberships: &mut [[u8; 5]; 8],
        best_lengths: &mut [u8; 8],
    ) {
        let mut checking = active_targets;
        while checking != 0 {
            let target_pos = checking.trailing_zeros() as usize;
            checking &= checking - 1;
            let delta = target_values[target_pos] - current_penalties[target_pos] - loss;
            if chosen_len != 0 && freed >= needs[target_pos] {
                if delta > 0 && delta > best_deltas[target_pos] {
                    best_deltas[target_pos] = delta;
                    best_memberships[target_pos] = *chosen;
                    best_lengths[target_pos] = chosen_len as u8;
                }
                active_targets &= !(1 << target_pos);
            } else if delta
                - lower[target_pos][pos][needs[target_pos].saturating_sub(freed) as usize]
                <= best_deltas[target_pos]
            {
                active_targets &= !(1 << target_pos);
            }
        }
        if active_targets == 0 {
            return;
        }
        if pos >= removal_count || chosen_len >= 5 {
            return;
        }

        for idx in pos..removal_count {
            let mut pair_correction = 0i64;
            for prior_pos in 0..chosen_len {
                pair_correction += removal_pairs[idx][chosen[prior_pos] as usize];
            }
            chosen[chosen_len] = idx as u8;
            let mut updating = active_targets;
            while updating != 0 {
                let target_pos = updating.trailing_zeros() as usize;
                updating &= updating - 1;
                current_penalties[target_pos] += target_penalties[target_pos][idx];
            }
            capacity_repair_search_bounded(
                removal_count,
                target_count,
                idx + 1,
                active_targets,
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
                lower,
                current_penalties,
                best_deltas,
                best_memberships,
                best_lengths,
            );
            let mut updating = active_targets;
            while updating != 0 {
                let target_pos = updating.trailing_zeros() as usize;
                updating &= updating - 1;
                current_penalties[target_pos] -= target_penalties[target_pos][idx];
            }
        }
    }

    fn apply_capacity_repair(
        state: &mut State,
        best_unused: &[usize],
        worst_used: &[usize],
    ) -> bool {
        if best_unused.is_empty()
            || worst_used.len() < 2
            || state.slack() >= state.ch.max_weight / 5
        {
            return false;
        }

        let mut targets: Vec<usize> = best_unused
            .iter()
            .copied()
            .filter(|&i| state.ch.weights[i] > state.slack())
            .collect();
        targets.sort_unstable_by_key(|&i| std::cmp::Reverse(state.contrib[i]));
        targets.truncate(8);
        if targets.is_empty() {
            return false;
        }

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
        let monotone = (0..target_count)
            .all(|t| (0..removal_count).all(|i| target_penalties[t][i] >= 0))
            && (0..removal_count).all(|i| {
                let positive_pairs: i64 = (0..removal_count)
                    .filter(|&j| j != i)
                    .map(|j| removal_pairs[i.max(j)][i.min(j)].max(0))
                    .sum();
                removal_values[i] >= positive_pairs
            });
        if monotone && needs[..target_count].iter().all(|&need| need <= 10) {
            let lower = capacity_lower_bounds(
                removal_count,
                target_count,
                &removal_weights,
                &removal_values,
                &removal_pairs,
                &target_penalties,
            );
            capacity_repair_search_bounded(
                removal_count,
                target_count,
                0,
                ((1u16 << target_count) - 1) as u8,
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
                &lower,
                &mut current_penalties,
                &mut best_deltas,
                &mut best_memberships,
                &mut best_lengths,
            );
        } else if monotone {
            capacity_repair_search_pruned(
                removal_count,
                target_count,
                0,
                ((1u16 << target_count) - 1) as u8,
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
        } else {
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
        }

        let mut best_move: Option<(i64, usize)> = None;
        for target_pos in 0..target_count {
            let delta = best_deltas[target_pos];
            if delta > 0
                && best_move
                    .as_ref()
                    .map_or(true, |(best_delta, _)| delta > *best_delta)
            {
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
            for sj in (si + 1)..sel_ranked.len() {
                let r2 = sel_ranked[sj].0;
                let w_r2 = state.ch.weights[r2] as i64;
                let c_r2 = state.contrib[r2] as i64;
                let freed_weight = w_r1 + w_r2;
                let removed_syn = unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }  as i64;
                let lost = c_r1 + c_r2 - removed_syn;
                let budget = state.slack() as i64 + freed_weight;

                for ui in 0..unsel_ranked.len() {
                    let a1 = unsel_ranked[ui].0;
                    let w_a1 = state.ch.weights[a1] as i64;
                    if w_a1 > budget {
                        continue;
                    }
                    let c_a1 = state.contrib[a1] as i64
                        - unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r1) }  as i64
                        - unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r2) }  as i64;
                    for uj in (ui + 1)..unsel_ranked.len() {
                        let a2 = unsel_ranked[uj].0;
                        let w_a2 = state.ch.weights[a2] as i64;
                        if w_a1 + w_a2 > budget {
                            continue;
                        }
                        let c_a2 = state.contrib[a2] as i64
                            - unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r1) }  as i64
                            - unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r2) }  as i64;
                        let added_syn = unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }  as i64;
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
        } else {
            false
        }
    }

    fn local_search_vnd_fast_certified(state: &mut State) -> bool {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..79 {
            if apply_best_add(state) {
                continue;
            }
            selected_buf.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    selected_buf.push(i);
                }
            }
            if apply_best_swap_1_1(state, &selected_buf) {
                continue;
            }
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
            if apply_best_add(state) {
                continue;
            }
            selected_buf.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    selected_buf.push(i);
                }
            }
            if apply_best_swap_1_1(state, &selected_buf) {
                continue;
            }
            if apply_pair_add(state) {
                continue;
            }
            if apply_swap_2_2_bounded(state, k) {
                continue;
            }
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
        let mut best_unused: Vec<usize> = Vec::with_capacity(32);
        let mut worst_used: Vec<usize> = Vec::with_capacity(32);
        let mut chain_unused: Vec<usize> = Vec::with_capacity(32);
        let mut chain_used: Vec<usize> = Vec::with_capacity(32);
        for _ in 0..299 {
            if apply_best_add(state) {
                continue;
            }
            selected_buf.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    selected_buf.push(i);
                }
            }
            if apply_best_swap_1_1(state, &selected_buf) {
                continue;
            }
            if apply_pair_add(state) {
                continue;
            }
            if apply_swap_2_2_bounded(state, 25) {
                continue;
            }
            build_windows_into(state, 32, &mut best_unused, &mut worst_used);
            build_ejection_windows_into(
                state,
                &best_unused,
                &worst_used,
                &mut chain_unused,
                &mut chain_used,
            );
            if apply_chain_move(state, &chain_unused, &chain_used) {
                continue;
            }
            if apply_reverse_chain(state, &chain_unused, &chain_used) {
                continue;
            }
            return true;
        }
        false
    }

    fn local_search_vnd_heavy(state: &mut State) {
        let _ = local_search_vnd_heavy_certified(state);
    }

    fn crossover_frequency(
        population: &[SolState],
        extra: &[SolState],
        ch: &Challenge,
        rng: &mut Rng,
    ) -> Vec<bool> {
        let n = ch.num_items;
        let pop_size = population.len() + extra.len();
        let mut freq = vec![0usize; n];
        for sol in population.iter().chain(extra.iter()) {
            for (f, &b) in freq.iter_mut().zip(&sol.bits[..n]) {
                *f += b as usize;
            }
        }
        let threshold = (pop_size * 3) / 4;
        let mut child_bits = vec![false; n];
        let mut child_weight: u32 = 0;
        let mut consensus: Vec<usize> = Vec::new();
        let mut exploratory: Vec<usize> = Vec::new();
        for i in 0..n {
            if freq[i] > threshold {
                consensus.push(i);
            } else if freq[i] > 0 {
                exploratory.push(i);
            }
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

    fn crossover_uniform(
        sol_a: &SolState,
        sol_b: &SolState,
        ch: &Challenge,
        rng: &mut Rng,
    ) -> Vec<bool> {
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
            if bits[i] {
                continue;
            }
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
        let mut removed = core::mem::take(&mut state.diff_rm_buf);
        let mut added = core::mem::take(&mut state.diff_add_buf);
        removed.clear();
        added.clear();
        if fast4(8) && bits.len() >= n {
            removed.reserve(n + 1);
            added.reserve(n + 1);
            unsafe {
                let rp = removed.as_mut_ptr();
                let ap = added.as_mut_ptr();
                let sp = state.selected_bit.as_ptr();
                let bp = bits.as_ptr();
                let (mut kr, mut ka) = (0usize, 0usize);
                macro_rules! rm { ($i:expr) => {{ let i = $i; rp.add(kr).write(i); kr += (*sp.add(i) & !*bp.add(i)) as usize; }}; }
                macro_rules! ad { ($i:expr) => {{ let i = $i; ap.add(ka).write(i); ka += (*bp.add(i) & !*sp.add(i)) as usize; }}; }
                let mut j = n;
                while j >= 8 {
                    rm!(j - 1); rm!(j - 2); rm!(j - 3); rm!(j - 4);
                    rm!(j - 5); rm!(j - 6); rm!(j - 7); rm!(j - 8);
                    j -= 8;
                }
                while j > 0 { rm!(j - 1); j -= 1; }
                let mut i = 0usize;
                while i + 8 <= n {
                    ad!(i); ad!(i + 1); ad!(i + 2); ad!(i + 3);
                    ad!(i + 4); ad!(i + 5); ad!(i + 6); ad!(i + 7);
                    i += 8;
                }
                while i < n { ad!(i); i += 1; }
                removed.set_len(kr);
                added.set_len(ka);
            }
            state.apply_items(&removed, false);
            state.apply_items(&added, true);
            state.diff_rm_buf = removed;
            state.diff_add_buf = added;
            return;
        }
        removed.extend((0..n).rev().filter(|&i| state.selected_bit[i] && !bits[i]));
        added.extend((0..n).filter(|&i| bits[i] && !state.selected_bit[i]));
        state.apply_items(&removed, false);
        state.apply_items(&added, true);
        state.diff_rm_buf = removed;
        state.diff_add_buf = added;
    }

    fn build_windows_into(
        state: &State,
        k: usize,
        best_unused: &mut Vec<usize>,
        worst_used: &mut Vec<usize>,
    ) {
        best_unused.clear();
        worst_used.clear();
        let mut cache = state.windows.borrow_mut();
        if cache.is_packed() {
            cache.build(
                &state.contrib,
                &state.selected_bit,
                k,
                best_unused,
                worst_used,
            );
            return;
        }
        let n = state.ch.num_items;
        const LDIV4: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        let mut unused_r: Vec<(usize, i64)> = Vec::with_capacity(n);
        let mut used_r: Vec<(usize, i64)> = Vec::with_capacity(n);
        for i in 0..n {
            let (c, w, sel) = unsafe {
                (
                    *state.contrib.get_unchecked(i),
                    *state.ch.weights.get_unchecked(i),
                    *state.selected_bit.get_unchecked(i),
                )
            };
            let r = c as i64 * unsafe { *LDIV4.get_unchecked((w as usize).min(10)) };
            if sel {
                used_r.push((i, r));
            } else {
                unused_r.push((i, r));
            }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.cmp(&a.1));
        }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.cmp(&b.1));
        }
        best_unused.extend(unused_r[..ku].iter().map(|x| x.0));
        worst_used.extend(used_r[..ks].iter().map(|x| x.0));
    }

    fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
        let mut unused = Vec::with_capacity(k.min(state.ch.num_items));
        let mut used = Vec::with_capacity(k.min(state.ch.num_items));
        build_windows_into(state, k, &mut unused, &mut used);
        (unused, used)
    }

    fn augment_synergy_neighbors_into(
        state: &State,
        best_unused: &[usize],
        augmented: &mut Vec<usize>,
    ) {
        augmented.clear();
        augmented.extend_from_slice(best_unused);
        super::exact_sort::density_order(
            augmented,
            &state.contrib,
            &state.ch.weights,
            state.positive_weights && state.max_item_weight <= 10,
        );
        augmented.truncate(32);

        let mut present_cell = state.present_buf.borrow_mut();
        if present_cell.len() != state.ch.num_items {
            present_cell.clear();
            present_cell.resize(state.ch.num_items, false);
        }
        let present: &mut Vec<bool> = &mut present_cell;
        for &item in augmented.iter() {
            present[item] = true;
        }
        let n_anchors = augmented.len().min(8);
        for a_idx in 0..n_anchors {
            if augmented.len() >= 56 {
                break;
            }
            let anchor = augmented[a_idx];
            let mut buf = [(0usize, 0i32); 4];
            let count = if let Some(c) =
                collect_cached_synergy_neighbors_buf(state, anchor, present, &mut buf)
            {
                c
            } else {
                let row = &state.ch.interaction_values[anchor];
                let mut c = 0;
                for item in 0..state.ch.num_items {
                    if state.selected_bit[item] || present[item] {
                        continue;
                    }
                    let synergy = row[item];
                    if synergy <= 0 {
                        continue;
                    }
                    if c < 4 {
                        buf[c] = (item, synergy);
                        c += 1;
                        buf[..c].sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
                    } else if synergy > buf[3].1 {
                        buf[3] = (item, synergy);
                        buf.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
                    }
                }
                c
            };
            for &(item, _) in &buf[..count] {
                if augmented.len() >= 56 {
                    break;
                }
                present[item] = true;
                augmented.push(item);
            }
        }
        for &item in augmented.iter() {
            present[item] = false;
        }
    }

    fn screen_compound_candidates_into(
        state: &State,
        best_unused: &[usize],
        worst_used: &[usize],
        compound_unused: &mut Vec<usize>,
        compound_used: &mut Vec<usize>,
        scored: &mut Vec<(usize, i64)>,
    ) {
        compound_unused.clear();
        compound_unused.extend_from_slice(best_unused);
        super::exact_sort::density_order(
            compound_unused,
            &state.contrib,
            &state.ch.weights,
            state.positive_weights && state.max_item_weight <= 10,
        );
        compound_unused.truncate(48);

        compound_used.clear();
        compound_used.extend(worst_used.iter().take(24).copied());

        scored.clear();
        for &i in compound_unused.iter() {
            let three = unsafe {
                super::exact_density::top_three_sum(
                    &state.ch.interaction_values[i],
                    compound_unused,
                    i,
                )
            };
            let removal_penalty: i64 = compound_used.iter()
                .map(|&r| (unsafe { *state.ch.interaction_values.get_unchecked(i).get_unchecked(r) }  as i64).max(0))
                .sum();
            scored.push((i, state.contrib[i] as i64 + three - removal_penalty / 2));
        }
        scored.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
        scored.truncate(40);

        compound_unused.clear();
        for &(i, _) in scored.iter() {
            compound_unused.push(i);
        }
    }

    fn apply_swap_2_2_windowed(
        state: &mut State,
        best_unused: &[usize],
        worst_used: &[usize],
    ) -> bool {
        let unused = &best_unused[..best_unused.len().min(24)];
        let used = &worst_used[..worst_used.len().min(24)];
        let weights = &state.ch.weights;
        let contrib = &state.contrib;
        let interaction = |a: usize, b: usize| (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) })  as i64;
        let slack = state.slack() as i64;
        use super::exact_pairs::{best_exchange, Pair};
        if unused.len() < 2 || used.len() < 2 {
            return false;
        }
        let mut additions = Vec::with_capacity(unused.len() * (unused.len() - 1) / 2);
        let mut removals = Vec::with_capacity(used.len() * (used.len() - 1) / 2);
        for (i, &a) in unused.iter().enumerate() {
            for &b in unused.iter().skip(i + 1) {
                additions.push(Pair {
                    a,
                    b,
                    weight: weights[a] as i64 + weights[b] as i64,
                    value: contrib[a] as i64 + contrib[b] as i64 + interaction(a, b),
                    ordinal: additions.len(),
                });
            }
        }
        for (i, &a) in used.iter().enumerate() {
            for &b in used.iter().skip(i + 1) {
                removals.push(Pair {
                    a,
                    b,
                    weight: weights[a] as i64 + weights[b] as i64,
                    value: contrib[a] as i64 + contrib[b] as i64 - interaction(a, b),
                    ordinal: removals.len(),
                });
            }
        }
        let movement = best_exchange(&mut additions, &mut removals, slack, interaction);
        if let Some((r, a)) = movement {
            state.replace_two_with_two(r.a, r.b, a.a, a.b);
            true
        } else {
            false
        }
    }

    fn local_search_vnd_windowed_certified(state: &mut State, window_k: usize) -> bool {
        let mut exact_pending = Vec::new();
        let mut best_unused: Vec<usize> = Vec::with_capacity(window_k);
        let mut worst_used: Vec<usize> = Vec::with_capacity(window_k);
        let mut augmented_unused: Vec<usize> = Vec::with_capacity(64);
        let mut compound_unused: Vec<usize> = Vec::with_capacity(48);
        let mut compound_used: Vec<usize> = Vec::with_capacity(24);
        let mut scored_buf: Vec<(usize, i64)> = Vec::with_capacity(48);
        let mut chain_unused: Vec<usize> = Vec::with_capacity(32);
        let mut chain_used: Vec<usize> = Vec::with_capacity(32);
        for exact_step in 0..79 {
            let exact_hash = state.hash;
            let exact_hit = VND_PATHS.with(|c| {
                c.borrow().lookup(
                    window_k,
                    exact_hash,
                    &state.selected_bit,
                    &state.contrib,
                    state.total_value,
                    state.total_weight,
                    79 - exact_step,
                )
            });
            if let Some((output, steps)) = exact_hit {
                state.selected_bit.clone_from(&output.bits);
                state.contrib.clone_from(&output.contrib);
                state.total_value = output.value;
                state.total_weight = output.weight;
                state.hash = output.hash;
                VND_PATHS.with(|c| {
                    c.borrow_mut()
                        .publish(window_k, exact_pending, exact_step + steps, output)
                });
                return true;
            }

            build_windows_into(state, window_k, &mut best_unused, &mut worst_used);
            let slack = state.slack();
            if slack > 0 {
                let mut best_c = usize::MAX;
                let mut best_d = 0i32;
                for &c in &best_unused {
                    if state.ch.weights[c] > slack {
                        continue;
                    }
                    let d = state.contrib[c];
                    if d > best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                if best_c != usize::MAX {
                    state.add_item(best_c);
                    continue;
                }
            }

            {
                let movement = super::exact_single::best_fused(
                    &best_unused,
                    &worst_used,
                    &state.ch.weights,
                    &state.contrib,
                    state.slack(),
                    |r, a| (unsafe { *state.ch.interaction_values.get_unchecked(r).get_unchecked(a) }) ,
                );
                if let Some((c, rm)) = movement {
                    state.replace_item(rm, c);
                    continue;
                }
            }

            exact_pending.push((
                super::exact_paths::Snapshot::new(
                    &state.selected_bit,
                    &state.contrib,
                    state.total_value,
                    state.total_weight,
                    exact_hash,
                ),
                exact_step,
            ));
            augment_synergy_neighbors_into(state, &best_unused, &mut augmented_unused);
            screen_compound_candidates_into(
                state,
                &augmented_unused,
                &worst_used,
                &mut compound_unused,
                &mut compound_used,
                &mut scored_buf,
            );

            let slack = state.slack();
            if slack >= 2 {
                let mut best_d = 0i64;
                let mut best_a = 0usize;
                let mut best_b = 0usize;
                for ai in 0..compound_unused.len() {
                    let a = compound_unused[ai];
                    let wa = state.ch.weights[a];
                    if wa > slack {
                        continue;
                    }
                    let slack_rem = slack - wa;
                    let ca = state.contrib[a] as i64;
                    let row_a = unsafe { state.ch.interaction_values.get_unchecked(a) };
                    for bi in (ai + 1)..compound_unused.len() {
                        let b = compound_unused[bi];
                        if state.ch.weights[b] > slack_rem {
                            continue;
                        }
                        let d = ca
                            + state.contrib[b] as i64
                            + unsafe { *row_a.get_unchecked(b) } as i64;
                        if d > best_d {
                            best_d = d;
                            best_a = a;
                            best_b = b;
                        }
                    }
                }
                if best_d > 0 {
                    state.add_pair(best_a, best_b);
                    continue;
                }
            }

            if apply_swap_2_2_windowed(state, &compound_unused, &compound_used) {
                continue;
            }

            build_ejection_windows_into(
                state,
                &compound_unused,
                &compound_used,
                &mut chain_unused,
                &mut chain_used,
            );
            if apply_chain_move(state, &chain_unused, &chain_used) {
                continue;
            }
            if apply_reverse_chain(state, &chain_unused, &chain_used) {
                continue;
            }
            if apply_capacity_repair(state, &compound_unused, &compound_used) {
                continue;
            }

            let output = std::rc::Rc::new(super::exact_paths::Snapshot::new(
                &state.selected_bit,
                &state.contrib,
                state.total_value,
                state.total_weight,
                state.hash,
            ));
            VND_PATHS.with(|c| {
                c.borrow_mut()
                    .publish(window_k, exact_pending, exact_step + 1, output)
            });
            return true;
        }
        false
    }

    fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
        let _ = local_search_vnd_windowed_certified(state, window_k);
    }

    fn perturb_by_strategy(
        state: &mut State,
        strength: usize,
        stall_count: usize,
        strategy: usize,
        rng: &mut Rng,
        hp: &Hparams,
    ) {
        let mut removal_candidates = core::mem::take(&mut state.rm_buf);
        removal_candidates.clear();
        let n = state.ch.num_items;
        for i in 0..n {
            if state.selected_bit[i] {
                let score = match strategy {
                    0 => state.contrib[i] as i64,
                    1 => -(state.ch.weights[i] as i64),
                    2 => state.contrib[i] as i64 - state.ch.values[i] as i64,
                    3 => {
                        let w = (state.ch.weights[i] as i64).max(1);
                        dw(state.contrib[i] as i64 * 1000, w)
                    }
                    4 => {
                        let w = (state.ch.weights[i] as i64).max(1);
                        let density = dw(state.contrib[i] as i64 * 100, w);
                        state.ch.weights[i] as i64 - density
                    }
                    5 => {
                        let w = (state.ch.weights[i] as i64).max(1);
                        (state.contrib[i] as i64 * 10000) / (w * w)
                    }
                    6 => rng.next_u32() as i64,
                    _ => -(state.contrib[i] as i64),
                };
                removal_candidates.push((i, score));
            }
        }
        let sel_len = removal_candidates.len();
        if sel_len == 0 {
            state.rm_buf = removal_candidates;
            return;
        }

        let base_remove = (sel_len / hp.perturb_base_frac).max(2);
        let adaptive_mult = 1 + (stall_count / 2);
        let n_remove = (base_remove * adaptive_mult)
            .min(strength)
            .min(sel_len * 2 / hp.perturb_max_frac);
        removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        let mut removed = core::mem::take(&mut state.diff_rm_buf);
        removed.clear();
        for &(i, _) in removal_candidates.iter().take(n_remove) {
            removed.push(i);
        }
        state.apply_items(&removed, false);
        state.diff_rm_buf = removed;
        state.rm_buf = removal_candidates;
    }

    fn greedy_reconstruct(state: &mut State, strategy: usize) {
        if !state.positive_weights || !super::exact_sort::compatible() {
            return greedy_reconstruct_scan(state, strategy);
        }
        let n = state.ch.num_items;
        let mode = strategy % 4;
        state.ids_buf.clear();
        state.ids_buf.reserve(n);
        if fast4(2) {
            let out = &mut state.ids_buf;
            let sel = &state.selected_bit;
            unsafe {
                let op = out.as_mut_ptr();
                let sp = sel.as_ptr();
                let mut k = 0usize;
                let mut i = 0usize;
                macro_rules! one { ($i:expr) => {{ op.add(k).write($i); k += (!*sp.add($i)) as usize; }}; }
                while i + 8 <= n {
                    one!(i); one!(i + 1); one!(i + 2); one!(i + 3);
                    one!(i + 4); one!(i + 5); one!(i + 6); one!(i + 7);
                    i += 8;
                }
                while i < n { one!(i); i += 1; }
                out.set_len(k);
            }
        } else {
            let out = &mut state.ids_buf;
            let sel = &state.selected_bit;
            for i in 0..n {
                if !unsafe { *sel.get_unchecked(i) } {
                    out.push(i);
                }
            }
        }
        if state.keys_buf.len() < n {
            state.keys_buf.resize(n, 0);
        }
        {
            let ids = &state.ids_buf;
            let keys = &mut state.keys_buf;
            let contrib = &state.contrib;
            let weights = &state.ch.weights;
            let k4 = fast4(16);
            macro_rules! kl { ($e:expr) => {{
                let cn = ids.len();
                let mut t = 0usize;
                if k4 {
                    let ip = ids.as_ptr();
                    let kp = keys.as_mut_ptr();
                    while t + 4 <= cn {
                        unsafe {
                            let i0 = *ip.add(t); let i1 = *ip.add(t + 1); let i2 = *ip.add(t + 2); let i3 = *ip.add(t + 3);
                            *kp.add(i0) = $e(i0); *kp.add(i1) = $e(i1); *kp.add(i2) = $e(i2); *kp.add(i3) = $e(i3);
                        }
                        t += 4;
                    }
                }
                for &i in ids[t..].iter() { keys[i] = $e(i); }
            }}; }
            match mode {
                0 => {
                    kl!(|i: usize| (unsafe { *contrib.get_unchecked(i) }).wrapping_neg() as i64);
                }
                1 => {
                    kl!(|i: usize| -((unsafe { *contrib.get_unchecked(i) }) as i64));
                }
                2 => STATIC_RECONSTRUCT_SYNERGY.with(|c| {
                    let syn = c.borrow();
                    for &i in ids.iter() {
                        keys[i] = -(syn[i] + contrib[i] as i64 / 10);
                    }
                }),
                _ => {
                    kl!(|i: usize| {
                        let c = (unsafe { *contrib.get_unchecked(i) }) as i64;
                        let w = ((unsafe { *weights.get_unchecked(i) }) as i64).max(1);
                        -dw(c * 100, w)
                    });
                }
            }
        }
        let end = super::exact_sort::keyed_prefix(
            &mut state.ids_buf,
            &state.keys_buf,
            &state.ch.weights,
            state.ch.max_weight - state.total_weight,
            mode == 1,
        );
        let mut accepted = 0usize;
        let mut weight = state.total_weight;
        {
            let cap = state.ch.max_weight;
            let weights = &state.ch.weights;
            let items = &mut state.ids_buf;
            for p in 0..end {
                let i = items[p];
                if weight + weights[i] <= cap {
                    weight += weights[i];
                    items[accepted] = i;
                    accepted += 1;
                }
            }
        }
        let mut ids = core::mem::take(&mut state.ids_buf);
        state.apply_items(&ids[..accepted], true);
        state.ids_buf = ids;
    }

    fn greedy_reconstruct_scan(state: &mut State, strategy: usize) {
        let n = state.ch.num_items;
        let cap = state.ch.max_weight;
        let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

        match strategy % 4 {
            0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
            1 => candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a]
                    .cmp(&state.ch.weights[b])
                    .then(state.contrib[b].cmp(&state.contrib[a]))
            }),
            2 => {
                let limit = n.min(100);
                let mut keys = vec![0i64; n];
                for &i in &candidates {
                    let syn: i64 = state.ch.interaction_values[i]
                        .iter()
                        .take(limit)
                        .map(|&v| v as i64)
                        .sum();
                    keys[i] = -(syn + state.contrib[i] as i64 / 10);
                }
                candidates.sort_unstable_by_key(|&i| keys[i]);
            }
            _ => {
                let mut keys = vec![0i64; n];
                for &i in &candidates {
                    let w = (state.ch.weights[i] as i64).max(1);
                    keys[i] = -dw(state.contrib[i] as i64 * 100, w);
                }
                candidates.sort_unstable_by_key(|&i| keys[i]);
            }
        }

        let mut to_add = Vec::with_capacity(candidates.len());
        let mut w = state.total_weight;
        for &i in &candidates {
            if w + state.ch.weights[i] <= cap {
                w += state.ch.weights[i];
                to_add.push(i);
            }
        }
        state.apply_items(&to_add, true);
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
        hub_pair_n: usize,
        use_heavy_polish: bool,
        window_k: usize,
        pub ejc_thresh: usize,
        core_half_dp: usize,
        rush_mode: usize,
        elite_chain: usize,
        elite_k: usize,
        rng_stream: u64,
        fastmask: u64,
        fused_dual: usize,
        t42_dedup: usize,
    }

    impl Hparams {
        fn defaults() -> Self {
            Self {
                n_random_starts: 5,
                n_crossover_gen: 13,
                ils_rounds: 105,
                ils_restart_interval: 13,
                perturb_base_frac: 8,
                perturb_max_frac: 5,
                ils_vnd_level: 0,
                bounded_2_2_k: 0,
                n_full_restarts: 32,
                use_hub_pair: false,
                hub_pair_n: 4,
                use_heavy_polish: false,
                window_k: 213,
                core_half_dp: 53,
                ejc_thresh: 1,
                rush_mode: 0,
                elite_chain: 2,
                elite_k: 2,
                rng_stream: 0,
                fastmask: 63,
                fused_dual: 0,
                t42_dedup: 0,
            }
        }

        fn from_map(h: &Option<Map<String, Value>>) -> Self {
            let mut p = Self::defaults();
            if let Some(m) = h {
                if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64()) {
                    p.n_random_starts = v as usize;
                }
                if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64()) {
                    p.n_crossover_gen = v as usize;
                }
                if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) {
                    p.ils_rounds = v as usize;
                }
                if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) {
                    p.ils_restart_interval = v as usize;
                }
                if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()) {
                    p.perturb_base_frac = v as usize;
                }
                if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) {
                    p.perturb_max_frac = v as usize;
                }
                if let Some(v) = m.get("ils_vnd_level").and_then(|v| v.as_u64()) {
                    p.ils_vnd_level = v as usize;
                }
                if let Some(v) = m.get("bounded_2_2_k").and_then(|v| v.as_u64()) {
                    p.bounded_2_2_k = v as usize;
                }
                if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) {
                    p.n_full_restarts = v as usize;
                }
                if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) {
                    p.window_k = v as usize;
                }
                if let Some(v) = m.get("ejc_thresh").and_then(|v| v.as_u64()) {
                    p.ejc_thresh = v as usize;
                }
                if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) {
                    p.core_half_dp = v as usize;
                }
                if let Some(v) = m.get("rush_mode").and_then(|v| v.as_u64()) {
                    p.rush_mode = v as usize;
                }
                if let Some(v) = m.get("elite_chain").and_then(|v| v.as_u64()) {
                    p.elite_chain = v as usize;
                }
                if let Some(v) = m.get("elite_k").and_then(|v| v.as_u64()) {
                    p.elite_k = v as usize;
                }
                if let Some(v) = m.get("rng_stream").and_then(|v| v.as_u64()) {
                    p.rng_stream = v;
                }
                if let Some(v) = m.get("fastmask").and_then(|v| v.as_u64()) {
                    p.fastmask = v;
                }
                if let Some(v) = m.get("use_hub_pair").and_then(|v| v.as_u64()) {
                    p.use_hub_pair = v != 0;
                    p.hub_pair_n = if v != 0 { 4 } else { 0 };
                }
                if let Some(v) = m.get("hub_pair_n").and_then(|v| v.as_u64()) {
                    p.hub_pair_n = v as usize;
                }
                if let Some(v) = m.get("use_heavy_polish").and_then(|v| v.as_u64()) {
                    p.use_heavy_polish = v != 0;
                }
                if let Some(v) = m.get("fused_dual").and_then(|v| v.as_u64()) {
                    p.fused_dual = v as usize;
                }
                if let Some(v) = m.get("t42_dedup").and_then(|v| v.as_u64()) {
                    p.t42_dedup = v as usize;
                }
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

    fn build_deterministic_seed_groups(
        challenge: &Challenge,
        hp: &Hparams,
    ) -> (Vec<SolState>, Vec<SolState>) {
        let ch = hp.core_half_dp;
        let n = challenge.num_items;
        let mut leading = Vec::with_capacity(3);
        let mut st = State::new_empty(challenge);
        let reset_state = |st: &mut State| {
            st.selected_bit.fill(false);
            for i in 0..n {
                st.contrib[i] = challenge.values[i] as i32;
            }
            st.total_value = 0;
            st.total_weight = 0;
            st.hash = 0;
        };

        let n_greedy = 3;
        for variant in 0..n_greedy {
            if variant > 0 {
                reset_state(&mut st);
            }
            match variant {
                0 => build_greedy_density(&mut st),
                1 => build_greedy_value(&mut st),
                2 => build_greedy_synergy_weight(&mut st),
                _ => build_greedy_hub(&mut st),
            }
            dp_refinement_hp(&mut st, ch);
            if hp.use_heavy_polish {
                local_search_vnd_heavy(&mut st);
            } else {
                vnd_dispatch(&mut st, hp);
            }
            leading.push(st.clone_solution());
        }

        let anchor_limit = if challenge.num_items <= 500 { 3 } else { 1 };
        let mut trailing = Vec::with_capacity(anchor_limit + hp.hub_pair_n);
        for anchor in select_interaction_anchors(challenge, anchor_limit) {
            reset_state(&mut st);
            build_anchor_neighborhood_seed(&mut st, anchor);
            dp_refinement_hp(&mut st, ch);
            vnd_dispatch(&mut st, hp);
            trailing.push(st.clone_solution());
        }

        if hp.hub_pair_n > 0 {
            let pairs = hub_pairs_sorted(challenge);
            for k in 0..hp.hub_pair_n {
                reset_state(&mut st);
                build_hub_pair_kth(&mut st, k, &pairs);
                dp_refinement_hp(&mut st, ch);
                vnd_dispatch(&mut st, hp);
                trailing.push(st.clone_solution());
            }
        }

        (leading, trailing)
    }

    fn stream_seed(challenge: &Challenge, stream: u64) -> [u8; 32] {
        let mut sd = challenge.seed;
        for k in 0..8 {
            sd[k] ^= ((stream >> (8 * k)) & 0xff) as u8;
        }
        sd
    }

    fn run_one_instance(
        challenge: &Challenge,
        hp: &Hparams,
        rng_offset: usize,
        elites: &[SolState],
    ) -> (Solution, i64, SolState) {
        let mut rng = Rng::from_seed(&stream_seed(challenge, hp.rng_stream));
        for _ in 0..rng_offset * 100 {
            rng.next_u32();
        }
        let ch = hp.core_half_dp;
        let n = challenge.num_items;

        let mut population: Vec<SolState> = Vec::with_capacity(16);
        DETERMINISTIC_SEED_GROUPS.with(|cache| {
            let cached = cache.borrow();
            let groups = cached.as_ref().unwrap();
            population.extend(groups.0.iter().cloned());
        });

        let mut state = State::new_empty(challenge);
        let ctor_is_noop = challenge.values.iter().all(|&v| v == 0);
        let mut rand_member: Option<SolState> = None;
        let g1 = fast4(1) && ctor_is_noop;
        for mode in 4..(4 + hp.n_random_starts) {
            if ctor_is_noop {
                if let Some(m0) = rand_member.as_ref() {
                    population.push(m0.clone());
                    continue;
                }
            }
            if g1 && mode == 4 {
                if let Some(m0) = RAND_MEMBER4.with(|c| c.borrow().clone()) {
                    population.push(m0.clone());
                    rand_member = Some(m0);
                    continue;
                }
            }
            state.selected_bit.fill(false);
            for i in 0..n {
                state.contrib[i] = challenge.values[i] as i32;
            }
            state.total_value = 0;
            state.total_weight = 0;
            state.hash = 0;
            let m = if mode < 6 { mode } else { mode - 2 };
            construct_forward_incremental(&mut state, m, &mut rng);
            dp_refinement_hp(&mut state, ch);
            vnd_dispatch(&mut state, hp);
            let sol = state.clone_solution();
            if ctor_is_noop {
                rand_member = Some(sol.clone());
            }
            if g1 && mode == 4 {
                RAND_MEMBER4.with(|c| *c.borrow_mut() = Some(sol.clone()));
            }
            population.push(sol);
        }

        DETERMINISTIC_SEED_GROUPS.with(|cache| {
            let cached = cache.borrow();
            let groups = cached.as_ref().unwrap();
            population.extend(groups.1.iter().cloned());
        });

        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);

        let no_elites: [SolState; 0] = [];
        let seeded: &[SolState] = if hp.elite_chain > 0 { elites } else { &no_elites };
        let pop_cap = if hp.elite_chain == 1 { 8 + seeded.len() } else { 8 };
        if hp.elite_chain == 1 && !seeded.is_empty() {
            population.extend(seeded.iter().cloned());
            population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        }
        let parents: &[SolState] = if hp.elite_chain == 2 { seeded } else { &no_elites };

        for gen in 0..hp.n_crossover_gen {
            let ranked_len = population.len();
            let child_bits = crossover_frequency(&population, parents, challenge, &mut rng);
            set_state_from_bits(&mut state, &child_bits);
            dp_refinement_hp(&mut state, ch);
            vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());

            if population.len() >= 2 {
                let a = gen % population.len().min(4);
                if !parents.is_empty() {
                    let child_bits = crossover_uniform(
                        &population[a],
                        &parents[gen % parents.len()],
                        challenge,
                        &mut rng,
                    );
                    set_state_from_bits(&mut state, &child_bits);
                    dp_refinement_hp(&mut state, ch);
                    vnd_dispatch(&mut state, hp);
                    population.push(state.clone_solution());
                } else {
                    let b = (gen + 1) % population.len().min(4);
                    if a != b {
                        let child_bits =
                            crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                        set_state_from_bits(&mut state, &child_bits);
                        dp_refinement_hp(&mut state, ch);
                        vnd_dispatch(&mut state, hp);
                        population.push(state.clone_solution());
                    }
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
            population.truncate(pop_cap);
        }

        state.restore_solution(&population[0]);
        let mut best_val = state.total_value;
        let mut best_state = state.clone_solution();
        let mut vnd_fixed_point = false;

        let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
        tabu_hashes.push(state.hash);

        let mut stall_count = 0;
        let mut cnt = hp.ils_rounds;

        let mut snap = state.clone_solution();

        for round in 0..cnt {
            snap.bits.clone_from(&state.selected_bit);
            snap.contrib.clone_from(&state.contrib);
            snap.value = state.total_value;
            snap.weight = state.total_weight;
            snap.hash = state.hash;

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

                if hp.ils_restart_interval > 0
                    && stall_count > 0
                    && stall_count % hp.ils_restart_interval == 0
                {
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
                    perturb_by_strategy(
                        &mut state,
                        extra_strength,
                        stall_count + 3,
                        6,
                        &mut rng,
                        hp,
                    );
                    greedy_reconstruct(&mut state, 0);
                    vnd_fixed_point = vnd_dispatch_certified(&mut state, hp);
                }
                let h2 = state.hash;
                if tabu_hashes.len() < 128 {
                    tabu_hashes.push(h2);
                } else {
                    tabu_hashes[round % 128] = h2;
                }

                if state.total_value > best_val {
                    best_val = state.total_value;
                    best_state.bits.clone_from(&state.selected_bit);
                    best_state.contrib.clone_from(&state.contrib);
                    best_state.value = state.total_value;
                    best_state.weight = state.total_weight;
                    best_state.hash = state.hash;
                    stall_count = 0;
                    cnt += 3;
                }
            } else {
                stall_count = 0;
                let h = state.hash;
                if tabu_hashes.len() < 128 {
                    tabu_hashes.push(h);
                }
            }
        }

        state.restore_solution(&best_state);

        if hp.use_heavy_polish {
            loop {
                let v_before = state.total_value;
                local_search_vnd_heavy(&mut state);
                dp_refinement_hp(&mut state, ch);
                if state.total_value <= v_before {
                    break;
                }
            }
        } else {
            loop {
                let v_before = state.total_value;
                local_search_vnd_windowed(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, ch);
                if state.total_value <= v_before {
                    break;
                }
            }
        }

        if state.total_value > best_val {
            let elite = state.clone_solution();
            (
                Solution {
                    items: state.selected_items(),
                },
                state.total_value,
                elite,
            )
        } else {
            let items = (0..challenge.num_items)
                .filter(|&i| best_state.bits[i])
                .collect();
            (Solution { items }, best_val, best_state)
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
            LAST_T42_CANDIDATES.with(|c| c.borrow_mut().clear());
            FASTM4.store(hp.fastmask, std::sync::atomic::Ordering::Relaxed);
            RAND_MEMBER4.with(|c| *c.borrow_mut() = None);
            STATIC_RECONSTRUCT_SYNERGY.with(|c| {
                *c.borrow_mut() = challenge
                    .interaction_values
                    .iter()
                    .map(|row| {
                        row.iter()
                            .take(challenge.num_items.min(100))
                            .map(|&v| v as i64)
                            .sum()
                    })
                    .collect()
            });
            EJC_THRESH.with(|c| c.set(hp.ejc_thresh.min(255) as u8));
            let n_restarts = hp.n_full_restarts.max(1);
            FORWARD_DIVIDERS.with(|cache| {
                *cache.borrow_mut() = Some(
                    challenge
                        .weights
                        .iter()
                        .map(|&weight| ExactDivider::new(weight))
                        .collect(),
                );
            });
            INTERACTION_NEIGHBOR_CACHE.with(|cache| {
                *cache.borrow_mut() = Some(InteractionNeighborCache {
                    queries: vec![0; challenge.num_items],
                    rows: vec![None; challenge.num_items],
                });
            });
            if hp.rush_mode > 0 {
                let mut st = State::new_empty(challenge);
                build_greedy_density(&mut st);
                if hp.rush_mode >= 2 {
                    dp_refinement_hp(&mut st, hp.core_half_dp.min(24));
                    local_search_vnd_windowed(&mut st, hp.window_k.min(120));
                }
                let sol = if st.total_weight <= challenge.max_weight {
                    Some(Solution {
                        items: st.selected_items(),
                    })
                } else {
                    None
                };
                INTERACTION_NEIGHBOR_CACHE.with(|cache| {
                    *cache.borrow_mut() = None;
                });
                FORWARD_DIVIDERS.with(|cache| {
                    *cache.borrow_mut() = None;
                });
                return Ok(sol);
            }
            let deterministic_seed_groups = build_deterministic_seed_groups(challenge, &hp);
            let best_sol = if hp.fused_dual > 0 {
                solve_fused_dual(challenge, &hp, deterministic_seed_groups)
            } else {
                run_trajectory(challenge, &hp, deterministic_seed_groups, n_restarts, i64::MAX).0
            };

            INTERACTION_NEIGHBOR_CACHE.with(|cache| {
                *cache.borrow_mut() = None;
            });
            FORWARD_DIVIDERS.with(|cache| {
                *cache.borrow_mut() = None;
            });
            Ok(best_sol)
        }
    }

    thread_local! {
        static LAST_T42_CANDIDATES: std::cell::RefCell<Vec<Solution>> = std::cell::RefCell::new(Vec::new());
    }

    /// Drains the candidates the last `fused_dual` solve stored (empty after any other solve).
    pub fn collect_last_candidates() -> Vec<Solution> {
        LAST_T42_CANDIDATES.with(|c| core::mem::take(&mut *c.borrow_mut()))
    }

    /// Both hub_pair trajectories in one call, sharing the dividers, neighbor cache, synergy row sums and the
    /// leading + anchor seed groups. `groups.1` holds the anchor seeds first, then the hub-pair seeds, so its
    /// anchor prefix is exactly the `hub_pair_n = 0` trailing group (`hub_pair_n` is read nowhere else). Before
    /// trajectory 0 the per-call state a standalone `use_hub_pair = 0` call starts from is restored: RAND_MEMBER4
    /// is emptied and the DP / VND memos are cleared, as `solve_challenge` does on entry. Stores every distinct
    /// feasible trajectory best and elite in LAST_T42_CANDIDATES (best first) and returns the better best.
    fn solve_fused_dual(
        challenge: &Challenge,
        hp: &Hparams,
        groups: (Vec<SolState>, Vec<SolState>),
    ) -> Option<Solution> {
        let anchor_limit = if challenge.num_items <= 500 { 3 } else { 1 };
        let (leading, trailing_with_hub) = groups;
        let trailing_anchor: Vec<SolState> =
            trailing_with_hub[..anchor_limit.min(trailing_with_hub.len())].to_vec();

        let restarts_4 = if hp.t42_dedup >= 3 {
            hp.n_full_restarts.min(29).max(1)
        } else {
            hp.n_full_restarts.max(1)
        };
        let max_hub_extra = trailing_with_hub[anchor_limit.min(trailing_with_hub.len())..]
            .iter()
            .map(|s| s.value)
            .max()
            .unwrap_or(i64::MIN);
        let (best_4, q_4, elites_4, converged_r0_4) =
            run_trajectory(challenge, hp, (leading.clone(), trailing_with_hub), restarts_4, i64::MAX);
        if hp.t42_dedup >= 3 && converged_r0_4 {
            LAST_T42_CANDIDATES.with(|c| c.borrow_mut().clear());
            return best_4;
        }
        if hp.t42_dedup >= 3 {
            if let Some(m0_val) = RAND_MEMBER4.with(|c| c.borrow().as_ref().map(|s| s.value)) {
                let mut base_vals: Vec<i64> = leading.iter().chain(trailing_anchor.iter()).map(|s| s.value).collect();
                base_vals.extend(std::iter::repeat(m0_val).take(hp.n_random_starts));
                base_vals.sort_unstable_by(|a, b| b.cmp(a));
                let eighth_val = base_vals.get(7).copied().unwrap_or(i64::MIN);
                if max_hub_extra < eighth_val {
                    // Hub-pair seeds are strictly below the top-8 initial population cutoff, so Trajectory 0 is identical to Trajectory 4.
                    LAST_T42_CANDIDATES.with(|c| c.borrow_mut().clear());
                    return best_4;
                }
            }
        }

        RAND_MEMBER4.with(|c| *c.borrow_mut() = None);
        if hp.t42_dedup == 0 {
            DP_MEMO.with(|cell| cell.borrow_mut().clear());
            VND_PATHS.with(|cell| cell.borrow_mut().clear());
            DP_STEP_MEMO.with(|cell| cell.borrow_mut().clear());
        }
        let restarts_0 = match hp.t42_dedup {
            2 => 20,
            3 => 16,
            4 => 12,
            _ => hp.n_full_restarts.max(1),
        };
        let (best_0, q_0, elites_0, aborted_t0) =
            run_trajectory(challenge, hp, (leading, trailing_anchor), restarts_0, q_4);
        // Velocity exit (T42 dedup 3+): trajectory 0 ties trajectory 4 or was dominated >4000 pts at r=4.
        if hp.t42_dedup >= 3 && (q_4 == q_0 || aborted_t0) {
            LAST_T42_CANDIDATES.with(|c| c.borrow_mut().clear());
            return best_4;
        }

        let mut pool: Vec<(Solution, i64)> = Vec::new();
        let mut push = |sol: Solution, q: i64| {
            let weight: u64 = sol.items.iter().map(|&i| challenge.weights[i] as u64).sum();
            if weight <= challenge.max_weight as u64 && !pool.iter().any(|(s, _)| s.items == sol.items) {
                pool.push((sol, q));
            }
        };
        if let Some(s) = best_4.clone() {
            push(s, q_4);
        }
        if let Some(s) = best_0.clone() {
            push(s, q_0);
        }
        for e in elites_4.iter().chain(elites_0.iter()) {
            let items: Vec<usize> = (0..challenge.num_items).filter(|&i| e.bits[i]).collect();
            push(Solution { items }, e.value);
        }
        pool.sort_by_key(|(_, q)| std::cmp::Reverse(*q));
        LAST_T42_CANDIDATES.with(|c| *c.borrow_mut() = pool.into_iter().map(|(s, _)| s).collect());

        // A tie keeps trajectory 4, the default call's result.
        if q_0 > q_4 {
            best_0
        } else {
            best_4
        }
    }

    /// One trajectory on `groups`: `n_restarts` instances, then the 3-pass p9 polish. Returns the best
    /// solution, its value and the elite states carried between restarts.
    fn run_trajectory(
        challenge: &Challenge,
        hp: &Hparams,
        groups: (Vec<SolState>, Vec<SolState>),
        n_restarts: usize,
        q_4_floor: i64,
    ) -> (Option<Solution>, i64, Vec<SolState>, bool) {
        {
            DETERMINISTIC_SEED_GROUPS.with(|cache| {
                *cache.borrow_mut() = Some(groups);
            });
            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            let mut best_restart = 0usize;
            let mut best_hits = 0usize;
            let elite_k = if hp.elite_chain > 0 { hp.elite_k } else { 0 };
            let mut elites: Vec<SolState> = Vec::with_capacity(elite_k);
            for restart in 0..n_restarts {
                let (sol, val, elite) = run_one_instance(challenge, &hp, restart, &elites);
                if val > best_quality {
                    best_quality = val;
                    best_sol = Some(sol);
                    best_restart = restart;
                    best_hits = 1;
                } else if val == best_quality {
                    best_hits += 1;
                    if hp.t42_dedup >= 3 {
                        if best_restart == 0 && best_hits >= 3 {
                            break;
                        }
                        if q_4_floor == i64::MAX && best_restart >= 10 && best_hits >= 10 {
                            break;
                        }
                        if q_4_floor < i64::MAX && best_restart >= 6 && best_hits >= 6 && best_quality >= q_4_floor {
                            break;
                        }
                    }
                }
                if hp.t42_dedup >= 3 && q_4_floor < i64::MAX && restart >= 4 && best_quality + 4000 < q_4_floor {
                    break;
                }
                if elite_k > 0 {
                    elites.push(elite);
                    elites.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
                    let mut seen: Vec<u64> = Vec::with_capacity(elites.len());
                    elites.retain(|s| {
                        if seen.contains(&s.hash) {
                            false
                        } else {
                            seen.push(s.hash);
                            true
                        }
                    });
                    elites.truncate(elite_k);
                }
            }
            let aborted_t0 = hp.t42_dedup >= 3 && q_4_floor < i64::MAX && best_quality + 4000 < q_4_floor;
            let converged_r0 = (best_restart == 0 && best_hits >= 3) || aborted_t0;
            DETERMINISTIC_SEED_GROUPS.with(|cache| {
                *cache.borrow_mut() = None;
            });

            if !converged_r0 {
                let p9_k: usize = 35;
                let mut rng_p9 = {
                    let mut r = Rng::from_seed(&stream_seed(challenge, hp.rng_stream));
                    for _ in 0..7919 {
                        r.next_u32();
                    }
                    r
                };
                let n_items = challenge.num_items;
                let mut st = State::new_empty(challenge);
                if let Some(s) = &best_sol {
                    st.apply_items(&s.items, true);
                }
                let mut sel_asc: Vec<(usize, i32)> = Vec::with_capacity(n_items);
                let mut to_rm: Vec<usize> = Vec::with_capacity(p9_k);
                let mut complement: Vec<usize> = Vec::with_capacity(n_items);
                let mut to_add: Vec<usize> = Vec::with_capacity(p9_k);
                let mut target_bits = vec![false; n_items];

                for _pass in 0..3usize {
                    let current_items: &[usize] = match &best_sol {
                        Some(s) => &s.items,
                        None => break,
                    };

                    sel_asc.clear();
                    for &i in current_items {
                        sel_asc.push((i, st.contrib[i]));
                    }
                    sel_asc.sort_unstable_by_key(|&(_, c)| c);
                    to_rm.clear();
                    for &(i, _) in sel_asc.iter().take(p9_k) {
                        to_rm.push(i);
                    }
                    st.apply_items(&to_rm, false);

                    complement.clear();
                    for i in 0..n_items {
                        if !st.selected_bit[i] {
                            complement.push(i);
                        }
                    }
                    complement.sort_unstable_by_key(|&i| std::cmp::Reverse(st.contrib[i]));
                    let top_len = complement.len().min(2 * p9_k);
                    for i in 0..top_len {
                        let j = i + rng_p9.next_usize(top_len - i);
                        complement.swap(i, j);
                    }
                    to_add.clear();
                    let mut w = st.total_weight;
                    for ci in 0..top_len {
                        if to_add.len() >= p9_k {
                            break;
                        }
                        let idx = complement[ci];
                        if w + challenge.weights[idx] <= challenge.max_weight {
                            w += challenge.weights[idx];
                            to_add.push(idx);
                        }
                    }
                    st.apply_items(&to_add, true);

                    local_search_vnd_windowed(&mut st, hp.window_k);
                    dp_refinement_hp(&mut st, hp.core_half_dp);

                    if st.total_value > best_quality {
                        best_quality = st.total_value;
                        best_sol = Some(Solution {
                            items: st.selected_items(),
                        });
                    } else if _pass < 2 {
                        if let Some(s) = &best_sol {
                            target_bits.fill(false);
                            for &idx in &s.items {
                                target_bits[idx] = true;
                            }
                            set_state_from_bits(&mut st, &target_bits);
                        }
                    }
                }
            }

            (best_sol, best_quality, elites, converged_r0)
        }
    }
    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        struct ExactCleanup;
        impl Drop for ExactCleanup {
            fn drop(&mut self) {
                DP_MEMO.with(|cell| *cell.borrow_mut() = Vec::new());
                VND_PATHS.with(|cell| cell.borrow_mut().clear());
                DP_STEP_MEMO.with(|cell| *cell.borrow_mut() = Vec::new());
                STATIC_RECONSTRUCT_SYNERGY.with(|cell| *cell.borrow_mut() = Vec::new());
            }
        }
        let _exact_cleanup = ExactCleanup;

        DP_MEMO.with(|cell| cell.borrow_mut().clear());
        VND_PATHS.with(|cell| cell.borrow_mut().clear());
        DP_STEP_MEMO.with(|cell| cell.borrow_mut().clear());

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

/// Distinct feasible solutions from the last `fused_dual` solve on this thread, best first; drains the buffer.
pub fn collect_last_candidates() -> Vec<Solution> {
    inner_four::collect_last_candidates()
}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_compound {

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Group {
        pub ids: [usize; 3],
        pub len: usize,
        pub weight: i64,
        pub value: i64,
        pub ordinal: usize,
    }

    pub(super) fn groups<W, C, Q>(
        items: &[usize],
        count: usize,
        weight: W,
        value: C,
        q: Q,
        add: bool,
    ) -> Vec<Group>
    where
        W: Fn(usize) -> i64,
        C: Fn(usize) -> i64,
        Q: Fn(usize, usize) -> i64,
    {
        let mut out = Vec::new();
        for (i, &a) in items.iter().enumerate() {
            if count == 1 {
                out.push(Group {
                    ids: [a, 0, 0],
                    len: 1,
                    weight: weight(a),
                    value: value(a),
                    ordinal: out.len(),
                });
                continue;
            }
            for (j, &b) in items.iter().enumerate().skip(i + 1) {
                let pair = value(a) + value(b) + if add { q(a, b) } else { -q(a, b) };
                let pair_weight = weight(a) + weight(b);
                if count == 2 {
                    out.push(Group {
                        ids: [a, b, 0],
                        len: 2,
                        weight: pair_weight,
                        value: pair,
                        ordinal: out.len(),
                    });
                } else {
                    for &c in &items[j + 1..] {
                        let extra = q(a, c) + q(b, c);
                        out.push(Group {
                            ids: [a, b, c],
                            len: 3,
                            weight: pair_weight + weight(c),
                            value: pair + value(c) + if add { extra } else { -extra },
                            ordinal: out.len(),
                        });
                    }
                }
            }
        }
        out
    }

    pub(super) fn best<Q>(
        adds: &[Group],
        removes: &[Group],
        slack: i64,
        add_outer: bool,
        stop_nonpositive: bool,
        q: Q,
    ) -> Option<(Group, Group)>
    where
        Q: Fn(usize, usize) -> i64,
    {
        if adds.is_empty() || removes.is_empty() {
            return None;
        }
        let mut best = 0i64;
        let mut order = (usize::MAX, usize::MAX);
        let mut result = None;
        let consider = |r: Group, a: Group| {
            let mut delta = a.value - r.value;
            for &i in &a.ids[..a.len] {
                for &j in &r.ids[..r.len] {
                    delta -= q(i, j);
                }
            }
            let key = if add_outer {
                (a.ordinal, r.ordinal)
            } else {
                (r.ordinal, a.ordinal)
            };
            (delta, key)
        };
        if adds
            .iter()
            .chain(removes)
            .any(|g| !(0..=30).contains(&g.weight))
        {
            if add_outer {
                for &a in adds {
                    if stop_nonpositive && a.value <= 0 && best > 0 {
                        break;
                    }
                    for &r in removes {
                        if a.weight > slack + r.weight {
                            continue;
                        }
                        let (delta, key) = consider(r, a);
                        if delta > 0 && (delta > best || (delta == best && key < order)) {
                            best = delta;
                            order = key;
                            result = Some((r, a));
                        }
                    }
                }
            } else {
                for &r in removes {
                    for &a in adds {
                        if a.weight > slack + r.weight {
                            continue;
                        }
                        let (delta, key) = consider(r, a);
                        if delta > 0 && (delta > best || (delta == best && key < order)) {
                            best = delta;
                            order = key;
                            result = Some((r, a));
                        }
                    }
                }
            }
            return result;
        }
        let mut buckets: [Vec<usize>; 31] = Default::default();
        if add_outer {
            let mut maximum = [i64::MIN; 31];
            for a in adds {
                maximum[a.weight as usize] = maximum[a.weight as usize].max(a.value);
            }
            for w in 1..31 {
                maximum[w] = maximum[w].max(maximum[w - 1]);
            }
            for (i, r) in removes.iter().enumerate() {
                let budget = slack + r.weight;
                if budget >= 0 && maximum[budget.min(30) as usize] > r.value {
                    buckets[r.weight as usize].push(i);
                }
            }
            for b in &mut buckets {
                b.sort_unstable_by_key(|&i| (removes[i].value, removes[i].ordinal));
            }
            let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
            weights.sort_unstable_by_key(|&w| removes[buckets[w][0]].value);
            for &a in adds {
                if stop_nonpositive && a.value <= 0 && best > 0 {
                    break;
                }
                for &w in &weights {
                    if a.value - removes[buckets[w][0]].value < best {
                        break;
                    }
                    if a.weight > slack + w as i64 {
                        continue;
                    }
                    for &i in &buckets[w] {
                        let r = removes[i];
                        if a.value - r.value < best {
                            break;
                        }
                        let (delta, key) = consider(r, a);
                        if delta > 0 && (delta > best || (delta == best && key < order)) {
                            best = delta;
                            order = key;
                            result = Some((r, a));
                        }
                    }
                }
            }
        } else {
            let mut minimum = [i64::MAX; 31];
            for r in removes {
                minimum[r.weight as usize] = minimum[r.weight as usize].min(r.value);
            }
            for w in (0..30).rev() {
                minimum[w] = minimum[w].min(minimum[w + 1]);
            }
            for (i, a) in adds.iter().enumerate() {
                let need = (a.weight - slack).max(0);
                if need <= 30 && a.value > minimum[need as usize] {
                    buckets[a.weight as usize].push(i);
                }
            }
            for b in &mut buckets {
                b.sort_unstable_by_key(|&i| (std::cmp::Reverse(adds[i].value), adds[i].ordinal));
            }
            let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
            weights.sort_unstable_by_key(|&w| std::cmp::Reverse(adds[buckets[w][0]].value));
            for &r in removes {
                for &w in &weights {
                    if adds[buckets[w][0]].value - r.value < best {
                        break;
                    }
                    if w as i64 > slack + r.weight {
                        continue;
                    }
                    for &i in &buckets[w] {
                        let a = adds[i];
                        if a.value - r.value < best {
                            break;
                        }
                        let (delta, key) = consider(r, a);
                        if delta > 0 && (delta > best || (delta == best && key < order)) {
                            best = delta;
                            order = key;
                            result = Some((r, a));
                        }
                    }
                }
            }
        }
        result
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_construct {

    #[inline(always)]
    pub(super) fn positive_density(c: i32, w: u32) -> i64 {
        const M: [u64; 11] = [
            0,
            1u64 << 48,
            ((1u64 << 48) + 1) / 2,
            ((1u64 << 48) + 2) / 3,
            ((1u64 << 48) + 3) / 4,
            ((1u64 << 48) + 4) / 5,
            ((1u64 << 48) + 5) / 6,
            ((1u64 << 48) + 6) / 7,
            ((1u64 << 48) + 7) / 8,
            ((1u64 << 48) + 8) / 9,
            ((1u64 << 48) + 9) / 10,
        ];
        debug_assert!(c > 0 && (1..=10).contains(&w));
        ((((c as u32 as u64) * 1000) as u128 * M[w as usize] as u128) >> 48) as i64
    }

    #[inline(always)]
    pub(super) fn positive_density_narrow(c: i32, w: u32) -> i64 {
        const N: u64 = 1000u64 << 30;
        const M: [u64; 11] = [
            0,
            N,
            (N + 1) / 2,
            (N + 2) / 3,
            (N + 3) / 4,
            (N + 4) / 5,
            (N + 5) / 6,
            (N + 6) / 7,
            (N + 7) / 8,
            (N + 8) / 9,
            (N + 9) / 10,
        ];
        debug_assert!(c > 0 && c < (1 << 24) && (1..=10).contains(&w));
        ((c as u32 as u64 * M[w as usize]) >> 30) as i64
    }

    pub(super) struct Queue {
        small: Vec<u32>,
        wide: Option<Vec<u64>>,
        maxima: Vec<u64>,
        dirty: Vec<bool>,
        bits: Vec<u64>,
        counts: Vec<i32>,
        n: usize,
        len: usize,
    }
    impl Queue {
        pub(super) fn new(n: usize) -> Self {
            assert!(n <= u16::MAX as usize);
            let blocks = (n + 31) / 32;
            let words = (n + 63) / 64;
            Self {
                small: vec![0; blocks * 32],
                wide: None,
                maxima: vec![0; blocks],
                dirty: vec![false; blocks],
                bits: vec![0; words],
                counts: vec![0; words + 1],
                n,
                len: 0,
            }
        }
        #[inline]
        fn active(&mut self, id: usize, yes: bool) {
            let word = id / 64;
            self.bits[word] ^= 1u64 << (id & 63);
            let mut p = word + 1;
            let delta = if yes { 1 } else { -1 };
            while p < self.counts.len() {
                self.counts[p] += delta;
                p += p & p.wrapping_neg();
            }
        }
        #[inline]
        pub(super) fn rank(&self, id: usize) -> usize {
            let word = id / 64;
            let mask = (1u64 << (id & 63)) - 1;
            let mut rank = (self.bits[word] & mask).count_ones() as usize;
            let mut p = word;
            while p != 0 {
                rank += self.counts[p] as usize;
                p &= p - 1;
            }
            rank
        }
        #[inline]
        pub(super) fn set(&mut self, id: usize, score: i64) {
            debug_assert!(id < self.n && score >= 0 && score < (1i64 << 48));
            let code = score as u64 + 1;
            if code > u32::MAX as u64 && self.wide.is_none() {
                self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
            }
            let fresh = unsafe {
                if let Some(wide) = self.wide.as_mut() {
                    let p = wide.get_unchecked_mut(id);
                    let fresh = *p == 0;
                    *p = code;
                    fresh
                } else {
                    let p = self.small.get_unchecked_mut(id);
                    let fresh = *p == 0;
                    *p = code as u32;
                    fresh
                }
            };
            if fresh {
                self.active(id, true);
                self.len += 1;
            }
            unsafe {
                *self.dirty.get_unchecked_mut(id / 32) = true;
            }
        }
        #[inline]
        pub(super) fn remove(&mut self, id: usize) {
            debug_assert!(id < self.n);
            let existed = unsafe {
                if let Some(wide) = self.wide.as_mut() {
                    let p = wide.get_unchecked_mut(id);
                    let existed = *p != 0;
                    *p = 0;
                    existed
                } else {
                    let p = self.small.get_unchecked_mut(id);
                    let existed = *p != 0;
                    *p = 0;
                    existed
                }
            };
            if existed {
                self.active(id, false);
                self.len -= 1;
                unsafe {
                    *self.dirty.get_unchecked_mut(id / 32) = true;
                }
            }
        }
        #[inline]
        pub(super) fn assign(&mut self, id: usize, score: i64, active: bool) {
            let code = (score as u64 + 1) & 0u64.wrapping_sub(active as u64);
            if code > u32::MAX as u64 && self.wide.is_none() {
                self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
            }
            let existed = unsafe {
                if let Some(wide) = self.wide.as_mut() {
                    let p = wide.get_unchecked_mut(id);
                    let old = *p != 0;
                    *p = code;
                    old
                } else {
                    let p = self.small.get_unchecked_mut(id);
                    let old = *p != 0;
                    *p = code as u32;
                    old
                }
            };
            if existed != active {
                self.active(id, active);
                if active {
                    self.len += 1;
                } else {
                    self.len -= 1;
                }
            }
            unsafe {
                *self.dirty.get_unchecked_mut(id / 32) = true;
            }
        }
        #[inline]
        pub(super) unsafe fn assign_small(&mut self, id: usize, score: u32, active: bool) {
            debug_assert!(id < self.n && self.wide.is_none() && score < u32::MAX);
            let code = (score + 1) & 0u32.wrapping_sub(active as u32);
            let p = self.small.get_unchecked_mut(id);
            let existed = *p != 0;
            *p = code;
            if existed != active {
                self.active(id, active);
                if active {
                    self.len += 1;
                } else {
                    self.len -= 1;
                }
            }
            *self.dirty.get_unchecked_mut(id / 32) = true;
        }
        fn refresh(&mut self) {
            if let Some(wide) = self.wide.as_ref() {
                for block in 0..self.maxima.len() {
                    if self.dirty[block] {
                        self.maxima[block] = wide[block * 32..block * 32 + 32]
                            .iter()
                            .copied()
                            .max()
                            .unwrap();
                        self.dirty[block] = false;
                    }
                }
            } else {
                for block in 0..self.maxima.len() {
                    if self.dirty[block] {
                        self.maxima[block] = self.small[block * 32..block * 32 + 32]
                            .iter()
                            .copied()
                            .max()
                            .unwrap() as u64;
                        self.dirty[block] = false;
                    }
                }
            }
        }
        pub(super) fn len(&self) -> usize {
            self.len
        }
        pub(super) fn best(&mut self) -> Option<usize> {
            if self.len == 0 {
                return None;
            }
            self.refresh();
            let maximum = self.maxima.iter().copied().max().unwrap();
            let block = self.maxima.iter().position(|&v| v == maximum).unwrap();
            let pos = if let Some(wide) = self.wide.as_ref() {
                wide[block * 32..block * 32 + 32]
                    .iter()
                    .position(|&v| v == maximum)
                    .unwrap()
            } else {
                self.small[block * 32..block * 32 + 32]
                    .iter()
                    .position(|&v| v as u64 == maximum)
                    .unwrap()
            };
            Some(block * 32 + pos)
        }
        pub(super) fn finalists(&mut self, bonus: u64, out: &mut Vec<(usize, i64)>) {
            out.clear();
            if self.len == 0 {
                return;
            }
            self.refresh();
            let mut first = 0u64;
            let mut second = 0u64;
            for &key in &self.maxima {
                if key > first {
                    second = first;
                    first = key;
                } else {
                    second = second.max(key);
                }
            }
            let lower = second.saturating_sub(1).saturating_sub(bonus);
            for (block, &maximum) in self.maxima.iter().enumerate() {
                if maximum == 0 || maximum - 1 < lower {
                    continue;
                }
                if let Some(wide) = self.wide.as_ref() {
                    for (p, &code) in wide[block * 32..block * 32 + 32].iter().enumerate() {
                        if code > 0 && code - 1 >= lower {
                            out.push((block * 32 + p, (code - 1) as i64));
                        }
                    }
                } else {
                    for (p, &code) in self.small[block * 32..block * 32 + 32].iter().enumerate() {
                        if code > 0 && code as u64 - 1 >= lower {
                            out.push((block * 32 + p, (code - 1) as i64));
                        }
                    }
                }
            }
        }
    }

    pub(super) struct Jumps {
        tables: Vec<[[u64; 256]; 8]>,
    }

    impl Jumps {
        fn step(mut x: u64) -> u64 {
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            x
        }

        pub(super) fn new() -> Self {
            let mut powers = [0u64; 64];
            for bit in 0..64 {
                powers[bit] = Self::step(1u64 << bit);
            }
            let mut tables = Vec::with_capacity(16);
            for _ in 0..16 {
                let mut table = [[0u64; 256]; 8];
                for byte in 0..8 {
                    for value in 1usize..256 {
                        let rest = value & (value - 1);
                        table[byte][value] =
                            table[byte][rest] ^ powers[byte * 8 + value.trailing_zeros() as usize];
                    }
                }
                let mut squared = [0u64; 64];
                for bit in 0..64 {
                    squared[bit] = Self::apply(&table, powers[bit]);
                }
                tables.push(table);
                powers = squared;
            }
            Self { tables }
        }

        #[inline(always)]
        fn apply(t: &[[u64; 256]; 8], x: u64) -> u64 {
            t[0][(x & 255) as usize]
                ^ t[1][((x >> 8) & 255) as usize]
                ^ t[2][((x >> 16) & 255) as usize]
                ^ t[3][((x >> 24) & 255) as usize]
                ^ t[4][((x >> 32) & 255) as usize]
                ^ t[5][((x >> 40) & 255) as usize]
                ^ t[6][((x >> 48) & 255) as usize]
                ^ t[7][(x >> 56) as usize]
        }

        #[inline]
        pub(super) fn advance(&self, state: &mut u64, mut count: usize) {
            debug_assert!(count <= u16::MAX as usize);
            if count < 8 {
                for _ in 0..count {
                    *state = Self::step(*state);
                }
                return;
            }
            while count != 0 {
                let power = count.trailing_zeros() as usize;
                *state = Self::apply(&self.tables[power], *state);
                count &= count - 1;
            }
        }
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_density {
    pub(super) struct Density {
        groups: [Vec<usize>; 11],
        positions: Vec<usize>,
    }
    impl Density {
        pub(super) fn new(weights: &[u32], include: &[bool]) -> Option<Self> {
            if weights.iter().any(|&w| !(1..=10).contains(&w)) {
                return None;
            }
            if weights.len() > u32::MAX as usize {
                return None;
            }
            let mut positions = vec![0usize; weights.len()];
            let mut groups: [Vec<usize>; 11] = Default::default();
            for (i, &w) in weights.iter().enumerate() {
                if include[i] {
                    positions[i] = groups[w as usize].len();
                    groups[w as usize].push(i);
                }
            }
            Some(Self { groups, positions })
        }
        fn extreme<const MAX: bool>(ids: &[usize], contrib: &[i32]) -> u64 {
            let mut result = if MAX { 0u64 } else { u64::MAX };
            unsafe {
                let ip = ids.as_ptr();
                let cp = contrib.as_ptr();
                macro_rules! visit {
                    ($p:expr) => {{
                        let id = *ip.add($p);
                        let c = (*cp.add(id) as u32) ^ 0x80000000;
                        let key = ((c as u64) << 32)
                            | if MAX {
                                (u32::MAX - id as u32) as u64
                            } else {
                                id as u64
                            };
                        result = if MAX {
                            result.max(key)
                        } else {
                            result.min(key)
                        };
                    }};
                }
                let mut p = 0usize;
                let full = ids.len() / 8 * 8;
                while p < full {
                    visit!(p);
                    visit!(p + 1);
                    visit!(p + 2);
                    visit!(p + 3);
                    visit!(p + 4);
                    visit!(p + 5);
                    visit!(p + 6);
                    visit!(p + 7);
                    p += 8;
                }
                while p < ids.len() {
                    visit!(p);
                    p += 1;
                }
            }
            result
        }
        pub(super) fn pop(&mut self, contrib: &[i32], slack: u32, maximum: bool) -> Option<usize> {
            let mut chosen = None;
            let mut best = if maximum { i64::MIN } else { i64::MAX };
            let mut best_id = usize::MAX;
            for w in 1..=10usize.min(slack as usize) {
                let group = &self.groups[w];
                if group.is_empty() {
                    continue;
                }
                let key = if maximum {
                    Self::extreme::<true>(group, contrib)
                } else {
                    Self::extreme::<false>(group, contrib)
                };
                let value = ((key >> 32) as u32 ^ 0x80000000) as i32;
                let id = if maximum {
                    u32::MAX - key as u32
                } else {
                    key as u32
                } as usize;
                let pos = self.positions[id];
                if maximum && value <= 0 {
                    continue;
                }
                let score = value as i64 * 1000 / w as i64;
                if (if maximum { score > best } else { score < best })
                    || (score == best && id < best_id)
                {
                    best = score;
                    best_id = id;
                    chosen = Some((w, pos));
                }
            }
            chosen.map(|(w, pos)| {
                let id = self.groups[w].swap_remove(pos);
                if pos < self.groups[w].len() {
                    self.positions[self.groups[w][pos]] = pos;
                }
                id
            })
        }

        pub(super) fn pop_affinity(
            &mut self,
            contrib: &[i32],
            affinity: &[i32],
            slack: u32,
        ) -> Option<usize> {
            assert!(self.positions.len() <= u16::MAX as usize);
            assert_eq!(contrib.len(), self.positions.len());
            assert_eq!(affinity.len(), self.positions.len());
            let mut chosen = None;
            let mut best = -1i64;
            let mut best_id = usize::MAX;
            for w in 1..=10usize.min(slack as usize) {
                let ids = &self.groups[w];
                let mut maxima = [0u64; 4];
                unsafe {
                    let ip = ids.as_ptr();
                    let cp = contrib.as_ptr();
                    let ap = affinity.as_ptr();
                    macro_rules! visit {
                        ($p:expr,$lane:expr) => {{
                            let id = *ip.add($p);
                            let c = *cp.add(id);
                            let numerator = c as i64 * 20 + (*ap.add(id)).max(0) as i64 * 7;
                            let key = if c > 0 {
                                ((numerator as u64) << 16) | (0xffff - id) as u64
                            } else {
                                0
                            };
                            maxima[$lane] = maxima[$lane].max(key);
                        }};
                    }
                    let mut p = 0;
                    let full = ids.len() / 4 * 4;
                    while p < full {
                        visit!(p, 0);
                        visit!(p + 1, 1);
                        visit!(p + 2, 2);
                        visit!(p + 3, 3);
                        p += 4;
                    }
                    while p < ids.len() {
                        visit!(p, 0);
                        p += 1;
                    }
                }
                let maximum = maxima.into_iter().max().unwrap();
                if maximum == 0 {
                    continue;
                }
                let id = 0xffff - (maximum as usize & 0xffff);
                let score = (maximum >> 16) as i64 * 50 / w as i64;
                if score > best || (score == best && id < best_id) {
                    best = score;
                    best_id = id;
                    chosen = Some((w, self.positions[id]));
                }
            }
            chosen.map(|(w, pos)| {
                let id = self.groups[w].swap_remove(pos);
                if pos < self.groups[w].len() {
                    self.positions[self.groups[w][pos]] = pos;
                }
                id
            })
        }
    }

    pub(super) unsafe fn top_three_sum(row: &[i32], ids: &[usize], exclude: usize) -> i64 {
        let mut tops = [[0i32; 4]; 3];
        let mut p = 0;
        while p + 4 <= ids.len() {
            for lane in 0..4 {
                let id = *ids.get_unchecked(p + lane);
                let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
                tops[2][lane] = tops[2][lane].max(x.min(tops[1][lane]));
                tops[1][lane] = tops[1][lane].max(x.min(tops[0][lane]));
                tops[0][lane] = tops[0][lane].max(x);
            }
            p += 4;
        }
        let mut a = 0;
        let mut b = 0;
        let mut c = 0;
        for x in tops.into_iter().flatten() {
            c = c.max(x.min(b));
            b = b.max(x.min(a));
            a = a.max(x);
        }
        for &id in &ids[p..] {
            let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
            c = c.max(x.min(b));
            b = b.max(x.min(a));
            a = a.max(x);
        }
        a as i64 + b as i64 + c as i64
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_dp {
    pub fn fill(
        core: &[usize],
        weights: &[u32],
        values: &[i32],
        capacity: usize,
        choices: &mut Vec<u8>,
        scratch: &mut Vec<i32>,
    ) -> Option<usize> {
        let magnitude: i64 = core.iter().map(|&i| (values[i] as i64).abs()).sum();
        if magnitude >= (1i64 << 29) {
            return None;
        }
        const INIT: i32 = -(1 << 30);
        let width = capacity + 1;
        scratch.resize(2 * width, INIT);
        scratch[..2 * width].fill(INIT);
        choices.resize(core.len() * width, 0);
        choices[..core.len() * width].fill(0);
        scratch[0] = 0;
        let mut row = 0;
        let mut hi = 0usize;
        for (t, &item) in core.iter().enumerate() {
            let weight = weights[item] as usize;
            if weight > capacity {
                continue;
            }
            let value = values[item];
            let next_hi = (hi + weight).min(capacity);
            let (first, second) = scratch[..2 * width].split_at_mut(width);
            let (src, dst) = if row == 0 {
                (first, second)
            } else {
                (second, first)
            };
            dst[..weight].copy_from_slice(&src[..weight]);
            let source = &src[..=next_hi - weight];
            let prior = &src[weight..=next_hi];
            let output = &mut dst[weight..=next_hi];
            let bits = &mut choices[t * width + weight..=t * width + next_hi];
            for (((out, bit), &old), &from) in output.iter_mut().zip(bits).zip(prior).zip(source) {
                let candidate = from + value;
                *bit = (candidate > old) as u8;
                *out = candidate.max(old);
            }
            hi = next_hi;
            row ^= 1;
        }
        let last = &scratch[row * width..(row + 1) * width];
        Some((0..=capacity).max_by_key(|&w| last[w]).unwrap_or(0))
    }

    pub(super) fn strict_density_solution(
        packed: &[usize], selected: &[bool], weights: &[u32], capacity: u32,
    ) -> bool {
        assert_eq!(packed.len(), selected.len());
        assert_eq!(packed.len(), weights.len());
        let mut lowest_selected = i64::MAX;
        let mut highest_unused = i64::MIN;
        let mut weight = 0u64;
        for ((&key, &chosen), &w) in packed.iter().zip(selected).zip(weights) {
            let density = (key as i64) >> 16;
            lowest_selected = lowest_selected.min(if chosen { density } else { i64::MAX });
            highest_unused = highest_unused.max(if chosen { i64::MIN } else { density });
            weight += w as u64 * chosen as u64;
        }
        weight == capacity as u64 && lowest_selected > 0 && lowest_selected > highest_unused
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_memo {
    pub(super) fn fingerprint(bits: &[bool], parameter: usize) -> u64 {
        let mut hash = 0x517cc1b727220a95u64 ^ parameter as u64;
        let mut i = 0;
        while i + 8 <= bits.len() {
            let word = unsafe { (bits.as_ptr().add(i) as *const u64).read_unaligned() };
            hash = (hash ^ word).wrapping_mul(0x9e3779b97f4a7c15);
            i += 8;
        }
        for &bit in &bits[i..] {
            hash = (hash ^ bit as u64).wrapping_mul(0x9e3779b97f4a7c15);
        }
        hash ^= hash >> 32;
        hash
    }

    pub(super) fn address(bits: &[bool], parameter: usize) -> usize {
        fingerprint(bits, parameter) as usize & 511
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_neighbors {

    #[inline]
    fn offer(list: &mut Vec<u64>, slots: usize, cutoff: &mut u64, key: u64) {
        if key <= *cutoff {
            return;
        }
        let at = list.partition_point(|&old| old > key);
        list.insert(at, key);
        if list.len() > slots {
            list.pop();
        }
        if list.len() == slots {
            *cutoff = list[slots - 1];
        }
    }

    pub(super) fn build(
        weights: &[u32],
        matrix: &[Vec<i32>],
        row_ptr: &[u32],
        columns: &[u16],
        k: usize,
        weight_aware: bool,
    ) -> Option<Vec<Vec<usize>>> {
        let n = weights.len();
        if n > u16::MAX as usize || k == 0 || weights.iter().any(|&w| !(1..=10).contains(&w)) {
            return None;
        }
        const LCM: u64 = 232_792_560;
        const FACTOR: [u64; 21] = [
            0,
            LCM,
            LCM / 2,
            LCM / 3,
            LCM / 4,
            LCM / 5,
            LCM / 6,
            LCM / 7,
            LCM / 8,
            LCM / 9,
            LCM / 10,
            LCM / 11,
            LCM / 12,
            LCM / 13,
            LCM / 14,
            LCM / 15,
            LCM / 16,
            LCM / 17,
            LCM / 18,
            LCM / 19,
            LCM / 20,
        ];
        let absolute_slots = if weight_aware { (k + 1) / 2 } else { k };
        let mut friends = Vec::with_capacity(n);
        let mut absolute = Vec::with_capacity(absolute_slots.min(n) + 1);
        let mut density = Vec::with_capacity(k.min(n) + 1);
        for i in 0..n {
            absolute.clear();
            density.clear();
            let mut absolute_cutoff = 0;
            let mut density_cutoff = 0;
            let row = &matrix[i];
            for &column in &columns[row_ptr[i] as usize..row_ptr[i + 1] as usize] {
                let j = column as usize;
                let interaction = row[j];
                if i == j || interaction <= 0 {
                    continue;
                }
                if interaction > 2_000_000 {
                    return None;
                }
                let tie = (u16::MAX as usize - j) as u64;
                offer(
                    &mut absolute,
                    absolute_slots,
                    &mut absolute_cutoff,
                    ((interaction as u64) << 16) | tie,
                );
                if weight_aware {
                    let scaled = interaction as u64 * FACTOR[(weights[i] + weights[j]) as usize];
                    offer(&mut density, k, &mut density_cutoff, (scaled << 16) | tie);
                }
            }
            let mut neighbors = Vec::with_capacity(k.min(n));
            neighbors.extend(
                absolute
                    .iter()
                    .map(|&key| u16::MAX as usize - (key as usize & 0xffff)),
            );
            if weight_aware {
                for &key in &density {
                    if neighbors.len() >= k {
                        break;
                    }
                    let j = u16::MAX as usize - (key as usize & 0xffff);
                    if !neighbors.contains(&j) {
                        neighbors.push(j);
                    }
                }
            }
            friends.push(neighbors);
        }
        Some(friends)
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_pairs {
    #[derive(Clone, Copy, Debug)]
    pub struct Pair {
        pub a: usize,
        pub b: usize,
        pub weight: i64,
        pub value: i64,
        pub ordinal: usize,
    }

    pub fn best_exchange<F: Fn(usize, usize) -> i64>(
        additions: &mut [Pair],
        removals: &mut [Pair],
        slack: i64,
        cross: F,
    ) -> Option<(Pair, Pair)> {
        if additions.is_empty() || removals.is_empty() {
            return None;
        }
        if additions.iter().all(|p| (0..=20).contains(&p.weight)) {
            return best_by_weight(additions, removals, slack, cross);
        }
        additions.sort_unstable_by(|a, b| b.value.cmp(&a.value).then(a.ordinal.cmp(&b.ordinal)));
        removals.sort_unstable_by(|a, b| a.value.cmp(&b.value).then(a.ordinal.cmp(&b.ordinal)));
        let mut best = 0i64;
        let mut best_order = (usize::MAX, usize::MAX);
        let mut movement = None;
        for &r in removals.iter() {
            if additions[0].value - r.value < best {
                break;
            }
            let budget = slack + r.weight;
            for &a in additions.iter() {
                let bound = a.value - r.value;
                if bound < best {
                    break;
                }
                let order = (r.ordinal, a.ordinal);
                if bound <= 0 || (bound == best && order >= best_order) || a.weight > budget {
                    continue;
                }
                let delta =
                    bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((r, a));
                }
            }
        }
        movement
    }

    fn best_by_weight<F: Fn(usize, usize) -> i64>(
        additions: &mut [Pair],
        removals: &mut [Pair],
        slack: i64,
        cross: F,
    ) -> Option<(Pair, Pair)> {
        let mut minimum = [i64::MAX; 21];
        for r in removals.iter() {
            let budget = slack + r.weight;
            if budget >= 0 {
                let b = budget.min(20) as usize;
                minimum[b] = minimum[b].min(r.value);
            }
        }
        for w in (0..20).rev() {
            minimum[w] = minimum[w].min(minimum[w + 1]);
        }
        let mut ai: Vec<usize> = additions
            .iter()
            .enumerate()
            .filter(|(_, a)| a.value > minimum[a.weight as usize])
            .map(|(i, _)| i)
            .collect();
        if ai.is_empty() {
            return None;
        }
        ai.sort_unstable_by(|&i, &j| {
            let a = &additions[i];
            let b = &additions[j];
            a.weight
                .cmp(&b.weight)
                .then(b.value.cmp(&a.value))
                .then(a.ordinal.cmp(&b.ordinal))
        });
        let mut starts = [0usize; 22];
        for &i in &ai {
            starts[additions[i].weight as usize + 1] += 1;
        }
        for w in 1..22 {
            starts[w] += starts[w - 1];
        }
        let mut maxima = [i64::MIN; 21];
        let mut groups = [0usize; 21];
        let mut ng = 0;
        for w in 0..21 {
            if starts[w] != starts[w + 1] {
                maxima[w] = additions[ai[starts[w]]].value;
                groups[ng] = w;
                ng += 1;
            }
        }
        groups[..ng].sort_unstable_by_key(|&w| std::cmp::Reverse(maxima[w]));
        for w in 1..21 {
            maxima[w] = maxima[w].max(maxima[w - 1]);
        }
        let mut ri: Vec<usize> = removals
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                let budget = slack + r.weight;
                budget >= 0 && maxima[budget.min(20) as usize] > r.value
            })
            .map(|(i, _)| i)
            .collect();
        ri.sort_unstable_by(|&i, &j| {
            removals[i]
                .value
                .cmp(&removals[j].value)
                .then(removals[i].ordinal.cmp(&removals[j].ordinal))
        });
        let mut best = 0i64;
        let mut order = (usize::MAX, usize::MAX);
        let mut movement = None;
        for rindex in ri {
            let r = removals[rindex];
            if maxima[20] - r.value < best {
                break;
            }
            let limit = (slack + r.weight).min(20) as usize;
            if maxima[limit] - r.value < best {
                continue;
            }
            for &w in &groups[..ng] {
                let indices = &ai[starts[w]..starts[w + 1]];
                if additions[indices[0]].value - r.value < best {
                    break;
                }
                if w > limit {
                    continue;
                }
                for &aindex in indices {
                    let a = additions[aindex];
                    let bound = a.value - r.value;
                    if bound < best {
                        break;
                    }
                    let at = (r.ordinal, a.ordinal);
                    if bound <= 0 || (bound == best && at >= order) {
                        continue;
                    }
                    let delta =
                        bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
                    if delta > 0 && (delta > best || (delta == best && at < order)) {
                        best = delta;
                        order = at;
                        movement = Some((r, a));
                    }
                }
            }
        }
        movement
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_paths {
    use std::rc::Rc;
    pub(super) struct Snapshot {
        pub bits: Vec<bool>,
        pub contrib: Vec<i32>,
        pub value: i64,
        pub weight: u32,
        pub hash: u64,
    }
    impl Snapshot {
        pub(super) fn new(bits: &[bool], contrib: &[i32], value: i64, weight: u32, hash: u64) -> Self {
            Self {
                bits: bits.to_vec(),
                contrib: contrib.to_vec(),
                value,
                weight,
                hash,
            }
        }
    }
    struct Entry {
        parameter: usize,
        input: Snapshot,
        output: Rc<Snapshot>,
        steps: usize,
    }
    pub(super) struct Paths {
        slots: Vec<Option<Entry>>,
    }
    impl Paths {
        pub(super) fn new() -> Self {
            Self { slots: Vec::new() }
        }
        pub(super) fn clear(&mut self) {
            self.slots = Vec::new();
        }
        fn index(hash: u64, parameter: usize) -> usize {
            ((hash ^ (parameter as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_mul(0xbf58476d1ce4e5b9)
                >> 53) as usize
        }
        pub(super) fn lookup(
            &self,
            parameter: usize,
            hash: u64,
            bits: &[bool],
            contrib: &[i32],
            value: i64,
            weight: u32,
            remaining: usize,
        ) -> Option<(Rc<Snapshot>, usize)> {
            let entry = self.slots.get(Self::index(hash, parameter))?.as_ref()?;
            let s = &entry.input;
            if entry.parameter == parameter
                && entry.steps <= remaining
                && s.hash == hash
                && s.value == value
                && s.weight == weight
                && s.bits == bits
                && s.contrib == contrib
            {
                Some((Rc::clone(&entry.output), entry.steps))
            } else {
                None
            }
        }
        pub(super) fn publish(
            &mut self,
            parameter: usize,
            pending: Vec<(Snapshot, usize)>,
            end: usize,
            output: Rc<Snapshot>,
        ) {
            if self.slots.is_empty() {
                self.slots.resize_with(2048, || None);
            }
            for (input, start) in pending {
                let index = Self::index(input.hash, parameter);
                self.slots[index] = Some(Entry {
                    parameter,
                    input,
                    output: Rc::clone(&output),
                    steps: end - start,
                });
            }
        }
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_single {
    pub(super) fn best<F: Fn(usize, usize) -> i32>(
        unused: &[usize],
        used: &[usize],
        weights: &[u32],
        contrib: &[i32],
        slack: u32,
        cross: F,
    ) -> Option<(usize, usize)> {
        if unused.len() > 256 {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        if unused
            .iter()
            .chain(used)
            .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
        {
            let mut best = 0i32;
            let mut movement = None;
            for &r in used {
                for &a in unused {
                    if weights[a] > weights[r] + slack {
                        continue;
                    }
                    let delta = contrib[a]
                        .wrapping_sub(contrib[r])
                        .wrapping_sub(cross(r, a));
                    if delta > best {
                        best = delta;
                        movement = Some((a, r));
                    }
                }
            }
            return movement;
        }
        let mut max_add = [i32::MIN; 11];
        let mut min_remove = [i32::MAX; 12];
        for &a in unused {
            max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
        }
        for &r in used {
            min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
        }
        for w in 1..=10 {
            max_add[w] = max_add[w].max(max_add[w - 1]);
        }
        for w in (1..=10).rev() {
            min_remove[w] = min_remove[w].min(min_remove[w + 1]);
        }
        if !used
            .iter()
            .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
        {
            return None;
        }
        let mut storage = [0u64; 11 * 256];
        let mut counts = [0usize; 11];
        for (ordinal, &a) in unused.iter().enumerate() {
            let w = weights[a] as usize;
            let lower = weights[a].saturating_sub(slack).max(1) as usize;
            if contrib[a] <= min_remove[lower] {
                continue;
            }
            let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
            let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
            storage[w * 256 + counts[w]] = entry;
            counts[w] += 1;
        }
        for w in 1..=10 {
            let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
            bucket.sort_unstable_by(|a, b| b.cmp(a));
        }
        let mut buckets: [&[u64]; 11] = [&[]; 11];
        for w in 0..11 {
            let __lo = w * 256;
            let __n = counts[w];
            buckets[w] = &storage[__lo..__lo + __n];
        }
        let buckets = buckets;
        let mut best = 0i64;
        let mut best_order = (usize::MAX, usize::MAX);
        let mut movement = None;
        for (ri, &r) in used.iter().enumerate() {
            let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
            if max_add[limit] as i64 - contrib[r] as i64 <= best {
                continue;
            }
            for bucket in &buckets[1..=limit] {
                for &entry in bucket.iter() {
                    let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                    let bound = value as i64 - contrib[r] as i64;
                    if bound < best || bound <= 0 {
                        break;
                    }
                    let ai = u32::MAX as usize - entry as u32 as usize;
                    let order = (ri, ai);
                    if bound == best && order >= best_order {
                        continue;
                    }
                    let a = unused[ai];
                    let delta = bound - cross(r, a) as i64;
                    if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                        best = delta;
                        best_order = order;
                        movement = Some((a, r));
                    }
                }
            }
        }
        movement
    }

    pub(super) fn best_heap<F: Fn(usize, usize) -> i32>(
        unused: &[usize],
        used: &[usize],
        weights: &[u32],
        contrib: &[i32],
        slack: u32,
        cross: F,
    ) -> Option<(usize, usize)> {
        if unused
            .iter()
            .chain(used)
            .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
        {
            let mut best = 0i32;
            let mut movement = None;
            for &r in used {
                for &a in unused {
                    if weights[a] > weights[r] + slack {
                        continue;
                    }
                    let delta = contrib[a]
                        .wrapping_sub(contrib[r])
                        .wrapping_sub(cross(r, a));
                    if delta > best {
                        best = delta;
                        movement = Some((a, r));
                    }
                }
            }
            return movement;
        }
        let mut max_add = [i32::MIN; 11];
        let mut min_remove = [i32::MAX; 12];
        for &a in unused {
            max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
        }
        for &r in used {
            min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
        }
        for w in 1..=10 {
            max_add[w] = max_add[w].max(max_add[w - 1]);
        }
        for w in (1..=10).rev() {
            min_remove[w] = min_remove[w].min(min_remove[w + 1]);
        }
        if !used
            .iter()
            .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
        {
            return None;
        }
        let mut buckets: [Vec<u64>; 11] = Default::default();
        for (ordinal, &a) in unused.iter().enumerate() {
            let w = weights[a] as usize;
            let lower = weights[a].saturating_sub(slack).max(1) as usize;
            if contrib[a] <= min_remove[lower] {
                continue;
            }
            let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
            buckets[w].push((c << 32) | (u32::MAX as usize - ordinal) as u64);
        }
        for bucket in &mut buckets {
            bucket.sort_unstable_by(|a, b| b.cmp(a));
        }
        let mut best = 0i64;
        let mut best_order = (usize::MAX, usize::MAX);
        let mut movement = None;
        for (ri, &r) in used.iter().enumerate() {
            let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
            if max_add[limit] as i64 - contrib[r] as i64 <= best {
                continue;
            }
            for bucket in &buckets[1..=limit] {
                for &entry in bucket {
                    let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                    let bound = value as i64 - contrib[r] as i64;
                    if bound < best || bound <= 0 {
                        break;
                    }
                    let ai = u32::MAX as usize - entry as u32 as usize;
                    let order = (ri, ai);
                    if bound == best && order >= best_order {
                        continue;
                    }
                    let a = unused[ai];
                    let delta = bound - cross(r, a) as i64;
                    if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                        best = delta;
                        best_order = order;
                        movement = Some((a, r));
                    }
                }
            }
        }
        movement
    }

    pub(super) fn best_fused<F: Fn(usize, usize) -> i32>(
        unused: &[usize],
        used: &[usize],
        weights: &[u32],
        contrib: &[i32],
        slack: u32,
        cross: F,
    ) -> Option<(usize, usize)> {
        if unused.len() > 256 {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        let mut max_add = [i32::MIN; 11];
        let mut min_remove = [i32::MAX; 12];
        let mut bad = false;
        let mut ga = [[i32::MIN; 11]; 4];
        let mut gr = [[i32::MAX; 12]; 4];
        macro_rules! feed_add {
            ($lane:expr, $id:expr) => {{
                let id = $id;
                let w = weights[id];
                let c = contrib[id];
                bad |= !(1..=10).contains(&w) | ((c as i64).abs() >= (1i64 << 28));
                let wi = (w as usize).clamp(1, 10);
                let slot = &mut ga[$lane][wi];
                if c > *slot {
                    *slot = c;
                }
            }};
        }
        macro_rules! feed_rm {
            ($lane:expr, $id:expr) => {{
                let id = $id;
                let w = weights[id];
                let c = contrib[id];
                bad |= !(1..=10).contains(&w) | ((c as i64).abs() >= (1i64 << 28));
                let wi = (w as usize).clamp(1, 10);
                let slot = &mut gr[$lane][wi];
                if c < *slot {
                    *slot = c;
                }
            }};
        }
        let mut p = 0usize;
        let fu = unused.len() / 4 * 4;
        while p < fu {
            feed_add!(0, unused[p]);
            feed_add!(1, unused[p + 1]);
            feed_add!(2, unused[p + 2]);
            feed_add!(3, unused[p + 3]);
            p += 4;
        }
        while p < unused.len() {
            feed_add!(0, unused[p]);
            p += 1;
        }
        let mut p = 0usize;
        let fr = used.len() / 4 * 4;
        while p < fr {
            feed_rm!(0, used[p]);
            feed_rm!(1, used[p + 1]);
            feed_rm!(2, used[p + 2]);
            feed_rm!(3, used[p + 3]);
            p += 4;
        }
        while p < used.len() {
            feed_rm!(0, used[p]);
            p += 1;
        }
        if bad {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        for w in 1..=10 {
            max_add[w] = ga[0][w].max(ga[1][w]).max(ga[2][w]).max(ga[3][w]);
            min_remove[w] = gr[0][w].min(gr[1][w]).min(gr[2][w]).min(gr[3][w]);
        }

        let weight = |i: usize| unsafe { *weights.get_unchecked(i) };
        let contribution = |i: usize| unsafe { *contrib.get_unchecked(i) };
        for w in 1..=10 {
            max_add[w] = max_add[w].max(max_add[w - 1]);
        }
        for w in (1..=10).rev() {
            min_remove[w] = min_remove[w].min(min_remove[w + 1]);
        }
        if !used
            .iter()
            .any(|&r| max_add[(weight(r) as u64 + slack as u64).min(10) as usize] > contribution(r))
        {
            return None;
        }
        let mut storage = [0u64; 11 * 256];
        let mut counts = [0usize; 11];
        for (ordinal, &a) in unused.iter().enumerate() {
            let w = weight(a) as usize;
            let lower = weight(a).saturating_sub(slack).max(1) as usize;
            if contribution(a) <= unsafe { *min_remove.get_unchecked(lower) } {
                continue;
            }
            let c = ((contribution(a) as u32) ^ 0x80000000) as u64;
            let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
            let __i = unsafe { *counts.get_unchecked(w) };
            storage[w * 256 + __i] = entry;
            unsafe {
                *counts.get_unchecked_mut(w) += 1;
            }
        }
        for w in 1..=10 {
            let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
            bucket.sort_unstable_by(|a, b| b.cmp(a));
        }
        let mut buckets: [&[u64]; 11] = [&[]; 11];
        for w in 0..11 {
            let __lo = w * 256;
            let __n = counts[w];
            buckets[w] = &storage[__lo..__lo + __n];
        }
        let buckets = buckets;
        let mut best = 0i64;
        let mut best_order = (usize::MAX, usize::MAX);
        let mut movement = None;
        for (ri, &r) in used.iter().enumerate() {
            let limit = (weight(r) as u64 + slack as u64).min(10) as usize;
            if max_add[limit] as i64 - contribution(r) as i64 <= best {
                continue;
            }
            for bucket in &buckets[1..=limit] {
                for &entry in bucket.iter() {
                    let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                    let bound = value as i64 - contribution(r) as i64;
                    if bound < best || bound <= 0 {
                        break;
                    }
                    let ai = u32::MAX as usize - entry as u32 as usize;
                    let order = (ri, ai);
                    if bound == best && order >= best_order {
                        continue;
                    }
                    let a = unsafe { *unused.get_unchecked(ai) };
                    let delta = bound - cross(r, a) as i64;
                    if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                        best = delta;
                        best_order = order;
                        movement = Some((a, r));
                    }
                }
            }
        }
        movement
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_sort {

    #[inline(always)]
    fn compare_swap<F: Fn(usize, usize) -> bool>(v: &mut [usize], a: usize, b: usize, less: &F) {
        unsafe {
            let x = *v.get_unchecked(a);
            let y = *v.get_unchecked(b);
            let swap = less(y, x);
            *v.get_unchecked_mut(a) = if swap { y } else { x };
            *v.get_unchecked_mut(b) = if swap { x } else { y };
        }
    }

    pub(super) fn pack_density(out: &mut [usize], contrib: &[i32], weights: &[u32]) {
        const FACTOR: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        assert_eq!(out.len(), contrib.len());
        assert_eq!(out.len(), weights.len());
        let n = out.len();
        let mut i = 0;
        unsafe {
            let p = out.as_mut_ptr();
            let c = contrib.as_ptr();
            let w = weights.as_ptr();
            macro_rules! emit {
                ($i:expr) => {{
                    let i = $i;
                    let factor = FACTOR[(*w.add(i) as usize).clamp(1, 10)];
                    *p.add(i) = (((*c.add(i) as i64 * factor) << 16) as usize) | i;
                }};
            }
            let full = n / 16 * 16;
            while i < full {
                emit!(i);
                emit!(i + 1);
                emit!(i + 2);
                emit!(i + 3);
                emit!(i + 4);
                emit!(i + 5);
                emit!(i + 6);
                emit!(i + 7);
                emit!(i + 8);
                emit!(i + 9);
                emit!(i + 10);
                emit!(i + 11);
                emit!(i + 12);
                emit!(i + 13);
                emit!(i + 14);
                emit!(i + 15);
                i += 16;
            }
            while i < n {
                emit!(i);
                i += 1;
            }
        }
    }

    pub(super) fn density_order(
        ids: &mut [usize],
        contrib: &[i32],
        weights: &[u32],
        small_weights: bool,
    ) {
        if usize::BITS == 64 && weights.len() <= u16::MAX as usize && small_weights {
            const F: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
            for item in ids.iter_mut() {
                let i = *item;
                *item = (((contrib[i] as i64 * F[weights[i] as usize]) << 16) as usize) | i;
            }
            if compatible() {
                sort(ids, |a, b| ((a as i64) >> 16) > ((b as i64) >> 16));
            } else {
                ids.sort_unstable_by(|&a, &b| ((b as i64) >> 16).cmp(&((a as i64) >> 16)));
            }
            for item in ids {
                *item &= 0xffff;
            }
        } else {
            ids.sort_unstable_by(|&a, &b| {
                ((contrib[b] as i64) * (weights[a] as i64).max(1))
                    .cmp(&((contrib[a] as i64) * (weights[b] as i64).max(1)))
            });
        }
    }

    pub(super) fn keyed_prefix(
        ids: &mut [usize],
        keys: &[i64],
        weights: &[u32],
        capacity: u32,
        weight_first: bool,
    ) -> usize {
        if usize::BITS == 64 && weights.len() <= u16::MAX as usize {
            let mut fits = true;
            for item in ids.iter_mut() {
                let i = *item;
                let mut key = keys[i];
                if weight_first {
                    fits &= weights[i] <= 1024 && (-(1i64 << 31)..=(1i64 << 31)).contains(&key);
                    key = (weights[i] as i64)
                        .wrapping_mul(1i64 << 33)
                        .wrapping_add(key);
                }
                fits &= (-(1i64 << 47)..(1i64 << 47)).contains(&key);
                *item = ((key << 16) as usize) | i;
            }
            if fits {
                let end = capacity_prefix(
                    ids,
                    |a, b| ((a as i64) >> 16) < ((b as i64) >> 16),
                    |key| weights[key & 0xffff],
                    capacity,
                    0,
                );
                for item in ids {
                    *item &= 0xffff;
                }
                return end;
            }
            for item in ids.iter_mut() {
                *item &= 0xffff;
            }
        }
        capacity_prefix(
            ids,
            |a, b| {
                if weight_first {
                    weights[a]
                        .cmp(&weights[b])
                        .then(keys[a].cmp(&keys[b]))
                        .is_lt()
                } else {
                    keys[a] < keys[b]
                }
            },
            |i| weights[i],
            capacity,
            0,
        )
    }

    fn insertion<F: Fn(usize, usize) -> bool>(v: &mut [usize], start: usize, less: &F) {
        for i in start..v.len() {
            let x = v[i];
            let mut j = i;
            while j > 0 && less(x, v[j - 1]) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    }

    fn small<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
        let n = v.len();
        if n < 2 {
            return;
        }
        let middle = if n < 18 { n } else { n / 2 };
        for range in [0..middle, middle..n] {
            let part = &mut v[range];
            let sorted = if part.len() >= 13 {
                sort13_optimal(part, less);
                13
            } else if part.len() >= 9 {
                sort9_optimal(part, less);
                9
            } else {
                1
            };
            insertion(part, sorted, less);
        }
        if middle == n {
            return;
        }
        let mut scratch = [0usize; 32];
        let (mut a, mut b) = (0, middle);
        for dest in &mut scratch[..n] {
            if b == n || (a < middle && !less(v[b], v[a])) {
                *dest = v[a];
                a += 1;
            } else {
                *dest = v[b];
                b += 1;
            }
        }
        v.copy_from_slice(&scratch[..n]);
    }

    #[inline(always)]
    fn median<F: Fn(usize, usize) -> bool>(
        v: &[usize],
        a: usize,
        b: usize,
        c: usize,
        less: &F,
    ) -> usize {
        let x = less(v[a], v[b]);
        let y = less(v[a], v[c]);
        if x == y {
            if less(v[b], v[c]) ^ x {
                c
            } else {
                b
            }
        } else {
            a
        }
    }
    fn median_rec<F: Fn(usize, usize) -> bool>(
        v: &[usize],
        mut a: usize,
        mut b: usize,
        mut c: usize,
        n: usize,
        less: &F,
    ) -> usize {
        if n * 8 >= 64 {
            let k = n / 8;
            a = median_rec(v, a, a + 4 * k, a + 7 * k, k, less);
            b = median_rec(v, b, b + 4 * k, b + 7 * k, k, less);
            c = median_rec(v, c, c + 4 * k, c + 7 * k, k, less);
        }
        median(v, a, b, c, less)
    }
    fn pivot<F: Fn(usize, usize) -> bool>(v: &[usize], less: &F) -> usize {
        let k = v.len() / 8;
        if v.len() < 64 {
            median(v, 0, k * 4, k * 7, less)
        } else {
            median_rec(v, 0, k * 4, k * 7, k, less)
        }
    }

    fn partition<F: Fn(usize, usize) -> bool, const EQUAL: bool>(
        v: &mut [usize],
        pivot: usize,
        less: &F,
    ) -> usize {
        v.swap(0, pivot);
        let value = v[0];
        let n = v.len() - 1;
        if n == 0 {
            return 0;
        }
        unsafe {
            let p = v.as_mut_ptr().add(1);
            let saved = *p;
            let mut count = 0usize;
            let mut gap = 0usize;
            let mut right = 1usize;
            macro_rules! step {
                ($x:expr,$to:expr) => {{
                    let x = $x;
                    let yes = if EQUAL {
                        !less(value, x)
                    } else {
                        less(x, value)
                    };
                    *p.add(gap) = *p.add(count);
                    *p.add(count) = x;
                    gap = $to;
                    count += yes as usize;
                }};
            }
            macro_rules! batch4 {
                () => {{
                    let x0 = *p.add(right);
                    let x1 = *p.add(right + 1);
                    let x2 = *p.add(right + 2);
                    let x3 = *p.add(right + 3);
                    let y0 = if EQUAL {
                        !less(value, x0)
                    } else {
                        less(x0, value)
                    };
                    let y1 = if EQUAL {
                        !less(value, x1)
                    } else {
                        less(x1, value)
                    };
                    let y2 = if EQUAL {
                        !less(value, x2)
                    } else {
                        less(x2, value)
                    };
                    let y3 = if EQUAL {
                        !less(value, x3)
                    } else {
                        less(x3, value)
                    };
                    macro_rules! commit {
                        ($x:expr,$yes:expr) => {{
                            *p.add(gap) = *p.add(count);
                            *p.add(count) = $x;
                            gap = right;
                            count += $yes as usize;
                            right += 1;
                        }};
                    }
                    commit!(x0, y0);
                    commit!(x1, y1);
                    commit!(x2, y2);
                    commit!(x3, y3);
                }};
            }
            while right + 8 <= n {
                batch4!();
                batch4!();
            }
            while right < n {
                step!(*p.add(right), right);
                right += 1;
            }
            step!(saved, 0);
            let _ = gap;
            v.swap(0, count);
            count
        }
    }

    fn heap<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
        let n = v.len();
        for i in (0..n + n / 2).rev() {
            let mut node = if i >= n {
                i - n
            } else {
                v.swap(0, i);
                0
            };
            let len = i.min(n);
            loop {
                let mut child = node * 2 + 1;
                if child >= len {
                    break;
                }
                if child + 1 < len {
                    child += less(v[child], v[child + 1]) as usize;
                }
                if !less(v[node], v[child]) {
                    break;
                }
                v.swap(node, child);
                node = child;
            }
        }
    }

    fn quick<F, S>(
        mut v: &mut [usize],
        mut ancestor: Option<usize>,
        mut limit: u32,
        mut offset: usize,
        less: &F,
        sink: &mut S,
    ) -> bool
    where
        F: Fn(usize, usize) -> bool,
        S: FnMut(&[usize], usize) -> bool,
    {
        loop {
            if v.len() <= 32 {
                small(v, less);
                return sink(v, offset);
            }
            if limit == 0 {
                heap(v, less);
                return sink(v, offset);
            }
            limit -= 1;
            let pos = pivot(v, less);
            if ancestor.map_or(false, |a| !less(a, v[pos])) {
                let count = partition::<F, true>(v, pos, less) + 1;
                if !sink(&v[..count], offset) {
                    return false;
                }
                offset += count;
                v = &mut v[count..];
                ancestor = None;
                continue;
            }
            let count = partition::<F, false>(v, pos, less);
            let (left, right) = v.split_at_mut(count);
            let (p, right) = right.split_at_mut(1);
            if !quick(left, ancestor, limit, offset, less, sink) {
                return false;
            }
            if !sink(p, offset + count) {
                return false;
            }
            ancestor = Some(p[0]);
            offset += count + 1;
            v = right;
        }
    }

    pub(super) fn sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: F) {
        visit(v, &less, &mut |_, _| true);
    }

    fn visit<F, S>(v: &mut [usize], less: &F, sink: &mut S)
    where
        F: Fn(usize, usize) -> bool,
        S: FnMut(&[usize], usize) -> bool,
    {
        let n = v.len();
        if n <= 20 {
            insertion(v, 1, less);
            sink(v, 0);
            return;
        }
        let descending = less(v[1], v[0]);
        let mut run = 2;
        while run < n
            && if descending {
                less(v[run], v[run - 1])
            } else {
                !less(v[run], v[run - 1])
            }
        {
            run += 1;
        }
        if run == n {
            if descending {
                v.reverse();
            }
            sink(v, 0);
            return;
        }
        quick(v, None, 2 * (n | 1).ilog2(), 0, less, sink);
    }

    pub(super) fn capacity_prefix<F, W>(
        v: &mut [usize],
        less: F,
        weight: W,
        capacity: u32,
        pad: usize,
    ) -> usize
    where
        F: Fn(usize, usize) -> bool,
        W: Fn(usize) -> u32,
    {
        let n = v.len();
        let mut rem = capacity;
        let mut goal = n;
        if rem == 0 {
            goal = pad.min(n);
        }
        visit(v, &less, &mut |part, offset| {
            for (i, &item) in part.iter().enumerate() {
                let index = offset + i;
                if index >= goal {
                    return false;
                }
                if rem > 0 {
                    let w = weight(item);
                    if w <= rem {
                        rem -= w;
                        if rem == 0 {
                            goal = (index + pad + 1).min(n);
                        }
                    }
                }
            }
            offset + part.len() < goal
        });
        goal
    }

    pub(super) fn capacity_core<F, W>(
        v: &mut [usize],
        less: F,
        weight: W,
        capacity: u32,
        half: usize,
        max_weight: u32,
    ) where
        F: Fn(usize, usize) -> bool,
        W: Fn(usize) -> u32,
    {
        let n = v.len();
        let mut scan = CoreScan {
            rem: capacity as u64,
            goal: if capacity == 0 {
                half.saturating_add(1).min(n)
            } else {
                n
            },
            half,
            reserve: (half as u64 + 1).saturating_mul(max_weight as u64),
            n,
        };
        if n <= 20 {
            insertion(v, 1, &less);
            return;
        }
        let descending = less(v[1], v[0]);
        let mut run = 2;
        while run < n
            && if descending {
                less(v[run], v[run - 1])
            } else {
                !less(v[run], v[run - 1])
            }
        {
            run += 1;
        }
        if run == n {
            if descending {
                v.reverse();
            }
            return;
        }
        core_quick(v, None, 2 * (n | 1).ilog2(), 0, &less, &weight, &mut scan);
    }
    struct CoreScan {
        rem: u64,
        goal: usize,
        half: usize,
        reserve: u64,
        n: usize,
    }
    impl CoreScan {
        fn consume<W: Fn(usize) -> u32>(&mut self, part: &[usize], offset: usize, weight: &W) -> bool {
            for (i, &item) in part.iter().enumerate() {
                let index = offset + i;
                if index >= self.goal {
                    return false;
                }
                if self.rem > 0 {
                    let w = weight(item) as u64;
                    if w <= self.rem {
                        self.rem -= w;
                        if self.rem == 0 {
                            self.goal = (index + self.half + 1).min(self.n);
                        }
                    }
                }
            }
            offset + part.len() < self.goal
        }
    }
    fn core_quick<F, W>(
        mut v: &mut [usize],
        mut ancestor: Option<usize>,
        mut limit: u32,
        mut offset: usize,
        less: &F,
        weight: &W,
        scan: &mut CoreScan,
    ) -> bool
    where
        F: Fn(usize, usize) -> bool,
        W: Fn(usize) -> u32,
    {
        loop {
            if offset >= scan.goal {
                return false;
            }
            if scan.rem >= scan.reserve
                && !v.is_empty()
                && offset + v.len() <= scan.n.saturating_sub(scan.half.saturating_add(1))
            {
                let total: u64 = v.iter().map(|&i| weight(i) as u64).sum();
                if total <= scan.rem - scan.reserve {
                    scan.rem -= total;
                    return true;
                }
            }
            if v.len() <= 32 {
                small(v, less);
                return scan.consume(v, offset, weight);
            }
            if limit == 0 {
                heap(v, less);
                return scan.consume(v, offset, weight);
            }
            limit -= 1;
            let pos = pivot(v, less);
            if ancestor.map_or(false, |a| !less(a, v[pos])) {
                let count = partition::<F, true>(v, pos, less) + 1;
                if !scan.consume(&v[..count], offset, weight) {
                    return false;
                }
                offset += count;
                v = &mut v[count..];
                ancestor = None;
                continue;
            }
            let count = partition::<F, false>(v, pos, less);
            let (left, right) = v.split_at_mut(count);
            let (p, right) = right.split_at_mut(1);
            if !core_quick(left, ancestor, limit, offset, less, weight, scan) {
                return false;
            }
            if !scan.consume(p, offset + count, weight) {
                return false;
            }
            ancestor = Some(p[0]);
            offset += count + 1;
            v = right;
        }
    }

    fn sort9_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
        compare_swap(v, 0, 3, less);
        compare_swap(v, 1, 7, less);
        compare_swap(v, 2, 5, less);
        compare_swap(v, 4, 8, less);
        compare_swap(v, 0, 7, less);
        compare_swap(v, 2, 4, less);
        compare_swap(v, 3, 8, less);
        compare_swap(v, 5, 6, less);
        compare_swap(v, 0, 2, less);
        compare_swap(v, 1, 3, less);
        compare_swap(v, 4, 5, less);
        compare_swap(v, 7, 8, less);
        compare_swap(v, 1, 4, less);
        compare_swap(v, 3, 6, less);
        compare_swap(v, 5, 7, less);
        compare_swap(v, 0, 1, less);
        compare_swap(v, 2, 4, less);
        compare_swap(v, 3, 5, less);
        compare_swap(v, 6, 8, less);
        compare_swap(v, 2, 3, less);
        compare_swap(v, 4, 5, less);
        compare_swap(v, 6, 7, less);
        compare_swap(v, 1, 2, less);
        compare_swap(v, 3, 4, less);
        compare_swap(v, 5, 6, less);
    }

    fn sort13_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
        compare_swap(v, 0, 12, less);
        compare_swap(v, 1, 10, less);
        compare_swap(v, 2, 9, less);
        compare_swap(v, 3, 7, less);
        compare_swap(v, 5, 11, less);
        compare_swap(v, 6, 8, less);
        compare_swap(v, 1, 6, less);
        compare_swap(v, 2, 3, less);
        compare_swap(v, 4, 11, less);
        compare_swap(v, 7, 9, less);
        compare_swap(v, 8, 10, less);
        compare_swap(v, 0, 4, less);
        compare_swap(v, 1, 2, less);
        compare_swap(v, 3, 6, less);
        compare_swap(v, 7, 8, less);
        compare_swap(v, 9, 10, less);
        compare_swap(v, 11, 12, less);
        compare_swap(v, 4, 6, less);
        compare_swap(v, 5, 9, less);
        compare_swap(v, 8, 11, less);
        compare_swap(v, 10, 12, less);
        compare_swap(v, 0, 5, less);
        compare_swap(v, 3, 8, less);
        compare_swap(v, 4, 7, less);
        compare_swap(v, 6, 11, less);
        compare_swap(v, 9, 10, less);
        compare_swap(v, 0, 1, less);
        compare_swap(v, 2, 5, less);
        compare_swap(v, 6, 9, less);
        compare_swap(v, 7, 8, less);
        compare_swap(v, 10, 11, less);
        compare_swap(v, 1, 3, less);
        compare_swap(v, 2, 4, less);
        compare_swap(v, 5, 6, less);
        compare_swap(v, 9, 10, less);
        compare_swap(v, 1, 2, less);
        compare_swap(v, 3, 4, less);
        compare_swap(v, 5, 7, less);
        compare_swap(v, 6, 8, less);
        compare_swap(v, 2, 3, less);
        compare_swap(v, 4, 5, less);
        compare_swap(v, 6, 7, less);
        compare_swap(v, 8, 9, less);
        compare_swap(v, 3, 4, less);
        compare_swap(v, 5, 6, less);
    }

    fn compatible_uncached() -> bool {
            for n in [17usize, 31, 89, 257, 1001] {
                for distinct in [3usize, 19, 511] {
                    let mut a: Vec<usize> = (0..n).collect();
                    let key = |i: usize| i.wrapping_mul(7919).wrapping_add(i / 7) % distinct;
                    let mut b = a.clone();
                    a.sort_unstable_by(|&x, &y| key(y).cmp(&key(x)));
                    sort(&mut b, |x, y| key(x) > key(y));
                    if a != b {
                        return false;
                    }
                    for k in [0, n - 1, n / 3] {
                        let mut a: Vec<usize> = (0..n).collect();
                        let mut b = a.clone();
                        a.select_nth_unstable_by(k, |&x, &y| key(y).cmp(&key(x)));
                        select(&mut b, k, |x, y| key(x) > key(y));
                        if a != b {
                            return false;
                        }
                    }
                }
            }
            true
        }

    pub(super) fn compatible() -> bool {
        thread_local! { static CHECK: bool = compatible_uncached(); }
        CHECK.with(|c| *c)
    }

    pub(super) fn select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: F) {
        assert!(index < v.len());
        if index == 0 || index == v.len() - 1 {
            let p = extreme(v, index != 0, &less);
            v.swap(p, index);
            return;
        }
        let mut limit = 16;
        let mut ancestor = None;
        loop {
            if v.len() <= 16 {
                insertion(v, 1, &less);
                return;
            }
            if limit == 0 {
                median_select(v, index, &less);
                return;
            }
            limit -= 1;
            let pos = pivot(v, &less);
            if ancestor.map_or(false, |a| !less(a, v[pos])) {
                let count = partition::<F, true>(v, pos, &less) + 1;
                if count > index {
                    return;
                }
                v = &mut v[count..];
                index -= count;
                ancestor = None;
                continue;
            }
            let count = partition::<F, false>(v, pos, &less);
            if count == index {
                return;
            }
            if count < index {
                ancestor = Some(v[count]);
                v = &mut v[count + 1..];
                index -= count + 1;
            } else {
                v = &mut v[..count];
            }
        }
    }
    fn extreme<F: Fn(usize, usize) -> bool>(v: &[usize], max: bool, less: &F) -> usize {
        let mut chosen = 0;
        for i in 1..v.len() {
            if if max {
                less(v[chosen], v[i])
            } else {
                less(v[i], v[chosen])
            } {
                chosen = i;
            }
        }
        chosen
    }
    fn median_select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: &F) {
        loop {
            if v.len() <= 16 {
                insertion(v, 1, less);
                return;
            }
            if index == 0 || index == v.len() - 1 {
                let p = extreme(v, index != 0, less);
                v.swap(p, index);
                return;
            }
            let frac = if v.len() <= 1024 {
                v.len() / 12
            } else if v.len() <= 128 * 1024 {
                v.len() / 64
            } else {
                v.len() / 1024
            };
            let middle = frac / 2;
            let lo = v.len() / 2 - middle;
            let hi = frac + lo;
            let gap = (v.len() - 9 * frac) / 4;
            let mut a = lo - 4 * frac - gap;
            let mut b = hi + gap;
            for i in lo..hi {
                ninther(
                    v,
                    [a, i - frac, b, a + 1, i, b + 1, a + 2, i + frac, b + 2],
                    less,
                );
                a += 3;
                b += 3;
            }
            median_select(&mut v[lo..lo + frac], middle, less);
            let p = partition::<F, false>(v, lo + middle, less);
            if p == index {
                return;
            }
            if p > index {
                v = &mut v[..p];
            } else {
                v = &mut v[p + 1..];
                index -= p + 1;
            }
        }
    }
    fn median_index<F: Fn(usize, usize) -> bool>(
        v: &[usize],
        mut a: usize,
        b: usize,
        mut c: usize,
        less: &F,
    ) -> usize {
        if less(v[c], v[a]) {
            let __t = a; a = c; c = __t;
        }
        if less(v[c], v[b]) {
            return c;
        }
        if less(v[b], v[a]) {
            return a;
        }
        b
    }
    fn ninther<F: Fn(usize, usize) -> bool>(v: &mut [usize], positions: [usize; 9], less: &F) {
        let [a, mut b, c, mut d, e, mut f, g, mut h, i] = positions;
        b = median_index(v, a, b, c, less);
        h = median_index(v, g, h, i, less);
        if less(v[h], v[b]) {
            let __t = b; b = h; h = __t;
        }
        if less(v[f], v[d]) {
            let __t = d; d = f; f = __t;
        }
        if less(v[e], v[d]) {
        } else if less(v[f], v[e]) {
            d = f;
        } else {
            if less(v[e], v[b]) {
                v.swap(e, b);
            } else if less(v[h], v[e]) {
                v.swap(e, h);
            }
            return;
        }
        if less(v[d], v[b]) {
            d = b;
        } else if less(v[h], v[d]) {
            d = h;
        }
        v.swap(d, e);
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_terminal {
    struct Proof {
        parameter: usize,
        hash: u64,
        bits: Vec<bool>,
        contrib: Vec<i32>,
        value: i64,
        weight: u32,
    }
    pub(super) struct Terminal {
        slots: Vec<Option<Proof>>,
    }
    impl Terminal {
        pub(super) fn new() -> Self {
            Self { slots: Vec::new() }
        }
        pub(super) fn clear(&mut self) {
            self.slots = Vec::new();
        }
        fn index(hash: u64, parameter: usize) -> usize {
            ((hash ^ (parameter as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_mul(0xbf58476d1ce4e5b9)
                >> 53) as usize
        }
        pub(super) fn contains(
            &self,
            parameter: usize,
            hash: u64,
            bits: &[bool],
            contrib: &[i32],
            value: i64,
            weight: u32,
        ) -> bool {
            if let Some(Some(p)) = self.slots.get(Self::index(hash, parameter)) {
                p.parameter == parameter
                    && p.hash == hash
                    && p.value == value
                    && p.weight == weight
                    && p.bits == bits
                    && p.contrib == contrib
            } else {
                false
            }
        }
        pub(super) fn insert(
            &mut self,
            parameter: usize,
            hash: u64,
            bits: &[bool],
            contrib: &[i32],
            value: i64,
            weight: u32,
        ) {
            if self.slots.is_empty() {
                self.slots.resize_with(2048, || None);
            }
            let p = Proof {
                parameter,
                hash,
                bits: bits.to_vec(),
                contrib: contrib.to_vec(),
                value,
                weight,
            };
            self.slots[Self::index(hash, parameter)] = Some(p);
        }
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_updates {
    pub(super) fn apply(contrib: &mut [i32], q: &[Vec<i32>], ids: &[usize], add: bool) -> i64 {
        let n = contrib.len();
        let mut delta = 0i64;
        for group in ids.chunks(4) {
            for (p, &i) in group.iter().enumerate() {
                let mut c = contrib[i];
                for &prior in &group[..p] {
                    c = if add {
                        c.wrapping_add(q[prior][i])
                    } else {
                        c.wrapping_sub(q[prior][i])
                    };
                }
                delta = if add {
                    delta.wrapping_add(c as i64)
                } else {
                    delta.wrapping_sub(c as i64)
                };
            }
            if group.len() == 4 {
                let a = &q[group[0]][..n];
                let b = &q[group[1]][..n];
                let c = &q[group[2]][..n];
                let d = &q[group[3]][..n];
                if add {
                    for ((((out, &va), &vb), &vc), &vd) in contrib[..n]
                        .iter_mut()
                        .zip(a.iter())
                        .zip(b.iter())
                        .zip(c.iter())
                        .zip(d.iter())
                    {
                        *out = out
                            .wrapping_add(va)
                            .wrapping_add(vb)
                            .wrapping_add(vc)
                            .wrapping_add(vd);
                    }
                } else {
                    for ((((out, &va), &vb), &vc), &vd) in contrib[..n]
                        .iter_mut()
                        .zip(a.iter())
                        .zip(b.iter())
                        .zip(c.iter())
                        .zip(d.iter())
                    {
                        *out = out
                            .wrapping_sub(va)
                            .wrapping_sub(vb)
                            .wrapping_sub(vc)
                            .wrapping_sub(vd);
                    }
                }
            } else {
                for &id in group {
                    let row = &q[id];
                    if add {
                        for (c, &v) in contrib.iter_mut().zip(row) {
                            *c = c.wrapping_add(v);
                        }
                    } else {
                        for (c, &v) in contrib.iter_mut().zip(row) {
                            *c = c.wrapping_sub(v);
                        }
                    }
                }
            }
        }
        delta
    }
    pub(super) fn differences(
        selected: &[bool],
        target: &[u8],
        removed: &mut Vec<usize>,
        added: &mut Vec<usize>,
    ) {
        assert_eq!(selected.len(), target.len());
        let n = selected.len();
        removed.clear();
        added.clear();
        removed.reserve(n);
        added.reserve(n);
        let mut nr = 0;
        let mut na = 0;
        unsafe {
            let rp = removed.as_mut_ptr();
            let ap = added.as_mut_ptr();
            let sp = selected.as_ptr();
            let tp = target.as_ptr();
            macro_rules! emit {
                ($i:expr) => {{
                    let i = $i;
                    let s = *sp.add(i);
                    let t = *tp.add(i) != 0;
                    rp.add(nr).write(i);
                    ap.add(na).write(i);
                    nr += (s && !t) as usize;
                    na += (!s && t) as usize;
                }};
            }
            let mut i = 0;
            let full = n / 16 * 16;
            while i < full {
                emit!(i);
                emit!(i + 1);
                emit!(i + 2);
                emit!(i + 3);
                emit!(i + 4);
                emit!(i + 5);
                emit!(i + 6);
                emit!(i + 7);
                emit!(i + 8);
                emit!(i + 9);
                emit!(i + 10);
                emit!(i + 11);
                emit!(i + 12);
                emit!(i + 13);
                emit!(i + 14);
                emit!(i + 15);
                i += 16;
            }
            while i < n {
                emit!(i);
                i += 1;
            }
            removed.set_len(nr);
            added.set_len(na);
        }
    }

    pub(super) fn bool_bytes(bits: &[bool]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len()) }
    }

    pub(super) fn value_delta(
        contrib: &[i32],
        q: &[Vec<i32>],
        removed: &[usize],
        added: &[usize],
    ) -> i64 {
        let n = contrib.len();
        let mut r: Vec<i32> = removed.iter().map(|&i| contrib[i]).collect();
        let mut a: Vec<i32> = added.iter().map(|&i| contrib[i]).collect();
        for (p, &id) in removed.iter().enumerate() {
            let row = &q[id][..n];
            adjust::<false>(&mut r[p + 1..], &removed[p + 1..], row);
            adjust::<false>(&mut a, added, row);
        }
        for (p, &id) in added.iter().enumerate() {
            adjust::<true>(&mut a[p + 1..], &added[p + 1..], &q[id][..n]);
        }
        let removed_value: i64 = r.iter().map(|&c| c as i64).sum();
        let added_value: i64 = a.iter().map(|&c| c as i64).sum();
        added_value.wrapping_sub(removed_value)
    }
    fn adjust<const ADD: bool>(values: &mut [i32], ids: &[usize], row: &[i32]) {
        debug_assert_eq!(values.len(), ids.len());
        unsafe {
            let vp = values.as_mut_ptr();
            let ip = ids.as_ptr();
            let qp = row.as_ptr();
            macro_rules! visit {
                ($p:expr) => {{
                    let p = $p;
                    let old = *vp.add(p);
                    let q = *qp.add(*ip.add(p));
                    *vp.add(p) = if ADD {
                        old.wrapping_add(q)
                    } else {
                        old.wrapping_sub(q)
                    };
                }};
            }
            let mut p = 0;
            let full = ids.len() / 8 * 8;
            while p < full {
                visit!(p);
                visit!(p + 1);
                visit!(p + 2);
                visit!(p + 3);
                visit!(p + 4);
                visit!(p + 5);
                visit!(p + 6);
                visit!(p + 7);
                p += 8;
            }
            while p < ids.len() {
                visit!(p);
                p += 1;
            }
        }
    }

}

#[allow(dead_code, unused_imports, clippy::all)]
mod exact_windows {

    const SCALE: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    type Original = (usize, i64, i64);

    pub(super) struct Windows<'a> {
        weights: &'a [u32],
        packed: bool,
        unused: Vec<usize>,
        used: Vec<usize>,
        original_unused: Vec<Original>,
        original_used: Vec<Original>,
    }

    impl<'a> Windows<'a> {
        pub(super) fn new(weights: &'a [u32]) -> Self {
            let n = weights.len();
            let packed = usize::BITS == 64
                && n <= u16::MAX as usize
                && weights.iter().all(|&w| (1..=10).contains(&w));
            Self {
                weights,
                packed,
                unused: Vec::with_capacity(if packed { n } else { 0 }),
                used: Vec::with_capacity(if packed { n } else { 0 }),
                original_unused: Vec::with_capacity(if packed { 0 } else { n }),
                original_used: Vec::with_capacity(if packed { 0 } else { n }),
            }
        }

        pub(super) fn is_packed(&self) -> bool {
            self.packed
        }

        pub(super) fn build(
            &mut self,
            contrib: &[i32],
            selected: &[bool],
            k: usize,
            best_unused: &mut Vec<usize>,
            worst_used: &mut Vec<usize>,
        ) {
            let weights = self.weights;
            assert_eq!(weights.len(), contrib.len());
            assert_eq!(weights.len(), selected.len());
            best_unused.clear();
            worst_used.clear();
            if !self.packed {
                self.original_unused.clear();
                self.original_used.clear();
                for i in 0..weights.len() {
                    let score = (i, contrib[i] as i64, (weights[i] as i64).max(1));
                    if selected[i] {
                        self.original_used.push(score);
                    } else {
                        self.original_unused.push(score);
                    }
                }
                let ku = k.min(self.original_unused.len());
                let ks = k.min(self.original_used.len());
                if ku > 0 && ku < self.original_unused.len() {
                    self.original_unused
                        .select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
                }
                if ks > 0 && ks < self.original_used.len() {
                    self.original_used
                        .select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
                }
                best_unused.extend(self.original_unused[..ku].iter().map(|t| t.0));
                worst_used.extend(self.original_used[..ks].iter().map(|t| t.0));
                return;
            }

            self.unused.clear();
            self.used.clear();
            let mut nu = 0usize;
            let mut ns = 0usize;
            let unused = self.unused.as_mut_ptr();
            let used = self.used.as_mut_ptr();
            macro_rules! emit {
                ($i:expr) => {{
                    let i = $i;
                    let (c, factor, is_selected) = unsafe {
                        (
                            *contrib.get_unchecked(i) as i64,
                            *SCALE.get_unchecked(*weights.get_unchecked(i) as usize),
                            *selected.get_unchecked(i),
                        )
                    };
                    let entry = ((c * factor) << 16) as usize | i;
                    unsafe {
                        unused.add(nu).write(entry);
                        used.add(ns).write(entry);
                    }
                    ns += is_selected as usize;
                    nu += (!is_selected) as usize;
                }};
            }
            let mut i = 0usize;
            let end = weights.len() / 16 * 16;
            while i < end {
                emit!(i);
                emit!(i + 1);
                emit!(i + 2);
                emit!(i + 3);
                emit!(i + 4);
                emit!(i + 5);
                emit!(i + 6);
                emit!(i + 7);
                emit!(i + 8);
                emit!(i + 9);
                emit!(i + 10);
                emit!(i + 11);
                emit!(i + 12);
                emit!(i + 13);
                emit!(i + 14);
                emit!(i + 15);
                i += 16;
            }
            while i < weights.len() {
                emit!(i);
                i += 1;
            }
            unsafe {
                self.unused.set_len(nu);
                self.used.set_len(ns);
            }
            let ku = k.min(self.unused.len());
            let ks = k.min(self.used.len());
            if super::exact_sort::compatible() {
                if ku > 0 && ku < self.unused.len() {
                    super::exact_sort::select(&mut self.unused, ku - 1, |a, b| {
                        ((a as i64) >> 16) > ((b as i64) >> 16)
                    });
                }
                if ks > 0 && ks < self.used.len() {
                    super::exact_sort::select(&mut self.used, ks - 1, |a, b| {
                        ((a as i64) >> 16) < ((b as i64) >> 16)
                    });
                }
            } else {
                let ku = k.min(self.unused.len());
                let ks = k.min(self.used.len());
                if ku > 0 && ku < self.unused.len() {
                    self.unused.select_nth_unstable_by(ku - 1, |&a, &b| {
                        ((b as i64) >> 16).cmp(&((a as i64) >> 16))
                    });
                }
                if ks > 0 && ks < self.used.len() {
                    self.used.select_nth_unstable_by(ks - 1, |&a, &b| {
                        ((a as i64) >> 16).cmp(&((b as i64) >> 16))
                    });
                }
            }
            best_unused.extend(self.unused[..ku].iter().map(|&entry| entry & 0xffff));
            worst_used.extend(self.used[..ks].iter().map(|&entry| entry & 0xffff));
        }
    }

}
