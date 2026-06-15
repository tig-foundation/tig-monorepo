use super::instance::Instance;
use super::config::Config;
use super::solution::Individual;
use super::gene_pool::{GenePool, Metric};
use super::builder::Builder;
use super::operators::LocalOps;
use super::route_eval::RouteEval;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::seq::SliceRandom;
use super::*;
use anyhow::Result;
use std::time::Instant;

pub struct Evolution<'a> {
    pub data: &'a Instance,
    pub params: Config,
    pub population: GenePool<'a>,
    split_dp: Vec<i64>,
    split_pred: Vec<usize>,
    split_seg_offsets: Vec<usize>,
    split_seg_costs: Vec<i64>,
    split_seg_rev: Vec<u8>,
    last_feasible_cost: i64,
    proxy_cache: Vec<i64>,
}

impl<'a> Evolution<'a> {
    pub fn new(data: &'a Instance, params: Config) -> Self {
        let population = GenePool::new(data);
        let max_total = (data.nb_vehicles.max(1) + 1) * data.nb_nodes;
        Self {
            data,
            params,
            population,
            split_dp: Vec::with_capacity(max_total),
            split_pred: Vec::with_capacity(max_total),
            split_seg_offsets: Vec::with_capacity(data.nb_nodes + 1),
            split_seg_costs: Vec::with_capacity(max_total),
            split_seg_rev: Vec::with_capacity(max_total),
            last_feasible_cost: i64::MAX,
            proxy_cache: Vec::new(),
        }
    }

    fn repair_and_maybe_add(&mut self, ls: &mut LocalOps, rng: &mut SmallRng) {
        let mut repaired_routes1: Vec<Vec<usize>> = Vec::new();
        ls.runls(&mut repaired_routes1, rng, &self.params, true, 100);
        let repaired1 = Individual::new_from_routes(self.data, &self.params, repaired_routes1);

        if repaired1.load_excess == 0 && repaired1.tw_violation == 0 {
            if repaired1.cost < self.last_feasible_cost {
                self.last_feasible_cost = repaired1.cost;
            }
            self.population.add(repaired1, &self.params);
        }
    }

    pub fn generate_initial_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalOps, randomize: bool) {
        let mut routes: Vec<Vec<usize>> = Builder::build_routes(self.data, rng, randomize);
        ls.runls(&mut routes, rng, &self.params, false, 0);
        let ind = Individual::new_from_routes(self.data, &self.params, routes);
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;

        self.population.add(ind, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    fn near_feasible_repair_budget(&self, child: &Individual) -> Option<usize> {
        if child.load_excess == 0 && child.tw_violation == 0 {
            return None;
        }

        let load = child.load_excess as i64;
        let tw = child.tw_violation as i64;
        let load_limit = ((self.data.max_capacity as i64) / 8).clamp(1, 200);
        let route_count = child.nb_routes.max(1) as i64;
        let tw_limit = (((child.distance as i64) / route_count) / 8).clamp(12, 180);

        if load <= load_limit && tw <= tw_limit && load + tw <= load_limit + tw_limit {
            Some(if self.data.nb_nodes <= 120 {
                40
            } else if self.data.nb_nodes <= 300 {
                28
            } else {
                18
            })
        } else {
            None
        }
    }

    fn educate_split_child(&self, routes: &mut Vec<Vec<usize>>, rng: &mut SmallRng, ls: &mut LocalOps) {
        let mut owned = Vec::new();
        owned.append(routes);
        let probe = Individual::new_from_routes(self.data, &self.params, owned);
        let repair_limit = self.near_feasible_repair_budget(&probe);
        *routes = probe.routes;
        if let Some(repair_limit) = repair_limit {
            ls.runls(routes, rng, &self.params, true, repair_limit);
        }
        ls.runls(routes, rng, &self.params, false, 0);
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalOps) {
        self.last_feasible_cost = i64::MAX;
        self.ensure_proxy_cache();

        let (mut child_tour, child_units_data, primary_target, secondary_target) = {
            let p1 = self.population.get_binary_tournament(rng);
            let mut p2 = self.population.get_binary_tournament(rng);
            while std::ptr::eq(p1, p2) {
                p2 = self.population.get_binary_tournament(rng);
            }

            let lb = self.data.lb_vehicles;
            let ub = self.data.nb_vehicles;
            let t2 = self.extract_giant_tour(&p2.routes);

            let (child_tour, child_units_data) = {
                let (units, customer_to_unit, block_count) = self.build_common_blocks(&p1.routes, &t2);
                if block_count > 0 {
                    let p1_unit_routes = self.compress_routes_to_units(&p1.routes, &customer_to_unit);
                    let t2_units = self.compress_tour_to_units(&t2, &customer_to_unit, &units);
                    if t2_units.len() == units.len() {
                        let mut child_units = self.crossover_rbx_units(&p1_unit_routes, &t2_units, units.len(), rng);
                        self.mutate_unit_tour(&mut child_units, &units);
                        (Vec::new(), Some((child_units, units)))
                    } else {
                        (self.crossover_rbx(p1, &t2, rng), None)
                    }
                } else {
                    (self.crossover_rbx(p1, &t2, rng), None)
                }
            };

            if self.data.nb_nodes <= 250 {
                let primary_target = p1.nb_routes.clamp(lb, ub);
                let blended_target = ((p1.nb_routes + p2.nb_routes + 1) / 2).clamp(lb, ub);
                let secondary_target = if blended_target != primary_target {
                    Some(blended_target)
                } else {
                    let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 };
                    let noisy_target = (p1.nb_routes + extra).clamp(lb, ub);
                    if noisy_target != primary_target {
                        Some(noisy_target)
                    } else {
                        None
                    }
                };
                (child_tour, child_units_data, primary_target, secondary_target)
            } else {
                let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 };
                let primary_target = (p1.nb_routes + extra).clamp(lb, ub);
                (child_tour, child_units_data, primary_target, None)
            }
        };

        let mut child_routes = if let Some((child_units, units)) = child_units_data {
            if primary_target <= child_units.len() && secondary_target.map_or(true, |t| t <= child_units.len()) {
                if let Some(secondary_target) = secondary_target {
                    let targets = [primary_target, secondary_target];
                    self.split_units_with_targets(&child_units, &units, &targets)
                } else {
                    self.split_units_with_targets(&child_units, &units, &[primary_target])
                }
            } else {
                child_tour = self.expand_unit_tour(&child_units, &units);
                if let Some(secondary_target) = secondary_target {
                    let targets = [primary_target, secondary_target];
                    self.split_with_targets(&child_tour, &targets)
                } else {
                    self.split(&child_tour, primary_target)
                }
            }
        } else {
            self.mutate_tour(&mut child_tour, rng);
            if let Some(secondary_target) = secondary_target {
                let targets = [primary_target, secondary_target];
                self.split_with_targets(&child_tour, &targets)
            } else {
                self.split(&child_tour, primary_target)
            }
        };
        self.educate_split_child(&mut child_routes, rng, ls);
        let child = Individual::new_from_routes(self.data, &self.params, child_routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        if is_capa_feasible && is_tw_feasible {
            if child.cost < self.last_feasible_cost {
                self.last_feasible_cost = child.cost;
            }
        }

        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Option<(Vec<Vec<usize>>, i32)> {
        if let Some(save) = save_solution {
            let dummy_routes: Vec<Vec<usize>> = (1..self.data.nb_nodes).map(|i| vec![0, i, 0]).collect();
            let _ = save(&Solution { routes: dummy_routes });
        }

        let mut ls = LocalOps::new(self.data, self.params);

        let diversity_boost = if self.data.nb_nodes < 1000 { 3 } else { 1 };
        let init_total = self.params.mu_start + diversity_boost;
        if init_total > 0 {
            self.generate_initial_individual(rng, &mut ls, false);
            for _ in 1..init_total {
                self.generate_initial_individual(rng, &mut ls, true);
            }
        }

        let mut best_metric: Metric = self.population.best_metric();
        let mut best_saved_feasible_cost = self.population.best_feasible().map_or(i64::MAX, |best| best.cost);
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {
            self.generate_crossover_individual(rng, &mut ls);
            let latest_feasible_cost = self.last_feasible_cost;

            if it_total % self.params.nb_it_traces == 0 {
                self.population
                    .print_trace(it_total, it_noimprov, t0.elapsed().as_secs_f64(), &self.params);
            }

            let cur = self.population.best_metric();
            if cur.better_than(best_metric) {
                best_metric = cur;
                it_noimprov = 0;

                if latest_feasible_cost < best_saved_feasible_cost {
                    if let Some(best) = self.population.best_feasible() {
                        best_saved_feasible_cost = best.cost;
                        if let Some(save) = save_solution {
                            let _ = save(&Solution { routes: best.routes });
                        }
                    }
                }
            } else {
                it_noimprov += 1;
            }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let mut best_routes = best.routes.clone();
            ls.runls(&mut best_routes, rng, &self.params, false, 0);
            let best_after = Individual::new_from_routes(self.data, &self.params, best_routes);
            let chosen = if best_after.tw_violation == 0
                && best_after.load_excess == 0
                && best_after.distance < best.distance
            {
                best_after
            } else {
                best
            };
            if let Some(save) = save_solution {
                let _ = save(&Solution { routes: chosen.routes.clone() });
            }
            Some((chosen.routes, chosen.cost as i32))
        } else {
            None
        }
    }

    fn ensure_proxy_cache(&mut self) {
        if self.proxy_cache.is_empty() {
            let n = self.data.nb_nodes;
            let mut matrix = vec![0i64; n * n];
            for i in 0..n {
                let (xa, ya) = self.data.node_positions[i];
                let row_i = i * n;
                for j in i..n {
                    let (xb, yb) = self.data.node_positions[j];
                    let dx = (xa as f64) - (xb as f64);
                    let dy = (ya as f64) - (yb as f64);
                    let d = ((dx * dx + dy * dy).sqrt()) as i64;
                    matrix[row_i + j] = d;
                    matrix[j * n + i] = d;
                }
            }
            self.proxy_cache = matrix;
        }
    }

    fn proxy_dist(&self, a: usize, b: usize) -> i64 {
        self.proxy_cache[a * self.data.nb_nodes + b]
    }

    fn build_common_blocks(&self, routes: &[Vec<usize>], t2: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>, usize) {
        let n = self.data.nb_nodes.saturating_sub(1);

        let mut succ = vec![0usize; self.data.nb_nodes];
        let mut pred = vec![0usize; self.data.nb_nodes];
        if !t2.is_empty() {
            let mut prev = t2[0];
            for idx in 1..t2.len() {
                let id = t2[idx];
                succ[prev] = id;
                pred[id] = prev;
                prev = id;
            }
        }

        let mut any_shared_adj = false;
        'adj_scan: for route in routes {
            let end = route.len().saturating_sub(1);
            if end <= 2 {
                continue;
            }
            for p in 1..(end - 1) {
                let a = route[p];
                let b = route[p + 1];
                if succ[a] == b || pred[a] == b {
                    any_shared_adj = true;
                    break 'adj_scan;
                }
            }
        }

        if !any_shared_adj {
            let mut customer_to_unit = vec![usize::MAX; self.data.nb_nodes];
            let mut units: Vec<Vec<usize>> = Vec::with_capacity(n);
            for id in 1..self.data.nb_nodes {
                customer_to_unit[id] = units.len();
                units.push(vec![id]);
            }
            return (units, customer_to_unit, 0);
        }

        let mut pos2 = vec![usize::MAX; self.data.nb_nodes];
        for (idx, &id) in t2.iter().enumerate() {
            pos2[id] = idx;
        }

        let max_blocks = if n < 80 {
            4
        } else if n < 250 {
            7
        } else {
            10
        };

        let mut candidates: Vec<(i64, Vec<usize>)> = Vec::with_capacity(n.min(max_blocks * 12));
        for route in routes {
            let end = route.len().saturating_sub(1);
            if end <= 2 {
                continue;
            }

            for start in 1..end {
                let max_len = 4.min(end - start);
                if max_len < 2 {
                    continue;
                }

                for len in (2..=max_len).rev() {
                    let slice = &route[start..start + len];
                    let mut matched = false;

                    let start_fwd = pos2[slice[0]];
                    if start_fwd != usize::MAX && start_fwd + len <= t2.len() {
                        matched = true;
                        for k in 0..len {
                            if t2[start_fwd + k] != slice[k] {
                                matched = false;
                                break;
                            }
                        }
                    }

                    if !matched {
                        let start_rev = pos2[slice[len - 1]];
                        if start_rev != usize::MAX && start_rev + len <= t2.len() {
                            matched = true;
                            for k in 0..len {
                                if t2[start_rev + k] != slice[len - 1 - k] {
                                    matched = false;
                                    break;
                                }
                            }
                        }
                    }

                    if matched {
                        let left = route[start - 1];
                        let right = if start + len == end { 0 } else { route[start + len] };
                        let mut internal = 0i64;
                        for k in 0..(len - 1) {
                            internal += self.proxy_dist(slice[k], slice[k + 1]);
                        }
                        let boundary = self.proxy_dist(left, slice[0]) + self.proxy_dist(slice[len - 1], right)
                            - self.proxy_dist(left, right);
                        let score = (len as i64) * 1_000_000 + internal + boundary.max(0);
                        candidates.push((score, slice.to_vec()));
                        break;
                    }
                }
            }
        }

        let mut customer_to_unit = vec![usize::MAX; self.data.nb_nodes];
        let mut units: Vec<Vec<usize>> = Vec::with_capacity(n);

        if !candidates.is_empty() {
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            let mut block_count = 0usize;

            for (_, block) in candidates {
                if block_count == max_blocks {
                    break;
                }
                if block.iter().any(|&id| customer_to_unit[id] != usize::MAX) {
                    continue;
                }

                let uid = units.len();
                for &id in &block {
                    customer_to_unit[id] = uid;
                }
                units.push(block);
                block_count += 1;
            }

            for id in 1..self.data.nb_nodes {
                if customer_to_unit[id] == usize::MAX {
                    customer_to_unit[id] = units.len();
                    units.push(vec![id]);
                }
            }

            return (units, customer_to_unit, block_count);
        }

        for id in 1..self.data.nb_nodes {
            customer_to_unit[id] = units.len();
            units.push(vec![id]);
        }
        (units, customer_to_unit, 0)
    }

    fn compress_routes_to_units(&self, routes: &[Vec<usize>], customer_to_unit: &[usize]) -> Vec<Vec<usize>> {
        let mut compressed: Vec<Vec<usize>> = Vec::with_capacity(routes.len());
        for route in routes {
            let end = route.len().saturating_sub(1);
            let mut unit_route: Vec<usize> = Vec::with_capacity(end.saturating_sub(1));
            let mut last = usize::MAX;
            for p in 1..end {
                let uid = customer_to_unit[route[p]];
                if uid != last {
                    unit_route.push(uid);
                    last = uid;
                }
            }
            compressed.push(unit_route);
        }
        compressed
    }

    fn compress_tour_to_units(&self, tour: &[usize], customer_to_unit: &[usize], units: &[Vec<usize>]) -> Vec<usize> {
        let mut compressed: Vec<usize> = Vec::with_capacity(units.len());
        let mut i = 0usize;

        while i < tour.len() {
            let uid = customer_to_unit[tour[i]];
            compressed.push(uid);

            let block = &units[uid];
            if block.len() > 1 && i + block.len() <= tour.len() {
                let mut forward = true;
                for k in 0..block.len() {
                    if tour[i + k] != block[k] {
                        forward = false;
                        break;
                    }
                }
                if forward {
                    i += block.len();
                    continue;
                }

                let mut reverse = true;
                for k in 0..block.len() {
                    if tour[i + k] != block[block.len() - 1 - k] {
                        reverse = false;
                        break;
                    }
                }
                if reverse {
                    i += block.len();
                    continue;
                }
            }

            i += 1;
        }

        debug_assert_eq!(compressed.len(), units.len());
        compressed
    }

    fn expand_unit_tour(&self, unit_tour: &[usize], units: &[Vec<usize>]) -> Vec<usize> {
        let mut tour: Vec<usize> = Vec::with_capacity(self.data.nb_nodes.saturating_sub(1));
        for &uid in unit_tour {
            for &id in &units[uid] {
                tour.push(id);
            }
        }
        debug_assert_eq!(tour.len(), self.data.nb_nodes.saturating_sub(1));
        tour
    }

    fn mutate_unit_tour(&self, tour: &mut Vec<usize>, units: &[Vec<usize>]) {
        let n = tour.len();
        if n < 3 {
            return;
        }

        let unit_tail = |uid: usize| -> usize {
            if uid == usize::MAX { 0 } else { *units[uid].last().unwrap() }
        };
        let unit_head = |uid: usize| -> usize {
            if uid == usize::MAX { 0 } else { units[uid][0] }
        };

        let mut best_i = 0usize;
        let mut best_remove_gain = i64::MIN;
        for i in 0..n {
            let a = if i == 0 { usize::MAX } else { tour[i - 1] };
            let u = tour[i];
            let b = if i + 1 == n { usize::MAX } else { tour[i + 1] };
            let gain = self.proxy_dist(unit_tail(a), unit_head(u))
                + self.proxy_dist(unit_tail(u), unit_head(b))
                - self.proxy_dist(unit_tail(a), unit_head(b));
            if gain > best_remove_gain {
                best_remove_gain = gain;
                best_i = i;
            }
        }

        let i = best_i;
        let u = tour[i];
        let a = if i == 0 { usize::MAX } else { tour[i - 1] };
        let b = if i + 1 == n { usize::MAX } else { tour[i + 1] };

        let len = n - 1;
        let remove_delta = -(self.proxy_dist(unit_tail(a), unit_head(u))
            + self.proxy_dist(unit_tail(u), unit_head(b)))
            + self.proxy_dist(unit_tail(a), unit_head(b));

        let mut best_ins = i;
        let mut best_delta = 0i64;

        let mut next_idx = if i == 0 { 1 } else { 0 };
        let mut c = usize::MAX;
        let mut d = if next_idx < n { tour[next_idx] } else { usize::MAX };

        for ins in 0..=len {
            if ins != i {
                let delta = remove_delta
                    - self.proxy_dist(unit_tail(c), unit_head(d))
                    + self.proxy_dist(unit_tail(c), unit_head(u))
                    + self.proxy_dist(unit_tail(u), unit_head(d));
                if delta < best_delta {
                    best_delta = delta;
                    best_ins = ins;
                }
            }
            if ins == len {
                break;
            }
            c = d;
            next_idx += 1;
            if next_idx == i {
                next_idx += 1;
            }
            d = if next_idx < n { tour[next_idx] } else { usize::MAX };
        }

        if best_delta < 0 {
            let node = tour.remove(i);
            if best_ins <= tour.len() {
                tour.insert(best_ins, node);
            } else {
                tour.push(node);
            }
        }
    }

    fn crossover_rbx_units(&self, p1_unit_routes: &[Vec<usize>], t2: &[usize], total_units: usize, rng: &mut SmallRng) -> Vec<usize> {
        if total_units == 0 {
            return Vec::new();
        }

        let mut cand: Vec<usize> = Vec::new();
        for (idx, route) in p1_unit_routes.iter().enumerate() {
            if !route.is_empty() {
                cand.push(idx);
            }
        }
        if cand.is_empty() {
            return t2.to_vec();
        }

        cand.shuffle(rng);
        let keep = rng.gen_range(1..=cand.len().min(3));
        cand.truncate(keep);

        let mut used = vec![false; total_units];
        let mut child: Vec<usize> = Vec::with_capacity(total_units);

        for &ri in &cand {
            for &uid in &p1_unit_routes[ri] {
                if !used[uid] {
                    used[uid] = true;
                    child.push(uid);
                }
            }
        }

        for &uid in t2 {
            if !used[uid] {
                used[uid] = true;
                child.push(uid);
            }
        }

        debug_assert_eq!(child.len(), total_units);
        child
    }

    fn mutate_tour(&self, tour: &mut Vec<usize>, rng: &mut SmallRng) {
        let _ = rng;

        let n = tour.len();
        if n < 3 {
            return;
        }

        let mut best_i = 0usize;
        let mut best_remove_gain = i64::MIN;
        for i in 0..n {
            let a = if i == 0 { 0 } else { tour[i - 1] };
            let u = tour[i];
            let b = if i + 1 == n { 0 } else { tour[i + 1] };
            let gain = self.proxy_dist(a, u) + self.proxy_dist(u, b) - self.proxy_dist(a, b);
            if gain > best_remove_gain {
                best_remove_gain = gain;
                best_i = i;
            }
        }

        let i = best_i;
        let u = tour[i];
        let a = if i == 0 { 0 } else { tour[i - 1] };
        let b = if i + 1 == n { 0 } else { tour[i + 1] };

        let len = n - 1;
        let remove_delta = -(self.proxy_dist(a, u) + self.proxy_dist(u, b)) + self.proxy_dist(a, b);

        let mut best_ins = i;
        let mut best_delta = 0i64;

        let mut next_idx = if i == 0 { 1 } else { 0 };
        let mut c = 0usize;
        let mut d = if next_idx < n { tour[next_idx] } else { 0 };

        for ins in 0..=len {
            if ins != i {
                let delta = remove_delta - self.proxy_dist(c, d) + self.proxy_dist(c, u) + self.proxy_dist(u, d);
                if delta < best_delta {
                    best_delta = delta;
                    best_ins = ins;
                }
            }
            if ins == len {
                break;
            }
            c = d;
            next_idx += 1;
            if next_idx == i {
                next_idx += 1;
            }
            d = if next_idx < n { tour[next_idx] } else { 0 };
        }

        if best_delta < 0 {
            let node = tour.remove(i);
            if best_ins <= tour.len() {
                tour.insert(best_ins, node);
            } else {
                tour.push(node);
            }
        }
    }
    
    fn build_split_segment_cache(&mut self, giant: &[usize], cap_limit: i32) {
        let n = giant.len();
        self.split_seg_offsets.clear();
        self.split_seg_offsets.resize(n + 1, 0);
        self.split_seg_costs.clear();
        self.split_seg_rev.clear();

        let depot = RouteEval::singleton(self.data, 0);
        let mut offset = 0usize;
        for i in 0..n {
            self.split_seg_offsets[i] = offset;

            let first = RouteEval::singleton(self.data, giant[i]);
            let mut acc_fwd = RouteEval::join2(self.data, &depot, &first);
            let mut acc_rev_customers = first;
            let mut j = i + 1;
            loop {
                let mut best_cost = RouteEval::eval2(self.data, &self.params, &acc_fwd, &depot);
                let mut rev_flag = 0u8;

                if j > i + 1 {
                    let acc_rev = RouteEval::join2(self.data, &depot, &acc_rev_customers);
                    let rev_cost = RouteEval::eval2(self.data, &self.params, &acc_rev, &depot);
                    if rev_cost < best_cost {
                        best_cost = rev_cost;
                        rev_flag = 1;
                    }
                }

                self.split_seg_costs.push(best_cost);
                self.split_seg_rev.push(rev_flag);
                offset += 1;

                if acc_fwd.load > cap_limit || j == n {
                    break;
                }

                let next = RouteEval::singleton(self.data, giant[j]);
                acc_fwd = RouteEval::join2(self.data, &acc_fwd, &next);
                acc_rev_customers = RouteEval::join2(self.data, &next, &acc_rev_customers);
                j += 1;
            }

            self.split_seg_offsets[i + 1] = offset;
        }
    }

    fn build_split_segment_cache_units(&mut self, unit_tour: &[usize], units: &[Vec<usize>], cap_limit: i32) {
        let n = unit_tour.len();
        self.split_seg_offsets.clear();
        self.split_seg_offsets.resize(n + 1, 0);
        self.split_seg_costs.clear();
        self.split_seg_rev.clear();

        let mut unit_fwd: Vec<RouteEval> = Vec::with_capacity(units.len());
        let mut unit_rev: Vec<RouteEval> = Vec::with_capacity(units.len());
        for unit in units {
            let mut fwd = RouteEval::singleton(self.data, unit[0]);
            for &id in unit.iter().skip(1) {
                let next = RouteEval::singleton(self.data, id);
                fwd = RouteEval::join2(self.data, &fwd, &next);
            }

            let mut rev = RouteEval::singleton(self.data, unit[unit.len() - 1]);
            for &id in unit.iter().rev().skip(1) {
                let next = RouteEval::singleton(self.data, id);
                rev = RouteEval::join2(self.data, &rev, &next);
            }

            unit_fwd.push(fwd);
            unit_rev.push(rev);
        }

        let depot = RouteEval::singleton(self.data, 0);
        let mut offset = 0usize;
        for i in 0..n {
            self.split_seg_offsets[i] = offset;

            let first_uid = unit_tour[i];
            let mut acc_fwd = RouteEval::join2(self.data, &depot, &unit_fwd[first_uid]);
            let mut acc_rev_customers = unit_rev[first_uid];
            let mut j = i + 1;
            loop {
                let mut best_cost = RouteEval::eval2(self.data, &self.params, &acc_fwd, &depot);
                let mut rev_flag = 0u8;

                let acc_rev = RouteEval::join2(self.data, &depot, &acc_rev_customers);
                let rev_cost = RouteEval::eval2(self.data, &self.params, &acc_rev, &depot);
                if rev_cost < best_cost {
                    best_cost = rev_cost;
                    rev_flag = 1;
                }

                self.split_seg_costs.push(best_cost);
                self.split_seg_rev.push(rev_flag);
                offset += 1;

                if acc_fwd.load > cap_limit || j == n {
                    break;
                }

                let next_uid = unit_tour[j];
                acc_fwd = RouteEval::join2(self.data, &acc_fwd, &unit_fwd[next_uid]);
                acc_rev_customers = RouteEval::join2(self.data, &unit_rev[next_uid], &acc_rev_customers);
                j += 1;
            }

            self.split_seg_offsets[i + 1] = offset;
        }
    }

    fn split_units_with_targets(&mut self, unit_tour: &Vec<usize>, units: &[Vec<usize>], targets: &[usize]) -> Vec<Vec<usize>> {
        let n = unit_tour.len();
        if n == 0 {
            return Vec::new();
        }

        let fallback_target = [1usize];
        let target_list = if targets.is_empty() {
            &fallback_target[..]
        } else {
            targets
        };

        let inf = i64::MAX / 4;
        let stride = n + 1;
        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let max_k = target_list
            .iter()
            .copied()
            .map(|t| t.max(1).min(n))
            .max()
            .unwrap_or(1);

        self.build_split_segment_cache_units(unit_tour, units, cap_limit);

        let pred_total = (max_k + 1) * stride;
        let dp_total = 2 * stride;
        self.split_dp.resize(dp_total, inf);
        self.split_pred.resize(pred_total, 0);
        self.split_dp[..dp_total].fill(inf);
        self.split_dp[0] = 0;

        let seg_offsets = &self.split_seg_offsets;
        let seg_costs = &self.split_seg_costs;
        let mut row_end_costs = vec![inf; max_k + 1];

        for kk in 1..=max_k {
            let curr_start = (kk & 1) * stride;
            let prev_start = ((kk - 1) & 1) * stride;
            let pred_row = &mut self.split_pred[kk * stride..(kk + 1) * stride];

            let (prev_row, curr_row): (&[i64], &mut [i64]) = if prev_start == 0 {
                let (prev_and_before, curr_and_after) = self.split_dp.split_at_mut(curr_start);
                (&prev_and_before[..stride], &mut curr_and_after[..stride])
            } else {
                let (curr_and_before, prev_and_after) = self.split_dp.split_at_mut(prev_start);
                (&prev_and_after[..stride], &mut curr_and_before[..stride])
            };

            curr_row.fill(inf);

            for i in (kk - 1)..n {
                let base = prev_row[i];
                if base >= inf {
                    continue;
                }

                let mut j = i + 1;
                let start = seg_offsets[i];
                let end = seg_offsets[i + 1];
                for &cost in &seg_costs[start..end] {
                    let cand = base + cost;
                    if cand < curr_row[j] {
                        curr_row[j] = cand;
                        pred_row[j] = i;
                    }
                    j += 1;
                }
            }

            row_end_costs[kk] = curr_row[n];
        }

        let mut best_k = 1usize;
        let mut best_val = inf;
        for &target in target_list {
            let capped_target = target.max(1).min(max_k);
            let mut cand_k = capped_target;
            let mut cand_val = row_end_costs[capped_target];

            if cand_val >= inf {
                cand_k = 1usize;
                cand_val = row_end_costs[1];
                for kk in 2..=capped_target {
                    let val = row_end_costs[kk];
                    if val < cand_val {
                        cand_val = val;
                        cand_k = kk;
                    }
                }
            }

            if cand_val < best_val || (cand_val == best_val && cand_k < best_k) {
                best_val = cand_val;
                best_k = cand_k;
            }
        }

        if best_val >= inf {
            let mut routes: Vec<Vec<usize>> = Vec::with_capacity(self.data.nb_nodes.saturating_sub(1));
            for &uid in unit_tour {
                for &id in &units[uid] {
                    routes.push(vec![0, id, 0]);
                }
            }
            return routes;
        }

        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(best_k);
        let mut j = n;
        for kk in (1..=best_k).rev() {
            let i = self.split_pred[kk * stride + j];
            let seg_idx = self.split_seg_offsets[i] + (j - i - 1);
            let reverse = self.split_seg_rev[seg_idx] != 0;
            let mut r: Vec<usize> = Vec::new();
            r.push(0);
            if reverse {
                for p in (i..j).rev() {
                    let uid = unit_tour[p];
                    for &id in units[uid].iter().rev() {
                        r.push(id);
                    }
                }
            } else {
                for p in i..j {
                    let uid = unit_tour[p];
                    for &id in &units[uid] {
                        r.push(id);
                    }
                }
            }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
    }

    fn split_with_targets(&mut self, giant: &Vec<usize>, targets: &[usize]) -> Vec<Vec<usize>> {
        let n = giant.len();
        if n == 0 {
            return Vec::new();
        }

        let fallback_target = [1usize];
        let target_list = if targets.is_empty() {
            &fallback_target[..]
        } else {
            targets
        };

        let inf = i64::MAX / 4;
        let stride = n + 1;
        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let max_k = target_list
            .iter()
            .copied()
            .map(|t| t.max(1).min(n))
            .max()
            .unwrap_or(1);

        self.build_split_segment_cache(giant, cap_limit);

        let pred_total = (max_k + 1) * stride;
        let dp_total = 2 * stride;
        self.split_dp.resize(dp_total, inf);
        self.split_pred.resize(pred_total, 0);
        self.split_dp[..dp_total].fill(inf);
        self.split_dp[0] = 0;

        let seg_offsets = &self.split_seg_offsets;
        let seg_costs = &self.split_seg_costs;
        let mut row_end_costs = vec![inf; max_k + 1];

        for kk in 1..=max_k {
            let curr_start = (kk & 1) * stride;
            let prev_start = ((kk - 1) & 1) * stride;
            let pred_row = &mut self.split_pred[kk * stride..(kk + 1) * stride];

            let (prev_row, curr_row): (&[i64], &mut [i64]) = if prev_start == 0 {
                let (prev_and_before, curr_and_after) = self.split_dp.split_at_mut(curr_start);
                (&prev_and_before[..stride], &mut curr_and_after[..stride])
            } else {
                let (curr_and_before, prev_and_after) = self.split_dp.split_at_mut(prev_start);
                (&prev_and_after[..stride], &mut curr_and_before[..stride])
            };

            curr_row.fill(inf);

            for i in (kk - 1)..n {
                let base = prev_row[i];
                if base >= inf {
                    continue;
                }

                let mut j = i + 1;
                let start = seg_offsets[i];
                let end = seg_offsets[i + 1];
                for &cost in &seg_costs[start..end] {
                    let cand = base + cost;
                    if cand < curr_row[j] {
                        curr_row[j] = cand;
                        pred_row[j] = i;
                    }
                    j += 1;
                }
            }

            row_end_costs[kk] = curr_row[n];
        }

        let mut best_k = 1usize;
        let mut best_val = inf;
        for &target in target_list {
            let capped_target = target.max(1).min(max_k);
            let mut cand_k = capped_target;
            let mut cand_val = row_end_costs[capped_target];

            if cand_val >= inf {
                cand_k = 1usize;
                cand_val = row_end_costs[1];
                for kk in 2..=capped_target {
                    let val = row_end_costs[kk];
                    if val < cand_val {
                        cand_val = val;
                        cand_k = kk;
                    }
                }
            }

            if cand_val < best_val || (cand_val == best_val && cand_k < best_k) {
                best_val = cand_val;
                best_k = cand_k;
            }
        }

        if best_val >= inf {
            let mut routes: Vec<Vec<usize>> = Vec::with_capacity(n);
            for &id in giant {
                routes.push(vec![0, id, 0]);
            }
            return routes;
        }

        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(best_k);
        let mut j = n;
        for kk in (1..=best_k).rev() {
            let i = self.split_pred[kk * stride + j];
            let seg_idx = self.split_seg_offsets[i] + (j - i - 1);
            let reverse = self.split_seg_rev[seg_idx] != 0;
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            if reverse {
                for p in (i..j).rev() {
                    r.push(giant[p]);
                }
            } else {
                for p in i..j {
                    r.push(giant[p]);
                }
            }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
    }

    pub fn split(&mut self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        self.split_with_targets(giant, &[target_routes])
    }

    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let depot_pos = self.data.node_positions[0];
        let x0 = depot_pos.0 as f64;
        let y0 = depot_pos.1 as f64;
        let mut route_angles: Vec<(f64, usize)> = Vec::with_capacity(routes.len());

        for r_idx in 0..routes.len() {
            let r = &routes[r_idx];
            let end = r.len().saturating_sub(1);
            if end <= 1 {
                continue;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for p in 1..end {
                let pos = self.data.node_positions[r[p]];
                sum_x += pos.0 as f64;
                sum_y += pos.1 as f64;
            }
            let cnt = (end - 1) as f64;
            let bx = sum_x / cnt;
            let by = sum_y / cnt;
            route_angles.push(((by - y0).atan2(bx - x0), r_idx));
        }

        route_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tour = Vec::with_capacity(self.data.nb_nodes - 1);
        for &(_, r_idx) in &route_angles {
            let r = &routes[r_idx];
            let end = r.len().saturating_sub(1);
            for p in 1..end {
                tour.push(r[p]);
            }
        }
        tour
    }

    fn crossover_rbx(&self, p1: &Individual, t2: &Vec<usize>, rng: &mut SmallRng) -> Vec<usize> {
        let n = self.data.nb_nodes - 1;
        if n == 0 {
            return Vec::new();
        }

        let mut cand: Vec<usize> = Vec::new();
        for (idx, r) in p1.routes.iter().enumerate() {
            if r.len() > 2 {
                cand.push(idx);
            }
        }
        if cand.is_empty() {
            return t2.clone();
        }

        cand.shuffle(rng);
        let keep = rng.gen_range(1..=cand.len().min(3));
        cand.truncate(keep);

        let mut used = vec![false; self.data.nb_nodes];
        let mut child: Vec<usize> = Vec::with_capacity(n);

        for &ri in &cand {
            let r = &p1.routes[ri];
            for &id in r.iter().skip(1).take(r.len() - 2) {
                if !used[id] {
                    used[id] = true;
                    child.push(id);
                }
            }
        }

        for &id in t2 {
            if !used[id] {
                used[id] = true;
                child.push(id);
            }
        }

        debug_assert_eq!(child.len(), n);
        child
    }
}