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
use tig_challenges::vehicle_routing::*;
use anyhow::Result;
use std::time::Instant;

pub struct Evolution<'a> {
    pub data: &'a Instance,
    pub params: Config,
    pub population: GenePool<'a>,
    split_dp: Vec<i64>,
    split_pred: Vec<usize>,
    last_feasible_cost: i64,
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
            last_feasible_cost: i64::MAX,
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

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalOps) {
        self.last_feasible_cost = i64::MAX;

        let p1 = self.population.get_binary_tournament(rng);
        let mut p2 = self.population.get_binary_tournament(rng);
        while std::ptr::eq(p1, p2) {
            p2 = self.population.get_binary_tournament(rng);
        }
        let t2 = self.extract_giant_tour(&p2.routes);
        let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 };
        let target_routes = (p1.nb_routes + extra).clamp(self.data.lb_vehicles, self.data.nb_vehicles);

        let mut child_tour = self.crossover_rbx(p1, &t2, rng);
        self.mutate_tour(&mut child_tour, rng);

        let mut child_routes = self.split(&child_tour, target_routes);
        ls.runls(&mut child_routes, rng, &self.params, false, 0);
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

    fn mutate_tour(&mut self, tour: &mut Vec<usize>, rng: &mut SmallRng) {
        let _ = rng;

        let n = tour.len();
        if n < 3 {
            return;
        }

        let dist = |a: usize, b: usize| -> i64 {
            let (xa, ya) = self.data.node_positions[a];
            let (xb, yb) = self.data.node_positions[b];
            let dx = (xa as f64) - (xb as f64);
            let dy = (ya as f64) - (yb as f64);
            ((dx * dx + dy * dy).sqrt()) as i64
        };

        let mut best_i = 0usize;
        let mut best_remove_gain = i64::MIN;
        for i in 0..n {
            let a = if i == 0 { 0 } else { tour[i - 1] };
            let u = tour[i];
            let b = if i + 1 == n { 0 } else { tour[i + 1] };
            let gain = dist(a, u) + dist(u, b) - dist(a, b);
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
        let remove_delta = -(dist(a, u) + dist(u, b)) + dist(a, b);

        let mut best_ins = i;
        let mut best_delta = 0i64;

        let mut next_idx = if i == 0 { 1 } else { 0 };
        let mut c = 0usize;
        let mut d = if next_idx < n { tour[next_idx] } else { 0 };

        for ins in 0..=len {
            if ins != i {
                let delta = remove_delta - dist(c, d) + dist(c, u) + dist(u, d);
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

    pub fn split(&mut self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        let n = giant.len();
        if n == 0 {
            return Vec::new();
        }

        let k = target_routes.max(1);
        let inf = i64::MAX / 4;
        let stride = n + 1;
        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let depot = RouteEval::singleton(self.data, 0);

        if k == 2 || k == 3 {
            let mut row1 = vec![inf; stride];
            row1[0] = 0;

            let mut acc = RouteEval::join2(self.data, &depot, &RouteEval::singleton(self.data, giant[0]));
            for j in 1..=n {
                row1[j] = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                if acc.load > cap_limit {
                    break;
                }
                if j < n {
                    let next = RouteEval::singleton(self.data, giant[j]);
                    acc = RouteEval::join2(self.data, &acc, &next);
                }
            }

            let mut row2 = vec![inf; stride];
            let mut pred2 = vec![0usize; stride];
            for i in 1..n {
                let base = row1[i];
                if base >= inf {
                    continue;
                }

                let mut acc = RouteEval::join2(self.data, &depot, &RouteEval::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    if cand < row2[j] {
                        row2[j] = cand;
                        pred2[j] = i;
                    }
                    if acc.load > cap_limit {
                        break;
                    }
                    if j < n {
                        let next = RouteEval::singleton(self.data, giant[j]);
                        acc = RouteEval::join2(self.data, &acc, &next);
                    }
                }
            }

            if k == 2 {
                let mut best_k = 2usize;
                let mut best_val = row2[n];
                if best_val >= inf {
                    best_k = 1usize;
                    best_val = row1[n];
                    if row2[n] < best_val {
                        best_val = row2[n];
                        best_k = 2;
                    }
                }

                if best_val >= inf {
                    let mut routes: Vec<Vec<usize>> = Vec::with_capacity(n);
                    for &id in giant {
                        routes.push(vec![0, id, 0]);
                    }
                    return routes;
                }

                if best_k == 1 {
                    let mut r: Vec<usize> = Vec::with_capacity(n + 2);
                    r.push(0);
                    for &id in giant {
                        r.push(id);
                    }
                    r.push(0);
                    return vec![r];
                }

                let i = pred2[n];
                let mut routes: Vec<Vec<usize>> = Vec::with_capacity(2);

                let mut r1: Vec<usize> = Vec::with_capacity(i + 2);
                r1.push(0);
                for p in 0..i {
                    r1.push(giant[p]);
                }
                r1.push(0);
                routes.push(r1);

                let mut r2: Vec<usize> = Vec::with_capacity((n - i) + 2);
                r2.push(0);
                for p in i..n {
                    r2.push(giant[p]);
                }
                r2.push(0);
                routes.push(r2);

                return routes;
            }

            let mut row3 = vec![inf; stride];
            let mut pred3 = vec![0usize; stride];
            for i in 2..n {
                let base = row2[i];
                if base >= inf {
                    continue;
                }

                let mut acc = RouteEval::join2(self.data, &depot, &RouteEval::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    if cand < row3[j] {
                        row3[j] = cand;
                        pred3[j] = i;
                    }
                    if acc.load > cap_limit {
                        break;
                    }
                    if j < n {
                        let next = RouteEval::singleton(self.data, giant[j]);
                        acc = RouteEval::join2(self.data, &acc, &next);
                    }
                }
            }

            let mut best_k = 3usize;
            let mut best_val = row3[n];
            if best_val >= inf {
                best_k = 1usize;
                best_val = row1[n];
                if row2[n] < best_val {
                    best_val = row2[n];
                    best_k = 2;
                }
                if row3[n] < best_val {
                    best_val = row3[n];
                    best_k = 3;
                }
            }

            if best_val >= inf {
                let mut routes: Vec<Vec<usize>> = Vec::with_capacity(n);
                for &id in giant {
                    routes.push(vec![0, id, 0]);
                }
                return routes;
            }

            if best_k == 1 {
                let mut r: Vec<usize> = Vec::with_capacity(n + 2);
                r.push(0);
                for &id in giant {
                    r.push(id);
                }
                r.push(0);
                return vec![r];
            }

            if best_k == 2 {
                let i = pred2[n];
                let mut routes: Vec<Vec<usize>> = Vec::with_capacity(2);

                let mut r1: Vec<usize> = Vec::with_capacity(i + 2);
                r1.push(0);
                for p in 0..i {
                    r1.push(giant[p]);
                }
                r1.push(0);
                routes.push(r1);

                let mut r2: Vec<usize> = Vec::with_capacity((n - i) + 2);
                r2.push(0);
                for p in i..n {
                    r2.push(giant[p]);
                }
                r2.push(0);
                routes.push(r2);

                return routes;
            }

            let i2 = pred3[n];
            let i1 = pred2[i2];
            let mut routes: Vec<Vec<usize>> = Vec::with_capacity(3);

            let mut r1: Vec<usize> = Vec::with_capacity(i1 + 2);
            r1.push(0);
            for p in 0..i1 {
                r1.push(giant[p]);
            }
            r1.push(0);
            routes.push(r1);

            let mut r2: Vec<usize> = Vec::with_capacity((i2 - i1) + 2);
            r2.push(0);
            for p in i1..i2 {
                r2.push(giant[p]);
            }
            r2.push(0);
            routes.push(r2);

            let mut r3: Vec<usize> = Vec::with_capacity((n - i2) + 2);
            r3.push(0);
            for p in i2..n {
                r3.push(giant[p]);
            }
            r3.push(0);
            routes.push(r3);

            return routes;
        }

        let total = (k + 1) * stride;
        self.split_dp.resize(total, inf);
        self.split_pred.resize(total, 0);
        self.split_dp.fill(inf);
        self.split_dp[0] = 0;

        for kk in 1..=k {
            let row_start = kk * stride;
            let prev_start = row_start - stride;
            let (prev_all, curr_and_after) = self.split_dp.split_at_mut(row_start);
            let prev_row = &prev_all[prev_start..row_start];
            let curr_row = &mut curr_and_after[..stride];
            let pred_row = &mut self.split_pred[row_start..row_start + stride];

            for i in (kk - 1)..n {
                let base = prev_row[i];
                if base >= inf {
                    continue;
                }

                let mut acc = RouteEval::join2(self.data, &depot, &RouteEval::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    if cand < curr_row[j] {
                        curr_row[j] = cand;
                        pred_row[j] = i;
                    }
                    if acc.load > cap_limit {
                        break;
                    }
                    if j < n {
                        let next = RouteEval::singleton(self.data, giant[j]);
                        acc = RouteEval::join2(self.data, &acc, &next);
                    }
                }
            }
        }

        let mut best_k = k;
        let mut best_val = self.split_dp[k * stride + n];
        if best_val >= inf {
            best_k = 1usize;
            best_val = self.split_dp[stride + n];
            for kk in 2..=k {
                let val = self.split_dp[kk * stride + n];
                if val < best_val {
                    best_val = val;
                    best_k = kk;
                }
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
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            for p in i..j {
                r.push(giant[p]);
            }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
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