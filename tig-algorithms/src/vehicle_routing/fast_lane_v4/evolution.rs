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
}

impl<'a> Evolution<'a> {
    pub fn new(data: &'a Instance, params: Config) -> Self {
        let population = GenePool::new(data);
        Self {
            data,
            params,
            population,
            split_dp: Vec::new(),
            split_pred: Vec::new(),
        }
    }

    fn repair_and_maybe_add(&mut self, ls: &mut LocalOps, rng: &mut SmallRng) {
        let mut repaired_routes1: Vec<Vec<usize>> = Vec::new();
        ls.runls(&mut repaired_routes1, rng, &self.params, true, 100);
        let repaired1 = Individual::new_from_routes(self.data, &self.params, repaired_routes1);

        if repaired1.load_excess == 0 && repaired1.tw_violation == 0 {
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
        for it in 0..(self.params.mu_start + diversity_boost) {
            self.generate_initial_individual(rng, &mut ls, it > 0);
        }

        let mut best_metric: Metric = self.population.best_metric();
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {
            self.generate_crossover_individual(rng, &mut ls);

            if it_total % self.params.nb_it_traces == 0 {
                self.population
                    .print_trace(it_total, it_noimprov, t0.elapsed().as_secs_f64(), &self.params);
            }

            let cur = self.population.best_metric();
            if cur.better_than(best_metric) {
                best_metric = cur;
                it_noimprov = 0;

                if let Some(best) = self.population.best_feasible() {
                    if let Some(save) = save_solution {
                        let _ = save(&Solution { routes: best.routes });
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
        let map_after = |k: usize| -> usize { if k < i { tour[k] } else { tour[k + 1] } };

        let mut best_ins = i;
        let mut best_delta = 0i64;

        for ins in 0..=len {
            if ins == i {
                continue;
            }
            let c = if ins == 0 { 0 } else { map_after(ins - 1) };
            let d = if ins == len { 0 } else { map_after(ins) };

            let delta = -(dist(a, u) + dist(u, b)) + dist(a, b) - dist(c, d) + dist(c, u) + dist(u, d);
            if delta < best_delta {
                best_delta = delta;
                best_ins = ins;
            }
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
        let total = (k + 1) * stride;

        self.split_dp.resize(total, inf);
        self.split_pred.resize(total, 0);
        self.split_dp.fill(inf);
        self.split_dp[0] = 0;

        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let depot = RouteEval::singleton(self.data, 0);

        for kk in 1..=k {
            for i in (kk - 1)..n {
                let base = self.split_dp[(kk - 1) * stride + i];
                if base >= inf {
                    continue;
                }

                let mut acc = RouteEval::join2(self.data, &depot, &RouteEval::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = RouteEval::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    let cell = kk * stride + j;
                    if cand < self.split_dp[cell] {
                        self.split_dp[cell] = cand;
                        self.split_pred[cell] = i;
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
        let (x0, y0) = (self.data.node_positions[0].0 as f64, self.data.node_positions[0].1 as f64);
        let mut route_angles: Vec<(f64, usize)> = Vec::with_capacity(routes.len());

        for (r_idx, r) in routes.iter().enumerate() {
            if r.len() <= 2 {
                continue;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                cnt += 1;
            }
            let bx = sum_x / (cnt as f64);
            let by = sum_y / (cnt as f64);
            let angle = (by - y0).atan2(bx - x0);
            route_angles.push((angle, r_idx));
        }

        route_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tour = Vec::with_capacity(self.data.nb_nodes - 1);
        for &(_, r_idx) in &route_angles {
            let r = &routes[r_idx];
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                if id != 0 {
                    tour.push(id);
                }
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

        if child.len() < n {
            for id in 1..self.data.nb_nodes {
                if !used[id] {
                    used[id] = true;
                    child.push(id);
                    if child.len() == n {
                        break;
                    }
                }
            }
        } else if child.len() > n {
            child.truncate(n);
        }
        child
    }
}