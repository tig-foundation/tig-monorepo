use super::problem::Problem;
use super::params::Params;
use super::individual::Individual;
use super::constructive::Constructive;
use super::local_search::LocalSearch;
use super::population::{Population, BestMetric};
use super::sequence::Sequence;
use rand::{rngs::SmallRng, Rng};
use std::time::Instant;
use anyhow::Result;
use tig_challenges::vehicle_routing::*;

pub struct Genetic<'a> {
    pub data: &'a Problem,
    pub params: Params,
    pub population: Population<'a>,
}

impl<'a> Genetic<'a> {
    pub fn new(data: &'a Problem, params: Params) -> Self {
        let population = Population::new(data);
        Self { data, params, population }
    }

    /// Run a repair pass from the current LS state (hot start to maintain efficiency)
    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        let mut repaired_routes1: Vec<Vec<usize>> = Vec::new();
        ls.runls(&mut repaired_routes1, rng, self.params, true, 100); // First attempt with 100x increase
        let repaired1 = Individual::new_from_routes(self.data, &self.params, repaired_routes1);
        assert!(repaired1.nb_routes <= self.data.nb_vehicles, "Too many routes after LS");

        if repaired1.load_excess == 0 && repaired1.tw_violation == 0 {
            self.population.add(repaired1, &self.params);
        }
    }

    pub fn generate_initial_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch, randomize: bool) {
        let mut routes: Vec<Vec<usize>> = Constructive::build_routes(self.data, rng, randomize);
        ls.runls(&mut routes, rng, self.params, false,0);
        let ind = Individual::new_from_routes(self.data, &self.params, routes);
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;
        assert!(ind.nb_routes <= self.data.nb_vehicles, "Too many routes after LS");

        // Add solution, record feasibility for parameters adaptation and optionally repair
        self.population.add(ind, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible { self.repair_and_maybe_add(ls, rng); }
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        assert!(
            self.population.feasible.indivs.len() + self.population.infeasible.indivs.len() >= 2,
            "Need at least 2 individuals for parent selection"
        );
        // Select two parents (repick if they are the same)
        let p1 = self.population.get_binary_tournament(rng);
        let mut p2 = self.population.get_binary_tournament(rng);
        while std::ptr::eq(p1, p2) { p2 = self.population.get_binary_tournament(rng); }
        let t1 = self.extract_giant_tour(&p1.routes);
        let t2 = self.extract_giant_tour(&p2.routes);
        let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 }; // small chance of an extra route
        let target_routes = (p1.nb_routes + extra).clamp(self.data.lb_vehicles, self.data.nb_vehicles);

        // Crossover and local search
        let child_tour = self.crossover_ox(&t1, &t2, rng);
        let mut child_routes = self.split(&child_tour, target_routes);
        ls.runls(&mut child_routes, rng, self.params, false, 0);
        let child = Individual::new_from_routes(self.data, &self.params, child_routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        // Add solution, record feasibility and optionally repair
        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible { self.repair_and_maybe_add(ls, rng); }
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Option<(Vec<Vec<usize>>,i32)> {
        let mut ls = LocalSearch::new(self.data, self.params);

        println!("----- BUILDING INITIAL POPULATION");
        for it in 0..self.params.mu_start {
            self.generate_initial_individual(rng, &mut ls, it > 0);
        }

        println!("----- STARTING GENETIC ALGORITHM");
        let mut best_metric: BestMetric = self.population.best_metric();
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {

            // Generates a new individual by crosser and LS
            self.generate_crossover_individual(rng, &mut ls);

            // Prints population statistics
            if it_total % self.params.nb_it_traces == 0 {
                self.population.print_trace(
                    it_total, it_noimprov, t0.elapsed().as_secs_f64(), &self.params );
            }

            // Track best solution
            let cur = self.population.best_metric();
            if cur.better_than(best_metric) {
                best_metric = cur;
                it_noimprov = 0;

                // If a challenge save_solution callback is provided, save the solution
                if let Some(best) = self.population.best_feasible()  {
                    if let Some(save) = save_solution {
                        let _ = save(&Solution {routes: best.routes});
                    }
                }
            }
            else { it_noimprov += 1; }
            it_total += 1;
        }

        println!(
            "----- GENETIC ALGORITHM FINISHED AFTER {} ITERATIONS. TIME SPENT: {:.4} SECONDS",
            it_total,
            t0.elapsed().as_secs_f64()
        );

        if let Some(best) = self.population.best_feasible() {
            println!(
                "----- FOUND SOLUTION WITH COST {} AND {} ROUTES",
                best.cost,
                best.nb_routes
            );
            Some((best.routes, best.cost as i32))
        } else {
            println!("----- NO FEASIBLE SOLUTION FOUND");
            None
        }
    }

    /// Split and return routes as vectors of node ids (with 0-delimiters).
    pub fn split(&self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        let n = giant.len();
        debug_assert!(n > 0, "By design we should never split an empty solution");
        debug_assert!(target_routes <= n, "By design target_routes should be <= number of clients");

        let k = target_routes;
        let inf = i64::MAX / 4;

        // dp[kk][j] = best cost to cover first j clients with exactly kk non-empty routes
        let mut dp = vec![vec![inf; n + 1]; k + 1];
        let mut pred = vec![vec![0usize; n + 1]; k + 1];
        dp[0][0] = 0;

        // Capacity pruning factor (e.g., 1.5 × capacity) to accelerate the inner loop
        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let depot = Sequence::singleton(self.data, 0);

        for kk in 1..=k {
            for i in (kk - 1)..n {
                let base = dp[kk - 1][i];
                if base >= inf { continue; }

                // Build last segment on the fly from depot -> giant[i] -> ... and prune by load
                let mut acc = Sequence::join2(self.data, &depot, &Sequence::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = Sequence::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    if cand < dp[kk][j] { dp[kk][j] = cand; pred[kk][j] = i; }
                    if acc.load > cap_limit { break; }
                    if j < n {
                        let next = Sequence::singleton(self.data, giant[j]);
                        acc = Sequence::join2(self.data, &acc, &next);
                    }
                }
            }
        }

        assert!(dp[k][n] < inf, "Exact-K Split failed unexpectedly");

        // Backtrack routes
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(k);
        let mut j = n;
        for kk in (1..=k).rev() {
            let i = pred[kk][j];
            assert!(i < j, "Split backtrack produced an empty segment");
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            for p in i..j { r.push(giant[p]); }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
    }

    /// Build a giant tour from GA routes, ordering routes by polar angle of their barycenter.
    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let (x0, y0) = (self.data.node_positions[0].0 as f64, self.data.node_positions[0].1 as f64);
        let mut route_angles: Vec<(f64, usize)> = Vec::new();

        for (r_idx, r) in routes.iter().enumerate() {
            if r.len() <= 2 { continue; } // skip [0, 0]
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                debug_assert!(id != 0, "Depot should not appear inside a route");
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                cnt += 1;
            }
            debug_assert!(cnt > 0, "Non-empty route must contain at least one client");
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
                if id != 0 { tour.push(id); }
            }
        }
        debug_assert_eq!(tour.len(), self.data.nb_nodes - 1, "Giant tour must contain all clients exactly once");
        tour
    }

    /// Standard OX crossover on two giant tours.
    pub fn crossover_ox(&self, parent1: &Vec<usize>, parent2: &Vec<usize>, rng: &mut SmallRng) -> Vec<usize> {
        let n = self.data.nb_nodes - 1;
        debug_assert_eq!(n, parent1.len(), "Parents must have same size as #clients");
        debug_assert_eq!(n, parent2.len(), "Parents must have same size as #clients");
        debug_assert!(n > 1, "OX requires at least 2 clients");

        let mut child = vec![0usize; n];
        let mut used = vec![false; self.data.nb_nodes];

        let start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        while end == start { end = rng.gen_range(0..n); }

        let stop = (end + 1) % n;
        let mut j = start;
        while j % n != stop {
            let idx = j % n;
            let v = parent1[idx];
            child[idx] = v;
            used[v] = true;
            j += 1;
        }

        let mut pos = stop;
        for t in 0..n {
            let v = parent2[(stop + t) % n];
            if !used[v] {
                child[pos] = v;
                used[v] = true;
                pos = (pos + 1) % n;
            }
        }
        debug_assert!(child.iter().all(|&x| x != 0), "Child giant tour must be fully filled");
        child
    }
}
