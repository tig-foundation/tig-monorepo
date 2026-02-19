// TIG submission module for `vehicle_routing`.
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::collections::HashSet;
use tig_challenges::vehicle_routing::{Challenge, Solution};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let params = Params::from_hyperparameters(hyperparameters);
    let solver = Solver::new(challenge, params);

    let mut incumbent = solver.solomon_fallback()?;
    let mut incumbent_distance = challenge.evaluate_total_distance(&incumbent)?;
    save_solution(&incumbent)?;

    if let Some(best_state) = solver.optimize_population() {
        let candidate = Solution {
            routes: best_state.routes,
        };
        if let Ok(distance) = challenge.evaluate_total_distance(&candidate) {
            if distance < incumbent_distance {
                incumbent = candidate;
                incumbent_distance = distance;
                save_solution(&incumbent)?;
            }
        }
    }

    Ok(())
}

pub fn help() {
    println!("Hyperparameters (optional):");
    println!("  population_size: usize (default 7, range 4..=14)");
    println!("  generations: usize (default 20, range 4..=80)");
    println!("  max_moves: usize (default 2400, range 200..=30000)");
    println!("  relink_steps: usize (default 30, range 0..=200)");
    println!("  max_children_per_gen: usize (default 2, range 1..=4)");
    println!("  profile_count: usize (default 6, range 1..=8)");
    println!("  dual_step: f64 (default 0.35, range 0.01..=2.0)");
    println!("  dual_decay: f64 (default 0.97, range 0.80..=1.00)");
    println!("  diversity_threshold: f64 (default 0.28, range 0.0..=1.0)");
}

#[derive(Clone, Copy)]
struct Params {
    population_size: usize,
    generations: usize,
    max_moves: usize,
    relink_steps: usize,
    max_children_per_gen: usize,
    profile_count: usize,
    dual_step: f64,
    dual_decay: f64,
    diversity_threshold: f64,
}

impl Params {
    fn from_hyperparameters(hp: &Option<Map<String, Value>>) -> Self {
        let defaults = Self {
            population_size: 7,
            generations: 20,
            max_moves: 2400,
            relink_steps: 30,
            max_children_per_gen: 2,
            profile_count: 6,
            dual_step: 0.35,
            dual_decay: 0.97,
            diversity_threshold: 0.28,
        };

        let Some(map) = hp else {
            return defaults;
        };

        Self {
            population_size: get_usize(map, "population_size", defaults.population_size, 4, 14),
            generations: get_usize(map, "generations", defaults.generations, 4, 80),
            max_moves: get_usize(map, "max_moves", defaults.max_moves, 200, 30_000),
            relink_steps: get_usize(map, "relink_steps", defaults.relink_steps, 0, 200),
            max_children_per_gen: get_usize(
                map,
                "max_children_per_gen",
                defaults.max_children_per_gen,
                1,
                4,
            ),
            profile_count: get_usize(map, "profile_count", defaults.profile_count, 1, 8),
            dual_step: get_f64(map, "dual_step", defaults.dual_step, 0.01, 2.0),
            dual_decay: get_f64(map, "dual_decay", defaults.dual_decay, 0.80, 1.00),
            diversity_threshold: get_f64(
                map,
                "diversity_threshold",
                defaults.diversity_threshold,
                0.0,
                1.0,
            ),
        }
    }
}

fn get_usize(map: &Map<String, Value>, key: &str, default: usize, min: usize, max: usize) -> usize {
    map.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
        .clamp(min, max)
}

fn get_f64(map: &Map<String, Value>, key: &str, default: f64, min: f64, max: f64) -> f64 {
    map.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
        .unwrap_or(default)
        .clamp(min, max)
}

#[derive(Clone, Copy)]
struct Profile {
    seed_distance_w: f64,
    seed_due_w: f64,
    seed_tight_w: f64,
    insertion_distance_w: f64,
    insertion_shift_w: f64,
    insertion_slack_w: f64,
    dual_insert_w: f64,
    dual_priority_w: f64,
    regret_w: f64,
    new_route_penalty: f64,
    phase_w: f64,
}

#[derive(Clone, Copy)]
struct RouteStats {
    distance: i32,
    end_time: i32,
    min_slack: i32,
    total_wait: i32,
}

#[derive(Clone)]
struct State {
    routes: Vec<Vec<usize>>,
    route_demands: Vec<i32>,
    route_stats: Vec<RouteStats>,
    total_distance: i32,
}

impl State {
    fn recompute_total_distance(&mut self) {
        self.total_distance = self.route_stats.iter().map(|s| s.distance).sum();
    }
}

#[derive(Clone)]
struct Individual {
    state: State,
    edges: HashSet<(usize, usize)>,
    distance: i32,
}

#[derive(Clone)]
struct InsertionOption {
    route_idx: Option<usize>,
    pos: usize,
    new_stats: RouteStats,
    score: f64,
}

#[derive(Clone)]
struct InsertionDecision {
    node: usize,
    priority: f64,
    option: InsertionOption,
}

struct Solver<'a> {
    challenge: &'a Challenge,
    params: Params,
}

impl<'a> Solver<'a> {
    fn new(challenge: &'a Challenge, params: Params) -> Self {
        Self { challenge, params }
    }

    fn profiles(&self) -> Vec<Profile> {
        vec![
            Profile {
                seed_distance_w: 1.1,
                seed_due_w: 1.4,
                seed_tight_w: 120.0,
                insertion_distance_w: 1.0,
                insertion_shift_w: 0.10,
                insertion_slack_w: 2.8,
                dual_insert_w: 0.9,
                dual_priority_w: 1.5,
                regret_w: 1.8,
                new_route_penalty: 35.0,
                phase_w: 8.0,
            },
            Profile {
                seed_distance_w: 0.9,
                seed_due_w: 1.9,
                seed_tight_w: 240.0,
                insertion_distance_w: 0.95,
                insertion_shift_w: 0.22,
                insertion_slack_w: 4.8,
                dual_insert_w: 1.2,
                dual_priority_w: 2.0,
                regret_w: 2.2,
                new_route_penalty: 65.0,
                phase_w: 12.0,
            },
            Profile {
                seed_distance_w: 1.6,
                seed_due_w: 0.7,
                seed_tight_w: 70.0,
                insertion_distance_w: 1.35,
                insertion_shift_w: 0.07,
                insertion_slack_w: 1.7,
                dual_insert_w: 0.6,
                dual_priority_w: 1.0,
                regret_w: 1.4,
                new_route_penalty: 10.0,
                phase_w: 5.0,
            },
            Profile {
                seed_distance_w: 1.2,
                seed_due_w: 1.1,
                seed_tight_w: 160.0,
                insertion_distance_w: 1.05,
                insertion_shift_w: 0.30,
                insertion_slack_w: 3.5,
                dual_insert_w: 1.0,
                dual_priority_w: 1.7,
                regret_w: 2.0,
                new_route_penalty: 48.0,
                phase_w: 10.0,
            },
            Profile {
                seed_distance_w: 1.3,
                seed_due_w: 1.3,
                seed_tight_w: 190.0,
                insertion_distance_w: 1.0,
                insertion_shift_w: 0.18,
                insertion_slack_w: 4.1,
                dual_insert_w: 1.3,
                dual_priority_w: 1.9,
                regret_w: 2.3,
                new_route_penalty: 58.0,
                phase_w: 9.0,
            },
            Profile {
                seed_distance_w: 0.8,
                seed_due_w: 2.1,
                seed_tight_w: 260.0,
                insertion_distance_w: 0.9,
                insertion_shift_w: 0.35,
                insertion_slack_w: 5.4,
                dual_insert_w: 1.5,
                dual_priority_w: 2.2,
                regret_w: 2.5,
                new_route_penalty: 74.0,
                phase_w: 14.0,
            },
            Profile {
                seed_distance_w: 1.5,
                seed_due_w: 0.9,
                seed_tight_w: 95.0,
                insertion_distance_w: 1.2,
                insertion_shift_w: 0.12,
                insertion_slack_w: 2.3,
                dual_insert_w: 0.8,
                dual_priority_w: 1.2,
                regret_w: 1.6,
                new_route_penalty: 24.0,
                phase_w: 6.0,
            },
            Profile {
                seed_distance_w: 1.0,
                seed_due_w: 1.6,
                seed_tight_w: 210.0,
                insertion_distance_w: 0.98,
                insertion_shift_w: 0.27,
                insertion_slack_w: 4.6,
                dual_insert_w: 1.4,
                dual_priority_w: 2.1,
                regret_w: 2.4,
                new_route_penalty: 70.0,
                phase_w: 13.0,
            },
        ]
    }

    fn optimize_population(&self) -> Option<State> {
        let profiles = self.profiles();
        let profile_count = self.params.profile_count.min(profiles.len()).max(1);

        let mut duals = self.initialize_duals();
        let mut population = self.build_initial_population(&profiles[..profile_count], &duals);
        if population.is_empty() {
            return None;
        }

        population.sort_by_key(|ind| ind.distance);
        let mut best = population[0].clone();

        for gen in 0..self.params.generations {
            population.sort_by_key(|ind| ind.distance);
            if population[0].distance < best.distance {
                best = population[0].clone();
            }

            self.update_duals(&mut duals, &best.state);

            let elite_count = population.len().min(4).max(1);
            for child_idx in 0..self.params.max_children_per_gen {
                let p1_idx = (gen + child_idx) % elite_count;
                let mut p2_idx = (gen * 2 + child_idx + 1) % elite_count;
                if p2_idx == p1_idx {
                    p2_idx = (p2_idx + 1) % elite_count;
                }

                let parent_a = &population[p1_idx].state;
                let parent_b = &population[p2_idx].state;

                let profile = profiles[(gen + child_idx) % profile_count];
                let phase = gen * self.params.max_children_per_gen + child_idx;

                if let Some(mut child) = self.crossover(parent_a, parent_b, phase % 2, profile, &duals, phase)
                {
                    self.path_relink(&mut child, parent_b, profile, &duals, phase);
                    let ls_budget = (self.params.max_moves / (self.params.max_children_per_gen + 1)).max(80);
                    self.local_search(&mut child, ls_budget);

                    if !self.validate_state(&child) {
                        continue;
                    }

                    let individual = self.to_individual(child);
                    if individual.distance < best.distance {
                        best = individual.clone();
                    }
                    self.insert_population(&mut population, individual);
                }
            }
        }

        Some(best.state)
    }

    fn initialize_duals(&self) -> Vec<f64> {
        let mut duals = vec![0.0; self.challenge.num_nodes];
        for node in 1..self.challenge.num_nodes {
            let tw_width =
                (self.challenge.due_times[node] - self.challenge.ready_times[node]).max(1) as f64;
            let urgency =
                (self.challenge.due_times[0] - self.challenge.due_times[node]).max(0) as f64;
            let demand_p = self.challenge.demands[node] as f64 / self.challenge.max_capacity as f64;
            duals[node] = (36.0 / tw_width + 0.03 * urgency + 8.0 * demand_p).clamp(0.0, 120.0);
        }
        duals
    }

    fn update_duals(&self, duals: &mut [f64], state: &State) {
        for d in duals.iter_mut().skip(1) {
            *d *= self.params.dual_decay;
        }

        for route in &state.routes {
            if let Some(slacks) = self.route_node_slacks(route) {
                for (node, slack) in slacks {
                    if slack < 25 {
                        duals[node] += self.params.dual_step * (25 - slack) as f64;
                    } else {
                        duals[node] -= self.params.dual_step * 0.2;
                    }
                    duals[node] = duals[node].clamp(0.0, 240.0);
                }
            }
        }
    }

    fn build_initial_population(&self, profiles: &[Profile], duals: &[f64]) -> Vec<Individual> {
        let mut population = Vec::new();

        for i in 0..(self.params.population_size * 2) {
            let profile = profiles[i % profiles.len()];
            let phase = i;
            if let Some(mut state) = self.construct_with_dual_regret(profile, duals, phase) {
                self.local_search(&mut state, (self.params.max_moves / 6).max(60));
                if !self.validate_state(&state) {
                    continue;
                }

                let individual = self.to_individual(state);
                self.insert_population(&mut population, individual);
            }
        }

        if population.is_empty() {
            if let Ok(solution) = self.solomon_fallback() {
                if let Some(state) = self.state_from_solution(&solution) {
                    population.push(self.to_individual(state));
                }
            }
        }

        population
    }

    fn to_individual(&self, state: State) -> Individual {
        let edges = self.extract_edges(&state.routes);
        let distance = state.total_distance;
        Individual {
            state,
            edges,
            distance,
        }
    }

    fn insert_population(&self, population: &mut Vec<Individual>, candidate: Individual) {
        if population
            .iter()
            .any(|ind| ind.distance == candidate.distance && ind.edges == candidate.edges)
        {
            return;
        }

        if population.len() < self.params.population_size {
            population.push(candidate);
            return;
        }

        let mut worst_idx = 0usize;
        for i in 1..population.len() {
            if population[i].distance > population[worst_idx].distance {
                worst_idx = i;
            }
        }

        let min_diversity = population
            .iter()
            .map(|ind| self.edge_distance(&ind.edges, &candidate.edges))
            .fold(1.0_f64, |a, b| a.min(b));

        let worst_distance = population[worst_idx].distance;
        let accept = candidate.distance < worst_distance
            || (min_diversity >= self.params.diversity_threshold
                && candidate.distance <= worst_distance + 180);

        if accept {
            population[worst_idx] = candidate;
        }
    }

    fn edge_distance(&self, a: &HashSet<(usize, usize)>, b: &HashSet<(usize, usize)>) -> f64 {
        let denom = a.len().max(b.len()).max(1) as f64;
        let overlap = a.intersection(b).count() as f64;
        1.0 - overlap / denom
    }

    fn construct_with_dual_regret(
        &self,
        profile: Profile,
        duals: &[f64],
        phase: usize,
    ) -> Option<State> {
        let mut state = State {
            routes: Vec::new(),
            route_demands: Vec::new(),
            route_stats: Vec::new(),
            total_distance: 0,
        };

        let mut unrouted: Vec<usize> = (1..self.challenge.num_nodes).collect();

        while !unrouted.is_empty() {
            let mut chosen: Option<InsertionDecision> = None;

            for &node in &unrouted {
                let mut options = self.collect_insertions(&state, node, profile, duals, phase);
                if options.is_empty() {
                    continue;
                }
                options.sort_by(|a, b| a.score.total_cmp(&b.score));

                let best = options[0].clone();
                let second = if options.len() > 1 {
                    options[1].score
                } else {
                    best.score + 95.0
                };

                let regret = second - best.score;
                let priority = profile.regret_w * regret
                    + profile.dual_priority_w * duals[node]
                    + profile.phase_w * self.phase_term(node, phase)
                    - best.score;

                if chosen.as_ref().map_or(true, |c| priority > c.priority) {
                    chosen = Some(InsertionDecision {
                        node,
                        priority,
                        option: best,
                    });
                }
            }

            if let Some(decision) = chosen {
                self.apply_insertion(&mut state, decision.node, &decision.option);
                if let Some(pos) = unrouted.iter().position(|&n| n == decision.node) {
                    unrouted.swap_remove(pos);
                }
                continue;
            }

            if state.routes.len() >= self.challenge.fleet_size {
                return None;
            }

            let seed = self.select_seed(&unrouted, profile, duals, phase);
            let singleton = vec![0, seed, 0];
            let stats = self.route_stats(&singleton)?;
            state.routes.push(singleton);
            state.route_demands.push(self.challenge.demands[seed]);
            state.route_stats.push(stats);
            state.recompute_total_distance();

            if let Some(pos) = unrouted.iter().position(|&n| n == seed) {
                unrouted.swap_remove(pos);
            }
        }

        if !self.validate_state(&state) {
            return None;
        }

        Some(state)
    }

    fn select_seed(&self, unrouted: &[usize], profile: Profile, duals: &[f64], phase: usize) -> usize {
        unrouted
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = self.seed_score(a, profile, duals, phase);
                let sb = self.seed_score(b, profile, duals, phase);
                sa.total_cmp(&sb)
            })
            .unwrap_or(unrouted[0])
    }

    fn seed_score(&self, node: usize, profile: Profile, duals: &[f64], phase: usize) -> f64 {
        let depot_d = self.challenge.distance_matrix[0][node] as f64;
        let urgency = (self.challenge.due_times[0] - self.challenge.due_times[node]).max(0) as f64;
        let tw_width = (self.challenge.due_times[node] - self.challenge.ready_times[node]).max(1) as f64;

        profile.seed_distance_w * depot_d
            + profile.seed_due_w * urgency
            + profile.seed_tight_w * (1.0 / tw_width)
            + profile.dual_priority_w * duals[node]
            + profile.phase_w * self.phase_term(node, phase)
    }

    fn phase_term(&self, node: usize, phase: usize) -> f64 {
        let x = (node as u64)
            .wrapping_mul(1103515245)
            .wrapping_add((phase as u64).wrapping_mul(12345))
            % 1024;
        x as f64 / 1024.0
    }

    fn collect_insertions(
        &self,
        state: &State,
        node: usize,
        profile: Profile,
        duals: &[f64],
        phase: usize,
    ) -> Vec<InsertionOption> {
        let mut out = Vec::new();

        for route_idx in 0..state.routes.len() {
            if state.route_demands[route_idx] + self.challenge.demands[node] > self.challenge.max_capacity {
                continue;
            }

            let route = &state.routes[route_idx];
            let old_stats = state.route_stats[route_idx];

            for pos in 1..route.len() {
                let mut candidate = route.clone();
                candidate.insert(pos, node);
                let Some(new_stats) = self.route_stats(&candidate) else {
                    continue;
                };

                let delta_distance = (new_stats.distance - old_stats.distance) as f64;
                let delta_shift = (new_stats.end_time - old_stats.end_time).max(0) as f64;
                let slack_pressure = 1.0 / (1.0 + new_stats.min_slack.max(0) as f64);
                let wait_pressure = 1.0 / (1.0 + new_stats.total_wait.max(0) as f64);

                let score = profile.insertion_distance_w * delta_distance
                    + profile.insertion_shift_w * delta_shift
                    + profile.insertion_slack_w * 80.0 * slack_pressure
                    + 24.0 * wait_pressure
                    - profile.dual_insert_w * duals[node]
                    + profile.phase_w * self.phase_term(node, phase);

                out.push(InsertionOption {
                    route_idx: Some(route_idx),
                    pos,
                    new_stats,
                    score,
                });
            }
        }

        if state.routes.len() < self.challenge.fleet_size {
            let singleton = vec![0, node, 0];
            if let Some(stats) = self.route_stats(&singleton) {
                let score = (2 * self.challenge.distance_matrix[0][node]) as f64
                    + profile.new_route_penalty
                    - profile.dual_insert_w * duals[node]
                    + profile.phase_w * self.phase_term(node, phase);
                out.push(InsertionOption {
                    route_idx: None,
                    pos: 1,
                    new_stats: stats,
                    score,
                });
            }
        }

        out
    }

    fn apply_insertion(&self, state: &mut State, node: usize, option: &InsertionOption) {
        if let Some(route_idx) = option.route_idx {
            state.routes[route_idx].insert(option.pos, node);
            state.route_demands[route_idx] += self.challenge.demands[node];
            state.route_stats[route_idx] = option.new_stats;
        } else {
            state.routes.push(vec![0, node, 0]);
            state.route_demands.push(self.challenge.demands[node]);
            state.route_stats.push(option.new_stats);
        }
        state.recompute_total_distance();
    }

    fn crossover(
        &self,
        parent_a: &State,
        parent_b: &State,
        anchor: usize,
        profile: Profile,
        duals: &[f64],
        phase: usize,
    ) -> Option<State> {
        let mut routes = Vec::new();
        for (idx, route) in parent_a.routes.iter().enumerate() {
            if (idx + anchor) % 2 == 0 {
                routes.push(route.clone());
            }
        }

        if routes.is_empty() && !parent_a.routes.is_empty() {
            routes.push(parent_a.routes[0].clone());
        }

        let mut used = vec![false; self.challenge.num_nodes];
        used[0] = true;
        let mut route_demands = Vec::new();
        let mut route_stats = Vec::new();

        for route in &routes {
            let mut demand = 0;
            for &node in &route[1..route.len() - 1] {
                if used[node] {
                    return None;
                }
                used[node] = true;
                demand += self.challenge.demands[node];
            }
            if demand > self.challenge.max_capacity {
                return None;
            }
            let stats = self.route_stats(route)?;
            route_demands.push(demand);
            route_stats.push(stats);
        }

        let mut state = State {
            routes,
            route_demands,
            route_stats,
            total_distance: 0,
        };
        state.recompute_total_distance();

        let mut order = Vec::new();
        let mut listed = vec![false; self.challenge.num_nodes];

        for route in &parent_b.routes {
            for &node in &route[1..route.len() - 1] {
                if !used[node] && !listed[node] {
                    listed[node] = true;
                    order.push(node);
                }
            }
        }

        for node in 1..self.challenge.num_nodes {
            if !used[node] && !listed[node] {
                order.push(node);
            }
        }

        for node in order {
            let mut options = self.collect_insertions(&state, node, profile, duals, phase + node);
            if options.is_empty() {
                return None;
            }
            options.sort_by(|a, b| a.score.total_cmp(&b.score));
            let best = options[0].clone();
            self.apply_insertion(&mut state, node, &best);
        }

        if !self.validate_state(&state) {
            return None;
        }

        Some(state)
    }

    fn path_relink(
        &self,
        current: &mut State,
        guide: &State,
        profile: Profile,
        duals: &[f64],
        phase: usize,
    ) {
        if self.params.relink_steps == 0 {
            return;
        }

        let guide_pairs: Vec<(usize, usize)> = guide
            .routes
            .iter()
            .flat_map(|route| {
                route
                    .windows(2)
                    .filter(|w| w[0] != 0 && w[1] != 0)
                    .map(|w| (w[0], w[1]))
                    .collect::<Vec<_>>()
            })
            .collect();

        if guide_pairs.is_empty() {
            return;
        }

        for step in 0..self.params.relink_steps {
            let current_edges = self.extract_edges(&current.routes);
            let missing: Vec<(usize, usize)> = guide_pairs
                .iter()
                .copied()
                .filter(|pair| !current_edges.contains(pair))
                .collect();
            if missing.is_empty() {
                break;
            }

            let mut best_state: Option<State> = None;
            let mut best_score = f64::NEG_INFINITY;
            let old_distance = current.total_distance;

            for &(u, v) in missing.iter().take(48) {
                let Some(candidate) = self.relocate_after_pair(current, u, v) else {
                    continue;
                };

                let candidate_edges = self.extract_edges(&candidate.routes);
                let new_missing_count = guide_pairs
                    .iter()
                    .filter(|pair| !candidate_edges.contains(pair))
                    .count();

                let missing_gain = (missing.len() as i32 - new_missing_count as i32) as f64;
                let dist_delta = (candidate.total_distance - old_distance) as f64;
                let score = 600.0 * missing_gain - dist_delta
                    + profile.phase_w * self.phase_term(v, phase + step)
                    + 0.5 * duals[v.min(duals.len() - 1)];

                if score > best_score {
                    best_score = score;
                    best_state = Some(candidate);
                }
            }

            if let Some(candidate) = best_state {
                if best_score > -35.0 {
                    *current = candidate;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn relocate_after_pair(&self, state: &State, u: usize, v: usize) -> Option<State> {
        if u == v || u == 0 || v == 0 {
            return None;
        }

        let positions = self.node_positions(&state.routes);
        let (u_route, u_pos) = positions[u]?;
        let (v_route, v_pos) = positions[v]?;

        if u_route == v_route && u_pos + 1 == v_pos {
            return None;
        }

        let target_pos = u_pos + 1;
        self.relocate_move(state, v_route, v_pos, u_route, target_pos)
    }

    fn node_positions(&self, routes: &[Vec<usize>]) -> Vec<Option<(usize, usize)>> {
        let mut positions = vec![None; self.challenge.num_nodes];
        for (r, route) in routes.iter().enumerate() {
            for p in 1..route.len() - 1 {
                positions[route[p]] = Some((r, p));
            }
        }
        positions
    }

    fn local_search(&self, state: &mut State, max_moves: usize) {
        let mut used = 0usize;

        loop {
            if used >= max_moves {
                break;
            }

            if self.apply_two_opt(state) {
                used += 1;
                continue;
            }
            if self.apply_relocate(state) {
                used += 1;
                continue;
            }
            if self.apply_swap(state) {
                used += 1;
                continue;
            }
            if self.apply_two_opt_star(state) {
                used += 1;
                continue;
            }

            break;
        }
    }

    fn apply_two_opt(&self, state: &mut State) -> bool {
        for r in 0..state.routes.len() {
            let len = state.routes[r].len();
            if len <= 4 {
                continue;
            }
            let old_distance = state.route_stats[r].distance;

            for i in 1..len - 2 {
                for j in i + 1..len - 1 {
                    let mut candidate = state.routes[r].clone();
                    candidate[i..=j].reverse();

                    let Some(stats) = self.route_stats(&candidate) else {
                        continue;
                    };

                    if stats.distance < old_distance {
                        state.routes[r] = candidate;
                        state.route_stats[r] = stats;
                        state.recompute_total_distance();
                        return true;
                    }
                }
            }
        }

        false
    }

    fn apply_relocate(&self, state: &mut State) -> bool {
        let route_count = state.routes.len();

        for from_route in 0..route_count {
            for from_pos in 1..state.routes[from_route].len() - 1 {
                for to_route in 0..route_count {
                    let to_len = state.routes[to_route].len();
                    for to_pos in 1..to_len {
                        if from_route == to_route && (to_pos == from_pos || to_pos == from_pos + 1) {
                            continue;
                        }
                        let Some(candidate) =
                            self.relocate_move(state, from_route, from_pos, to_route, to_pos)
                        else {
                            continue;
                        };

                        if candidate.total_distance < state.total_distance {
                            *state = candidate;
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    fn relocate_move(
        &self,
        state: &State,
        from_route: usize,
        from_pos: usize,
        to_route: usize,
        to_pos: usize,
    ) -> Option<State> {
        if from_route >= state.routes.len() || to_route >= state.routes.len() {
            return None;
        }

        let mut routes = state.routes.clone();
        let mut demands = state.route_demands.clone();
        let mut stats = state.route_stats.clone();

        if from_route == to_route {
            let route = &mut routes[from_route];
            if from_pos >= route.len() - 1 {
                return None;
            }
            let node = route.remove(from_pos);
            let mut insert_pos = to_pos.min(route.len() - 1);
            if from_pos < to_pos {
                insert_pos = insert_pos.saturating_sub(1);
            }
            if insert_pos == 0 || insert_pos >= route.len() {
                return None;
            }
            route.insert(insert_pos, node);

            let new_stats = self.route_stats(route)?;
            stats[from_route] = new_stats;

            let mut new_state = State {
                routes,
                route_demands: demands,
                route_stats: stats,
                total_distance: 0,
            };
            new_state.recompute_total_distance();
            return Some(new_state);
        }

        let node = routes[from_route][from_pos];
        let demand = self.challenge.demands[node];

        if demands[to_route] + demand > self.challenge.max_capacity {
            return None;
        }

        routes[from_route].remove(from_pos);
        demands[from_route] -= demand;

        let mut to_index = to_route;

        if routes[from_route].len() == 2 {
            routes.remove(from_route);
            demands.remove(from_route);
            stats.remove(from_route);
            if from_route < to_route {
                to_index = to_route - 1;
            }
        } else {
            let fs = self.route_stats(&routes[from_route])?;
            stats[from_route] = fs;
        }

        if to_index >= routes.len() {
            return None;
        }

        let ins_pos = to_pos.min(routes[to_index].len() - 1);
        if ins_pos == 0 || ins_pos >= routes[to_index].len() {
            return None;
        }

        routes[to_index].insert(ins_pos, node);
        demands[to_index] += demand;
        let ts = self.route_stats(&routes[to_index])?;
        stats[to_index] = ts;

        let mut new_state = State {
            routes,
            route_demands: demands,
            route_stats: stats,
            total_distance: 0,
        };
        new_state.recompute_total_distance();
        Some(new_state)
    }

    fn apply_swap(&self, state: &mut State) -> bool {
        for r1 in 0..state.routes.len() {
            for r2 in r1 + 1..state.routes.len() {
                for i in 1..state.routes[r1].len() - 1 {
                    let n1 = state.routes[r1][i];
                    for j in 1..state.routes[r2].len() - 1 {
                        let n2 = state.routes[r2][j];

                        let d1 = state.route_demands[r1] - self.challenge.demands[n1]
                            + self.challenge.demands[n2];
                        let d2 = state.route_demands[r2] - self.challenge.demands[n2]
                            + self.challenge.demands[n1];
                        if d1 > self.challenge.max_capacity || d2 > self.challenge.max_capacity {
                            continue;
                        }

                        let mut new_r1 = state.routes[r1].clone();
                        let mut new_r2 = state.routes[r2].clone();
                        new_r1[i] = n2;
                        new_r2[j] = n1;

                        let Some(stats1) = self.route_stats(&new_r1) else {
                            continue;
                        };
                        let Some(stats2) = self.route_stats(&new_r2) else {
                            continue;
                        };

                        let old_pair = state.route_stats[r1].distance + state.route_stats[r2].distance;
                        let new_pair = stats1.distance + stats2.distance;

                        if new_pair < old_pair {
                            state.routes[r1] = new_r1;
                            state.routes[r2] = new_r2;
                            state.route_demands[r1] = d1;
                            state.route_demands[r2] = d2;
                            state.route_stats[r1] = stats1;
                            state.route_stats[r2] = stats2;
                            state.recompute_total_distance();
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    fn apply_two_opt_star(&self, state: &mut State) -> bool {
        for r1 in 0..state.routes.len() {
            for r2 in r1 + 1..state.routes.len() {
                let route1 = &state.routes[r1];
                let route2 = &state.routes[r2];
                if route1.len() <= 3 || route2.len() <= 3 {
                    continue;
                }

                for i in 1..route1.len() - 2 {
                    for j in 1..route2.len() - 2 {
                        let mut new_r1 = route1[..=i].to_vec();
                        new_r1.extend_from_slice(&route2[j + 1..]);

                        let mut new_r2 = route2[..=j].to_vec();
                        new_r2.extend_from_slice(&route1[i + 1..]);

                        let d1 = self.route_demand(&new_r1);
                        let d2 = self.route_demand(&new_r2);
                        if d1 > self.challenge.max_capacity || d2 > self.challenge.max_capacity {
                            continue;
                        }

                        let Some(stats1) = self.route_stats(&new_r1) else {
                            continue;
                        };
                        let Some(stats2) = self.route_stats(&new_r2) else {
                            continue;
                        };

                        let old_pair = state.route_stats[r1].distance + state.route_stats[r2].distance;
                        let new_pair = stats1.distance + stats2.distance;
                        if new_pair < old_pair {
                            state.routes[r1] = new_r1;
                            state.routes[r2] = new_r2;
                            state.route_demands[r1] = d1;
                            state.route_demands[r2] = d2;
                            state.route_stats[r1] = stats1;
                            state.route_stats[r2] = stats2;
                            state.recompute_total_distance();
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    fn route_demand(&self, route: &[usize]) -> i32 {
        route
            .iter()
            .filter(|&&node| node != 0)
            .map(|&node| self.challenge.demands[node])
            .sum()
    }

    fn route_stats(&self, route: &[usize]) -> Option<RouteStats> {
        if route.len() < 3 || route[0] != 0 || *route.last()? != 0 {
            return None;
        }

        let mut curr_time = 0;
        let mut curr_node = 0usize;
        let mut distance = 0;
        let mut min_slack = i32::MAX;
        let mut total_wait = 0;

        for (idx, &next_node) in route.iter().enumerate().skip(1) {
            let travel = self.challenge.distance_matrix[curr_node][next_node];
            curr_time += travel;
            distance += travel;

            if idx == route.len() - 1 {
                if curr_time > self.challenge.due_times[0] {
                    return None;
                }
                min_slack = min_slack.min(self.challenge.due_times[0] - curr_time);
            } else {
                if curr_time > self.challenge.due_times[next_node] {
                    return None;
                }
                if curr_time < self.challenge.ready_times[next_node] {
                    total_wait += self.challenge.ready_times[next_node] - curr_time;
                    curr_time = self.challenge.ready_times[next_node];
                }
                min_slack = min_slack.min(self.challenge.due_times[next_node] - curr_time);
                curr_time += self.challenge.service_time;
            }

            curr_node = next_node;
        }

        if min_slack == i32::MAX {
            min_slack = self.challenge.due_times[0] - curr_time;
        }

        Some(RouteStats {
            distance,
            end_time: curr_time,
            min_slack,
            total_wait,
        })
    }

    fn route_node_slacks(&self, route: &[usize]) -> Option<Vec<(usize, i32)>> {
        if route.len() < 3 || route[0] != 0 || *route.last()? != 0 {
            return None;
        }

        let mut curr_time = 0;
        let mut curr_node = 0usize;
        let mut out = Vec::with_capacity(route.len() - 2);

        for idx in 1..route.len() {
            let node = route[idx];
            curr_time += self.challenge.distance_matrix[curr_node][node];

            if idx == route.len() - 1 {
                if curr_time > self.challenge.due_times[0] {
                    return None;
                }
            } else {
                if curr_time > self.challenge.due_times[node] {
                    return None;
                }
                if curr_time < self.challenge.ready_times[node] {
                    curr_time = self.challenge.ready_times[node];
                }
                let slack = self.challenge.due_times[node] - curr_time;
                out.push((node, slack));
                curr_time += self.challenge.service_time;
            }

            curr_node = node;
        }

        Some(out)
    }

    fn extract_edges(&self, routes: &[Vec<usize>]) -> HashSet<(usize, usize)> {
        let mut edges = HashSet::new();
        for route in routes {
            for pair in route.windows(2) {
                edges.insert((pair[0], pair[1]));
            }
        }
        edges
    }

    fn validate_state(&self, state: &State) -> bool {
        let solution = Solution {
            routes: state.routes.clone(),
        };
        self.challenge.evaluate_total_distance(&solution).is_ok()
    }

    fn state_from_solution(&self, solution: &Solution) -> Option<State> {
        if self.challenge.evaluate_total_distance(solution).is_err() {
            return None;
        }

        let mut route_demands = Vec::with_capacity(solution.routes.len());
        let mut route_stats = Vec::with_capacity(solution.routes.len());

        for route in &solution.routes {
            route_demands.push(self.route_demand(route));
            route_stats.push(self.route_stats(route)?);
        }

        let mut state = State {
            routes: solution.routes.clone(),
            route_demands,
            route_stats,
            total_distance: 0,
        };
        state.recompute_total_distance();
        Some(state)
    }

    fn solomon_fallback(&self) -> Result<Solution> {
        let mut routes = Vec::new();

        let mut nodes: Vec<usize> = (1..self.challenge.num_nodes).collect();
        nodes.sort_by(|&a, &b| self.challenge.distance_matrix[0][a].cmp(&self.challenge.distance_matrix[0][b]));

        let mut remaining = vec![true; self.challenge.num_nodes];
        remaining[0] = false;

        while let Some(node) = nodes.pop() {
            if !remaining[node] {
                continue;
            }
            remaining[node] = false;

            let mut route = vec![0, node, 0];
            let mut route_demand = self.challenge.demands[node];

            while let Some((best_node, best_pos)) = self.solomon_find_best_insertion(
                &route,
                remaining
                    .iter()
                    .enumerate()
                    .filter(|(n, flag)| {
                        **flag && route_demand + self.challenge.demands[*n] <= self.challenge.max_capacity
                    })
                    .map(|(n, _)| n)
                    .collect(),
            ) {
                remaining[best_node] = false;
                route_demand += self.challenge.demands[best_node];
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        let solution = Solution { routes };
        if self.challenge.evaluate_total_distance(&solution).is_err() {
            return Err(anyhow!("Fallback constructor produced an infeasible solution"));
        }

        Ok(solution)
    }

    fn solomon_is_feasible(
        &self,
        route: &[usize],
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
    ) -> bool {
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            curr_time += self.challenge.distance_matrix[curr_node][next_node];
            if curr_time > self.challenge.due_times[next_node] {
                return false;
            }
            curr_time = curr_time.max(self.challenge.ready_times[next_node]) + self.challenge.service_time;
            curr_node = next_node;
        }
        true
    }

    fn solomon_find_best_insertion(
        &self,
        route: &[usize],
        remaining_nodes: Vec<usize>,
    ) -> Option<(usize, usize)> {
        let alpha1 = 1;
        let alpha2 = 0;
        let lambda = 1;

        let mut best_c2 = None;
        let mut best = None;

        for insert_node in remaining_nodes {
            let mut best_c1 = None;

            let mut curr_time = 0;
            let mut curr_node = 0;

            for pos in 1..route.len() {
                let next_node = route[pos];

                let new_arrival_time = self.challenge.ready_times[insert_node]
                    .max(curr_time + self.challenge.distance_matrix[curr_node][insert_node]);
                if new_arrival_time > self.challenge.due_times[insert_node] {
                    continue;
                }

                let old_arrival_time = self.challenge.ready_times[next_node]
                    .max(curr_time + self.challenge.distance_matrix[curr_node][next_node]);

                let c11 = self.challenge.distance_matrix[curr_node][insert_node]
                    + self.challenge.distance_matrix[insert_node][next_node]
                    - self.challenge.distance_matrix[curr_node][next_node];

                let c12 = new_arrival_time - old_arrival_time;
                let c1 = -(alpha1 * c11 + alpha2 * c12);
                let c2 = lambda * self.challenge.distance_matrix[0][insert_node] + c1;

                if best_c1.map_or(true, |x| c1 > x)
                    && best_c2.map_or(true, |x| c2 > x)
                    && self.solomon_is_feasible(
                        route,
                        insert_node,
                        new_arrival_time + self.challenge.service_time,
                        pos,
                    )
                {
                    best_c1 = Some(c1);
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                curr_time = self.challenge.ready_times[next_node]
                    .max(curr_time + self.challenge.distance_matrix[curr_node][next_node])
                    + self.challenge.service_time;
                curr_node = next_node;
            }
        }

        best
    }
}
