use super::instance::{Instance, NodeData};
use super::config::Config;
use super::evolution::Evolution;
use anyhow::Result;
use super::*;
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, SeedableRng, Rng};
use std::time::Instant;
use std::cell::RefCell;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Instance {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut service_times = vec![challenge.service_time; nb_nodes];
        service_times[0] = 0;

        let total_demand: f64 = challenge.demands.iter().map(|&d| d as f64).sum();
        let ratio = total_demand / challenge.max_capacity as f64;
        let lb_vehicles = ratio.ceil() as usize;

        let node_data: Vec<NodeData> = (0..nb_nodes).map(|i| NodeData {
            start_tw: challenge.ready_times[i],
            end_tw: challenge.due_times[i],
            service_time: service_times[i],
            demand: challenge.demands[i],
        }).collect();

        Instance {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            demands: challenge.demands.clone(),
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix: challenge.distance_matrix.iter().flatten().copied().collect(),
            service_times,
            start_tw: challenge.ready_times.clone(),
            end_tw: challenge.due_times.clone(),
            node_data,
        }
    }
}

pub struct Solver;

impl Solver {
    fn solve_with_portfolio(
        data: &Instance,
        base_params: Config,
        challenge: &Challenge,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        const WALL_BUDGET_MS: u128 = 248_000;
        let best_cost_tracker: RefCell<i32> = RefCell::new(i32::MAX);

        let wrapped_save = |solution: &Solution| -> Result<()> {
            if let Some(saver) = save_solution {
                match challenge.evaluate_total_distance(solution) {
                    Ok(cost) => {
                        let mut best = best_cost_tracker.borrow_mut();
                        if cost < *best {
                            *best = cost;
                            saver(solution)?;
                        }
                    }
                    Err(_) => {}
                }
            }
            Ok(())
        };

        let mut best_solution: Option<(Solution, i32, usize)> = None;

        // Entry A: Deep absorber with level 4, allowing swap3 override
        {
            if t0.elapsed().as_millis() > WALL_BUDGET_MS {
                return Ok(best_solution);
            }
            let mut config_a = base_params;
            config_a.exploration_level = 4;
            config_a.allow_swap3 = true;
            config_a.granularity = 30;
            config_a.granularity2 = 20;
            config_a.nb_it_adapt_penalties = 20;
            config_a.nb_it_traces = 100;
            config_a.mu = 15;
            config_a.mu_start = 30;
            config_a.lambda = 15;
            config_a.nb_close = 4;
            config_a.nb_elite = 3;
            config_a.max_it_noimprov = 500;
            config_a.max_it_total = 5_000;
            config_a.target_ratio = 0.2;
            // Override for deep budget absorption
            config_a.max_it_total = 1_000_000;
            config_a.max_it_noimprov = 100_000;

            let mut rng_a = SmallRng::from_seed(data.seed);
            let mut ga_a = Evolution::new(&data, config_a);
            if let Some((routes, cost)) = ga_a.run(&mut rng_a, t0, Some(&wrapped_save)) {
                best_solution = Some((Solution { routes: routes.clone() }, cost, routes.len()));
            }
        }

        // Entry B: Quick seed-perturbed insurance probe—gates at 65% budget and uses tight iteration cap
        {
            let elapsed = t0.elapsed().as_millis();
            if elapsed > WALL_BUDGET_MS * 65 / 100 {
                return Ok(best_solution);
            }
            let mut config_b = base_params;
            config_b.exploration_level = 3;
            config_b.allow_swap3 = true;
            // Widen granularity/neighborhood for structural diversity orthogonal to Entry A
            config_b.granularity = 50;
            config_b.granularity2 = 30;
            config_b.mu = 5;
            config_b.mu_start = 10;
            config_b.lambda = 5;
            config_b.nb_close = 3;
            config_b.nb_elite = 2;
            config_b.nb_it_adapt_penalties = 20;
            config_b.nb_it_traces = 20;
            config_b.target_ratio = 0.0;
            config_b.max_it_total = 50_000;
            config_b.max_it_noimprov = 5_000;

            let mut perturbed_seed = data.seed;
            perturbed_seed[0] ^= 0x5A;
            perturbed_seed[1] ^= 0xA5;

            let mut rng_b = SmallRng::from_seed(perturbed_seed);
            let mut ga_b = Evolution::new(&data, config_b);
            if let Some((routes, cost)) = ga_b.run(&mut rng_b, t0, Some(&wrapped_save)) {
                let curr_cost = best_cost_tracker.borrow().clone();
                if cost < curr_cost {
                    best_solution = Some((Solution { routes: routes.clone() }, cost, routes.len()));
                }
            }
        }

        // Entry C: Late-gated micro-probe with population-axis diversification
        {
            let elapsed = t0.elapsed().as_millis();
            let budget_remaining = if elapsed < WALL_BUDGET_MS {
                WALL_BUDGET_MS - elapsed
            } else {
                0
            };

            if budget_remaining > 20_000 {
                let mut config_c = base_params;
                config_c.exploration_level = 3;
                config_c.allow_swap3 = true;
                config_c.granularity = 40;
                config_c.granularity2 = 20;
                config_c.nb_it_adapt_penalties = 20;
                config_c.nb_it_traces = 20;
                config_c.mu = 5;
                config_c.mu_start = 30;
                config_c.lambda = 15;
                config_c.nb_close = 2;
                config_c.nb_elite = 4;
                config_c.target_ratio = 0.2;
                config_c.max_it_total = 120_000;
                config_c.max_it_noimprov = 8_000;

                let mut perturbed_seed_c = data.seed;
                perturbed_seed_c[0] ^= 0xC3;
                perturbed_seed_c[1] ^= 0x3C;

                let mut rng_c = SmallRng::from_seed(perturbed_seed_c);
                let mut ga_c = Evolution::new(&data, config_c);
                if let Some((routes, cost)) = ga_c.run(&mut rng_c, t0, Some(&wrapped_save)) {
                    let curr_cost = best_cost_tracker.borrow().clone();
                    if cost < curr_cost {
                        best_solution = Some((Solution { routes: routes.clone() }, cost, routes.len()));
                    }
                }
            }
        }

        Ok(best_solution)
    }

    fn solve(
        data: Instance,
        params: Config,
        challenge: &Challenge,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        Self::solve_with_portfolio(&data, params, challenge, t0, save_solution)
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let data = TigLoader::load(challenge);
        let params = Config::initialize(hyperparameters, data.nb_nodes);
        match Self::solve(data, params, challenge, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}
