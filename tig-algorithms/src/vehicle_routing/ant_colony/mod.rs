// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[derive(Clone, Debug)]
struct Ant {
    tour: Vec<usize>,
    tour_length: f64,
}

impl Ant {
    fn new(num_nodes: usize) -> Self {
        Self {
            tour: Vec::with_capacity(num_nodes),
            tour_length: 0.0,
        }
    }
}



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let num_nodes = challenge.difficulty.num_nodes;
    let distance_matrix = challenge.distance_matrix.iter().map(|row| row.iter().map(|&d| d as f64).collect::<Vec<f64>>()).collect::<Vec<_>>();
    let max_capacity = challenge.max_capacity as f64;
    let demands = challenge.demands.iter().map(|&d| d as f64).collect::<Vec<_>>();

    let num_ants = 10;
    let max_iterations = 1000;
    let alpha = 1.0;
    let beta = 2.0;
    let evaporation_rate = 0.5;
    let initial_pheromone = 1.0 / (num_nodes as f64);

    let mut pheromone_matrix = vec![vec![initial_pheromone; num_nodes]; num_nodes];
    let mut rng = StdRng::seed_from_u64(42);

    let mut best_tour: Vec<usize> = Vec::new();
    let mut best_tour_length = f64::MAX;

    for _ in 0..max_iterations {
        let mut ants = Vec::new();

        for _ in 0..num_ants {
            let mut ant = Ant::new(num_nodes);
            let start_node = rng.gen_range(0..num_nodes);
            ant.tour.push(start_node);
            let mut visited = vec![false; num_nodes];
            visited[start_node] = true;

            while ant.tour.len() < num_nodes {
                let current_node = *ant.tour.last().unwrap();
                let mut probabilities = vec![0.0; num_nodes];
                let mut total_probability = 0.0;

                for next_node in 0..num_nodes {
                    if !visited[next_node] {
                        let pheromone = pheromone_matrix[current_node][next_node].powf(alpha);
                        let heuristic = (1.0 / distance_matrix[current_node][next_node]).powf(beta);
                        probabilities[next_node] = pheromone * heuristic;
                        total_probability += probabilities[next_node];
                    }
                }

                let mut cumulative_probability = 0.0;
                let r: f64 = rng.gen();
                let mut next_node = 0;

                for (i, &prob) in probabilities.iter().enumerate() {
                    cumulative_probability += prob / total_probability;
                    if r <= cumulative_probability {
                        next_node = i;
                        break;
                    }
                }

                ant.tour.push(next_node);
                visited[next_node] = true;
            }

            ant.tour.push(ant.tour[0]);
            ant.tour_length = calculate_tour_length(&ant.tour, &distance_matrix);
            ants.push(ant);
        }

        for ant in &ants {
            if ant.tour_length < best_tour_length {
                best_tour = ant.tour.clone();
                best_tour_length = ant.tour_length;
            }
        }

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                pheromone_matrix[i][j] *= (1.0 - evaporation_rate);
            }
        }

        for ant in &ants {
            for k in 0..ant.tour.len() - 1 {
                let i = ant.tour[k];
                let j = ant.tour[k + 1];
                pheromone_matrix[i][j] += 1.0 / ant.tour_length;
                pheromone_matrix[j][i] += 1.0 / ant.tour_length;
            }
        }
    }

    let _ = save_solution(&Solution {
        routes: vec![best_tour],
    });
    return Ok(());
}

fn calculate_tour_length(tour: &[usize], distance_matrix: &[Vec<f64>]) -> f64 {
    let mut length = 0.0;
    for i in 0..tour.len() - 1 {
        length += distance_matrix[tour[i]][tour[i + 1]];
    }
    length
}