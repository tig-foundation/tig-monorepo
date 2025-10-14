use std::collections::HashSet;

use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub size: usize,
    #[cfg(not(feature = "hide_verification"))]
    pub better_than_baseline: u32,
    #[cfg(feature = "hide_verification")]
    better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            size: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }
}

impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.size as i32, self.better_than_baseline as i32]
    }
}

impl_base64_serde! {
     Solution {
        route: Vec<usize>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self { route: Vec::new() }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub node_positions: Vec<(i32, i32)>,
    pub distance_matrix: Vec<Vec<f32>>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline_route: Vec<usize>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline_distance: f32,
    #[cfg(feature = "hide_verification")]
    baseline_route: Vec<usize>,
    #[cfg(feature = "hide_verification")]
    baseline_distance: f32,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        if difficulty.size < 3 {
            return Err(anyhow!("Size must be at least 3"));
        }
        let mut rng = SmallRng::from_seed(seed.clone());
        let num_nodes = difficulty.size;

        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(num_nodes);
        let mut node_positions_set: HashSet<(i32, i32)> = HashSet::with_capacity(num_nodes);
        while node_positions.len() < num_nodes {
            let pos = (rng.gen_range(0..=1000), rng.gen_range(0..=1000));
            if node_positions_set.contains(&pos) {
                continue;
            }
            node_positions.push(pos.clone());
            node_positions_set.insert(pos);
        }

        let distance_matrix: Vec<Vec<f32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| {
                        let dx = (from.0 - to.0) as f32;
                        let dy = (from.1 - to.1) as f32;
                        dx.hypot(dy)
                    })
                    .collect()
            })
            .collect();

        let mut unvisited = (0..num_nodes).collect::<HashSet<_>>();
        let mut baseline_route = Vec::with_capacity(num_nodes);
        let mut current_node = 0;
        baseline_route.push(current_node);
        while baseline_route.len() < num_nodes {
            unvisited.remove(&current_node);
            let next_node = unvisited
                .iter()
                .min_by(|&&a, &&b| {
                    distance_matrix[current_node][a]
                        .partial_cmp(&distance_matrix[current_node][b])
                        .unwrap()
                })
                .cloned()
                .unwrap();
            baseline_route.push(next_node);
            current_node = next_node;
        }

        let baseline_distance = calc_total_distance(&distance_matrix, &baseline_route)?;
        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            node_positions,
            distance_matrix,
            baseline_route,
            baseline_distance,
        })
    }

    pub fn calc_total_distance(&self, solution: &Solution) -> Result<f32> {
        calc_total_distance(&self.distance_matrix, &solution.route)
    }

    conditional_pub!(
        fn verify_solution(&self, solution: &Solution) -> Result<()> {
            let total_distance = self.calc_total_distance(solution)?;
            let btb = self.difficulty.better_than_baseline as f32 / 1000.0;
            let total_distance_threshold = self.baseline_distance * (1.0 - btb);
            let actual_btb = (1.0 - total_distance / self.baseline_distance) * 100.0;
            if total_distance > total_distance_threshold {
                Err(anyhow!(
                    "Total distance ({}) is greater than threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    total_distance,
                    total_distance_threshold,
                    self.baseline_distance,
                    actual_btb
                ))
            } else {
                println!(
                    "Total distance ({}) is less than or equal to threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    total_distance,
                    total_distance_threshold,
                    self.baseline_distance,
                    actual_btb
                );
                Ok(())
            }
        }
    );
}

fn calc_total_distance(distance_matrix: &Vec<Vec<f32>>, route: &Vec<usize>) -> Result<f32> {
    if route.len() != distance_matrix.len() {
        return Err(anyhow!(
            "Route length ({}) does not match number of nodes ({})",
            route.len(),
            distance_matrix.len()
        ));
    }
    let visited = route.iter().cloned().collect::<HashSet<usize>>();
    if visited.len() != route.len() {
        return Err(anyhow!("Route contains duplicate nodes"));
    }
    if route.iter().any(|&node| node >= distance_matrix.len()) {
        return Err(anyhow!("Route contains invalid nodes",));
    }
    let total_distance = route
        .windows(2)
        .map(|w| distance_matrix[w[0]][w[1]])
        .sum::<f32>()
        + distance_matrix[route[route.len() - 1]][route[0]];
    Ok(total_distance)
}
