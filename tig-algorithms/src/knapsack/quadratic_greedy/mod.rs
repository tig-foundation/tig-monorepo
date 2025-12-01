use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
    use anyhow::{anyhow, Result};
    use tig_challenges::knapsack::*;


    pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
        let mut solution = Solution {
            sub_solutions: Vec::new(),
        };
        for sub_instance in &challenge.sub_instances {
            match solve_sub_instance(sub_instance)? {
                Some(sub_solution) => solution.sub_solutions.push(sub_solution),
                None => return Ok(None),
            }
        }
        Ok(Some(solution))
    }

    pub fn solve_sub_instance(challenge: &SubInstance) -> Result<Option<SubSolution>> {
        let n = challenge.weights.len();
        let mut items: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let potential_profit = challenge.values[i] as f64
                    + challenge.interaction_values[i].iter().sum::<i32>() as f64;
                let density = potential_profit / challenge.weights[i] as f64;
                (i, density)
            })
            .collect();

        // Sort items by density in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected = vec![false; n];
        let mut total_weight = 0;
        let mut total_value = 0;

        // Greedy selection
        for (i, _) in items.iter() {
            if total_weight + challenge.weights[*i] <= challenge.max_weight {
                selected[*i] = true;
                total_weight += challenge.weights[*i];
                total_value += challenge.values[*i] as i32;
                for j in 0..n {
                    if selected[j] && j != *i {
                        total_value += challenge.interaction_values[*i][j];
                    }
                }
            }
        }

        // Local search improvement
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    if selected[i] && !selected[j] {
                        let weight_diff = challenge.weights[j] as i32 - challenge.weights[i] as i32;
                        if total_weight as i32 + weight_diff <= challenge.max_weight as i32 {
                            let mut value_diff = challenge.values[j] as i32 - challenge.values[i] as i32;
                            for k in 0..n {
                                if selected[k] && k != i && k != j {
                                    value_diff += challenge.interaction_values[j][k] - challenge.interaction_values[i][k];
                                }
                            }
                            if value_diff > 0 {
                                selected[i] = false;
                                selected[j] = true;
                                total_weight = (total_weight as i32 + weight_diff) as u32;
                                total_value += value_diff;
                                improved = true;
                            }
                        }
                    }
                }
            }
        }

        let solution_items: Vec<usize> = selected.iter().enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if total_value as u32 >= challenge.baseline_value {
            Ok(Some(SubSolution { items: solution_items }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
