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

    pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
        let max_weight = challenge.max_weight;
        let baseline_value = challenge.baseline_value;
        let num_items = challenge.num_items;

        let weights: Vec<u32> = challenge.weights.iter().map(|&w| w).collect();
        let values: Vec<u32> = challenge.values.iter().map(|&v| v).collect();

        let mut sorted_items: Vec<(usize, u32, u32)> =
            (0..num_items).map(|i| (i, values[i], weights[i])).collect();
        sorted_items.sort_unstable_by(|a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));

        let mut current_solution = vec![false; num_items];
        let mut best_solution = vec![false; num_items];
        let mut best_profit = 0;

        fn calculate_upper_bound(
            i: usize,
            w: u32,
            p: u32,
            max_weight: u32,
            sorted_items: &[(usize, u32, u32)],
        ) -> u32 {
            let mut ub = p;
            let mut remaining_weight = max_weight - w;

            for &(_, value, weight) in &sorted_items[i..] {
                if weight <= remaining_weight {
                    remaining_weight -= weight;
                    ub += value;
                } else {
                    ub += value * remaining_weight / weight;
                    break;
                }
            }
            ub
        }

        fn branch_and_bound(
            i: usize,
            w: u32,
            p: u32,
            max_weight: u32,
            num_items: usize,
            sorted_items: &[(usize, u32, u32)],
            current_solution: &mut [bool],
            best_solution: &mut [bool],
            best_profit: &mut u32,
            baseline_value: u32,
        ) -> u32 {
            let ub = calculate_upper_bound(i, w, p, max_weight, sorted_items);
            if ub <= *best_profit || ub < baseline_value {
                return *best_profit;
            }

            if p > *best_profit {
                *best_profit = p;
                best_solution.copy_from_slice(current_solution);
            }

            if i < num_items {
                let (item_index, value, weight) = sorted_items[i];
                if w + weight <= max_weight {
                    current_solution[item_index] = true;
                    branch_and_bound(
                        i + 1,
                        w + weight,
                        p + value,
                        max_weight,
                        num_items,
                        sorted_items,
                        current_solution,
                        best_solution,
                        best_profit,
                        baseline_value,
                    );
                    current_solution[item_index] = false;
                }

                branch_and_bound(
                    i + 1,
                    w,
                    p,
                    max_weight,
                    num_items,
                    sorted_items,
                    current_solution,
                    best_solution,
                    best_profit,
                    baseline_value,
                );
            }

            *best_profit
        }

        let _ = branch_and_bound(
            0,
            0,
            0,
            max_weight,
            num_items,
            &sorted_items,
            &mut current_solution,
            &mut best_solution,
            &mut best_profit,
            baseline_value,
        );

        if best_profit >= baseline_value {
            Ok(Some(SubSolution {
                items: best_solution
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &included)| if included { Some(i) } else { None })
                    .collect(),
            }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
