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
    use std::cmp::Ordering;

    struct Item {
        index: usize,
        weight: usize,
        value: usize,
        ratio: f64,
    }


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
        let max_weight = challenge.max_weight as usize;
        let baseline_value = challenge.baseline_value as usize;
        let num_items = challenge.difficulty.num_items;

        let mut items: Vec<Item> = challenge.weights.iter().zip(challenge.values.iter()).enumerate()
            .map(|(i, (&w, &v))| Item {
                index: i,
                weight: w as usize,
                value: v as usize,
                ratio: v as f64 / w as f64,
            })
            .collect();

        items.sort_unstable_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap_or(Ordering::Equal));

        let mut best_value = 0;
        let mut best_solution = vec![];
        let mut current_solution = vec![];

        fn branch_and_bound(
            items: &[Item],
            index: usize,
            current_weight: usize,
            current_value: usize,
            max_weight: usize,
            baseline_value: usize,
            best_value: &mut usize,
            best_solution: &mut Vec<usize>,
            current_solution: &mut Vec<usize>,
        ) {
            if current_value > *best_value && current_value >= baseline_value {
                *best_value = current_value;
                best_solution.clear();
                best_solution.extend(current_solution.iter().cloned());
            }

            if index >= items.len() {
                return;
            }

            let mut upper_bound = current_value;
            let mut remaining_weight = max_weight - current_weight;

            for item in &items[index..] {
                if item.weight <= remaining_weight {
                    upper_bound += item.value;
                    remaining_weight -= item.weight;
                } else {
                    upper_bound += (item.ratio * remaining_weight as f64) as usize;
                    break;
                }
            }

            if upper_bound <= *best_value || upper_bound < baseline_value {
                return;
            }

            let item = &items[index];
            if current_weight + item.weight <= max_weight {
                current_solution.push(item.index);
                branch_and_bound(
                    items,
                    index + 1,
                    current_weight + item.weight,
                    current_value + item.value,
                    max_weight,
                    baseline_value,
                    best_value,
                    best_solution,
                    current_solution,
                );
                current_solution.pop();
            }

            branch_and_bound(
                items,
                index + 1,
                current_weight,
                current_value,
                max_weight,
                baseline_value,
                best_value,
                best_solution,
                current_solution,
            );
        }

        branch_and_bound(
            &items,
            0,
            0,
            0,
            max_weight,
            baseline_value,
            &mut best_value,
            &mut best_solution,
            &mut current_solution,
        );

        if best_value >= baseline_value {
            Ok(Some(SubSolution { items: best_solution }))
        } else {
            Ok(None)
        }
    }
}