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

        // Sort items by value-to-weight ratio in descending order
        let mut sorted_items: Vec<usize> = (0..num_items).collect();
        sorted_items.sort_by(|&a, &b| {
            let ratio_a = challenge.values[a] as f64 / challenge.weights[a] as f64;
            let ratio_b = challenge.values[b] as f64 / challenge.weights[b] as f64;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        // Initialize combinations with a single empty combo
        let mut combinations: Vec<(Vec<bool>, u32, u32)> = vec![(vec![false; num_items], 0, 0)];

        let mut items = Vec::new();
        for &item in &sorted_items {
            // Create new combos with the current item
            let mut new_combinations: Vec<(Vec<bool>, u32, u32)> = combinations
                .iter()
                .map(|(combo, value, weight)| {
                    let mut new_combo = combo.clone();
                    new_combo[item] = true;
                    (
                        new_combo,
                        value + challenge.values[item],
                        weight + challenge.weights[item],
                    )
                })
                .filter(|&(_, _, weight)| weight <= max_weight) // Keep only combos within weight limit
                .collect();

            // Check if any new combination meets the minimum value requirement
            if let Some((combo, _, _)) = new_combinations
                .iter()
                .find(|&&(_, value, _)| value >= baseline_value)
            {
                items = combo
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &included)| if included { Some(i) } else { None })
                    .collect();
                break;
            }

            // Merge new_combinations with existing combinations
            combinations.append(&mut new_combinations);

            // Deduplicate combinations by keeping the highest value for each weight
            combinations.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.cmp(&a.1))); // Sort by weight, then by value
            combinations.dedup_by(|a, b| a.2 == b.2 && a.1 <= b.1); // Deduplicate by weight, keeping highest value
        }

        Ok(Some(SubSolution { items }))
    }
}

pub fn help() {
    println!("No help information available.");
}
