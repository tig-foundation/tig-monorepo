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
    // TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge


    use tig_challenges::knapsack::*;
    use anyhow::Result;

    // use anyhow::{anyhow, Result};
    // use tig_challenges::knapsack::{SubInstance, SubSolution};


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
        let max_weight = challenge.max_weight as usize;
        let num_items = challenge.num_items;

        // Initialize a DP table where dp[w] holds the max value achievable with weight w
        let mut dp = vec![0; max_weight + 1];
        let mut keep = vec![vec![false; max_weight + 1]; num_items];

        for i in 0..num_items {
            let item_weight = challenge.weights[i] as usize;
            let item_value = challenge.values[i];

            for w in (item_weight..=max_weight).rev() {
                if dp[w - item_weight] + item_value as u32 > dp[w] {
                    dp[w] = dp[w - item_weight] + item_value as u32;
                    keep[i][w] = true;
                }
            }
        }

        // Find the optimal value achievable
        let optimal_value = dp[max_weight];

        // Check if the optimal value meets the minimum value requirement
        if optimal_value < challenge.baseline_value {
            return Ok(None);
        }

        // Reconstruct the list of items to pick
        let mut items = Vec::new();
        let mut w = max_weight;
        for i in (0..num_items).rev() {
            if keep[i][w] {
                items.push(i);
                w -= challenge.weights[i] as usize;
            }
        }
        items.sort(); // Ensure items are in the original order for the solution

        Ok(Some(SubSolution { items }))
    }
}

pub fn help() {
    println!("No help information available.");
}
