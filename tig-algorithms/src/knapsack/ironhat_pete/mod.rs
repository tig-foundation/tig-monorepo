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
    use anyhow::Result;

    const DEBUG: bool = false;

    macro_rules! debug_log {
        ($($arg:tt)*) => {
            if DEBUG {
                println!($($arg)*);
            }
        };
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

    pub fn solve_sub_instance(challenge: &SubInstance) -> Result<Option<SubSolution>> {
        let max_weight = challenge.max_weight as usize;
        let num_items = challenge.num_items;

        debug_log!("SubInstance data: {:?}", challenge);
        debug_log!("Max weight: {}, Number of items: {}", max_weight, num_items);

        let mut dp = vec![0; max_weight + 1];
        let mut item_included = vec![vec![false; max_weight + 1]; num_items];

        let mut items: Vec<(usize, usize)> = (0..num_items)
            .map(|i| (challenge.values[i] as usize, challenge.weights[i] as usize))
            .collect();
        items.sort_by(|a, b| (b.0 as f64 / b.1 as f64).partial_cmp(&(a.0 as f64 / a.1 as f64)).unwrap());

        let upper_bound = {
            let mut bound = 0;
            let mut remaining_weight = max_weight;
            for (value, weight) in &items {
                if *weight <= remaining_weight {
                    bound += value;
                    remaining_weight -= weight;
                } else {
                    bound += value * remaining_weight / weight;
                    break;
                }
            }
            bound
        };

        debug_log!("Upper bound for optimization: {}", upper_bound);

        for i in 0..num_items {
            let item_weight = challenge.weights[i] as usize;
            let item_value = challenge.values[i] as usize;

            debug_log!("Processing item {}: weight = {}, value = {}", i, item_weight, item_value);

            for w in (item_weight..=max_weight).rev() {
                if dp[w - item_weight] + item_value > dp[w] {
                    dp[w] = dp[w - item_weight] + item_value;
                    item_included[i][w] = true;
                    debug_log!("Updated dp[{}] to {} by including item {}", w, dp[w], i);
                }
            }
        }

        let optimal_value = dp[max_weight];
        debug_log!("Optimal value achievable: {}", optimal_value);

        if optimal_value < challenge.baseline_value as usize {
            debug_log!("Optimal value {} is less than minimum required value {}", optimal_value, challenge.baseline_value);
            return Ok(None);
        }

        let mut items = Vec::new();
        let mut w = max_weight;
        for i in (0..num_items).rev() {
            if item_included[i][w] {
                items.push(i);
                w -= challenge.weights[i] as usize;
                debug_log!("Item {} included, remaining weight {}", i, w);
            }
        }
        items.sort();
        debug_log!("Final items list: {:?}", items);

        Ok(Some(SubSolution { items }))
    }
}

pub fn help() {
    println!("No help information available.");
}
