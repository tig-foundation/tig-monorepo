/*!
Copyright [year copyright work created] [name of copyright owner]

Identity of Submitter [name of person or entity that submits the Work to TIG]

UAI [UAI (if applicable)]

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// REMOVE BELOW SECTION IF UNUSED
/*
REFERENCES AND ACKNOWLEDGMENTS

This implementation is based on or inspired by existing work. Citations and
acknowledgments below:

1. Academic Papers:
   - [Author(s), "Paper Title", DOI (if available)]

2. Code References:
   - [Author(s), URL]

3. Other:
   - [Author(s), Details]

*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Serialize, Deserialize)]
pub struct MyHyperparameters {
    // Optionally define your hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn solve_challenge(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<Option<Solution>> {
    // Boiler plate for looping through and solving sub-instances
    // You can modify this function if you want
    let hyperparameters = match hyperparameters {
        Some(hyperparameters) => {
            serde_json::from_value::<MyHyperparameters>(Value::Object(hyperparameters.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => MyHyperparameters {},
    };

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

pub fn solve_sub_instance(instance: &SubInstance) -> Result<Option<SubSolution>> {
    // If you need random numbers, recommend using SmallRng with instance.seed:
    // use rand::{rngs::SmallRng, Rng, SeedableRng};
    // let mut rng = SmallRng::from_seed(instance.seed);

    // If you need HashMap or HashSet, make sure to use a deterministic hasher for consistent runtime_signature:
    // use crate::{seeded_hasher, HashMap, HashSet};
    // let hasher = seeded_hasher(&instance.seed);
    // let map = HashMap::with_hasher(hasher);

    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(None) if your algorithm finds no solution or needs to exit early
    // return Ok(SubSolution { .. }) if your algorithm finds a solution
    Err(anyhow!("Not implemented"))
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
