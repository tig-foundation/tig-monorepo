/*!
Copyright 2026 stinger

Identity of Submitter: stinger

UAI: null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/agreements

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

use anyhow::Result;
use tig_challenges::satisfiability::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    super::weighted::solve(challenge, save_solution, &super::weighted::Params {
        weighted_restarts: 80,
        flips_multiplier: 60,
        cb_exp: 26,
        cambium_interval_divisor: 4,
        smooth_every: 3,
        perturb_pct: 12,
        crossover_pct: 20,
        crossover_bias: 70,
        stagnation_factor: 15,
        fast_restarts: 0,
        fast_flips_multiplier: 0,
    })
}