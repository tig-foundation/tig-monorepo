/*!
Copyright [yyyy] [name of copyright owner]

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::vector_search::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(None) if your algorithm finds no solution or needs to exit early
    // return Ok(Solution { .. }) if your algorithm finds a solution
    Err(anyhow!("Not implemented"))
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
