/*!
Copyright 2024 Clarence Callahan

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use std::collections::{HashMap, HashSet};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solution = Solution {
        variables: vec![false; challenge.difficulty.num_variables],
    };
    let mut vars_map = HashMap::<i32, HashSet<usize>>::new();
    for (idx, clause) in challenge.clauses.iter().enumerate() {
        for v in clause.iter() {
            vars_map.entry(*v).or_insert(HashSet::new()).insert(idx);
        }
    }
    while !vars_map.is_empty() {
        let mut lens = vars_map
            .iter()
            .map(|v| (v.0.clone(), v.1.len()))
            .collect::<Vec<_>>();
        lens.sort_by(|a, b| b.1.cmp(&a.1));
        let s = &lens[0];
        if s.1 == 0 {
            break;
        }
        solution.variables[(s.0.abs() - 1) as usize] = s.0 > 0;
        let c = vars_map.remove(&s.0).unwrap();
        vars_map.remove(&-s.0);
        vars_map.retain(|_, v| {
            *v = v.difference(&c).cloned().collect();
            !v.is_empty()
        });
    }

    Ok(Some(solution))
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
