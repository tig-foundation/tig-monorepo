/*!
Copyright 2024 Kouraf

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};


#[derive(Debug)]
struct KDTreeNode {
    point: Vec<f32>,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
}

#[derive(Debug)]
struct KDTree {
    root: Option<Box<KDTreeNode>>,
    k: usize,
}

impl KDTreeNode {
    pub fn nearest_neighbor(&self, target: &[f32], best: &mut Option<(Vec<f32>, f32)>, depth: usize, k: usize) -> Option<(Vec<f32>, f32)> {
        let axis = depth % k;
        let dist = euclidean_distance(&self.point, target);

        if best.is_none() || dist < best.clone().unwrap().1 {
            *best = Some((self.point.clone(), dist));
        }

        let next_branch = if target[axis] < self.point[axis] {
            &self.left
        } else {
            &self.right
        };

        if let Some(branch) = next_branch {
            branch.nearest_neighbor(target, best, depth + 1, k);
        }

        let other_branch = if target[axis] < self.point[axis] {
            &self.right
        } else {
            &self.left
        };

        if let Some(branch) = other_branch {
            if (target[axis] - self.point[axis]).abs() < best.clone().unwrap().1 {
                branch.nearest_neighbor(target, best, depth + 1, k);
            }
        }

        best.clone()
    }
}

impl KDTree {
    pub fn new(points: Vec<Vec<f32>>, depth: usize, k: usize) -> Option<Box<KDTreeNode>> {
        if points.is_empty() {
            return None;
        }

        let axis = depth % k;
        let mut sorted_points = points.clone();
        sorted_points.sort_by(|a, b| a[axis].partial_cmp(&b[axis]).unwrap());

        let median = sorted_points.len() / 2;

        Some(Box::new(KDTreeNode {
            point: sorted_points[median].clone(),
            left: KDTree::new(sorted_points[..median].to_vec(), depth + 1, k),
            right: KDTree::new(sorted_points[median + 1..].to_vec(), depth + 1, k),
        }))
    }

    pub fn nearest_neighbor(&self, target: &[f32]) -> Option<(Vec<f32>, f32)> {
        if let Some(root) = &self.root {
            let mut best = None;
            root.nearest_neighbor(target, &mut best, 0, self.k);
            best
        } else {
            None
        }
    }
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let k = challenge.vector_database[0].len(); // Dimension of vectors
    let tree = KDTree {
        root: KDTree::new(challenge.vector_database.clone(), 0, k),
        k,
    };

    let mut indexes = Vec::new();

    for query in &challenge.query_vectors {
        if let Some((best_point, _)) = tree.nearest_neighbor(query) {
            indexes.push(
                challenge.vector_database.iter().position(|x| x == &best_point).unwrap()
            );
        }
    }

    let solution = Solution { indexes };
    Ok(Some(solution))
    
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x1, &x2)| (x1 - x2) * (x1 - x2))
        .sum::<f32>()
        .sqrt()
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
