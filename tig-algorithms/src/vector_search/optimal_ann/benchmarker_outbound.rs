/*!
Copyright 2024 Just

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};

pub fn solve_challenge(c: &Challenge) -> Result<Option<Solution>> {
    // Build the k-d tree
    let mut kdtree = KDTree { root: None };
    let mut database: Vec<_> = c.vector_database.iter().collect();
    kdtree.root = KDTree::build(&mut database, 0);
    let mut indexes = Vec::new();
    let mut total_dist = 0.0;
    for query in &c.query_vectors {
        let mut best = None;
        kdtree.nearest_neighbor_search(query, &mut best);
        if let Some((nearest, dist)) = best {
            total_dist += dist;
            if total_dist / (c.query_vectors.len() as f32) > c.max_distance {
                return Ok(None);
            }
            if let Some(index) = c.vector_database.iter().position(|v| v == nearest) {
                indexes.push(index);
            }
        }
    }
    Ok(Some(Solution { indexes }))
}

struct Node {
    vector: Vec<f32>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

struct KDTree {
    root: Option<Box<Node>>,
}

impl KDTree {
    #[inline]
    fn build(vectors: &mut [&Vec<f32>], depth: usize) -> Option<Box<Node>> {
        if vectors.is_empty() {
            return None;
        }
        
        let k = vectors[0].len();
        let axis = depth % k;

        // Sort vectors by the current axis
        vectors.sort_unstable_by(|&a, &b| a[axis].partial_cmp(&b[axis]).unwrap());
        let median = vectors.len() / 2;

        Some(Box::new(Node {
            vector: vectors[median].clone(),
            left: KDTree::build(&mut vectors[..median], depth + 1),
            right: KDTree::build(&mut vectors[median + 1..], depth + 1),
        }))
    }
}

impl KDTree {
    fn nearest_neighbor_search<'a>(&'a self, query: &Vec<f32>, best: &mut Option<(&'a Vec<f32>, f32)>) {
        fn search<'a>(node: &'a Option<Box<Node>>, query: &Vec<f32>, best: &mut Option<(&'a Vec<f32>, f32)>, depth: usize) {
            if let Some(n) = node {
                let k: usize = query.len();
                let axis = depth % k;
                
                // Compute Euclidean distance between query and current node vector
                let dist = euclidean_distance(query, &n.vector);
                if best.is_none() || dist < best.unwrap().1 {
                    *best = Some((&n.vector, dist));
                }
                
                // Determine which subtree to search next
                let diff = query[axis] - n.vector[axis];
                let (next, other) = if diff < 0.0 { (&n.left, &n.right) } else { (&n.right, &n.left) };
                
                search(next, query, best, depth + 1);
                
                // If the current best distance is greater than the distance to the splitting plane, search the other subtree
                if diff.abs() < best.unwrap().1 {
                    search(other, query, best, depth + 1);
                }
            }
        }

        search(&self.root, query, best, 0);
    }
}

#[inline(always)]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let len = a.len(); // Assume a and b are of the same length
    for i in 0..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
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
