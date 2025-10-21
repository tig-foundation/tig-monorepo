use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde_json::{Map, Value};
use tig_challenges::vector_search::{Challenge, Solution};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
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
}