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
   use anyhow::Ok;
   use tig_challenges::vector_search::*;

   const DIMENSIONS: usize = 250;
   const VECTOR_COUNT: usize = 100000;
   const BATCH_SIZE: usize = 8;
   const KD_TREE_MAX_DEPTH: usize = 16;

   struct AlignedVectors {
       data: Vec<f32>,
       len: usize,
   }

   impl AlignedVectors {
       fn new(vectors: &[Vec<f32>]) -> Self {
           let mut data = Vec::with_capacity(vectors.len() * DIMENSIONS);
           for vector in vectors {
               data.extend_from_slice(vector);
           }
           Self {
               data,
               len: vectors.len(),
           }
       }

       #[inline(always)]
       fn get_vector(&self, index: usize) -> &[f32] {
           let start = index * DIMENSIONS;
           &self.data[start..start + DIMENSIONS]
       }
   }

   struct KDNode {
       point_idx: usize,
       axis: u8,
       left: Option<Box<KDNode>>,
       right: Option<Box<KDNode>>,
   }

   impl KDNode {
       fn new(point_idx: usize, axis: u8) -> Self {
           Self {
               point_idx,
               axis,
               left: None,
               right: None,
           }
       }
   }

   #[inline(always)]
   fn calculate_distance(a: &[f32], b: &[f32], early_stop: Option<f32>) -> f32 {
       let mut sum = 0.0;
       let mut i = 0;

       while i + BATCH_SIZE <= DIMENSIONS {
           let mut batch_sum = 0.0;
           let mut j = 0;
           while j < BATCH_SIZE {
               let diff = unsafe {
                   *a.get_unchecked(i + j) - *b.get_unchecked(i + j)
               };
               batch_sum += diff * diff;
               j += 1;
           }
           sum += batch_sum;
           if let Some(threshold) = early_stop {
               if sum > threshold {
                   return f32::MAX;
               }
           }
           i += BATCH_SIZE;
       }

       while i < DIMENSIONS {
           let diff = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
           sum += diff * diff;
           i += 1;
       }
       sum
   }

   fn partition(vectors: &AlignedVectors, indices: &mut [usize], axis: usize, mid: usize) -> usize {
       let (mut left, right) = (0, indices.len() - 1);
       let pivot_value = unsafe {
           vectors.get_vector(*indices.get_unchecked(mid)).get_unchecked(axis)
       };
       indices.swap(mid, right);

       let mut store = left;
       while left < right {
           let curr_value = unsafe {
               vectors.get_vector(*indices.get_unchecked(left)).get_unchecked(axis)
           };
           if curr_value < pivot_value {
               indices.swap(store, left);
               store += 1;
           }
           left += 1;
       }
       indices.swap(store, right);
       store
   }

   fn build_kd_tree(vectors: &AlignedVectors, indices: &mut [usize], depth: usize) -> Option<Box<KDNode>> {
       if indices.is_empty() || depth >= KD_TREE_MAX_DEPTH {
           return None;
       }

       let axis = depth % DIMENSIONS;
       let mid = indices.len() / 2;
       let split_idx = partition(vectors, indices, axis, mid);

       let mut node = Box::new(KDNode::new(indices[split_idx], axis as u8));

       if split_idx > 0 {
           node.left = build_kd_tree(vectors, &mut indices[..split_idx], depth + 1);
       }
       if split_idx + 1 < indices.len() {
           node.right = build_kd_tree(vectors, &mut indices[split_idx + 1..], depth + 1);
       }

       Some(node)
   }

   fn search_nearest(
       vectors: &AlignedVectors,
       node: &KDNode,
       target: &[f32],
       best: &mut (f32, usize),
       depth: usize,
   ) {
       let point = vectors.get_vector(node.point_idx);
       let dist = calculate_distance(point, target, Some(best.0));

       if dist < best.0 {
           best.0 = dist;
           best.1 = node.point_idx;
       }

       let axis = node.axis as usize;
       let diff = target[axis] - point[axis];
       let diff_squared = diff * diff;

       let (first, second) = if diff < 0.0 {
           (&node.left, &node.right)
       } else {
           (&node.right, &node.left)
       };

       if let Some(ref subtree) = first {
           search_nearest(vectors, subtree, target, best, depth + 1);
       }

       if diff_squared < best.0 {
           if let Some(ref subtree) = second {
               search_nearest(vectors, subtree, target, best, depth + 1);
           }
       }
   }

   pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
       let db_vectors = AlignedVectors::new(&challenge.vector_database);
       let query_vectors = AlignedVectors::new(&challenge.query_vectors);

       let mut indices: Vec<usize> = (0..VECTOR_COUNT).collect();
       let kd_tree = build_kd_tree(&db_vectors, &mut indices, 0)
           .expect("KD-tree construction failed");

       let mut solution_indices = Vec::with_capacity(challenge.query_vectors.len());
       let mut total_distance = 0.0;

       for i in 0..query_vectors.len {
           let query = query_vectors.get_vector(i);
           let mut best = (f32::MAX, 0);
           search_nearest(&db_vectors, &kd_tree, query, &mut best, 0);

           let actual_distance = calculate_distance(query, db_vectors.get_vector(best.1), None).sqrt();
           total_distance += actual_distance;

           if total_distance / (i + 1) as f32 > challenge.max_distance {
               return Ok(None);
           }

           solution_indices.push(best.1);
       }

       Ok(Some(Solution {
           indexes: solution_indices,
       }))
   }
}

pub fn help() {
    println!("No help information available.");
}
