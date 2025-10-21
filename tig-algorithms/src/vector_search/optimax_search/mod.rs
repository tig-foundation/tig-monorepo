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
   use std::cmp::Ordering;
   use std::collections::BinaryHeap;

   struct KDNode<'a> {
       point: &'a [f32],
       left: Option<Box<KDNode<'a>>>,
       right: Option<Box<KDNode<'a>>>,
       index: usize,
   }

   impl<'a> KDNode<'a> {
       fn new(point: &'a [f32], index: usize) -> Self {
           KDNode {
               point,
               left: None,
               right: None,
               index,
           }
       }
   }
   fn quickselect_by<F>(arr: &mut [(&[f32], usize)], k: usize, compare: &F)
   where
       F: Fn(&(&[f32], usize), &(&[f32], usize)) -> Ordering,
   {
       if arr.len() <= 1 {
           return;
       }

       let pivot_index = partition(arr, compare);
       if k < pivot_index {
           quickselect_by(&mut arr[..pivot_index], k, compare);
       } else if k > pivot_index {
           quickselect_by(&mut arr[pivot_index + 1..], k - pivot_index - 1, compare);
       }
   }

   fn partition<F>(arr: &mut [(&[f32], usize)], compare: &F) -> usize
   where
       F: Fn(&(&[f32], usize), &(&[f32], usize)) -> Ordering,
   {
       let pivot_index = arr.len() >> 1;
       arr.swap(pivot_index, arr.len() - 1);

       let mut store_index = 0;
       for i in 0..arr.len() - 1 {
           if compare(&arr[i], &arr[arr.len() - 1]) == Ordering::Less {
               arr.swap(i, store_index);
               store_index += 1;
           }
       }
       arr.swap(store_index, arr.len() - 1);
       store_index
   }

   fn build_kd_tree<'a>(points: &mut [(&'a [f32], usize)]) -> Option<Box<KDNode<'a>>> {
       if points.is_empty() {
           return None;
       }

       const NUM_DIMENSIONS: usize = 250;
       let mut stack: Vec<(usize, usize, usize, Option<*mut KDNode<'a>>, bool)> = Vec::new();
       let mut root: Option<Box<KDNode<'a>>> = None;

       stack.push((0, points.len(), 0, None, false));

       while let Some((start, end, depth, parent_ptr, is_left)) = stack.pop() {
           if start >= end {
               continue;
           }

           let axis = depth % NUM_DIMENSIONS;
           let median = (start + end) / 2;
           quickselect_by(&mut points[start..end], median - start, &|a, b| {
               a.0[axis].partial_cmp(&b.0[axis]).unwrap()
           });

           let (median_point, median_index) = points[median];
           let mut new_node = Box::new(KDNode::new(median_point, median_index));
           let new_node_ptr: *mut KDNode = &mut *new_node;

           if let Some(parent_ptr) = parent_ptr {
               unsafe {
                   if is_left {
                       (*parent_ptr).left = Some(new_node);
                   } else {
                       (*parent_ptr).right = Some(new_node);
                   }
               }
           } else {
               root = Some(new_node);
           }

           stack.push((median + 1, end, depth + 1, Some(new_node_ptr), false));
           stack.push((start, median, depth + 1, Some(new_node_ptr), true));
       }

       root
   }

   #[inline(always)]
   fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
       let mut sum = 0.0;
       for i in 0..a.len() {
           let diff = a[i] - b[i];
           sum += diff * diff;
       }
       sum
   }

   #[inline(always)]
   fn early_stopping_distance(a: &[f32], b: &[f32], current_min: f32) -> f32 {
       let mut sum = 0.0;
       let mut i = 0;
       while i + 3 < a.len() {
           let diff0 = a[i] - b[i];
           let diff1 = a[i + 1] - b[i + 1];
           let diff2 = a[i + 2] - b[i + 2];
           let diff3 = a[i + 3] - b[i + 3];

           sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

           if sum > current_min {
               return f32::MAX;
           }

           i += 4;
       }

       while i < a.len() {
           let diff = a[i] - b[i];
           sum += diff * diff;

           if sum > current_min {
               return f32::MAX;
           }

           i += 1;
       }

       sum
   }

   fn nearest_neighbor_search<'a>(
       root: &Option<Box<KDNode<'a>>>,
       target: &[f32],
       best: &mut (f32, Option<usize>),
   ) {
       let num_dimensions = target.len();
       let mut stack = Vec::with_capacity(64);

       if let Some(node) = root {
           stack.push((node.as_ref(), 0));
       }

       while let Some((node, depth)) = stack.pop() {
           let axis = depth % num_dimensions;
           let dist = early_stopping_distance(&node.point, target, best.0);

           if dist < best.0 {
               best.0 = dist;
               best.1 = Some(node.index);
           }

           let diff = target[axis] - node.point[axis];
           let sqr_diff = diff * diff;

           if sqr_diff < best.0 {
               if let Some(farther_node) = if diff < 0.0 { &node.right } else { &node.left } {
                   stack.push((farther_node.as_ref(), depth + 1));
               }
           }

           if let Some(nearer_node) = if diff < 0.0 { &node.left } else { &node.right } {
               stack.push((nearer_node.as_ref(), depth + 1));
           }
       }
   }

   fn calculate_mean_vector(vectors: &[&[f32]]) -> Vec<f32> {
       let num_vectors = vectors.len();
       let num_dimensions = 250;

       let mut mean_vector = vec![0.0; num_dimensions];

       for vector in vectors {
           for i in 0..num_dimensions {
               mean_vector[i] += vector[i];
           }
       }

       for i in 0..num_dimensions {
           mean_vector[i] /= num_vectors as f32;
       }

       mean_vector
   }

   #[derive(Debug)]
   struct FloatOrd(f32);

   impl PartialEq for FloatOrd {
       fn eq(&self, other: &Self) -> bool {
           self.0 == other.0
       }
   }

   impl Eq for FloatOrd {}

   impl PartialOrd for FloatOrd {
       fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
           self.0.partial_cmp(&other.0)
       }
   }

   impl Ord for FloatOrd {
       fn cmp(&self, other: &Self) -> Ordering {

           self.partial_cmp(other).unwrap_or(Ordering::Equal)
       }
   }

   fn filter_relevant_vectors<'a>(
       database: &'a [Vec<f32>],
       query_vectors: &[Vec<f32>],
       k: usize,
   ) -> Vec<(&'a [f32], usize)> {
       let query_refs: Vec<&[f32]> = query_vectors.iter().map(|v| &v[..]).collect();
       let mean_query_vector = calculate_mean_vector(&query_refs);

       let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

       for (index, vector) in database.iter().enumerate() {
           let dist = squared_euclidean_distance(&mean_query_vector, vector);
           let ord_dist = FloatOrd(dist);
           if heap.len() < k {
               heap.push((ord_dist, index));
           } else if let Some(&(FloatOrd(top_dist), _)) = heap.peek() {
               if dist < top_dist {
                   heap.pop();
                   heap.push((ord_dist, index));
               }
           }
       }
       let result: Vec<(&'a [f32], usize)> = heap
           .into_iter()
           .map(|(_, index)| (&database[index][..], index))
           .collect();

       result
   }

   pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
       let query_count = challenge.query_vectors.len();

       let subset_size = match query_count {
           10..=19 if challenge.difficulty.better_than_baseline <= 470 => 4200,
           10..=19 if challenge.difficulty.better_than_baseline > 470 => 4200,
           20..=28 if challenge.difficulty.better_than_baseline <= 465 => 3000,
           20..=28 if challenge.difficulty.better_than_baseline > 465 => 6000, // need more fuel
           29..=50 if challenge.difficulty.better_than_baseline <= 480 => 2000,
           29..=45 if challenge.difficulty.better_than_baseline > 480 => 6000,
           46..=50 if challenge.difficulty.better_than_baseline > 480 => 5000, // need more fuel
           51..=70 if challenge.difficulty.better_than_baseline <= 480 => 3000,
           51..=70 if challenge.difficulty.better_than_baseline > 480 => 3000, // need more fuel
           71..=100 if challenge.difficulty.better_than_baseline <= 480 => 1500,
           71..=100 if challenge.difficulty.better_than_baseline > 480 => 2500, // need more fuel
           _ => 1000,                                                             // need more fuel
       };
       let subset = filter_relevant_vectors(
           &challenge.vector_database,
           &challenge.query_vectors,
           subset_size,
       );


       let kd_tree = build_kd_tree(&mut subset.clone());


       let mut best_indexes = Vec::with_capacity(challenge.query_vectors.len());

       for query in challenge.query_vectors.iter() {
           let mut best = (std::f32::MAX, None);
           nearest_neighbor_search(&kd_tree, query, &mut best);

           if let Some(best_index) = best.1 {
               best_indexes.push(best_index);
           }
       }


       Ok(Some(Solution {
           indexes: best_indexes,
       }))
   }
}