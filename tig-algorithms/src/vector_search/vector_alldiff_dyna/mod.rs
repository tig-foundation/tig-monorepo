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
mod dead_code {   use anyhow::Ok;
   use tig_challenges::vector_search::*;
   use std::cmp::Ordering;
   use std::collections::BinaryHeap;
   use rand::{Rng, SeedableRng};
   use rand::rngs::SmallRng;

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

       let num_dimensions = points[0].0.len();
       let mut stack: Vec<(usize, usize, usize, Option<*mut KDNode<'a>>, bool)> = Vec::new();
       let mut root: Option<Box<KDNode<'a>>> = None;

       stack.push((0, points.len(), 0, None, false));

       while let Some((start, end, depth, parent_ptr, is_left)) = stack.pop() {
           if start >= end {
               continue;
           }

           let axis = depth % num_dimensions;
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
       let mut i = 0;
       let len = a.len();

       if a.len() != b.len() || a.len() < 8 {
           return f32::MAX;
       }

       while i + 7 < len {
           unsafe {
               let diff0 = *a.get_unchecked(i) - *b.get_unchecked(i);
               let diff1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
               let diff2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
               let diff3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
               let diff4 = *a.get_unchecked(i + 4) - *b.get_unchecked(i + 4);
               let diff5 = *a.get_unchecked(i + 5) - *b.get_unchecked(i + 5);
               let diff6 = *a.get_unchecked(i + 6) - *b.get_unchecked(i + 6);
               let diff7 = *a.get_unchecked(i + 7) - *b.get_unchecked(i + 7);

               sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 +
                   diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
           }

           i += 8;
       }

       while i < len {
           unsafe {
               let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
               sum += diff * diff;
           }
           i += 1;
       }
       sum
   }

   #[inline(always)]
   fn early_stopping_distance(a: &[f32], b: &[f32], current_min: f32) -> f32 {
       let mut sum = 0.0;
       let mut i = 0;
       let len = a.len();

       if a.len() != b.len() || a.len() < 8 {
           return f32::MAX;
       }

       while i + 7 < len {
           unsafe {
               let diff0 = *a.get_unchecked(i) - *b.get_unchecked(i);
               let diff1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
               let diff2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
               let diff3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
               let diff4 = *a.get_unchecked(i + 4) - *b.get_unchecked(i + 4);
               let diff5 = *a.get_unchecked(i + 5) - *b.get_unchecked(i + 5);
               let diff6 = *a.get_unchecked(i + 6) - *b.get_unchecked(i + 6);
               let diff7 = *a.get_unchecked(i + 7) - *b.get_unchecked(i + 7);

               sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 +
                   diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
           }

           if sum > current_min {
               return f32::MAX;
           }

           i += 8;
       }

       while i < len {
           unsafe {
               let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
               sum += diff * diff;
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
           let dist = early_stopping_distance(node.point, target, best.0);

           if dist < best.0 {
               best.0 = dist;
               best.1 = Some(node.index);
           }

           let diff = target[axis] - node.point[axis];
           let sqr_diff = diff * diff;

           let (nearer, farther) = if diff < 0.0 {
               (&node.left, &node.right)
           } else {
               (&node.right, &node.left)
           };

           if let Some(nearer_node) = nearer {
               stack.push((nearer_node.as_ref(), depth + 1));
           }

           if sqr_diff < best.0 {
               if let Some(farther_node) = farther {
                   stack.push((farther_node.as_ref(), depth + 1));
               }
           }
       }
   }

   fn calculate_mean_vector(vectors: &[&[f32]]) -> Vec<f32> {
       if vectors.is_empty() {
           return Vec::new();
       }

       let num_vectors = vectors.len();
       let num_dimensions = vectors[0].len();

       let mut mean_vector = vec![0.0f64; num_dimensions];

       for vector in vectors {
           for i in 0..num_dimensions {
               mean_vector[i] += vector[i] as f64;
           }
       }
       for i in 0..num_dimensions {
           mean_vector[i] /= num_vectors as f64;
       }
       mean_vector.into_iter().map(|x| x as f32).collect()
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
   ) -> Vec<(f32, &'a [f32], usize)> {
       let query_refs: Vec<&[f32]> = query_vectors.iter().map(|v| &v[..]).collect();
       let mean_query_vector = calculate_mean_vector(&query_refs);

       let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

       for (index, vector) in database.iter().enumerate() {
           if heap.len() < k 
           {
               let dist = squared_euclidean_distance(&mean_query_vector, vector);
               let ord_dist = FloatOrd(dist);

               heap.push((ord_dist, index));
           } else if let Some(&(FloatOrd(top_dist), _)) = heap.peek() 
           {
               let dist = early_stopping_distance(&mean_query_vector, vector, top_dist);
               let ord_dist = FloatOrd(dist);
               if dist < top_dist {
                   heap.pop();
                   heap.push((ord_dist, index));
               }
           }
       }
       heap.into_sorted_vec()
           .into_iter()
           .map(|(FloatOrd(dist), index)| (dist, &database[index][..], index))
           .collect()
   }

   pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
       let query_count = challenge.query_vectors.len();

       let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

       let max_fuel = 10000000000.0;
       let base_fuel = 760000000.0;
       let alpha = 1720.0 * challenge.num_queries as f64;

       let m = ((max_fuel - base_fuel) / alpha) as usize;
       let n = (m as f32 * 1.2) as usize;

       let closest_vectors = filter_relevant_vectors(
           &challenge.vector_database,
           &challenge.query_vectors,
           n,
       );

       let (m_slice, r_slice) = closest_vectors.split_at(m);
       let m_vectors: Vec<_> = m_slice.to_vec();
       let r_vectors: Vec<_> = r_slice.to_vec();

       let mut kd_tree_vectors: Vec<(&[f32], usize)> = m_vectors.iter().map(|&(_, v, i)| (v, i)).collect();
       let kd_tree = build_kd_tree(&mut kd_tree_vectors);

       let mut best_indexes = Vec::with_capacity(query_count);
       let mut distances = Vec::with_capacity(query_count);

       for query in &challenge.query_vectors {
           let mut best = (std::f32::MAX, None);
           nearest_neighbor_search(&kd_tree, query, &mut best);

           distances.push(best.0);
           best_indexes.push(best.1.unwrap_or(0));
       }

       let difficulty_factor = (challenge.difficulty.better_than_baseline as f32 - 505.0) / (570.0 - 505.0);
       let brute_force_ratio = 0.1 + (0.02 * difficulty_factor);  
       let brute_force_count = (query_count as f32 * brute_force_ratio) as usize;
       let mut distance_indices: Vec<_> = distances.iter().enumerate().collect();
       distance_indices.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
       let high_distance_indices: Vec<_> = distance_indices.into_iter()
           .take(brute_force_count)
           .map(|(index, _)| index)
           .collect();

       for &query_index in &high_distance_indices {
           let query = &challenge.query_vectors[query_index];
           let mut best = (distances[query_index], best_indexes[query_index]);
           let current_min = best.0;

           for &(_, vec, index) in &r_vectors {
               let dist = early_stopping_distance(query, vec, current_min);
               if dist < best.0 {
                   best = (dist, index);
               }
           }

           let intensive_ratio = 0.2 + (0.1 * difficulty_factor);  
           if query_index < (brute_force_count as f32 * intensive_ratio) as usize {
               let distance_ratio = distances[query_index] / challenge.max_distance;

               let base_sample = if query_index < 3 {
                   (25000.0 + (5000.0 * difficulty_factor)) as usize
               } else if query_index < 8 {
                   (10000.0 + (2000.0 * difficulty_factor)) as usize
               } else {
                   (3000.0 + (1000.0 * difficulty_factor)) as usize
               };


               let scaling_factor = if distance_ratio < 1.1 {
                   2.5 + (0.5 * difficulty_factor)  
               } else if distance_ratio < 1.3 {
                   2.0 + (0.3 * difficulty_factor)  
               } else {
                   1.5 + (0.2 * difficulty_factor)  
               };

               let sample_count = (base_sample as f32 * scaling_factor) as usize;

               for _ in 0..sample_count {
                   let idx = rng.gen_range(0..challenge.vector_database.len());
                   let dist = early_stopping_distance(query, &challenge.vector_database[idx], best.0);
                   if dist < best.0 {
                       best = (dist, idx);
                       if dist <= challenge.max_distance {
                           break;
                       }
                   }
               }
           }

           best_indexes[query_index] = best.1;
           distances[query_index] = best.0;
       }

       Ok(Some(Solution {
           indexes: best_indexes,
       }))
   }
}

pub fn help() {
    println!("No help information available.");
}
