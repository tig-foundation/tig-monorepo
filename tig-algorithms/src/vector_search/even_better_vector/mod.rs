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
   use std::collections::{BinaryHeap, HashSet};

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
       let len = a.len();

       if a.len() != b.len() || len < 8 {
           return f32::MAX;
       }

       let mut a_ptr = a.as_ptr();
       let mut b_ptr = b.as_ptr();
       let end_ptr = unsafe { a_ptr.add(len - 7) };

       while a_ptr < end_ptr {
           unsafe {
               let a0 = *a_ptr.add(0);
               let a1 = *a_ptr.add(1);
               let a2 = *a_ptr.add(2);
               let a3 = *a_ptr.add(3);
               let a4 = *a_ptr.add(4);
               let a5 = *a_ptr.add(5);
               let a6 = *a_ptr.add(6);
               let a7 = *a_ptr.add(7);

               let b0 = *b_ptr.add(0);
               let b1 = *b_ptr.add(1);
               let b2 = *b_ptr.add(2);
               let b3 = *b_ptr.add(3);
               let b4 = *b_ptr.add(4);
               let b5 = *b_ptr.add(5);
               let b6 = *b_ptr.add(6);
               let b7 = *b_ptr.add(7);

               let block_sum = (a0 - b0).powi(2)
                   + (a1 - b1).powi(2)
                   + (a2 - b2).powi(2)
                   + (a3 - b3).powi(2)
                   + (a4 - b4).powi(2)
                   + (a5 - b5).powi(2)
                   + (a6 - b6).powi(2)
                   + (a7 - b7).powi(2);

               sum += block_sum;
           }

           if sum > current_min {
               return f32::MAX;
           }

           a_ptr = unsafe { a_ptr.add(8) };
           b_ptr = unsafe { b_ptr.add(8) };
       }


       let remaining = len - (unsafe { a_ptr.offset_from(a.as_ptr()) } as usize);
       for i in 0..remaining {
           unsafe {
               let diff = *a_ptr.add(i) - *b_ptr.add(i);
               sum += diff * diff;
           }
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

   fn ejection_chain_search<'a>(
       query: &[f32],
       database: &'a [Vec<f32>],
       initial_idx: usize,
       initial_dist: f32,
       candidates: &[(f32, &'a [f32], usize)],
       max_chain_length: usize,
   ) -> (f32, usize) {
       let mut best_idx = initial_idx;
       let mut best_dist = initial_dist;
       let mut visited = HashSet::new();
       visited.insert(initial_idx);

       let mut current_idx = initial_idx;
       let mut current_chain_length = 0;

       while current_chain_length < max_chain_length {
           current_chain_length += 1;
           let mut improved = false;

           let current_vector = &database[current_idx];

           let mut neighbors = Vec::with_capacity(5);
           for &(_, vec, idx) in candidates {
               if !visited.contains(&idx) {
                   let dist_to_current = squared_euclidean_distance(current_vector, vec);
                   if dist_to_current < f32::MAX {
                       neighbors.push((dist_to_current, idx));
                   }
               }
           }

           neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
           for &(_, neighbor_idx) in neighbors.iter().take(3) {
               let dist = early_stopping_distance(query, &database[neighbor_idx], best_dist);
               if dist < best_dist {
                   best_dist = dist;
                   best_idx = neighbor_idx;
                   current_idx = neighbor_idx;
                   visited.insert(neighbor_idx);
                   improved = true;
                   break;
               }
           }

           if !improved {
               if neighbors.is_empty() {
                   break;
               }
               let random_pick = (std::time::SystemTime::now()
                   .duration_since(std::time::UNIX_EPOCH)
                   .unwrap()
                   .subsec_nanos() as usize) % neighbors.len();
               current_idx = neighbors[random_pick].1;
               visited.insert(current_idx);
           }
       }

       let time_based_seed = std::time::SystemTime::now()
           .duration_since(std::time::UNIX_EPOCH)
           .unwrap()
           .subsec_nanos() as usize;

       for i in 0..7 {
           let random_idx = (time_based_seed + i * 97) % database.len();
           if !visited.contains(&random_idx) {
               let dist = early_stopping_distance(query, &database[random_idx], best_dist);
               if dist < best_dist {
                   best_dist = dist;
                   best_idx = random_idx;
               }
           }
       }

       for &(_, _, idx) in candidates.iter().take(15) {
           if !visited.contains(&idx) {
               let dist = early_stopping_distance(query, &database[idx], best_dist);
               if dist < best_dist {
                   best_dist = dist;
                   best_idx = idx;
               }
           }
       }

       (best_dist, best_idx)
   }

   fn calculate_mean_vector(vectors: &[&[f32]]) -> Vec<f32> {
       let num_vectors = vectors.len();
       let num_dimensions = 250;

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

       let max_fuel = 10000000000.0;
       let base_fuel = 760000000.0;
       let alpha = 1630.0 * challenge.difficulty.num_queries as f64;

       let m = ((max_fuel - base_fuel) / alpha) as usize;
       let n = (m as f32 * 1.2) as usize;
       let r = n - m;  

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

       let improvement_threshold = distances.iter().fold(0.0, |acc, &x| acc + x) / (distances.len() as f32) * 1.15;
       let brute_force_count = (query_count as f32 * 0.12) as usize;
       let mut distance_indices: Vec<_> = distances.iter().enumerate().collect();
       distance_indices.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
       let high_distance_indices: Vec<_> = distance_indices.into_iter()
           .take(brute_force_count)
           .map(|(index, _)| index)
           .collect();

       for &query_index in &high_distance_indices {
           let query = &challenge.query_vectors[query_index];
           let initial_dist = distances[query_index];
           let initial_idx = best_indexes[query_index];

           if initial_dist > improvement_threshold {
               let (improved_dist, improved_idx) = ejection_chain_search(
                   query,
                   &challenge.vector_database,
                   initial_idx,
                   initial_dist,
                   &closest_vectors,
                   7
               );

               if improved_dist < initial_dist {
                   best_indexes[query_index] = improved_idx;
                   distances[query_index] = improved_dist;
               } else {
                   let mut best = (initial_dist, initial_idx);

                   for &(_, vec, index) in &r_vectors {
                       let dist = early_stopping_distance(query, vec, best.0);
                       if dist < best.0 {
                           best = (dist, index);
                       }
                   }

                   best_indexes[query_index] = best.1;
                   distances[query_index] = best.0;
               }
           } else {
               let mut best = (initial_dist, initial_idx);

               for &(_, vec, index) in &r_vectors {
                   let dist = early_stopping_distance(query, vec, best.0);
                   if dist < best.0 {
                       best = (dist, index);
                   }
               }

               best_indexes[query_index] = best.1;
           }
       }

       for i in 0..10 {
           let time_seed = std::time::SystemTime::now()
               .duration_since(std::time::UNIX_EPOCH)
               .unwrap()
               .subsec_nanos() as usize;

           let random_query_idx = high_distance_indices[time_seed % high_distance_indices.len()];
           let query = &challenge.query_vectors[random_query_idx];
           let current_dist = distances[random_query_idx];

           let random_db_idx = time_seed % challenge.vector_database.len();
           let dist = early_stopping_distance(query, &challenge.vector_database[random_db_idx], current_dist);

           if dist < current_dist {
               best_indexes[random_query_idx] = random_db_idx;
               distances[random_query_idx] = dist;
           }
       }

       Ok(Some(Solution {
           indexes: best_indexes,
       }))
   }
}