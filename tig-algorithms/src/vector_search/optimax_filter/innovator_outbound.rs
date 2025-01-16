/*!
Copyright 2024 bw-dev36

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Ok;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use tig_challenges::vector_search::*;

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
fn early_stopping_distance(a: &[f32], b: &[f32], current_min: f32) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    let dimensions_to_check = 248;

    while i + 7 < dimensions_to_check {
        let diff0 = a[i] - b[i];
        let diff1 = a[i + 1] - b[i + 1];
        let diff2 = a[i + 2] - b[i + 2];
        let diff3 = a[i + 3] - b[i + 3];
        let diff4 = a[i + 4] - b[i + 4];
        let diff5 = a[i + 5] - b[i + 5];
        let diff6 = a[i + 6] - b[i + 6];
        let diff7 = a[i + 7] - b[i + 7];

        sum += diff0 * diff0
            + diff1 * diff1
            + diff2 * diff2
            + diff3 * diff3
            + diff4 * diff4
            + diff5 * diff5
            + diff6 * diff6
            + diff7 * diff7;

        if sum > current_min {
            return f32::MAX;
        }

        i += 8;
    }

    sum
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

#[inline(always)]
fn weighted_squared_euclidean_distance(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += weights[i] * diff * diff;
    }
    sum
}

#[inline(always)]
fn calculate_mean_and_variance(vectors: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
    let num_vectors = vectors.len();
    let num_dimensions = 250;

    let mut mean_vector = vec![0.0; num_dimensions];
    let mut sum_squares = vec![0.0; num_dimensions];

    for vector in vectors {
        for i in 0..num_dimensions {
            mean_vector[i] += vector[i];
            sum_squares[i] += vector[i] * vector[i];
        }
    }

    let mut variances = vec![0.0; num_dimensions];
    for i in 0..num_dimensions {
        mean_vector[i] /= num_vectors as f32;
        let mean_square = sum_squares[i] / num_vectors as f32;
        let square_mean = mean_vector[i] * mean_vector[i];
        variances[i] = mean_square - square_mean;
    }

    (mean_vector, variances)
}

fn filter_relevant_vectors<'a>(
    database: &'a [Vec<f32>],
    query_vectors: &[Vec<f32>],
    k: usize,
    convergence_threshold: f32,
) -> Option<Vec<(&'a [f32], usize)>> {
    let query_refs: Vec<&[f32]> = query_vectors.iter().map(|v| &v[..]).collect();
    let (mean_query_vector, variances) = calculate_mean_and_variance(&query_refs);

    let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

    for (index, vector) in database.iter().enumerate() {
        let dist = weighted_squared_euclidean_distance(&mean_query_vector, vector, &variances);

        let ord_dist = FloatOrd(dist);
        if heap.len() < k {
            heap.push((ord_dist, index));
        } else if let Some(&(FloatOrd(top_dist), _)) = heap.peek() {
            if dist < top_dist {
                heap.pop();
                heap.push((ord_dist, index));

                if (top_dist - dist).abs() < convergence_threshold {
                    return None;
                }
            }
        }
    }

    let result: Vec<(&'a [f32], usize)> = heap
        .into_iter()
        .map(|(_, index)| (&database[index][..], index))
        .collect();

    Some(result)
}

fn nearest_neighbor_search<'a>(
    root: &Option<Box<KDNode<'a>>>,
    target: &[f32],
    best: &mut (f32, Option<usize>)
) {
    let num_dimensions = 250;
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

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let query_count = challenge.query_vectors.len();

    let subset_size = match query_count {
        10..=19 if challenge.difficulty.better_than_baseline <= 470 => 4200,
        100..=117 => 4500,
        118..=146 => 4000,
        147..=190 => 3000,
        191..=280 => 2000,
        281..=380 => 1500,
        _ => 1000,
    };

    let distance_threshold = 0.00001;
    let subset = filter_relevant_vectors(
        &challenge.vector_database,
        &challenge.query_vectors,
        subset_size,
        distance_threshold,
    );

    if subset.is_none() {
        return Ok(Some(Solution { indexes: vec![] }));
    }

    let kd_tree = build_kd_tree(&mut subset.unwrap().clone());

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

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;
    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        
        extern "C" __global__ void merge(float* distances, int* indices, float* temp_distances, int* temp_indices, int left, int mid, int right) {
            int i = left;
            int j = mid + 1;
            int k = left;

            while (i <= mid && j <= right) {
                if (distances[i] <= distances[j]) {
                    temp_distances[k] = distances[i];
                    temp_indices[k] = indices[i];
                    i++;
                } else {
                    temp_distances[k] = distances[j];
                    temp_indices[k] = indices[j];
                    j++;
                }
                k++;
            }

            while (i <= mid) {
                temp_distances[k] = distances[i];
                temp_indices[k] = indices[i];
                i++;
                k++;
            }

            while (j <= right) {
                temp_distances[k] = distances[j];
                temp_indices[k] = indices[j];
                j++;
                k++;
            }

            for (i = left; i <= right; i++) {
                distances[i] = temp_distances[i];
                indices[i] = temp_indices[i];
            }
        }

        extern "C" __global__ void merge_sort_parallel(float* distances, int* indices, float* temp_distances, int* temp_indices, int num_vectors) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            int curr_size = 1;

            while (curr_size < num_vectors) {
                int left = idx * (curr_size * 2);

                if (left < num_vectors) {
                    int mid = min(left + curr_size - 1, num_vectors - 1);
                    int right = min(left + 2 * curr_size - 1, num_vectors - 1);

                    merge(distances, indices, temp_distances, temp_indices, left, mid, right);
                }

                curr_size *= 2;

                __syncthreads();
            }
        }



        extern "C" __global__ void calculate_mean_and_variance(
        float* vectors, float* mean, float* variances, int num_vectors, int num_dimensions) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < num_dimensions) {
                float sum = 0.0;
                float sum_squares = 0.0;
                
                for (int i = 0; i < num_vectors; ++i) {
                    float val = vectors[i * num_dimensions + idx];
                    sum += val;
                    sum_squares += val * val;
                }
                
                mean[idx] = sum / num_vectors;
                float mean_square = sum_squares / num_vectors;
                float square_mean = mean[idx] * mean[idx];
                variances[idx] = mean_square - square_mean;
            }
        }

        extern "C" __global__ void calculate_distances(
                float* mean_query_vector, float* database, 
                float* variances, float* distances, 
                int num_vectors, int num_dimensions) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_vectors) {
                float dist = 0.0;
                for (int d = 0; d < num_dimensions; ++d) {
                    float diff = mean_query_vector[d] - database[idx * num_dimensions + d];
                    dist += variances[d] * diff * diff;
                }
                distances[idx] = dist;
            }
        }
      
        
        "#,

        funcs: &[
            "calculate_distances",
            "calculate_mean_and_variance",
            "merge_sort_parallel",
            "merge",
        ],
    });

    #[cfg(feature = "cuda")]
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let query_count = challenge.query_vectors.len();

        let subset_size = match query_count {
            10..=19 if challenge.difficulty.better_than_baseline <= 470 => 4200,
            100..=117 => 5000,
            118..=146 => 4000,
            147..=190 => 3000,
            191..=280 => 2000,
            281..=380 => 1500,
            _ => 1000,
        };

        //GPU calls with so many memory transfer are sub optimal comapred to CPU, It's sad to call cuda for now
        // Need a better solution to manage memory...

        let subset = cuda_filter_relevant_vectors(
            &challenge.vector_database,
            &challenge.query_vectors,
            subset_size,
            0.00001,
            dev,
            funcs,
        )?;

        if subset.is_none() {
            return Ok(Some(Solution { indexes: vec![] }));
        }

        let kd_tree = build_kd_tree(&mut subset.unwrap().clone());

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

    #[cfg(feature = "cuda")]
    fn cuda_calculate_mean_and_variance(
        query_refs: &[Vec<f32>],
        dev: &Arc<CudaDevice>,
        funcs: &mut HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let num_vectors = query_refs.len();
        let num_dimensions = 250;

        let flattened_queries: Vec<f32> = query_refs.iter().flatten().cloned().collect();

        let queries_dev = dev.htod_sync_copy(&flattened_queries)?;

        let mut mean_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(num_dimensions)?;
        let mut variances_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(num_dimensions)?;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((num_dimensions as u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            funcs.remove("calculate_mean_and_variance").unwrap().launch(
                cfg,
                (
                    &queries_dev,
                    &mut mean_dev,
                    &mut variances_dev,
                    num_vectors as i32,
                    num_dimensions as i32,
                ),
            )
        }?;

        Ok((mean_dev, variances_dev))
    }

    #[cfg(feature = "cuda")]
    fn cuda_filter_relevant_vectors<'a>(
        database: &'a [Vec<f32>],
        query_vectors: &[Vec<f32>],
        k: usize,
        convergence_threshold: f32,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Vec<(&'a [f32], usize)>>> {
        let (mean_query_dev, variances_dev) =
            cuda_calculate_mean_and_variance(query_vectors, dev, &mut funcs)?;

        let num_vectors = database.len();
        let num_dimensions = 250;
        let flattened_database: Vec<f32> = database.iter().flatten().cloned().collect();
        let database_dev = dev.htod_sync_copy(&flattened_database)?;
        let mut distances_dev = dev.alloc_zeros::<f32>(num_vectors)?;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((num_vectors as u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let _ = dev.synchronize();

        unsafe {
            funcs.remove("calculate_distances").unwrap().launch(
                cfg,
                (
                    &mean_query_dev,
                    &database_dev,
                    &variances_dev,
                    &mut distances_dev,
                    num_vectors as i32,
                    num_dimensions as i32,
                ),
            )
        }?;

        let mut distances_host = vec![0.0f32; num_vectors];

        dev.dtoh_sync_copy_into(&distances_dev, &mut distances_host)?;

        let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

        for (i, &distance) in distances_host.iter().enumerate() {
            let ord_dist = FloatOrd(distance);
            if heap.len() < k {
                heap.push((ord_dist, i));
            } else if let Some(&(FloatOrd(top_dist), _)) = heap.peek() {
                if distance < top_dist {
                    heap.pop();
                    heap.push((ord_dist, i as usize));

                    if (top_dist - distance).abs() < convergence_threshold {
                        return Ok(None);
                    }
                }
            }
        }

        let result: Vec<(&[f32], usize)> = heap
            .into_iter()
            .map(|(_, index)| (&database[index][..], index))
            .collect();

        Ok(Some(result))
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
