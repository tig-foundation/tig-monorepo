/*!
Copyright 2024 mcmoid

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
use tig_challenges::vector_search::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Fork<'a> {
    coordinate: &'a [f32],
    sx: Option<Box<Fork<'a>>>,
    dx: Option<Box<Fork<'a>>>,
    i: usize,
}

impl<'a> Fork<'a> {
    fn new(coordinate: &'a [f32], i: usize) -> Self {
        Fork {
            coordinate,
            sx: None,
            dx: None,
            i,
        }
    }
}
fn nth_element_by<F>(list: &mut [(&[f32], usize)], k: usize, evaluate: &F)
where
    F: Fn(&(&[f32], usize), &(&[f32], usize)) -> Ordering,
{
    if list.len() <= 1 {
        return;
    }

    let split_index = split(list, evaluate);
    if k < split_index {
        nth_element_by(&mut list[..split_index], k, evaluate);
    } else if k > split_index {
        nth_element_by(&mut list[split_index + 1..], k - split_index - 1, evaluate);
    }
}

fn split<F>(list: &mut [(&[f32], usize)], evaluate: &F) -> usize
where
    F: Fn(&(&[f32], usize), &(&[f32], usize)) -> Ordering,
{
    let split_index = list.len() >> 1;
    list.swap(split_index, list.len() - 1);

    let mut write_index = 0;
    for i in 0..list.len() - 1 {
        if evaluate(&list[i], &list[list.len() - 1]) == Ordering::Less {
            list.swap(i, write_index);
            write_index += 1;
        }
    }
    list.swap(write_index, list.len() - 1);
    write_index
}

fn create_fork_tree<'a>(coordinates: &mut [(&'a [f32], usize)]) -> Option<Box<Fork<'a>>> {
    if coordinates.is_empty() {
        return None;
    }

    const DIMENSION_COUNT: usize = 250;
    let mut list: Vec<(usize, usize, usize, Option<*mut Fork<'a>>, bool)> = Vec::new();
    let mut origin: Option<Box<Fork<'a>>> = None;

    list.push((0, coordinates.len(), 0, None, false));

    while let Some((begin, end, height, ancestor_ref, is_sx)) = list.pop() {
        if begin >= end {
            continue;
        }

        let dimension = height % DIMENSION_COUNT;
        let middle = (begin + end) / 2;
        nth_element_by(&mut coordinates[begin..end], middle - begin, &|a, b| {
            a.0[dimension].partial_cmp(&b.0[dimension]).unwrap()
        });

        let (middle_coordinate, middle_index) = coordinates[middle];
        let mut new_fork = Box::new(Fork::new(middle_coordinate, middle_index));
        let new_fork_ptr: *mut Fork = &mut *new_fork;

        if let Some(ancestor_ref) = ancestor_ref {
            unsafe {
                if is_sx {
                    (*ancestor_ref).sx = Some(new_fork);
                } else {
                    (*ancestor_ref).dx = Some(new_fork);
                }
            }
        } else {
            origin = Some(new_fork);
        }

        list.push((middle + 1, end, height + 1, Some(new_fork_ptr), false));
        list.push((begin, middle, height + 1, Some(new_fork_ptr), true));
    }

    origin
}

#[inline(always)]
fn l2_norm_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut total = 0.0;
    for i in 0..a.len() {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
    }
    total
}

#[inline(always)]
fn l2_norm_squared_limited(a: &[f32], b: &[f32], c : f32) -> f32 {
    let mut total = 0.0;
    for i in 0..180 {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
    }
    if total > c {
        total;
    }
    for i in 180..a.len() {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
    }
    total
}
#[inline(always)]
fn cutoff_dist(a: &[f32], b: &[f32], current_min: f32) -> f32 {
    let mut total = 0.0;
    let mut i = 0;
    let len = a.len();

    if a.len() != b.len() || a.len() < 8 {
        return f32::MAX;
    }

    while i + 7 < len {
        unsafe {
            let d0 = *a.get_unchecked(i) - *b.get_unchecked(i);
            let d1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
            let d2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
            let d3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
            let d4 = *a.get_unchecked(i + 4) - *b.get_unchecked(i + 4);
            let d5 = *a.get_unchecked(i + 5) - *b.get_unchecked(i + 5);
            let d6 = *a.get_unchecked(i + 6) - *b.get_unchecked(i + 6);
            let d7 = *a.get_unchecked(i + 7) - *b.get_unchecked(i + 7);

            total += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 +
                d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        }

        if total > current_min {
            return f32::MAX;
        }

        i += 8;
    }

    while i < len {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
        i += 1;
    }
    total
}

fn proximity_search<'a>(
    origin: &Option<Box<Fork<'a>>>,
    dest: &[f32],
    optimal: &mut (f32, Option<usize>),
) {
    let DIMENSION_COUNT = dest.len();
    let mut list = Vec::with_capacity(64);

    if let Some(fork) = origin {
        list.push((fork.as_ref(), 0));
    }

    while let Some((fork, height)) = list.pop() {
        let dimension = height % DIMENSION_COUNT;
        let dist = cutoff_dist(&fork.coordinate, dest, optimal.0);

        if dist < optimal.0 {
            optimal.0 = dist;
            optimal.1 = Some(fork.i);
        }

        let d = dest[dimension] - fork.coordinate[dimension];
        let sqr_d = d * d;

        let (proximal, remoter) = if d < 0.0 {
            (&fork.sx, &fork.dx)
        } else {
            (&fork.dx, &fork.sx)
        };

        if let Some(proximal_fork) = proximal {
            list.push((proximal_fork.as_ref(), height + 1));
        }

        if sqr_d < optimal.0 {
            if let Some(remoter_fork) = remoter {
                list.push((remoter_fork.as_ref(), height + 1));
            }
        }
    }
}

fn compute_centroid(arrays: &[&[f32]]) -> Vec<f32> {
    let num_arrays = arrays.len();
    let DIMENSION_COUNT = 250;

    let mut average_array = vec![0.0; DIMENSION_COUNT];

    for array in arrays {
        for i in 0..DIMENSION_COUNT {
            average_array[i] += array[i];
        }
    }

    for i in 0..DIMENSION_COUNT {
        average_array[i] /= num_arrays as f32;
    }

    average_array
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

fn extract_matching_arrays<'a>(
    data_store: &'a [Vec<f32>],
    fetch_arrays: &[Vec<f32>],
    k: usize,
) -> Vec<(&'a [f32], usize)> {
    let fetch_refs: Vec<&[f32]> = fetch_arrays.iter().map(|v| &v[..]).collect();
    let average_query_ref = compute_centroid(&fetch_refs);

    let mut priority_queue: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

    for (i, array) in data_store.iter().enumerate() {
        if priority_queue.len() < k 
        {
            let dist = l2_norm_squared(&average_query_ref, array);
            let ord_dist = FloatOrd(dist);

            priority_queue.push((ord_dist, i));
        } else if let Some(&(FloatOrd(top_dist), _)) = priority_queue.peek() 
        {
            let dist = l2_norm_squared_limited(&average_query_ref, array, top_dist);
            let ord_dist = FloatOrd(dist);
            if dist < top_dist {
                priority_queue.pop();
                priority_queue.push((ord_dist, i));
            }
        }
    }
    
    priority_queue
        .into_iter()
        .map(|(_, i)| (&data_store[i][..], i))
        .collect()
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let query_count = challenge.query_vectors.len();

    let max_fuel = 2000000000.0;
    let base_fuel = 760000000.0;
    let alpha = 1700.0 * challenge.difficulty.num_queries as f64;

    let subgroup_size = ((max_fuel - base_fuel) / alpha) as usize;
    let subgroup = extract_matching_arrays(
        &challenge.vector_database,
        &challenge.query_vectors,
        subgroup_size,
    );


    let fork_tree: Option<Box<Fork<'_>>> = create_fork_tree(&mut subgroup.clone());
    let mut optimal_indexes = Vec::with_capacity(challenge.query_vectors.len());

    for query in challenge.query_vectors.iter() {
        let mut optimal = (std::f32::MAX, None);
        proximity_search(&fork_tree, query, &mut optimal);

        if let Some(optimal_index) = optimal.1 {
            optimal_indexes.push(optimal_index);
        }
    }


    Ok(Some(Solution {
        indexes: optimal_indexes,
    }))
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