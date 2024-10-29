// optimized pareto impl

use
{
    std::
    {
        sync::
        {
            Arc,
            Mutex
        }  
    },
    ndarray::
    {
        s
    },
    crate::
    {
        Point,
        Frontier
    }
};

use ndarray::{Array2, ArrayView2, Axis};
use std::cmp::Ordering;

fn change_directions(
    costs:                                  &mut Array2<i32>, 
    larger_is_better_objectives:                    Option<&[usize]>)
{
    let n_objectives                                    = costs.shape()[1];

    if let Some(larger_is_better) = larger_is_better_objectives 
    {
        if larger_is_better.is_empty() 
        {
            return;
        }

        if larger_is_better.iter().any(|&i| i >= n_objectives) 
        {
            panic!("The indices specified in larger_is_better_objectives must be in [0, n_objectives(={})), but got {:?}", n_objectives, larger_is_better);
        }

        for &i in larger_is_better 
        {
            costs.slice_mut(s![.., i]).mapv_inplace(|x| -x);
        }
    }
}

fn is_pareto_front_2d(
    costs:                                  ArrayView2<i32>
)                                                   -> Vec<bool> 
{
    let n_observations                                  = costs.shape()[0];
    if n_observations == 0 
    {
        return vec![];
    }

    let mut indices                                     : Vec<usize> = (0..n_observations).collect();
    indices.sort_by_key(|&i| costs[[i, 0]]);

    let mut on_front                                    = vec![true; n_observations];
    let mut stack                                       = Vec::with_capacity(n_observations);
    for &curr_idx in indices.iter() 
    {
        // Remove points from stack that are dominated by current point
        while let Some(&top_idx) = stack.last() 
        {
            if costs[[top_idx, 1]] <= costs[[curr_idx, 1]] 
            {
                break;
            }

            stack.pop();
        }
        
        // If stack is not empty, current point is dominated
        if let Some(&top_idx) = stack.last() 
        {
            if costs[[top_idx, 1]] <= costs[[curr_idx, 1]] 
            {
                on_front[curr_idx]                      = false;
            }
        }
        
        // Add current point to stack
        stack.push(curr_idx);
    }

    return on_front;
}

fn is_pareto_front_nd(
    costs:                                  ArrayView2<i32>
) 
                                                    -> Vec<bool> 
{
    let n_observations                                  = costs.shape()[0];
    let mut on_front                                    = vec![false; n_observations];
    let mut nondominated_indices                        : Vec<usize> = (0..n_observations).collect();
    let mut costs                                       = costs.to_owned();

    while !costs.is_empty() 
    {
        let nondominated_and_not_top                    : Vec<bool> = costs
            .axis_iter(Axis(0))
            .map(|row| row.iter().zip(costs.row(0).iter()).any(|(a, b)| a < b))
            .collect();

        on_front[nondominated_indices[0]]               = true;

        let mut __idx                                   = Vec::with_capacity(nondominated_and_not_top.len());
        for i in 0..nondominated_and_not_top.len()
        {
            if nondominated_and_not_top[i]
            {
                __idx.push(i);
            }
        }
        
        costs                                           = costs.select(Axis(0), &__idx);
        nondominated_indices                            = nondominated_indices
            .into_iter()
            .zip(nondominated_and_not_top)
            .filter(|&(_, b)| b)
            .map(|(i, _)| i)
            .collect();
    }

    return on_front;
}

pub fn o_is_pareto_front(
    costs:                                  ArrayView2<i32>,
    larger_is_better_objectives:            Option<&[usize]>,
    assume_unique_lexsorted:                bool
)                                                   -> Vec<bool> 
{
    change_directions(&mut costs.to_owned(), larger_is_better_objectives);
    let apply_unique                                    = larger_is_better_objectives.is_some() || !assume_unique_lexsorted;

    let (unique_costs, order_inv)                       = if apply_unique 
    {
        let (unique, indices)                           = unique_with_indices(costs.view());

        (unique.to_owned(), Some(indices))
    } 
    else 
    {
        (costs.to_owned(), None)
    };

    let on_front                                        = if costs.shape()[1] == 2 
    {
        is_pareto_front_2d(unique_costs.view())
    } 
    else 
    {
        is_pareto_front_nd(unique_costs.view())
    };

    if let Some(inv) = order_inv 
    {
        return inv.iter().map(|&i| on_front[i]).collect();
    } 
    else 
    {
        return on_front;
    }
}

fn _nondominated_rank(costs: ArrayView2<i32>) -> Vec<usize> 
{
    let (n_observations, n_obj)                         = costs.dim();
    
    if n_obj == 1 
    {
        let mut sorted                                  = costs.column(0).to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mut rank                                    = vec![0; n_observations];
        for (i, &val) in costs.column(0).iter().enumerate() 
        {
            rank[i]                                     = sorted
                .binary_search_by(|probe| probe.partial_cmp(&val)
                .unwrap_or(Ordering::Equal))
                .unwrap();
        }

        return rank;
    }

    let mut ranks                                       = vec![0; n_observations];
    let mut rank                                        = 0;
    let mut indices                                     : Vec<usize> = (0..n_observations).collect();
    let mut costs                                       = costs.to_owned();

    while !indices.is_empty() 
    {
        let indices_len                                 = indices.len();

        let on_front                                    = o_is_pareto_front(costs.view(), None, true);
        for (idx, &is_front) in indices.iter().zip(on_front.iter()) 
        { 
            if is_front 
            {
                ranks[*idx]                             = rank;
            }
        }
        
        indices                                         = Vec::new();
        for idx in 0..indices_len
        {
            if !on_front[idx]
            {
                indices.push(idx);
            }
        }

        costs                                           = costs.select(Axis(0), &indices);
        rank                                            += 1;
    }

    return ranks;
}

// Note: The tie_break functionality is not implemented in this Rust version
pub fn o_nondominated_rank(
    costs:                                  ArrayView2<i32>,
    larger_is_better_objectives:            Option<&[usize]>
)                                                   -> Vec<usize> 
{
    let (_, n_obj)                                      = costs.dim();
    change_directions(&mut costs.to_owned(), larger_is_better_objectives);
    
    let (unique_costs, order_inv)                       = unique_with_indices(costs.view());
    let ranks                                           = _nondominated_rank(unique_costs.view());

    return order_inv.iter().map(|&i| ranks[i]).collect();
}

// Helper function to get unique rows with inverse indices
fn unique_with_indices(
    arr:                                    ArrayView2<i32>
)                                                   -> (Array2<i32>, Vec<usize>) 
{
    let mut unique_vec                                  : Vec<(Vec<i32>, usize)> = Vec::new();
    let mut inverse                                     = vec![0; arr.shape()[0]];

    for (i, row) in arr.outer_iter().enumerate() 
    {
        let row_vec: Vec<i32>                           = row.to_vec();
        match unique_vec.iter().position(|(v, _)| v == &row_vec) 
        {
            Some(pos) =>
            {
                inverse[i]                              = pos;
            },

            None => 
            {
                inverse[i]                              = unique_vec.len();
                unique_vec.push((row_vec, i));
            }
        }
    }

    unique_vec.sort_by(|a, b| a.1.cmp(&b.1));
    let unique_arr                                      = Array2::from_shape_vec(
        (unique_vec.len(), arr.shape()[1]), 
        unique_vec.into_iter().flat_map(|(v, _)| v).collect()
    )
    .unwrap();

    return (unique_arr, inverse);
}