// optimized pareto impl

fn is_pareto_front_2d(
    costs:                          &Vec<Vec<i32>>
)                                           -> Vec<bool> 
{
    let n_observations                          = costs.len();
    if n_observations == 0 
    {
        return vec![];
    }

    let mut indices                             : Vec<usize> = (0..n_observations).collect();
    indices.sort_by_key(|&i| costs[i][0]);

    let mut on_front                            = vec![true; n_observations];
    let mut stack                               = Vec::with_capacity(n_observations);
    for &curr_idx in indices.iter() 
    {
        // Remove points from stack that are dominated by current point
        while let Some(&top_idx) = stack.last() 
        {
            let cost1: &Vec<i32>                = &costs[top_idx];
            let cost2: &Vec<i32>                = &costs[curr_idx];

            if cost1[1] <= cost2[1]
            {
                break;
            }

            stack.pop();
        }
        
        // If stack is not empty, current point is dominated
        if let Some(&top_idx) = stack.last() 
        {
            let cost1: &Vec<i32>                = &costs[top_idx];
            let cost2: &Vec<i32>                = &costs[curr_idx];

            if cost1[1] <= cost2[1]
            {
                on_front[curr_idx]              = false;
            }
        }
        
        // Add current point to stack
        stack.push(curr_idx);
    }

    return on_front;
}

pub fn o_is_pareto_front(
    costs:                          &Vec<Vec<i32>>,
    assume_unique_lexsorted:        bool
)                                           -> Vec<bool> 
{
    let apply_unique                            = !assume_unique_lexsorted;
    let (unique_costs, order_inv)               = if apply_unique 
    {
        let (unique, indices)                   = unique_with_indices(costs);
        
        (Some(unique), Some(indices))
    } 
    else 
    {
        (None, None)
    };

    let on_front                                = if unique_costs.is_some() 
    { 
        is_pareto_front_2d(&unique_costs.unwrap())
    }
    else
    {
        is_pareto_front_2d(costs)
    };

    if let Some(inv) = order_inv 
    {
        return inv.iter().map(|&i| on_front[i]).collect();
    } 
    
    return on_front;
}

use std::collections::HashMap;
pub fn unique_with_indices(
    arr:                                &Vec<Vec<i32>>
)                                           -> (Vec<Vec<i32>>, Vec<usize>) 
{
    let n                                       = arr.len();
    let mut unique                              = Vec::with_capacity(n);
    let mut indices                             = Vec::with_capacity(n);
    let mut seen                                = HashMap::with_capacity(n);
    
    for (i, point) in arr.iter().enumerate() 
    {
        if let Some(&idx) = seen.get(point) 
        {
            indices.push(idx);
        } 
        else 
        {
            seen.insert(point, i);
            unique.push(point.clone());
            indices.push(i);
        }
    }
    
    return (unique, indices);
}