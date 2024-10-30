// optimized pareto impl

use std::collections::HashSet;
use std::cmp::Ordering;

fn change_directions(
    costs:                          &mut Vec<Vec<i32>>,
    larger_is_better_objectives:    Option<&[usize]>
) 
{
    if costs.is_empty() 
    {
        return;
    }

    let n_objectives                            = costs[0].len();

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

        for point in costs.iter_mut() 
        {
            for &i in larger_is_better 
            {
                point[i]                        = -point[i];
            }
        }
    }
}

fn is_pareto_front_2d(
    costs:                          &HashSet<Vec<i32>>
)                                           -> Vec<bool> 
{
    let n_observations = costs.len();
    if n_observations == 0 
    {
        return vec![];
    }

    let mut indices                             : Vec<usize> = (0..n_observations).collect();
    indices.sort_by_key(|&i| costs.iter().nth(i).unwrap()[0]);

    let mut on_front                            = vec![true; n_observations];
    let mut stack                               = Vec::with_capacity(n_observations);
    for &curr_idx in indices.iter() 
    {
        // Remove points from stack that are dominated by current point
        while let Some(&top_idx) = stack.last() 
        {
            if costs.iter().nth(top_idx).unwrap()[1] <= costs.iter().nth(curr_idx).unwrap()[1]
            {
                break;
            }

            stack.pop();
        }
        
        // If stack is not empty, current point is dominated
        if let Some(&top_idx) = stack.last() 
        {
            if costs.iter().nth(top_idx).unwrap()[1] <= costs.iter().nth(curr_idx).unwrap()[1]
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
    costs:                          &HashSet<Vec<i32>>,
    assume_unique_lexsorted:        bool
)                                           -> Vec<bool> 
{
    //let mut costs_copy                                  = costs.clone();
    //change_directions(&mut costs_copy, larger_is_better_objectives);

    let costs_copy                                      = costs;

    let apply_unique                                    = !assume_unique_lexsorted;
    let (unique_costs, order_inv)                       = if apply_unique 
    {
        let (unique, indices)                           = unique_with_indices(&costs_copy);
        
        (Some(unique), Some(indices))
    } 
    else 
    {
        (None, None)
    };

    let on_front                                        = if unique_costs.is_some() 
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
fn unique_with_indices(
    arr:                            &HashSet<Vec<i32>>
)                                           -> (HashSet<Vec<i32>>, Vec<usize>) 
{
    let n                                       = arr.len();
    let mut unique                              = HashSet::with_capacity(n);
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
            unique.insert(point.clone());
            
            indices.push(i);
        }
    }
    
    return (unique, indices);
}