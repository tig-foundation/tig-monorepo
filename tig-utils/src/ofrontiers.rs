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
    costs:                          &HashSet<Vec<i32>>
)                                           -> Vec<bool> 
{
    return is_pareto_front_2d(costs);
}

fn unique_with_indices(
    arr:                            &HashSet<Vec<i32>>
)                                           -> (HashSet<Vec<i32>>, Vec<usize>) 
{
    let mut unique_vec                          : Vec<(Vec<i32>, usize)> = Vec::new();
    let mut inverse                             = vec![0; arr.len()];
    let arr_vec: Vec<_>                         = arr.iter().collect();

    for (i, row) in arr_vec.iter().enumerate() 
    {
        match unique_vec.iter().position(|(v, _)| v == *row) 
        {
            Some(pos) => 
            {
                inverse[i]                      = pos;
            },

            None => 
            {
                inverse[i]                      = unique_vec.len();
                unique_vec.push(((*row).clone(), i));
            }
        }
    }

    unique_vec.sort_by(|a, b| a.1.cmp(&b.1));
    let unique_arr: HashSet<_>                  = unique_vec.into_iter().map(|(v, _)| v).collect();

    return (unique_arr, inverse);
}