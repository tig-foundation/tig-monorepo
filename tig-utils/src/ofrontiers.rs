// optimized pareto impl
pub type Point      = Vec<i32>;
pub type Frontier   = Vec<Point>;

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

// will be about 1.3x faster if we use this and cache it somehow instead of calling it repeatedely on the same points
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
            seen.insert(point, unique.len());
            unique.push(point.clone());
            indices.push(unique.len() - 1);
        }
    }
    
    return (unique, indices);
}

#[derive(PartialEq)]
pub enum ParetoCompare
{
    ADominatesB,
    Equal,
    BDominatesA
}

pub fn pareto_compare(
    point:                          &Point, 
    other:                          &Point
)                                           -> ParetoCompare 
{
    let mut a_dominate_b = false;
    let mut b_dominate_a = false;
    for (a_val, b_val) in point.iter().zip(other) 
    {
        if a_val < b_val 
        {
            b_dominate_a                        = true;
        } 
        else if a_val > b_val 
        {
            a_dominate_b                        = true;
        }
    }

    if a_dominate_b == b_dominate_a 
    {
        return ParetoCompare::Equal;
    } 
    else if a_dominate_b 
    {
        return ParetoCompare::ADominatesB;
    } 
    else 
    {
        return ParetoCompare::BDominatesA;
    }
}

#[derive(PartialEq)]
pub enum PointCompareFrontiers
{
    Below,
    Within,
    Above
}

pub fn pareto_within(
    point:                          &Vec<Point>,
    lower_frontier:                 &Frontier,
    upper_frontier:                 &Frontier
)                                           -> PointCompareFrontiers
{
    for point_ in lower_frontier
    {
        if pareto_compare(point, point_) == ParetoCompare::BDominatesA
        {
            return PointCompareFrontiers::Below;
        }
    }

    for point_ in upper_frontier
    {
        if pareto_compare(point, point_) == ParetoCompare::ADominatesB
        {
            return PointCompareFrontiers::Above;
        }
    }

    return PointCompareFrontiers::Within;
}