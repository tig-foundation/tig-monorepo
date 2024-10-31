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
    point:                          &Point,
    lower_frontier:                 &Frontier,
    upper_frontier:                 &Frontier
)                                           -> PointCompareFrontiers
{
    for point_ in lower_frontier.iter()
    {
        if pareto_compare(point, point_) == ParetoCompare::BDominatesA
        {
            return PointCompareFrontiers::Below;
        }
    }

    for point_ in upper_frontier.iter()
    {
        if pareto_compare(point, point_) == ParetoCompare::ADominatesB
        {
            return PointCompareFrontiers::Above;
        }
    }

    return PointCompareFrontiers::Within;
}

pub fn scale_point(
    point:                          &Point,
    min_point:                      &Point,
    max_point:                      &Point,
    multiplier:                     f64
)                                           -> Point
{
    return point.iter()
        .enumerate()
        .map(|(i, value)|
        {
            let offset                          = ((value - min_point[i] + 1) as f64) * multiplier;
            (min_point[i] + offset.ceil() as i32 - 1).clamp(min_point[i], max_point[i])   
        })
        .collect();
}

pub fn scale_frontier(
    frontier:                       &Frontier,
    min_point:                      &Point,
    max_point:                      &Point,
    multiplier:                     f64
)                                           -> Frontier
{
    if frontier.is_empty()
    {
        return vec![];
    }

    let scaled_frontier                         = frontier
        .iter()
        .map(|point| scale_point(&point, min_point, max_point, multiplier))
        .collect();

    if multiplier > 1.0
    {
        return pareto_frontier(&scaled_frontier);
    }

    let mirrored_frontier                      = scaled_frontier
        .into_iter()
        .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
        .collect::<Frontier>();

    return pareto_frontier(&mirrored_frontier)
        .iter()
        .map(|d| d.iter().map(|x| -x).collect())
        .collect();
}

pub fn o_pareto_algorithm(
    points:                         &Vec<Vec<i32>>, 
    only_one:                       bool
)                                       -> Vec<Vec<Point>>
{
    if points.len() == 0
    {
        return vec![];
    }

    let mut frontiers               = Vec::new();
    let mut remaining_points        : Option<Vec<Vec<i32>>> = None;

    while true
    {
        let points_                 = if remaining_points.is_some() { &remaining_points.unwrap() } else { points };
        let on_front                = o_is_pareto_front(points_, false);

        // Extract frontier points
        let frontier                : Vec<_> = points_
            .iter()
            .zip(on_front.iter())
            .filter(|(_, &is_front)| is_front)
            .map(|(point, _)| point.to_vec())
            .collect();

        frontiers.push(frontier);

        let new_points              : Vec<_> = points_
            .iter()
            .zip(on_front.iter())
            .filter(|(_, &is_front)| !is_front)
            .map(|(point, _)| point.to_vec())
            .collect();

        if new_points.is_empty() 
        {
            break;
        }

        remaining_points            = Some(new_points);

        if only_one 
        {
            break;
        }
    }

    return frontiers;
}

pub fn pareto_frontier(
    frontier:                       &Frontier,
)
                                            -> Frontier
{
    return o_pareto_algorithm(frontier, true).first().unwrap().to_vec();
}