use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
    use tig_challenges::vehicle_routing::*;


    pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
        let mut solution = Solution {
            sub_solutions: Vec::new(),
        };
        for sub_instance in &challenge.sub_instances {
            match solve_sub_instance(sub_instance)? {
                Some(sub_solution) => solution.sub_solutions.push(sub_solution),
                None => return Ok(None),
            }
        }
        Ok(Some(solution))
    }

    pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
        let dist: &Vec<Vec<i32>> = &challenge.distance_matrix;
        let capacity: i32 = challenge.max_capacity;
        let num_forks: usize = challenge.num_nodes;

        let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
    
        if challenge.baseline_total_distance as f32 / max_dist < 0.57 { return Ok(None) }

        let mut values: Vec<(i32, usize, usize)> = Vec::with_capacity((num_forks-1)*(num_forks-2)/2);
        for v1 in 1..num_forks {
            for v2 in (v1 + 1)..num_forks {
                values.push((dist[v1][0] + dist[0][v2] - dist[v1][v2], v1, v2));
            }
        }

        values.sort_unstable_by(|a, b| b.0.cmp(&a.0));    
    
        let mut paths: Vec<Option<Vec<usize>>> = (0..num_forks).map(|i| Some(vec![i])).collect();
        paths[0] = None;
        let mut path_demands: Vec<i32> = challenge.demands.clone();
   
        for (idx, v1, v2) in values {
            if idx < 0 { break; }

            if paths[v1].is_none() || paths[v2].is_none() {
                continue;
            }

            let sx_path = paths[v1].as_ref().unwrap();
            let dx_path = paths[v2].as_ref().unwrap();
            let mut sx_startfork = sx_path[0];
            let dx_startfork = dx_path[0];
            let sx_endfork = sx_path[sx_path.len() - 1];
            let mut dx_endfork = dx_path[dx_path.len() - 1];
            let merged_demand = path_demands[sx_startfork] + path_demands[dx_startfork];

            if sx_startfork == dx_startfork || merged_demand > capacity { continue; }

            let mut sx_path = paths[v1].take().unwrap();
            let mut dx_path = paths[v2].take().unwrap();
            paths[sx_startfork] = None;
            paths[dx_startfork] = None;
            paths[sx_endfork] = None;
            paths[dx_endfork] = None;

            if sx_startfork == v1 {
                sx_path.reverse();
                sx_startfork = sx_endfork;
            }
            if dx_endfork == v2 {
                dx_path.reverse();
                dx_endfork = dx_startfork;
            }

            let mut new_path = sx_path;
            new_path.extend(dx_path);

            paths[sx_startfork] = Some(new_path.clone());
            paths[dx_endfork] = Some(new_path);
            path_demands[sx_startfork] = merged_demand;
            path_demands[dx_endfork] = merged_demand;
        }

        Ok(Some(SubSolution {
            routes: paths
            .into_iter()
            .enumerate()
            .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
            .map(|(_, mut x)| {
                let mut path = vec![0];
                path.append(x.as_mut().unwrap());
                path.push(0);
                path
            })
            .collect(),
        }))
    }
}

pub fn help() {
    println!("No help information available.");
}
