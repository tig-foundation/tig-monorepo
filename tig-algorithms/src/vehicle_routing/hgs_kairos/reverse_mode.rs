use super::constructive::Constructive;
use super::genetic::Genetic;
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::problem::Problem;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;
use tig_challenges::vehicle_routing::*;

const MAX_PENALTY: usize = 10_000;
const REPAIR_FACTORS: [usize; 3] = [5, 20, 100];
const NB_MASTER_STARTS: usize = 5;
/// Below this node count a subproblem is handed back unchanged instead of being re-solved.
const MIN_SUBPROBLEM_NODES: usize = 5;

/// A route is `[depot, clients..., depot]`, so the client count is `len - 2`.
#[inline]
fn nb_clients_in_route(route: &[usize]) -> usize {
    route.len().saturating_sub(2)
}

fn route_barycenter(data: &Problem, route: &[usize]) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cnt = 0usize;
    for &id in route.iter().skip(1).take(nb_clients_in_route(route)) {
        sx += data.node_positions[id].0 as f64;
        sy += data.node_positions[id].1 as f64;
        cnt += 1;
    }
    (sx / cnt as f64, sy / cnt as f64)
}

fn cluster_route_indices(
    data: &Problem,
    routes: &[Vec<usize>],
    k: usize,
    rng: &mut SmallRng,
) -> Vec<Vec<usize>> {
    let m = routes.len();
    if m == 0 {
        return Vec::new();
    }
    debug_assert!(
        routes.iter().all(|r| r.len() > 2),
        "Reverse-mode decomposition should not receive empty routes"
    );
    let kk = k.max(1).min(m);

    let barycenters: Vec<(f64, f64)> = routes.iter().map(|r| route_barycenter(data, r)).collect();
    let route_clients: Vec<usize> = routes.iter().map(|r| nb_clients_in_route(r)).collect();

    let mut route_indices: Vec<usize> = (0..m).collect();
    route_indices.shuffle(rng);
    let center_routes: Vec<usize> = route_indices[..kk].to_vec();

    let mut assigned = vec![false; m];
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); kk];
    let mut cluster_clients: Vec<usize> = vec![0; kk];

    for c in 0..kk {
        let rid = center_routes[c];
        assigned[rid] = true;
        clusters[c].push(rid);
        cluster_clients[c] += route_clients[rid];
    }

    let ordered_candidates: Vec<Vec<usize>> = center_routes
        .iter()
        .map(|&center_rid| {
            let (cx, cy) = barycenters[center_rid];
            let mut cand: Vec<(f64, usize)> = (0..m)
                .filter(|&rid| rid != center_rid)
                .map(|rid| {
                    let (x, y) = barycenters[rid];
                    let dx = x - cx;
                    let dy = y - cy;
                    (dx * dx + dy * dy, rid)
                })
                .collect();
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            cand.into_iter().map(|(_, rid)| rid).collect()
        })
        .collect();

    let mut candidate_pos = vec![0usize; kk];
    let mut remaining = m - kk;
    while remaining > 0 {
        let mut best_cluster: Option<usize> = None;

        for c in 0..kk {
            while candidate_pos[c] < ordered_candidates[c].len()
                && assigned[ordered_candidates[c][candidate_pos[c]]]
            {
                candidate_pos[c] += 1;
            }
            if candidate_pos[c] == ordered_candidates[c].len() {
                continue;
            }
            if best_cluster.is_none() || cluster_clients[c] < cluster_clients[best_cluster.unwrap()]
            {
                best_cluster = Some(c);
            }
        }

        let c = best_cluster.expect("At least one cluster should still have assignable routes");
        let rid = ordered_candidates[c][candidate_pos[c]];
        candidate_pos[c] += 1;
        assigned[rid] = true;
        clusters[c].push(rid);
        cluster_clients[c] += route_clients[rid];
        remaining -= 1;
    }

    clusters
}

/// Builds a standalone `Problem` over `clients`, plus the local-to-global node index map.
/// Local index 0 is always the depot, so `local_to_global[0] == 0`.
fn build_subproblem(
    data: &Problem,
    clients: &[usize],
    vehicles_hint: usize,
) -> (Problem, Vec<usize>) {
    let mut local_to_global = Vec::with_capacity(clients.len() + 1);
    local_to_global.push(0);
    local_to_global.extend_from_slice(clients);

    let nb_nodes = local_to_global.len();
    let mut node_positions = Vec::with_capacity(nb_nodes);
    let mut node_data = Vec::with_capacity(nb_nodes);
    for &gid in &local_to_global {
        node_positions.push(data.node_positions[gid]);
        node_data.push(data.node_data[gid]);
    }

    let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
    for i in 0..nb_nodes {
        let gi = local_to_global[i];
        for j in 0..nb_nodes {
            let gj = local_to_global[j];
            distance_matrix[i * nb_nodes + j] = data.dm(gi, gj);
        }
    }

    let total_demand: i64 = clients.iter().map(|&id| data.nd(id).demand as i64).sum();
    let lb_vehicles =
        ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
    let nb_vehicles = vehicles_hint
        .max(lb_vehicles)
        .max(1)
        .min(clients.len().max(1));
    let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);

    (
        Problem {
            seed: data.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            is_vrptw: data.is_vrptw,
            fixed_distance_offset: 0,
            max_capacity: data.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_positions,
            node_data,
        },
        local_to_global,
    )
}

fn map_sub_routes_to_global(
    sub_routes: &[Vec<usize>],
    local_to_global: &[usize],
) -> Vec<Vec<usize>> {
    sub_routes
        .iter()
        .map(|r| r.iter().map(|&lid| local_to_global[lid]).collect())
        .collect()
}

/// Rebuilt routes for the clusters already processed, followed by the untouched base routes
/// of the clusters from `next_cluster_idx` onwards.
fn build_progressive_master_routes(
    phase_base_routes: &[Vec<usize>],
    clusters: &[Vec<usize>],
    rebuilt_routes: &[Vec<usize>],
    next_cluster_idx: usize,
) -> Vec<Vec<usize>> {
    let mut routes = rebuilt_routes.to_vec();
    for route_group in clusters.iter().skip(next_cluster_idx) {
        for &rid in route_group {
            routes.push(phase_base_routes[rid].clone());
        }
    }
    routes
}

unsafe extern "C" {
    static __fuel_remaining: u64;
}
#[inline(always)]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

#[derive(Clone)]
struct MasterState {
    routes: Vec<Vec<usize>>,
    ind: Individual,
}

fn promote_if_improved(
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    incumbent: &mut MasterState,
    candidate_routes: &[Vec<usize>],
    candidate_ind: &Individual,
) {
    if candidate_ind.cost < incumbent.ind.cost {
        incumbent.routes = candidate_routes.to_vec();
        incumbent.ind = candidate_ind.clone();
        if let Some(save) = save_solution {
            let _ = save(&Solution {
                routes: incumbent.routes.clone(),
            });
        }
    }
}

fn increase_penalties_for_infeasibility(params: &mut Params, ind: &Individual) {
    let had_capa = ind.load_excess > 0;
    let had_tw = ind.tw_violation > 0;
    if had_capa {
        params.penalty_capa = (2 * params.penalty_capa).min(MAX_PENALTY);
    }
    if had_tw {
        params.penalty_tw = (2 * params.penalty_tw).min(MAX_PENALTY);
    }
}

pub(super) fn solve_reversed_mode(
    data: Problem,
    mut params: Params,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
) -> Result<Option<(Solution, i32, usize)>> {
    let mut rng = SmallRng::from_seed(data.seed);
    let data_arc = Arc::new(data);

    let mut ls = LocalSearch::new(Arc::clone(&data_arc), params, &mut rng);
    let mut best_constructive_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;
    let mut best_ls_repair_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;

    for run_idx in 0..NB_MASTER_STARTS {
        let randomize = run_idx > 0;
        let constructed_routes = Constructive::build_routes(data_arc.as_ref(), &mut rng, randomize);
        let constructed_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, constructed_routes.clone());
        if constructed_ind.load_excess == 0 && constructed_ind.tw_violation == 0 {
            let replace = best_constructive_feasible
                .as_ref()
                .map_or(true, |(_, best)| constructed_ind.cost < best.cost);
            if replace {
                best_constructive_feasible = Some((constructed_routes.clone(), constructed_ind));
            }
        }

        let routes_after_ls = ls.run_from_routes(&constructed_routes, &[], params, &mut rng);
        let mut candidate_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, routes_after_ls.clone());
        let mut candidate_routes = routes_after_ls;

        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            increase_penalties_for_infeasibility(&mut params, &candidate_ind);
            for factor in REPAIR_FACTORS {
                let repaired_routes = ls.continue_repair(&mut rng, params, factor);
                let repaired_ind = Individual::new_from_routes(
                    data_arc.as_ref(),
                    &params,
                    repaired_routes.clone(),
                );
                if repaired_ind.load_excess == 0 && repaired_ind.tw_violation == 0 {
                    candidate_routes = repaired_routes;
                    candidate_ind = repaired_ind;
                    break;
                }
            }
        }

        if candidate_ind.load_excess == 0 && candidate_ind.tw_violation == 0 {
            let replace = best_ls_repair_feasible
                .as_ref()
                .map_or(true, |(_, best)| candidate_ind.cost < best.cost);
            if replace {
                best_ls_repair_feasible = Some((candidate_routes, candidate_ind));
            }
        }
    }

    let mut incumbent = if let Some((routes, ind)) = best_ls_repair_feasible {
        MasterState { routes, ind }
    } else if let Some((routes, ind)) = best_constructive_feasible {
        MasterState { routes, ind }
    } else {
        panic!("No feasible constructive solution found among the reversed-mode starts");
    };

    if let Some(save) = save_solution {
        let _ = save(&Solution {
            routes: incumbent.routes.clone(),
        });
    }

    let fuel_start = fuel_left();
    let fuel_budget = ((fuel_start as f64) * params.fuel_use).max(1.0) as u64;

    let mut exploration_schedule: Vec<usize> = Vec::new();
    if params.exploration_level == 0 {
        for _ in 0..params.decomp_nb_phases {
            exploration_schedule.push(0);
        }
    } else {
        for level in 1..=params.exploration_level {
            for _ in 0..params.decomp_nb_phases {
                exploration_schedule.push(level);
            }
        }
    }

    // The schedule can grow while the loop runs, hence the index rather than an iterator.
    let mut phase_idx = 0usize;
    let mut ot_rounds: i64 = 0;
    while phase_idx < exploration_schedule.len() {
        let sub_exploration_level = exploration_schedule[phase_idx];
        let spent = fuel_start.saturating_sub(fuel_left());
        let prog = (spent as f64) / (fuel_budget as f64);
        if params.giveup_distance > 0
            && params.giveup_at > 0.0
            && prog >= params.giveup_at
            && (incumbent.ind.cost as i64) > params.giveup_distance
        {
            break;
        }
        if params.stop_distance > 0 && (incumbent.ind.cost as i64) <= params.stop_distance {
            break;
        }
        let phase_base = incumbent.clone();
        let k = ((data_arc.nb_nodes as f64) / (params.decomp_target_size as f64)).round() as usize;
        let clusters = cluster_route_indices(data_arc.as_ref(), &phase_base.routes, k, &mut rng);
        let mut residual_free_routes =
            (data_arc.nb_vehicles as i32) - (phase_base.routes.len() as i32);

        let mut rebuilt_routes: Vec<Vec<usize>> = Vec::new();
        for (cid, route_group) in clusters.iter().enumerate() {
            let mut clients: Vec<usize> = Vec::new();
            for &rid in route_group {
                clients.extend_from_slice(
                    &phase_base.routes[rid][1..phase_base.routes[rid].len().saturating_sub(1)],
                );
            }
            if clients.is_empty() {
                continue;
            }

            let (sub_problem, local_to_global) = build_subproblem(
                data_arc.as_ref(),
                &clients,
                ((route_group.len() as i32) + residual_free_routes).max(1) as usize,
            );
            if sub_problem.nb_nodes < MIN_SUBPROBLEM_NODES {
                rebuilt_routes.extend(
                    route_group
                        .iter()
                        .map(|&rid| phase_base.routes[rid].clone()),
                );
                continue;
            }
            let mut global_to_local = vec![usize::MAX; data_arc.nb_nodes];
            for (lid, &gid) in local_to_global.iter().enumerate() {
                global_to_local[gid] = lid;
            }
            let seed_sub_routes: Vec<Vec<usize>> = route_group
                .iter()
                .map(|&rid| {
                    phase_base.routes[rid]
                        .iter()
                        .map(|&gid| global_to_local[gid])
                        .collect()
                })
                .collect();
            let mut params_sub = Params::preset(sub_exploration_level, &sub_problem);
            params_sub.decomp_nb_phases = 0;
            let seed_sub_ind =
                Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
            debug_assert!(
                seed_sub_ind.load_excess == 0 && seed_sub_ind.tw_violation == 0,
                "Invariant violated: subproblem seed should always be feasible in reversed mode"
            );
            let seed_sub_route_count = seed_sub_routes.len();
            let mut sub_ga = Genetic::new(sub_problem, params_sub);
            let chosen_sub_routes: Vec<Vec<usize>> =
                match sub_ga.run(&mut rng, None, Some(&seed_sub_routes)) {
                    Some((sub_routes, _sub_cost)) => {
                        let cand_sub_ind = Individual::new_from_routes(
                            sub_ga.root_data.as_ref(),
                            &sub_ga.params,
                            sub_routes.clone(),
                        );
                        if cand_sub_ind.cost < seed_sub_ind.cost {
                            sub_routes
                        } else {
                            seed_sub_routes
                        }
                    }
                    None => seed_sub_routes,
                };
            residual_free_routes +=
                (seed_sub_route_count as i32) - (chosen_sub_routes.len() as i32);
            let sub_routes_global = map_sub_routes_to_global(&chosen_sub_routes, &local_to_global);

            rebuilt_routes.extend(sub_routes_global);

            let progressive_master_routes = build_progressive_master_routes(
                &phase_base.routes,
                &clusters,
                &rebuilt_routes,
                cid + 1,
            );
            let progressive_master_ind = Individual::new_from_routes(
                data_arc.as_ref(),
                &params,
                progressive_master_routes.clone(),
            );
            promote_if_improved(
                save_solution,
                &mut incumbent,
                &progressive_master_routes,
                &progressive_master_ind,
            );
        }

        let merged_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, rebuilt_routes.clone());
        debug_assert!(
            merged_ind.load_excess == 0 && merged_ind.tw_violation == 0,
            "Invariant violated: merged master solution should be feasible before global LS"
        );
        promote_if_improved(save_solution, &mut incumbent, &rebuilt_routes, &merged_ind);
        let mut candidate_routes = ls.run_from_routes(&rebuilt_routes, &[], params, &mut rng);
        let mut candidate_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, candidate_routes.clone());
        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            increase_penalties_for_infeasibility(&mut params, &candidate_ind);
            let mut repaired: Option<(Vec<Vec<usize>>, Individual)> = None;
            for factor in REPAIR_FACTORS {
                let routes = ls.continue_repair(&mut rng, params, factor);
                let ind = Individual::new_from_routes(data_arc.as_ref(), &params, routes.clone());
                if ind.load_excess == 0 && ind.tw_violation == 0 {
                    repaired = Some((routes, ind));
                    break;
                }
            }
            if let Some((routes, ind)) = repaired {
                if ind.cost <= merged_ind.cost {
                    candidate_routes = routes;
                    candidate_ind = ind;
                } else {
                    candidate_routes = rebuilt_routes;
                    candidate_ind = merged_ind.clone();
                }
            } else {
                candidate_routes = rebuilt_routes;
                candidate_ind = merged_ind.clone();
            }
        }

        promote_if_improved(
            save_solution,
            &mut incumbent,
            &candidate_routes,
            &candidate_ind,
        );
        phase_idx += 1;

        if ot_rounds < params.ot_max
            && phase_idx >= exploration_schedule.len()
            && params.stop_distance > 0
            && params.ot_cap > 0.0
            && (incumbent.ind.cost as i64) > params.stop_distance
            && (incumbent.ind.cost as i64) <= params.stop_distance + params.ot_window
        {
            let spent = fuel_start.saturating_sub(fuel_left());
            if (spent as f64) < (fuel_start as f64) * params.ot_cap {
                ot_rounds += 1;
                let deepest = *exploration_schedule.last().unwrap_or(&1);
                let extra = params.decomp_nb_phases.max(1);
                for _ in 0..extra {
                    exploration_schedule.push(deepest);
                }
            }
        }
    }

    Ok(Some((
        Solution {
            routes: incumbent.routes.clone(),
        },
        incumbent.ind.cost as i32,
        incumbent.routes.len(),
    )))
}
