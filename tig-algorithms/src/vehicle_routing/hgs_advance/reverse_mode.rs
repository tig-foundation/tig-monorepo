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
use std::time::Instant;
use tig_challenges::vehicle_routing::*;

fn route_barycenter(data: &Problem, route: &[usize]) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cnt = 0usize;
    for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
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
    let route_clients: Vec<usize> = routes.iter().map(|r| r.len().saturating_sub(2)).collect();

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
            if best_cluster.is_none() || cluster_clients[c] < cluster_clients[best_cluster.unwrap()] {
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

fn build_subproblem(data: &Problem, clients: &[usize], vehicles_hint: usize) -> (Problem, Vec<usize>) {
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
    let lb_vehicles = ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
    let nb_vehicles = vehicles_hint.max(lb_vehicles).max(1).min(clients.len().max(1));

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
            node_positions,
            node_data,
        },
        local_to_global,
    )
}

fn map_sub_routes_to_global(sub_routes: &[Vec<usize>], local_to_global: &[usize]) -> Vec<Vec<usize>> {
    sub_routes
        .iter()
        .map(|r| r.iter().map(|&lid| local_to_global[lid]).collect())
        .collect()
}

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
            let _ = save(&Solution { routes: incumbent.routes.clone() });
        }
    }
}

#[derive(Clone)]
struct MasterState {
    routes: Vec<Vec<usize>>,
    ind: Individual,
}

pub(super) fn solve_reversed_mode(
    data: Problem,
    mut params: Params,
    t0: &Instant,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
) -> Result<Option<(Solution, i32, usize)>> {
    macro_rules! trace {
        ($($arg:tt)*) => {
            if params.display_traces {
                println!($($arg)*);
            }
        };
    }

    macro_rules! increase_penalties_for_infeasibility {
        ($ind:expr, $context:expr) => {{
            let had_capa = $ind.load_excess > 0;
            let had_tw = $ind.tw_violation > 0;
            let old_capa = params.penalty_capa;
            let old_tw = params.penalty_tw;
            if had_capa {
                params.penalty_capa = (2 * params.penalty_capa).min(10_000);
            }
            if had_tw {
                params.penalty_tw = (2 * params.penalty_tw).min(10_000);
            }
            trace!(
                "----- {}: INCREASING PENALTIES -> CAP {} -> {}, TW {} -> {}",
                $context,
                old_capa,
                params.penalty_capa,
                old_tw,
                params.penalty_tw
            );
        }};
    }

    let mut rng = SmallRng::from_seed(data.seed);
    let data_arc = Arc::new(data);

    let mut ls = LocalSearch::new(Arc::clone(&data_arc), params, &mut rng);
    let mut best_constructive_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;
    let mut best_ls_repair_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;

    for run_idx in 0..5 {
        let randomize = run_idx > 0;
        let constructed_routes = Constructive::build_routes(data_arc.as_ref(), &mut rng, randomize);
        let constructed_ind = Individual::new_from_routes(data_arc.as_ref(), &params, constructed_routes.clone());
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
            increase_penalties_for_infeasibility!(
                candidate_ind,
                format!("INITIAL MASTER START {} INFEASIBLE AFTER LS", run_idx + 1)
            );
            for factor in [5usize, 20, 100] {
                let repaired_routes = ls.continue_repair(&mut rng, params, factor);
                let repaired_ind =
                    Individual::new_from_routes(data_arc.as_ref(), &params, repaired_routes.clone());
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
    } else if let Some(sol) = best_constructive_feasible {
        trace!(
            "----- WARNING: MASTER LS+REPAIR FAILED TO RECOVER FEASIBILITY ON ALL STARTS, FALLING BACK TO BEST CONSTRUCTIVE SOLUTION"
        );
        let (routes, ind) = sol;
        MasterState { routes, ind }
    } else {
        panic!("No feasible constructive solution found among the 5 reversed-mode starts");
    };

    trace!("==============================================================");
    trace!("======================== REVERSED MODE =======================");
    trace!(
        "----- INITIAL GLOBAL SOLUTION: COST = {}, ROUTES = {}/{}, LB_ROUTES = {}",
        incumbent.ind.cost, incumbent.ind.nb_routes, data_arc.nb_vehicles, data_arc.lb_vehicles
    );
    trace!("==============================================================");

    if let Some(save) = save_solution {
        let _ = save(&Solution { routes: incumbent.routes.clone() });
    }

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
    let total_phases = exploration_schedule.len();

    for (phase_idx, &sub_exploration_level) in exploration_schedule.iter().enumerate() {
        let phase_global = phase_idx + 1;
        let phase_t0 = Instant::now();
        let phase_base = incumbent.clone();
        let k = ((data_arc.nb_nodes as f64) / (params.decomp_target_size as f64)).round() as usize;
        let clusters = cluster_route_indices(data_arc.as_ref(), &phase_base.routes, k, &mut rng);
        let nb_clusters = clusters.len();
        let mut residual_free_routes = (data_arc.nb_vehicles as i32) - (phase_base.routes.len() as i32);
        trace!(
            "----- REVERSED MODE PHASE {}/{}, EXP LEVEL = {}, {} CLUSTERS, COST = {}, ROUTES = {}/{}",
            phase_global,
            total_phases,
            sub_exploration_level,
            nb_clusters,
            phase_base.ind.cost,
            phase_base.ind.nb_routes,
            data_arc.nb_vehicles
        );

        let mut rebuilt_routes: Vec<Vec<usize>> = Vec::new();
        let mut phase_sub_improvements: Vec<i64> = Vec::new();
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
            if sub_problem.nb_nodes < 5 {
                rebuilt_routes.extend(route_group.iter().map(|&rid| phase_base.routes[rid].clone()));
                phase_sub_improvements.push(0);
                continue;
            }
            let mut global_to_local = vec![usize::MAX; data_arc.nb_nodes];
            for (lid, &gid) in local_to_global.iter().enumerate() {
                global_to_local[gid] = lid;
            }
            let seed_sub_routes: Vec<Vec<usize>> = route_group
                .iter()
                .map(|&rid| phase_base.routes[rid].iter().map(|&gid| global_to_local[gid]).collect())
                .collect();
            trace!(
                "----- SUBPROBLEM {:>3}/{:>3}: CLIENTS = {:>5}, FLEET SIZE LIMIT = {:>3}, RESIDUAL ROUTES = {:>3}",
                cid + 1,
                nb_clusters,
                clients.len(),
                sub_problem.nb_vehicles,
                residual_free_routes
            );
            let mut params_sub = Params::preset(sub_exploration_level, &sub_problem);
            params_sub.nb_it_traces = 1000;
            params_sub.decomp_nb_phases = 0;
            params_sub.display_traces = params.display_traces;
            let seed_sub_ind = Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
            debug_assert!(
                seed_sub_ind.load_excess == 0 && seed_sub_ind.tw_violation == 0,
                "Invariant violated: subproblem seed should always be feasible in reversed mode"
            );
            let seed_sub_route_count = seed_sub_routes.len();
            let mut sub_ga = Genetic::new(sub_problem, params_sub);
            let (chosen_sub_routes, chosen_sub_cost): (Vec<Vec<usize>>, i64) =
                match sub_ga.run(&mut rng, t0, None, Some(&seed_sub_routes)) {
                Some((sub_routes, sub_cost)) => {
                    let cand_sub_ind = Individual::new_from_routes(
                        sub_ga.root_data.as_ref(),
                        &sub_ga.params,
                        sub_routes.clone(),
                    );
                    if cand_sub_ind.cost < seed_sub_ind.cost {
                        (sub_routes, sub_cost as i64)
                    } else {
                        (seed_sub_routes, seed_sub_ind.cost)
                    }
                }
                None => (seed_sub_routes, seed_sub_ind.cost),
            };
            residual_free_routes += (seed_sub_route_count as i32) - (chosen_sub_routes.len() as i32);
            let sub_routes_global = map_sub_routes_to_global(&chosen_sub_routes, &local_to_global);
            phase_sub_improvements.push(seed_sub_ind.cost - chosen_sub_cost);

            rebuilt_routes.extend(sub_routes_global);

            let progressive_master_routes = build_progressive_master_routes(
                &phase_base.routes,
                &clusters,
                &rebuilt_routes,
                cid + 1,
            );
            let progressive_master_ind =
                Individual::new_from_routes(data_arc.as_ref(), &params, progressive_master_routes.clone());
            promote_if_improved(
                save_solution,
                &mut incumbent,
                &progressive_master_routes,
                &progressive_master_ind,
            );
        }

        let merged_ind = Individual::new_from_routes(data_arc.as_ref(), &params, rebuilt_routes.clone());
        debug_assert!(
            merged_ind.load_excess == 0 && merged_ind.tw_violation == 0,
            "Invariant violated: merged master solution should be feasible before global LS"
        );
        promote_if_improved(
            save_solution,
            &mut incumbent,
            &rebuilt_routes,
            &merged_ind,
        );
        let phase_impr_sum: i64 = phase_sub_improvements.iter().sum();
        trace!(
            "----- PHASE {} SUBPROBLEM IMPROVEMENTS: {:?} (SUM = {}) OVER {:.3} seconds",
            phase_global,
            phase_sub_improvements,
            phase_impr_sum,
            phase_t0.elapsed().as_secs_f64()
        );
        trace!(
            "----- PHASE {} DECOMP SOLUTION: COST = {}, ROUTES = {}/{}",
            phase_global, phase_base.ind.cost, phase_base.ind.nb_routes, data_arc.nb_vehicles
        );
        trace!(
            "----- PHASE {} MERGED SOLUTION: COST = {}, ROUTES = {}/{}",
            phase_global, merged_ind.cost, merged_ind.nb_routes, data_arc.nb_vehicles
        );
        let mut candidate_routes =
            ls.run_from_routes(&rebuilt_routes, &[], params, &mut rng);
        let mut candidate_ind = Individual::new_from_routes(data_arc.as_ref(), &params, candidate_routes.clone());
        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            trace!(
                "----- PHASE {} GLOBAL LS RETURNED INFEASIBLE WITH COST = {}, ROUTES = {}/{}, TRYING REPAIR",
                phase_global, candidate_ind.cost, candidate_ind.nb_routes, data_arc.nb_vehicles
            );
            increase_penalties_for_infeasibility!(
                candidate_ind,
                format!("PHASE {} GLOBAL LS INFEASIBLE", phase_global)
            );
            let mut repaired: Option<(Vec<Vec<usize>>, Individual)> = None;
            for factor in [5usize, 20, 100] {
                let routes = ls.continue_repair(&mut rng, params, factor);
                let ind = Individual::new_from_routes(data_arc.as_ref(), &params, routes.clone());
                if ind.load_excess == 0 && ind.tw_violation == 0 {
                    repaired = Some((routes, ind));
                    break;
                }
            }
            if let Some((routes, ind)) = repaired {
                if ind.cost <= merged_ind.cost {
                    trace!(
                        "----- PHASE {} REPAIR SUCCEEDED WITH COST = {}, ROUTES = {}/{}, KEEPING REPAIRED MASTER SOLUTION",
                        phase_global, ind.cost, ind.nb_routes, data_arc.nb_vehicles
                    );
                    candidate_routes = routes;
                    candidate_ind = ind;
                } else {
                    trace!(
                        "----- PHASE {} REPAIR WORSE THAN PRE-LS SOLUTION, ROLLBACK TO PRE-LS MASTER SOLUTION WITH COST = {}, ROUTES = {}/{}",
                        phase_global, merged_ind.cost, merged_ind.nb_routes, data_arc.nb_vehicles
                    );
                    candidate_routes = rebuilt_routes;
                    candidate_ind = merged_ind.clone();
                }
            } else {
                trace!(
                    "----- PHASE {} REPAIR FAILED, ROLLBACK TO PRE-LS MASTER SOLUTION WITH COST = {}, ROUTES = {}/{}",
                    phase_global, merged_ind.cost, merged_ind.nb_routes, data_arc.nb_vehicles
                );
                candidate_routes = rebuilt_routes;
                candidate_ind = merged_ind.clone();
            }
        }

        trace!(
            "----- PHASE {} AFTER GLOBAL LS: COST = {}, ROUTES = {}/{}",
            phase_global, candidate_ind.cost, candidate_ind.nb_routes, data_arc.nb_vehicles
        );
        promote_if_improved(
            save_solution,
            &mut incumbent,
            &candidate_routes,
            &candidate_ind,
        );
        trace!("==============================================================");
    }
    trace!(
        "----- REVERSED MODE FINAL: COST = {}, ROUTES = {}/{}",
        incumbent.ind.cost, incumbent.ind.nb_routes, data_arc.nb_vehicles
    );
    trace!("==============================================================");

    Ok(Some((
        Solution {
            routes: incumbent.routes.clone(),
        },
        incumbent.ind.cost as i32,
        incumbent.routes.len(),
    )))
}
