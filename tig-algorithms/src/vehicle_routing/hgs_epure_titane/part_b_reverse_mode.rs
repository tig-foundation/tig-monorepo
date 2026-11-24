use super::stopping::Stopping;
use super::constructive::Constructive;
use super::genetic::{canonical_routes, Genetic, MIN_FRESH_STARTS, REPAIR_FACTORS};
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vehicle_routing::*;

const MAX_PENALTY: usize = 10_000;
const NB_MASTER_STARTS: usize = 5;
const COLLAPSE_DIVERSITY: f64 = 0.02;
const MIN_SUBPROBLEM_NODES: usize = 5;
const SIZE_BUDGET_FLOOR_PERMILLE: usize = 600;
const MAX_CARRIED: usize = 64;

struct BlockDonors {
    donors: Vec<Vec<Vec<usize>>>,
    fallback: Vec<Vec<usize>>,
}

fn assemble_carried(
    blocks: &[BlockDonors],
    k: usize,
    nb_nodes: usize,
) -> Vec<Vec<Vec<usize>>> {
    let depth = blocks
        .iter()
        .map(|block| block.donors.len())
        .max()
        .unwrap_or(0)
        .min(k);
    let mut out: Vec<Vec<Vec<usize>>> = Vec::with_capacity(depth);
    let mut seen = vec![false; nb_nodes];
    for j in 0..depth {
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for block in blocks {
            match block.donors.get(j) {
                Some(solution) => routes.extend(solution.iter().cloned()),
                None => {
                    routes.extend(block.fallback.iter().cloned());
                }
            }
        }
        seen.fill(false);
        let mut covered = 0usize;
        let mut well_formed = true;
        for route in &routes {
            let len = route.len();
            if len < 3 || route[0] != 0 || route[len - 1] != 0 {
                well_formed = false;
                break;
            }
            for &id in &route[1..len - 1] {
                if id == 0 || id >= nb_nodes || seen[id] {
                    well_formed = false;
                    break;
                }
                seen[id] = true;
                covered += 1;
            }
            if !well_formed {
                break;
            }
        }
        if !well_formed || covered + 1 != nb_nodes {
            continue;
        }
        out.push(routes);
    }
    out
}

fn restrict_carried(
    carried: &[Vec<Vec<usize>>],
    sub: &Problem,
    global_to_local: &[usize],
    seed_sub_routes: &[Vec<usize>],
    cap: usize,
) -> Vec<Vec<Vec<usize>>> {
    let mut out: Vec<Vec<Vec<usize>>> = Vec::new();
    if cap == 0 {
        return out;
    }
    let nb_nodes = sub.nb_nodes;
    let seed_key = canonical_routes(seed_sub_routes);
    let mut keys: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut seen = vec![false; nb_nodes];
    for solution in carried {
        if out.len() >= cap {
            break;
        }
        seen.fill(false);
        let mut covered = 0usize;
        let mut well_formed = true;
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for route in solution {
            let len = route.len();
            if len < 3 || route[0] != 0 || route[len - 1] != 0 {
                well_formed = false;
                break;
            }
            let mut local: Vec<usize> = vec![0];
            for &gid in &route[1..len - 1] {
                let lid = match global_to_local.get(gid) {
                    Some(&lid) => lid,
                    None => {
                        well_formed = false;
                        break;
                    }
                };
                if lid == usize::MAX {
                    continue;
                }
                if lid == 0 || lid >= nb_nodes || seen[lid] {
                    well_formed = false;
                    break;
                }
                seen[lid] = true;
                covered += 1;
                local.push(lid);
            }
            if !well_formed {
                break;
            }
            if local.len() > 1 {
                local.push(0);
                routes.push(local);
            }
        }
        if !well_formed || covered + 1 != nb_nodes {
            continue;
        }
        if routes.len() > sub.nb_vehicles {
            continue;
        }
        let mut capa_ok = true;
        let mut tw_ok = true;
        for route in &routes {
            let mut seq = Sequence::singleton(sub, route[0]);
            for &id in &route[1..] {
                seq = Sequence::join2(sub, &seq, &Sequence::singleton(sub, id));
            }
            capa_ok &= seq.load <= sub.max_capacity;
            tw_ok &= seq.tw == 0;
        }
        if !capa_ok {
            continue;
        }
        if !tw_ok {
            continue;
        }
        let key = canonical_routes(&routes);
        if key == seed_key {
            continue;
        }
        if keys.contains(&key) {
            continue;
        }
        keys.push(key);
        out.push(routes);
    }
    out
}

#[inline]
fn scale_permille(value: usize, permille: usize) -> usize {
    value.saturating_mul(permille).saturating_add(500) / 1000
}

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
    mode: usize,
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

    if mode == 1 {
        let total: usize = route_clients.iter().sum();
        let (mut cx, mut cy) = (0.0, 0.0);
        for rid in 0..m {
            cx += barycenters[rid].0 * route_clients[rid] as f64;
            cy += barycenters[rid].1 * route_clients[rid] as f64;
        }
        cx /= total as f64;
        cy /= total as f64;
        let angles: Vec<f64> = barycenters.iter()
            .map(|&(x, y)| (y - cy).atan2(x - cx)).collect();
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| angles[a].total_cmp(&angles[b]));
        order.rotate_left(rng.gen_range(0..m));
        let target = total as f64 / kk as f64;
        let mut clusters = Vec::with_capacity(kk);
        let mut pos = 0usize;
        for c in 0..kk {
            let mut group = Vec::new();
            let mut clients = 0usize;
            while pos < m - (kk - c - 1)
                && (group.is_empty() || c + 1 == kk || (clients as f64) < target)
            {
                let rid = order[pos];
                group.push(rid);
                clients += route_clients[rid];
                pos += 1;
            }
            clusters.push(group);
        }
        return clusters;
    }

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

    if mode == 2 {
        for _ in 0..2 {
            let centers: Vec<(f64, f64)> = clusters.iter().map(|group| {
                let (mut sx, mut sy, mut weight) = (0.0, 0.0, 0usize);
                for &rid in group {
                    let count = route_clients[rid];
                    sx += barycenters[rid].0 * count as f64;
                    sy += barycenters[rid].1 * count as f64;
                    weight += count;
                }
                (sx / weight as f64, sy / weight as f64)
            }).collect();
            let mut next = vec![Vec::new(); centers.len()];
            for rid in 0..m {
                let (x, y) = barycenters[rid];
                let mut nearest = 0usize;
                let mut best_distance = f64::INFINITY;
                for (cid, &(cx, cy)) in centers.iter().enumerate() {
                    let dx = x - cx;
                    let dy = y - cy;
                    let distance = dx * dx + dy * dy;
                    if distance < best_distance {
                        best_distance = distance;
                        nearest = cid;
                    }
                }
                next[nearest].push(rid);
            }
            next.retain(|group| !group.is_empty());
            clusters = next;
        }
    }

    clusters
}

fn build_subproblem(
    data: &Problem,
    clients: &[usize],
    vehicles_hint: usize,
    cached: Option<Problem>,
) -> (Problem, Vec<usize>) {
    let mut local_to_global = Vec::with_capacity(clients.len() + 1);
    local_to_global.push(0);
    local_to_global.extend_from_slice(clients);

    if let Some(mut problem) = cached {
        problem.nb_vehicles = vehicles_hint
            .max(problem.lb_vehicles)
            .max(1)
            .min(clients.len().max(1));
        return (problem, local_to_global);
    }

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
    hyperparameters: &Option<Map<String, Value>>,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    mut stopping: Option<Stopping>,
) -> Result<Option<Solution>> {
    let mut rng = SmallRng::from_seed(data.seed);
    let data_arc = Arc::new(data);

    let mut ls = LocalSearch::new(Arc::clone(&data_arc), params);
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
        let rung_permille = [
            params.l1_permille,
            params.l2_permille,
            params.l3_permille,
            params.l4_permille,
        ];
        for level in 1..=params.exploration_level {
            let permille = match rung_permille.get(level - 1) {
                Some(&value) => value,
                None => 1000,
            };
            let nb_phases = (params.decomp_nb_phases * permille + 500) / 1000;
            for _ in 0..nb_phases {
                exploration_schedule.push(level);
            }
        }
    }

    let carry_k = params.inh_k.min(MAX_CARRIED);
    let carry_min_level = params.inh_min_level.max(1);
    let carry_active = carry_k > 0 && params.exploration_level > 0;
    let mut carried: Vec<Vec<Vec<usize>>> = Vec::new();
    let regional_cache_active = data_arc.nb_nodes == 1000;
    let mut regional_cache: Vec<(Vec<usize>, Problem)> = Vec::new();
    let mut structural_history = if data_arc.nb_nodes == 1000 {
        Some((
            vec![0.0f64; data_arc.nb_nodes],
            None::<Vec<(usize, usize)>>,
        ))
    } else {
        None
    };

    let mut phase_idx = 0usize;
    let mut ot_rounds: i64 = 0;
    let mut decompositions_since_polish = 0usize;
    const DECOMP_EXPLORE: u32 = 5;
    const DECOMP_DECAY: f64 = 0.5;
    const DECOMP_RATE_DECAY: f64 = 0.7;
    let decomp_active = data_arc.nb_nodes == 800;
    let mut score = [0.0f64; 3];
    let mut rate_ema = 0.0f64;
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
        if stopping.as_mut().map_or(false, |a| a.settled(incumbent.ind.cost as i64, phase_idx, exploration_schedule.len())) {
            break;
        }
        let phase_cost_before = if decomp_active { incumbent.ind.cost } else { 0 };
        let phase_fuel_before = if decomp_active { fuel_left() } else { 0 };
        let mode = if !decomp_active {
            0
        } else if phase_idx < score.len() {
            phase_idx
        } else {
            let mut best = 0usize;
            for candidate in 1..score.len() {
                if score[candidate] > score[best] {
                    best = candidate;
                }
            }
            if rng.gen_ratio(1, DECOMP_EXPLORE) {
                (best + 1 + rng.gen_range(0..score.len() - 1)) % score.len()
            } else {
                best
            }
        };
        let phase_base_routes = incumbent.routes.clone();
        let k = ((data_arc.nb_nodes as f64) / (params.decomp_target_size as f64)).round() as usize;
        let clusters = cluster_route_indices(data_arc.as_ref(), &phase_base_routes, k, &mut rng, mode);
        let mut residual_free_routes =
            (data_arc.nb_vehicles as i32) - (phase_base_routes.len() as i32);
        let receive =
            carry_active && sub_exploration_level >= carry_min_level && !carried.is_empty();
        let export = carry_active
            && match exploration_schedule.get(phase_idx + 1) {
                Some(&next_level) => next_level >= carry_min_level,
                None => ot_rounds < params.ot_max && sub_exploration_level >= carry_min_level,
            };
        let mut blocks: Vec<BlockDonors> = Vec::new();

        let mut phase_budgets: Vec<[usize; 2]> = Vec::new();
        if let Some((contraction, previous_links)) = structural_history.as_mut() {
            let mut links = vec![(0usize, 0usize); data_arc.nb_nodes];
            for route in &phase_base_routes {
                for window in route.windows(3) {
                    links[window[1]] = (window[0], window[2]);
                }
            }
            if let Some(previous) = previous_links.as_ref() {
                let mut counts = vec![0usize; clusters.len()];
                let mut changed = vec![0usize; clusters.len()];
                let mut rates = vec![0.0f64; clusters.len()];
                for (cid, group) in clusters.iter().enumerate() {
                    for &rid in group {
                        let route = &phase_base_routes[rid];
                        for &gid in &route[1..route.len() - 1] {
                            counts[cid] += 1;
                            changed[cid] += usize::from(links[gid] != previous[gid]);
                            rates[cid] += contraction[gid];
                        }
                    }
                    if counts[cid] + 1 < MIN_SUBPROBLEM_NODES {
                        changed[cid] = 0;
                        rates[cid] = 0.0;
                    } else {
                        rates[cid] /= counts[cid] as f64;
                    }
                }
                let changed_total: u128 = changed.iter().map(|&n| n as u128).sum();
                if changed_total > 0 {
                    let full_params = Params::initialize_at_level(
                        hyperparameters,
                        sub_exploration_level,
                        data_arc.as_ref(),
                    );
                    let it_permille = match params.sub_it_l4_permille {
                        Some(value) if sub_exploration_level >= 4 => value,
                        _ => params.sub_it_permille,
                    };
                    let mut full = [
                        full_params.max_it_noimprov,
                        full_params.max_it_total,
                    ];
                    if it_permille != 1000 {
                        for budget in &mut full {
                            *budget = scale_permille(*budget, it_permille);
                        }
                    }
                    for &count in &counts {
                        if count + 1 < MIN_SUBPROBLEM_NODES {
                            phase_budgets.push([0, 0]);
                            continue;
                        }
                        let size_permille = (count * 1000 / params.decomp_target_size)
                            .clamp(SIZE_BUDGET_FLOOR_PERMILLE, 1000);
                        phase_budgets.push([
                            scale_permille(full[0], size_permille).max(1),
                            scale_permille(full[1], size_permille).max(1),
                        ]);
                    }
                    for account in 0..2 {
                        let mut withdrawn = 0u128;
                        for cid in 0..clusters.len() {
                            let budget = phase_budgets[cid][account];
                            if changed[cid] == 0 && rates[cid] > 0.0 && budget > 1 {
                                let reduction = ((budget as f64 * rates[cid] / 5.0)
                                    .floor() as usize)
                                    .min(budget - 1);
                                phase_budgets[cid][account] -= reduction;
                                withdrawn += reduction as u128;
                            }
                        }
                        let available = withdrawn / 5;
                        let mut shares: Vec<(usize, u128, u128)> = changed
                            .iter()
                            .enumerate()
                            .filter(|(_, count)| **count > 0)
                            .map(|(cid, &count)| {
                                let numerator = available * count as u128;
                                (
                                    cid,
                                    numerator / changed_total,
                                    numerator % changed_total,
                                )
                            })
                            .collect();
                        let remainder = available
                            - shares.iter().map(|share| share.1).sum::<u128>();
                        shares.sort_by(|a, b| {
                            b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0))
                        });
                        for (rank, &(cid, share, _)) in shares.iter().enumerate() {
                            let extra = share + u128::from((rank as u128) < remainder);
                            let room = full[account]
                                .max(1)
                                .saturating_sub(phase_budgets[cid][account]);
                            phase_budgets[cid][account] += extra.min(room as u128) as usize;
                        }
                    }
                    for budget in &mut phase_budgets {
                        budget[0] = budget[0].min(budget[1]);
                    }
                }
            }
            contraction.fill(0.0);
            *previous_links = Some(links);
        }

        let mut next_regional_cache: Vec<(Vec<usize>, Problem)> = Vec::new();
        let mut rebuilt_routes: Vec<Vec<usize>> = Vec::new();
        let mut given_up = false;
        for (cid, route_group) in clusters.iter().enumerate() {
            let mut clients: Vec<usize> = Vec::new();
            for &rid in route_group {
                clients.extend_from_slice(
                    &phase_base_routes[rid][1..phase_base_routes[rid].len().saturating_sub(1)],
                );
            }
            if clients.is_empty() {
                continue;
            }

            let cached = if regional_cache_active {
                regional_cache
                    .iter()
                    .position(|(key, _)| key == &clients)
                    .map(|index| regional_cache.remove(index).1)
            } else {
                None
            };
            let (sub_problem, local_to_global) = build_subproblem(
                data_arc.as_ref(),
                &clients,
                ((route_group.len() as i32) + residual_free_routes).max(1) as usize,
                cached,
            );
            if sub_problem.nb_nodes < MIN_SUBPROBLEM_NODES {
                if export {
                    blocks.push(BlockDonors {
                        donors: Vec::new(),
                        fallback: route_group
                            .iter()
                            .map(|&rid| phase_base_routes[rid].clone())
                            .collect(),
                    });
                }
                rebuilt_routes.extend(
                    route_group
                        .iter()
                        .map(|&rid| phase_base_routes[rid].clone()),
                );
                if regional_cache_active {
                    next_regional_cache.push((clients, sub_problem));
                }
                continue;
            }
            let mut global_to_local = vec![usize::MAX; data_arc.nb_nodes];
            for (lid, &gid) in local_to_global.iter().enumerate() {
                global_to_local[gid] = lid;
            }
            let seed_sub_routes: Vec<Vec<usize>> = route_group
                .iter()
                .map(|&rid| {
                    phase_base_routes[rid]
                        .iter()
                        .map(|&gid| global_to_local[gid])
                        .collect()
                })
                .collect();
            let mut params_sub =
                Params::initialize_at_level(hyperparameters, sub_exploration_level, &sub_problem);
            params_sub.decomp_nb_phases = 0;
            if params.sub_mu_permille != 1000 {
                params_sub.mu_start =
                    scale_permille(params_sub.mu_start, params.sub_mu_permille).max(1);
            }
            let it_permille = match params.sub_it_l4_permille {
                Some(value) if sub_exploration_level >= 4 => value,
                _ => params.sub_it_permille,
            };
            if it_permille != 1000 {
                params_sub.max_it_noimprov =
                    scale_permille(params_sub.max_it_noimprov, it_permille);
                params_sub.max_it_total = scale_permille(params_sub.max_it_total, it_permille);
            }
            if data_arc.nb_nodes == 1000 {
                let size_permille =
                    ((sub_problem.nb_nodes - 1) * 1000 / params.decomp_target_size)
                        .clamp(SIZE_BUDGET_FLOOR_PERMILLE, 1000);
                params_sub.max_it_noimprov =
                    scale_permille(params_sub.max_it_noimprov, size_permille).max(1);
                params_sub.max_it_total =
                    scale_permille(params_sub.max_it_total, size_permille).max(1);
            }
            if let Some(budget) = phase_budgets.get(cid) {
                params_sub.max_it_noimprov = budget[0];
                params_sub.max_it_total = budget[1];
            }
            params_sub.ot_cap = if data_arc.nb_nodes == 1000 && sub_exploration_level >= 3 {
                COLLAPSE_DIVERSITY
            } else {
                0.0
            };
            let seed_sub_ind =
                Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
            debug_assert!(
                seed_sub_ind.load_excess == 0 && seed_sub_ind.tw_violation == 0,
                "Invariant violated: subproblem seed should always be feasible in reversed mode"
            );
            let seed_sub_route_count = seed_sub_routes.len();
            let migrants = if receive {
                let cap = carry_k.min(params_sub.mu_start.saturating_sub(MIN_FRESH_STARTS));
                restrict_carried(&carried, &sub_problem, &global_to_local, &seed_sub_routes, cap)
            } else {
                Vec::new()
            };
            let mut sub_ga = Genetic::new(sub_problem, params_sub);
            sub_ga.inherited = migrants;
            let chosen_sub_routes: Vec<Vec<usize>> =
                match sub_ga.run(&mut rng, None, Some(&seed_sub_routes)) {
                    Some(candidate) => {
                        if candidate.cost < seed_sub_ind.cost {
                            candidate.routes
                        } else {
                            seed_sub_routes
                        }
                    }
                    None => seed_sub_routes,
                };
            if let Some((contraction, _)) = structural_history.as_mut() {
                let covered: usize = sub_ga.client_expansion
                    .iter()
                    .skip(1)
                    .map(|clients| clients.len())
                    .sum();
                if covered > 0 {
                    let remaining = sub_ga.data.nb_nodes.saturating_sub(1);
                    let rate = covered.saturating_sub(remaining) as f64 / covered as f64;
                    for expansion in sub_ga.client_expansion.iter().skip(1) {
                        for &lid in expansion {
                            contraction[local_to_global[lid]] = rate;
                        }
                    }
                }
            }
            if export {
                let donors: Vec<Vec<Vec<usize>>> = sub_ga
                    .export_feasible(carry_k)
                    .iter()
                    .map(|routes| map_sub_routes_to_global(routes, &local_to_global))
                    .collect();
                blocks.push(BlockDonors {
                    donors,
                    fallback: route_group
                        .iter()
                        .map(|&rid| phase_base_routes[rid].clone())
                        .collect(),
                });
            }
            if regional_cache_active {
                let root_data = Arc::clone(&sub_ga.root_data);
                drop(sub_ga);
                if let Ok(problem) = Arc::try_unwrap(root_data) {
                    next_regional_cache.push((clients, problem));
                }
            }
            residual_free_routes +=
                (seed_sub_route_count as i32) - (chosen_sub_routes.len() as i32);
            let sub_routes_global = map_sub_routes_to_global(&chosen_sub_routes, &local_to_global);

            rebuilt_routes.extend(sub_routes_global);

            let progressive_master_routes = build_progressive_master_routes(
                &phase_base_routes,
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
            if data_arc.nb_nodes == 1000 && cid + 1 < clusters.len() {
                let done = phase_idx * k + cid + 1;
                let total = exploration_schedule.len() * k;
                if stopping.as_ref().map_or(false, |s| s.abandoned(incumbent.ind.cost as i64, done, total)) {
                    given_up = true;
                    break;
                }
            }
        }

        if given_up {
            break;
        }

        if regional_cache_active {
            regional_cache = next_regional_cache;
        }

        carried = if export {
            assemble_carried(&blocks, carry_k, data_arc.nb_nodes)
        } else {
            Vec::new()
        };

        let merged_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, rebuilt_routes.clone());
        debug_assert!(
            merged_ind.load_excess == 0 && merged_ind.tw_violation == 0,
            "Invariant violated: merged master solution should be feasible before global LS"
        );
        promote_if_improved(save_solution, &mut incumbent, &rebuilt_routes, &merged_ind);
        if data_arc.nb_nodes == 1000 {
            decompositions_since_polish += 1;
        }
        let polish = data_arc.nb_nodes != 1000
            || decompositions_since_polish >= 2
            || exploration_schedule
                .get(phase_idx + 1)
                .map_or(true, |&next_level| next_level != sub_exploration_level);
        if polish {
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
                        candidate_ind = merged_ind;
                    }
                } else {
                    candidate_routes = rebuilt_routes;
                    candidate_ind = merged_ind;
                }
            }

            promote_if_improved(
                save_solution,
                &mut incumbent,
                &candidate_routes,
                &candidate_ind,
            );
            decompositions_since_polish = 0;
        }
        if decomp_active {
            let fuel_used = phase_fuel_before.saturating_sub(fuel_left()).max(1);
            let r = (phase_cost_before - incumbent.ind.cost) as f64 / fuel_used as f64;
            let rel = if rate_ema == 0.0 { 1.0 } else { r / rate_ema };
            score[mode] = DECOMP_DECAY * score[mode] + (1.0 - DECOMP_DECAY) * rel;
            rate_ema = DECOMP_RATE_DECAY * rate_ema + 0.3 * r;
        }
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

    Ok(Some(Solution { routes: incumbent.routes }))
}
