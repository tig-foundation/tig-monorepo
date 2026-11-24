use super::compression::ProblemCompression;
use super::constructive::Constructive;
use super::genetic::{canonical_routes, Genetic, MIN_FRESH_STARTS, REPAIR_FACTORS};
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::problem::Problem;
use super::quality::{evaluate_total_distance, greedy_total_distance, quality_of};
use super::sequence::Sequence;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vehicle_routing::*;

const GAIN_P99_900: [i64; 32] = [
    74319, 69501, 64683, 59865, 55047, 50229, 45411, 40593,
    35775, 30957, 26139, 21321, 16503, 11685, 6867, 6452,
    6037, 5622, 5207, 4792, 4377, 3961, 3566, 3171,
    2777, 2382, 1987, 1592, 1198, 803, 401, 0,
];

const GAIN_P99_700: [i64; 32] = [
    69441, 38332, 38332, 23896, 18276, 18276, 14619, 12466,
    12466, 10599, 10191, 10191, 9028, 7483, 7483, 6028,
    5514, 5514, 4051, 3343, 3343, 3164, 2529, 2529,
    2023, 1831, 1831, 1079, 942, 942, 547, 547,
];

const G700_PROGRESS: [i64; 32] = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 9837, 1000000, 10025, 8826, 1000000, 7124, 6193, 1000000, 4629, 3477, 1000000, 3199, 2657, 1000000, 2284, 1870, 1000000, 1270, 932, 1000000, 661, 485, 1000000, 248, 1000000];

const MAX_PENALTY: usize = 10_000;
const TARGET_SCHEDULE_MIN_PHASES: usize = 6;
const MIN_SUBPROBLEM_NODES: usize = 5;
const MAX_CARRIED: usize = 64;

struct BlockDonors {
    donors: Vec<Vec<Vec<usize>>>,
    fallback: Vec<Vec<usize>>,
}

fn join_b(
    blocks: &[BlockDonors],
    k: usize,
    nb_nodes: usize,
    carried_rr: bool,
    carried_sim_permille: usize,
) -> Vec<Vec<Vec<usize>>> {
    let depth = blocks
        .iter()
        .map(|block| block.donors.len())
        .max()
        .unwrap_or(0)
        .min(k);
    let mut out: Vec<Vec<Vec<usize>>> = Vec::with_capacity(depth);
    let diversity_filter = carried_sim_permille > 0;
    let mut succ_scratch = if diversity_filter {
        vec![usize::MAX; nb_nodes]
    } else {
        Vec::new()
    };
    let mut accepted_succ: Vec<Vec<usize>> =
        Vec::with_capacity(if diversity_filter { depth } else { 0 });
    let nb_clients = nb_nodes.saturating_sub(1);
    let similarity_margin = if diversity_filter {
        (1.0 - carried_sim_permille.min(1000) as f64 / 1000.0) * (nb_clients as f64)
    } else {
        0.0
    };
    let mut seen = vec![false; nb_nodes];
    let mut next_donor = if carried_rr {
        vec![0usize; blocks.len()]
    } else {
        Vec::new()
    };
    for j in 0..depth {
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for (bi, block) in blocks.iter().enumerate() {
            if carried_rr {
                if block.donors.is_empty() {
                    routes.extend(block.fallback.iter().cloned());
                } else {
                    let idx = next_donor[bi] % block.donors.len();
                    routes.extend(block.donors[idx].iter().cloned());
                    next_donor[bi] += 1;
                }
            } else {
                match block.donors.get(j) {
                    Some(solution) => routes.extend(solution.iter().cloned()),
                    None => {
                        routes.extend(block.fallback.iter().cloned());
                    }
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
        if diversity_filter {
            succ_scratch.fill(usize::MAX);
            for route in &routes {
                let len = route.len();
                for pos in 1..len - 1 {
                    succ_scratch[route[pos]] = route[pos + 1];
                }
            }

            if out.len() >= 2 && nb_clients > 0 {
                let mut too_similar = false;
                for accepted in &accepted_succ {
                    let mut disagree = 0usize;
                    for c in 1..nb_nodes {
                        if succ_scratch[c] != accepted[c] {
                            disagree += 1;
                            if (disagree as f64) >= similarity_margin {
                                break;
                            }
                        }
                    }
                    if (disagree as f64) < similarity_margin {
                        too_similar = true;
                        break;
                    }
                }
                if too_similar {
                    continue;
                }
            }

            accepted_succ.push(succ_scratch.clone());
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
    params: &Params,
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

    if params.cluster_tw_weight > 0.0 || params.cluster_lloyd > 0 {
        let coordinates: Vec<(f64, f64, f64)> = routes
            .iter()
            .enumerate()
            .map(|(rid, route)| {
                let (x, y) = barycenters[rid];
                let t = if params.cluster_tw_weight > 0.0 {
                    let total: f64 = route
                        .iter()
                        .skip(1)
                        .take(route_clients[rid])
                        .map(|&id| {
                            let nd = data.nd(id);
                            (nd.start_tw as f64 + nd.end_tw as f64) / 2.0
                        })
                        .sum();
                    params.cluster_tw_weight * (total / route_clients[rid] as f64)
                } else {
                    0.0
                };
                (x, y, t)
            })
            .collect();
        let mut centers: Vec<(f64, f64, f64)> =
            center_routes.iter().map(|&rid| coordinates[rid]).collect();
        let mut clusters: Vec<Vec<usize>> = center_routes.iter().map(|&rid| vec![rid]).collect();

        for iteration in 0..=params.cluster_lloyd {
            if iteration > 0 {
                for c in 0..kk {
                    let (mut sx, mut sy, mut st) = (0.0, 0.0, 0.0);
                    let mut count = 0usize;
                    for &rid in &clusters[c] {
                        let (x, y, t) = coordinates[rid];
                        let weight = route_clients[rid] as f64;
                        sx += x * weight;
                        sy += y * weight;
                        st += t * weight;
                        count += route_clients[rid];
                    }
                    centers[c] = (sx / count as f64, sy / count as f64, st / count as f64);
                    clusters[c].clear();
                }
            }

            let mut assigned = vec![false; m];
            let mut cluster_clients = vec![0usize; kk];
            let mut remaining = m;
            for c in 0..kk {
                for &rid in &clusters[c] {
                    assigned[rid] = true;
                    cluster_clients[c] += route_clients[rid];
                    remaining -= 1;
                }
            }
            let ordered_candidates: Vec<Vec<usize>> = centers
                .iter()
                .map(|&(cx, cy, ct)| {
                    let mut cand: Vec<(f64, usize)> = (0..m)
                        .map(|rid| {
                            let (x, y, t) = coordinates[rid];
                            let dx = x - cx;
                            let dy = y - cy;
                            let dt = t - ct;
                            (dx * dx + dy * dy + dt * dt, rid)
                        })
                        .collect();
                    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    cand.into_iter().map(|(_, rid)| rid).collect()
                })
                .collect();
            let mut candidate_pos = vec![0usize; kk];
            while remaining > 0 {
                let c = (0..kk).min_by_key(|&c| cluster_clients[c]).unwrap();
                while assigned[ordered_candidates[c][candidate_pos[c]]] {
                    candidate_pos[c] += 1;
                }
                let rid = ordered_candidates[c][candidate_pos[c]];
                candidate_pos[c] += 1;
                assigned[rid] = true;
                clusters[c].push(rid);
                cluster_clients[c] += route_clients[rid];
                remaining -= 1;
            }
            debug_assert!(clusters.iter().all(|cluster| !cluster.is_empty()));
        }

        return clusters;
    }

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
            root_nb_nodes: data.root_nb_nodes,
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

fn stable_chains(
    seed_sub_routes: &[Vec<usize>],
    local_to_global: &[usize],
    succ_prev1: &[usize],
    succ_prev2: &[usize],
) -> Vec<Vec<usize>> {
    const STABLE_CHAIN_MAX: usize = 6;
    let mut chains = Vec::new();
    for route in seed_sub_routes {
        let mut chain: Vec<usize> = Vec::new();
        for &client in route {
            if client == 0 {
                if !chain.is_empty() {
                    chains.push(chain);
                    chain = Vec::new();
                }
                continue;
            }
            if let Some(&previous) = chain.last() {
                let from = local_to_global[previous];
                let to = local_to_global[client];
                if chain.len() == STABLE_CHAIN_MAX
                    || succ_prev1[from] != to
                    || succ_prev2[from] != to
                {
                    chains.push(chain);
                    chain = Vec::new();
                }
            }
            chain.push(client);
        }
        if !chain.is_empty() {
            chains.push(chain);
        }
    }
    chains
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

#[derive(Clone)]
struct MasterState {
    routes: Vec<Vec<usize>>,
    ind: Individual,
}

#[derive(Clone)]
struct TriageSnapshot {
    rng: SmallRng,
    ls: LocalSearch,
    params: Params,
    incumbent: MasterState,
    exploration_schedule: Vec<usize>,
    carried: Vec<Vec<Vec<usize>>>,
    penalty_memory: [Option<(usize, f64)>; 4],
    adaptive_target_size: usize,
    window_start_cost: i64,
    greedy_distance: Option<u32>,
    triage_sub_it_active: bool,
    phase_idx: usize,
    ot_rounds: i64,
    normal_snapshot: Option<(u32, i32, i32)>,
    band_added: usize,
    band_stalled: usize,
    band_previous_cost: i64,
    post_added: usize,
    post_stalled: usize,
    post_previous_cost: i64,
}

fn save_with_diag(
    challenge: &Challenge,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    routes: &[Vec<usize>],
    greedy_distance: &mut Option<u32>,
) -> Result<()> {
    if let Some(save) = save_solution {
        let solution = Solution {
            routes: routes.to_vec(),
        };
        save(&solution)?;
        let greedy = match *greedy_distance {
            Some(g) => g,
            None => {
                let g = greedy_total_distance(challenge)?;
                *greedy_distance = Some(g);
                g
            }
        };
        let distance = evaluate_total_distance(challenge, &solution)?;
        let quality = quality_of(greedy, distance);
        println!("DIAG quality={}", quality);
    }
    Ok(())
}

fn promote_if_improved(
    challenge: &Challenge,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    save_below: i64,
    incumbent: &mut MasterState,
    candidate_routes: &[Vec<usize>],
    candidate_ind: &Individual,
    greedy_distance: &mut Option<u32>,
) {
    if candidate_ind.cost < incumbent.ind.cost {
        incumbent.routes = candidate_routes.to_vec();
        incumbent.ind = candidate_ind.clone();
        if incumbent.ind.cost < save_below {
            let _ = save_with_diag(
                challenge,
                save_solution,
                &incumbent.routes,
                greedy_distance,
            );
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
    challenge: &Challenge,
    data: Problem,
    mut params: Params,
    hyperparameters: &Option<Map<String, Value>>,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    seed: Option<Vec<Vec<usize>>>,
    greedy: Option<u32>,
) -> Result<Option<Solution>> {
    let mut rng = SmallRng::from_seed(data.seed);
    let root_nb_nodes = data.root_nb_nodes;
    let data_arc = Arc::new(data);
    let seeded = seed.is_some();
    let nb_starts = if seeded {
        0
    } else if root_nb_nodes == 900 || root_nb_nodes == 600 {
        3
    } else {
        5
    };

    let mut ls = LocalSearch::new(Arc::clone(&data_arc), params);
    let mut best_constructive_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;
    let mut best_ls_repair_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;
    for run_idx in 0..nb_starts {
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
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let mut candidate_ind = Individual::new_from_evaluated_routes(
            data_arc.as_ref(),
            &params,
            routes_after_ls.clone(),
            distance,
            tw,
            load_excess,
        );
        let mut candidate_routes = routes_after_ls;

        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            increase_penalties_for_infeasibility(&mut params, &candidate_ind);
            for factor in REPAIR_FACTORS {
                let repaired_routes = ls.continue_repair(&mut rng, params, factor);
                let (distance, tw, load_excess) = ls.evaluated_metrics();
                let repaired_ind = Individual::new_from_evaluated_routes(
                    data_arc.as_ref(),
                    &params,
                    repaired_routes.clone(),
                    distance,
                    tw,
                    load_excess,
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

    let mut incumbent = if let Some(routes) = seed {
        let ind = Individual::new_from_routes(data_arc.as_ref(), &params, routes.clone());
        MasterState { routes, ind }
    } else if let Some((routes, ind)) = best_ls_repair_feasible {
        MasterState { routes, ind }
    } else if let Some((routes, ind)) = best_constructive_feasible {
        MasterState { routes, ind }
    } else {
        panic!("No feasible constructive solution found among the reversed-mode starts");
    };

    let mut greedy_distance: Option<u32> = greedy;
    let _ = save_with_diag(
        challenge,
        save_solution,
        &incumbent.routes,
        &mut greedy_distance,
    );

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
    let mut penalty_memory: [Option<(usize, f64)>; 4] = [None; 4];

    let (base_target_size, mut adaptive_target_size, mut window_start_cost) = if params.decomp_adapt
    {
        let base = if params.adapt_base > 0 {
            params.adapt_base
        } else {
            params.decomp_target_size.clamp(100, 500)
        };
        (base, base, incumbent.ind.cost)
    } else {
        (0, 0, 0)
    };

    let relay_budget = if challenge.num_nodes == 700 && seeded {
        let intensity = exploration_schedule.iter().fold(0u128, |total, &level| {
            let it_permille = match params.sub_it_l4_permille {
                Some(value) if level >= 4 => value,
                _ => params.sub_it_permille,
            };
            total.saturating_add(it_permille as u128)
        });
        Some((data_arc.nb_nodes.saturating_sub(1) as u128).saturating_mul(intensity))
    } else {
        None
    };
    let mut relay_spent = 0u128;
    let mut relay_exhausted = false;

    if relay_budget == Some(0) {
        return Ok(Some(Solution {
            routes: incumbent.routes,
        }));
    }

    if seeded {
        exploration_schedule.clear();
    }

    if matches!(challenge.num_nodes, 600 | 700 | 800 | 900 | 1000)
        && params.giveup_gap > 0
        && params.band_bar > 0
        && params.triage_permille > 0
    {
        let g = match greedy_distance {
            Some(distance) => distance,
            None => greedy_total_distance(challenge)?,
        };
        greedy_distance = Some(g);
        let distance = evaluate_total_distance(
            challenge,
            &Solution {
                routes: incumbent.routes.clone(),
            },
        )?;
        let q0 = quality_of(g, distance) as i64;
        let threshold: i64 = if params.giveup_below > 0 {
            params.band_bar.saturating_sub(params.giveup_below)
        } else {
            let entry_low = if challenge.num_nodes == 900 || challenge.num_nodes == 700 { params.band_low } else { params.triage_low };
            params.band_bar.saturating_sub(entry_low)
        };
        if q0.saturating_add(params.giveup_gap) < threshold {
            exploration_schedule.clear();
        }
    }
    let normal_len = exploration_schedule.len();
    let mut triage_phase = if params.triage_permille > 0 && params.band_bar > 0 {
        ((normal_len as u128 * params.triage_permille as u128 + 999) / 1000).min(normal_len as u128)
            as usize
    } else {
        normal_len
    };
    let mut aux_g: Option<u32> = None;
    let mut triage_sub_it_active = false;

    let mut phase_idx = 0usize;
    let mut ot_rounds: i64 = 0;
    let mut normal_snapshot: Option<(u32, i32, i32)> = None;
    let mut band_added = 0usize;
    let mut prog_costs = [0i64; 3];
    let mut band_stalled = 0usize;
    let mut band_previous_cost = incumbent.ind.cost;
    let mut post_added = 0usize;
    let mut post_stalled = 0usize;
    let mut post_previous_cost = incumbent.ind.cost;
    let mut fork_master: Option<TriageSnapshot> = None;
    let mut best_kept: Option<MasterState> = None;
    let triage_branches = if challenge.num_nodes == 1000 {
        params.triage_branches.max(1)
    } else {
        1
    };
    let mut branch_idx = 0usize;
    let mut minimum_branch_cost = i64::MAX;
    let mut save_below = i64::MAX;
    let mut hit_b = false;
    let line_gate = if challenge.num_nodes == 900 {
        super::super::part_b::stopping::line_quality(900, hyperparameters)
            .saturating_add(2000)
    } else {
        i64::MAX
    };
    let relay_line = if challenge.num_nodes == 700 {
        super::super::part_b::stopping::line_quality(700, hyperparameters)
    } else {
        i64::MAX
    };
    loop {
        let mut succ_prev1 = if challenge.num_nodes == 900 || challenge.num_nodes == 700 {
            vec![usize::MAX; data_arc.nb_nodes]
        } else {
            Vec::new()
        };
        let mut succ_prev2 = succ_prev1.clone();
        let mut stable_history_len = 0usize;
        let mut phase_gains = [0i64; 3];
        let mut phase_gain_count = 0usize;
        let mut phase_gain_cursor = 0usize;
        loop {
            if phase_idx == triage_phase && phase_idx < normal_len {
                let g = match greedy_distance {
                    Some(g) => g,
                    None => greedy_total_distance(challenge)?,
                };
                greedy_distance = Some(g);
                let distance = evaluate_total_distance(
                    challenge,
                    &Solution {
                        routes: incumbent.routes.clone(),
                    },
                )?;
                debug_assert_eq!(distance as i64, incumbent.ind.cost);
                let quality = quality_of(g, distance) as i64;
                if quality < params.band_bar.saturating_sub(params.triage_low) {
                    break;
                }
                if quality >= params.band_bar.saturating_sub(params.triage_boost_low)
                    && quality < params.band_bar.saturating_add(params.triage_boost_high)
                {
                    if params.triage_fork != 0 {
                        fork_master = Some(TriageSnapshot {
                            rng: rng.clone(),
                            ls: ls.clone(),
                            params,
                            incumbent: incumbent.clone(),
                            exploration_schedule: exploration_schedule.clone(),
                            carried: carried.clone(),
                            penalty_memory,
                            adaptive_target_size,
                            window_start_cost,
                            greedy_distance,
                            triage_sub_it_active,
                            phase_idx,
                            ot_rounds,
                            normal_snapshot,
                            band_added,
                            band_stalled,
                            band_previous_cost,
                            post_added,
                            post_stalled,
                            post_previous_cost,
                        });
                    } else {
                        if params.triage_sub_it_permille > 0 {
                            params.sub_it_permille = params.triage_sub_it_permille;
                            params.sub_it_l4_permille = Some(params.triage_sub_it_permille);
                            triage_sub_it_active = true;
                        }
                        let level = match params.band_level {
                            2 | 3 | 4 => params.band_level,
                            _ => 3,
                        };
                        for _ in 0..params.triage_extra_phases {
                            exploration_schedule.push(level);
                        }
                    }
                }
            }
            if phase_idx == exploration_schedule.len() {
                if params.band_bar <= 0 {
                    break;
                }
                let (g, d5, q5) = match normal_snapshot {
                    Some(snapshot) => snapshot,
                    None => {
                        let g = match greedy_distance {
                            Some(g) => g,
                            None => greedy_total_distance(challenge)?,
                        };
                        let d5 = evaluate_total_distance(
                            challenge,
                            &Solution {
                                routes: incumbent.routes.clone(),
                            },
                        )?;
                        let snapshot = (g, d5, quality_of(g, d5));
                        normal_snapshot = Some(snapshot);
                        snapshot
                    }
                };
                let mut distance = if band_added == 0 && post_added == 0 {
                    d5
                } else {
                    evaluate_total_distance(
                        challenge,
                        &Solution {
                            routes: incumbent.routes.clone(),
                        },
                    )?
                };
                let mut quality = quality_of(g, distance);
                if challenge.num_nodes == 700
                    && params.ejc_depth > 0
                    && post_added == 0
                    && (quality as i64) >= params.band_bar.saturating_sub(params.band_low)
                    && (quality as i64) < params.band_bar.saturating_add(params.band_high)
                {
                    let candidate_routes =
                        ls.eject_descent(&incumbent.routes, params, &mut rng);
                    let candidate_ind = Individual::new_from_routes(
                        data_arc.as_ref(),
                        &params,
                        candidate_routes.clone(),
                    );
                    if candidate_ind.load_excess == 0 && candidate_ind.tw_violation == 0 {
                        promote_if_improved(
                            challenge,
                            save_solution,
                            save_below,
                            &mut incumbent,
                            &candidate_routes,
                            &candidate_ind,
                            &mut greedy_distance,
                        );
                    }
                    distance = evaluate_total_distance(
                        challenge,
                        &Solution {
                            routes: incumbent.routes.clone(),
                        },
                    )?;
                    quality = quality_of(g, distance);
                }
                debug_assert_eq!(distance as i64, incumbent.ind.cost);
                debug_assert!(distance <= d5 && quality >= q5);
                if post_added > 0 {
                    debug_assert!(incumbent.ind.cost <= post_previous_cost);
                    post_stalled = if incumbent.ind.cost < post_previous_cost {
                        0
                    } else {
                        post_stalled + 1
                    };
                    if (quality as i64) >= params.band_bar.saturating_add(params.post_high)
                        || (quality as i64) >= relay_line
                        || post_added >= params.post_phases
                        || (params.post_stall > 0 && post_stalled >= params.post_stall)
                    {
                        break;
                    }
                    post_previous_cost = incumbent.ind.cost;
                    post_added += 1;
                    exploration_schedule.push(match params.band_level {
                        2 | 3 | 4 => params.band_level,
                        _ => 3,
                    });
                    continue;
                }
                let upper = params.band_bar.saturating_add(params.band_high);
                let band_finished = if band_added == 0 {
                    params.band_bar <= 0
                        || (quality as i64) < params.band_bar.saturating_sub(params.band_low)
                        || (quality as i64) >= upper
                } else {
                    band_stalled = if incumbent.ind.cost < band_previous_cost {
                        0
                    } else {
                        band_stalled + 1
                    };
                    (quality as i64) >= upper
                        || band_added >= params.band_phases
                        || (params.band_stall > 0 && band_stalled >= params.band_stall)
                };
                if band_finished {
                    if params.post_phases == 0
                        || (quality as i64) >= relay_line
                        || params.band_bar <= 0
                        || (quality as i64) < params.band_bar.saturating_sub(params.post_low)
                        || (quality as i64) >= params.band_bar.saturating_add(params.post_high)
                    {
                        break;
                    }
                    params.cluster_lloyd = params.band_lloyd;
                    if params.post_sub_it_permille > 0 {
                        params.sub_it_permille = params.post_sub_it_permille;
                        if params.band_level == 4 {
                            params.sub_it_l4_permille = Some(params.post_sub_it_permille);
                        }
                    }
                    post_previous_cost = incumbent.ind.cost;
                    post_added = 1;
                    exploration_schedule.push(match params.band_level {
                        2 | 3 | 4 => params.band_level,
                        _ => 3,
                    });
                    continue;
                }
                if band_added == 0 {
                    params.cluster_lloyd = params.band_lloyd;
                    if params.band_sub_it_permille > 0 && !triage_sub_it_active {
                        params.sub_it_permille = params.band_sub_it_permille;
                        if params.band_level == 4 {
                            params.sub_it_l4_permille = Some(params.band_sub_it_permille);
                        }
                    }
                }
                band_previous_cost = incumbent.ind.cost;
                band_added += 1;
                exploration_schedule.push(match params.band_level {
                    2 | 3 | 4 => params.band_level,
                    _ => 3,
                });
                continue;
            }
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
            if challenge.num_nodes == 900 {
                if let Some(g) = greedy_distance {
                    if quality_of(g, incumbent.ind.cost as i32) as i64 >= line_gate {
                        break;
                    }
                }
            }
            let phase_start_cost = if data_arc.nb_nodes == 800 && phase_idx < normal_len {
                Some(incumbent.ind.cost)
            } else {
                None
            };
            let phase_base_routes = incumbent.routes.clone();
            let stable_history = if (challenge.num_nodes == 900 || challenge.num_nodes == 700)
                && (band_added > 0 || post_added > 0)
            {
                let history = (stable_history_len == 2)
                    .then(|| (succ_prev1.clone(), succ_prev2.clone()));
                succ_prev2.copy_from_slice(&succ_prev1);
                succ_prev1.fill(usize::MAX);
                for route in &phase_base_routes {
                    for arc in route.windows(2) {
                        if arc[0] != 0 && arc[1] != 0 {
                            succ_prev1[arc[0]] = arc[1];
                        }
                    }
                }
                stable_history_len = (stable_history_len + 1).min(2);
                history
            } else {
                None
            };
            let k = if params.decomp_target_l2 > 0 && sub_exploration_level == 2 {
                ((data_arc.nb_nodes as f64) / (params.decomp_target_l2 as f64)).round() as usize
            } else if params.decomp_target_l3 > 0 && sub_exploration_level == 3 {
                ((data_arc.nb_nodes as f64) / (params.decomp_target_l3 as f64)).round() as usize
            } else if params.decomp_target_l4 > 0 && sub_exploration_level == 4 {
                ((data_arc.nb_nodes as f64) / (params.decomp_target_l4 as f64)).round() as usize
            } else if params.decomp_adapt {
                ((data_arc.nb_nodes as f64) / (adaptive_target_size as f64)).round() as usize
            } else if params.decomp_target_size_final > 0
                && exploration_schedule.len() >= TARGET_SCHEDULE_MIN_PHASES
            {
                let sched_len = exploration_schedule.len();
                let t = (phase_idx.min(sched_len) as f64) / (sched_len as f64);
                let init = params.decomp_target_size as f64;
                let fin = params.decomp_target_size_final as f64;
                let target_size = (init + (fin - init) * t).round().clamp(100.0, 500.0) as usize;
                ((data_arc.nb_nodes as f64) / (target_size as f64)).round() as usize
            } else {
                ((data_arc.nb_nodes as f64) / (params.decomp_target_size as f64)).round() as usize
            };
            let clusters =
                cluster_route_indices(data_arc.as_ref(), &phase_base_routes, k, &params, &mut rng);
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

                let (sub_problem, local_to_global) = build_subproblem(
                    data_arc.as_ref(),
                    &clients,
                    ((route_group.len() as i32) + residual_free_routes).max(1) as usize,
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
                let mut params_sub = Params::initialize_at_level(
                    hyperparameters,
                    sub_exploration_level,
                    &sub_problem,
                );
                if challenge.num_nodes == 900 && sub_exploration_level == 4 {
                    params_sub.lambda = 2;
                }
                params_sub.decomp_nb_phases = 0;
                if params.sub_mu_permille != 1000 {
                    params_sub.mu_start =
                        scale_permille(params_sub.mu_start, params.sub_mu_permille).max(1);
                }
                let it_permille = match params.sub_it_l4_permille {
                    Some(value) if sub_exploration_level >= 4 => value,
                    _ => params.sub_it_permille,
                };
                let relay_debit = if let Some(budget) = relay_budget {
                    let debit = (clients.len() as u128).saturating_mul(it_permille as u128);
                    if debit > budget.saturating_sub(relay_spent) {
                        relay_exhausted = true;
                        given_up = true;
                        break;
                    }
                    debit
                } else {
                    0
                };
                if it_permille != 1000 {
                    params_sub.max_it_noimprov =
                        scale_permille(params_sub.max_it_noimprov, it_permille);
                    params_sub.max_it_total = scale_permille(params_sub.max_it_total, it_permille);
                }
                let initial_penalty_capa = params_sub.penalty_capa;
                if params.pen_warm != 0 && (1..=4).contains(&sub_exploration_level) {
                    if let Some((last_tw, capa_ratio)) = penalty_memory[sub_exploration_level - 1] {
                        let mut warm_tw = last_tw as f64;
                        let mut warm_capa = (initial_penalty_capa as f64 * capa_ratio).round();
                        if params.pen_warm == 2 {
                            warm_tw = (params_sub.penalty_tw as f64 * warm_tw).sqrt().round();
                            warm_capa = (initial_penalty_capa as f64 * warm_capa).sqrt().round();
                        }
                        debug_assert!(warm_tw.is_finite() && warm_capa.is_finite());
                        params_sub.penalty_tw = warm_tw.clamp(1.0, MAX_PENALTY as f64) as usize;
                        params_sub.penalty_capa = warm_capa.clamp(1.0, MAX_PENALTY as f64) as usize;
                    }
                }
                let seed_sub_ind =
                    Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
                debug_assert!(
                    seed_sub_ind.load_excess == 0 && seed_sub_ind.tw_violation == 0,
                    "Invariant violated: subproblem seed should always be feasible in reversed mode"
                );
                let seed_sub_route_count = seed_sub_routes.len();
                let migrants = if receive {
                    let cap = carry_k.min(params_sub.mu_start.saturating_sub(MIN_FRESH_STARTS));
                    restrict_carried(
                        &carried,
                        &sub_problem,
                        &global_to_local,
                        &seed_sub_routes,
                        cap,
                    )
                } else {
                    Vec::new()
                };
                let compression = if !receive {
                    stable_history.as_ref().and_then(|(prev1, prev2)| {
                        ProblemCompression::from_chains(
                            &sub_problem,
                            stable_chains(&seed_sub_routes, &local_to_global, prev1, prev2),
                        )
                    })
                } else {
                    None
                };
                let (mut sub_ga, expansion) = match compression {
                    Some(compression) => (
                        Genetic::new(compression.compact, params_sub),
                        Some((sub_problem, compression.chains)),
                    ),
                    None => (Genetic::new(sub_problem, params_sub), None),
                };
                let compact_seed: Option<Vec<Vec<usize>>> =
                    expansion.as_ref().map(|(original, chains)| {
                        let mut old_to_new = vec![0usize; original.nb_nodes];
                        for (index, chain) in chains.iter().enumerate() {
                            for &client in chain {
                                old_to_new[client] = index + 1;
                            }
                        }
                        seed_sub_routes.iter().map(|route| {
                            let mut compact: Vec<usize> =
                                route.iter().map(|&id| old_to_new[id]).collect();
                            compact.dedup();
                            compact
                        }).collect()
                    });
                let expand_routes = |routes: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                    match expansion.as_ref() {
                        Some((_, chains)) => routes.into_iter().map(|route| {
                            let mut expanded = Vec::new();
                            for id in route {
                                if id == 0 {
                                    expanded.push(0);
                                } else {
                                    expanded.extend_from_slice(&chains[id - 1]);
                                }
                            }
                            expanded
                        }).collect(),
                        None => routes,
                    }
                };
                sub_ga.population.infeasible.damped_penalties = challenge.num_nodes == 900;
                sub_ga.inherited = migrants;
                if relay_budget.is_some() {
                    relay_spent = relay_spent.saturating_add(relay_debit);
                }
                let candidate = sub_ga.run(
                    &mut rng,
                    None,
                    Some(compact_seed.as_deref().unwrap_or(&seed_sub_routes)),
                ).map(|candidate| {
                    match expansion.as_ref() {
                        Some((original, _)) => Individual::new_from_routes(
                            original,
                            &params_sub,
                            expand_routes(candidate.routes),
                        ),
                        None => candidate,
                    }
                });
                let chosen_sub_routes: Vec<Vec<usize>> =
                    match candidate {
                        Some(candidate) => {
                            if candidate.cost < seed_sub_ind.cost {
                                candidate.routes
                            } else {
                                seed_sub_routes
                            }
                        }
                        None => seed_sub_routes,
                    };
                if params.pen_warm != 0 && (1..=4).contains(&sub_exploration_level) {
                    penalty_memory[sub_exploration_level - 1] =
                        (initial_penalty_capa != 0).then(|| {
                            (
                                sub_ga.params.penalty_tw,
                                sub_ga.params.penalty_capa as f64 / initial_penalty_capa as f64,
                            )
                        });
                }
                if export {
                    let donors: Vec<Vec<Vec<usize>>> = sub_ga
                        .export_feasible(carry_k)
                        .into_iter()
                        .map(|routes| {
                            map_sub_routes_to_global(&expand_routes(routes), &local_to_global)
                        })
                        .collect();
                    blocks.push(BlockDonors {
                        donors,
                        fallback: route_group
                            .iter()
                            .map(|&rid| phase_base_routes[rid].clone())
                            .collect(),
                    });
                }
                residual_free_routes +=
                    (seed_sub_route_count as i32) - (chosen_sub_routes.len() as i32);
                let sub_routes_global =
                    map_sub_routes_to_global(&chosen_sub_routes, &local_to_global);

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
                    challenge,
                    save_solution,
                    save_below,
                    &mut incumbent,
                    &progressive_master_routes,
                    &progressive_master_ind,
                    &mut greedy_distance,
                );
                if challenge.num_nodes == 900
                    && params.rival_abandon == 1
                    && branch_idx == 0
                    && phase_idx < normal_len
                    && band_added == 0
                    && post_added == 0
                    && params.band_bar > 0
                    && cid + 1 < clusters.len()
                {
                    if let Some(g) = greedy_distance {
                        let done = phase_idx * k + cid + 1;
                        let total = normal_len * k;
                        let q = quality_of(g, incumbent.ind.cost as i32) as i64;
                        let gain = super::super::part_b::stopping::gain_p99(900);
                        if q + gain[done * 32 / total]
                            < params.band_bar.saturating_sub(params.band_low)
                        {
                            given_up = true;
                            break;
                        }
                    }
                }
            }

            if given_up {
                break;
            }

            carried = if export {
                join_b(
                    &blocks,
                    carry_k,
                    data_arc.nb_nodes,
                    params.carried_rr,
                    params.carried_sim_permille,
                )
            } else {
                Vec::new()
            };

            let merged_ind =
                Individual::new_from_routes(data_arc.as_ref(), &params, rebuilt_routes.clone());
            debug_assert!(
                merged_ind.load_excess == 0 && merged_ind.tw_violation == 0,
                "Invariant violated: merged master solution should be feasible before global LS"
            );
            promote_if_improved(
                challenge,
                save_solution,
                save_below,
                &mut incumbent,
                &rebuilt_routes,
                &merged_ind,
                &mut greedy_distance,
            );
            let mut candidate_routes = ls.run_from_routes(&rebuilt_routes, &[], params, &mut rng);
            let (distance, tw, load_excess) = ls.evaluated_metrics();
            let mut candidate_ind = Individual::new_from_evaluated_routes(
                data_arc.as_ref(),
                &params,
                candidate_routes.clone(),
                distance,
                tw,
                load_excess,
            );
            if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
                increase_penalties_for_infeasibility(&mut params, &candidate_ind);
                let mut repaired: Option<(Vec<Vec<usize>>, Individual)> = None;
                for factor in REPAIR_FACTORS {
                    let routes = ls.continue_repair(&mut rng, params, factor);
                    let (distance, tw, load_excess) = ls.evaluated_metrics();
                    let ind = Individual::new_from_evaluated_routes(
                        data_arc.as_ref(),
                        &params,
                        routes.clone(),
                        distance,
                        tw,
                        load_excess,
                    );
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
                challenge,
                save_solution,
                save_below,
                &mut incumbent,
                &candidate_routes,
                &candidate_ind,
                &mut greedy_distance,
            );
            phase_idx += 1;

            if challenge.num_nodes == 900 && branch_idx == 0 && phase_idx < normal_len && band_added == 0 && post_added == 0 && params.band_bar > 0 {
                if let Some(g) = greedy_distance {
                    let q = quality_of(g, incumbent.ind.cost as i32) as i64;
                    let idx = (phase_idx * 32) / normal_len;
                    let gain = if params.rival_abandon == 1 {
                        super::super::part_b::stopping::gain_p99(900)[idx]
                    } else {
                        GAIN_P99_900[idx]
                    };
                    if q + gain < params.band_bar.saturating_sub(params.band_low) {
                        break;
                    }
                }
            }

            if challenge.num_nodes == 700 && phase_idx < normal_len && band_added == 0 && post_added == 0 && params.band_bar > 0 {
                let g = match greedy_distance {
                    Some(g) => g,
                    None => {
                        let g = greedy_total_distance(challenge)?;
                        greedy_distance = Some(g);
                        g
                    }
                };
                let q = quality_of(g, incumbent.ind.cost as i32) as i64;
                let idx = ((phase_idx * 32) / normal_len).min(31);
                let old = prog_costs[(phase_idx + 1) % 3];
                let cost = incumbent.ind.cost;
                prog_costs[phase_idx % 3] = cost;
                let mut gain = GAIN_P99_700[idx];
                if phase_idx >= 3 && old >= cost && (old - cost) * 1000 <= g as i64 {
                    gain = gain.min(G700_PROGRESS[idx]);
                }
                if q + gain < params.band_bar.saturating_sub(params.band_low) {
                    break;
                }
            }

            if let Some(start_cost) = phase_start_cost {
                let g = match greedy_distance {
                    Some(g) => g,
                    None => {
                        let g = greedy_total_distance(challenge)?;
                        greedy_distance = Some(g);
                        g
                    }
                };
                let cost_gain = start_cost - incumbent.ind.cost;
                phase_gains[phase_gain_cursor] =
                    quality_of(g, (g as i64 - cost_gain) as i32) as i64;
                phase_gain_cursor = (phase_gain_cursor + 1) % phase_gains.len();
                phase_gain_count = (phase_gain_count + 1).min(phase_gains.len());
                if phase_gain_count == phase_gains.len() {
                    let slope =
                        phase_gains.iter().sum::<i64>() as f64 / phase_gains.len() as f64;
                    let quality = quality_of(g, incumbent.ind.cost as i32) as f64;
                    let projected_quality = quality + slope * (normal_len - phase_idx) as f64;
                    if projected_quality < params.band_bar.saturating_sub(params.triage_low) as f64
                    {
                        break;
                    }
                }
            }

            if params.aux_flag {
                let aux_level = if params.aux_level > 0 {
                    params.aux_level
                } else {
                    params.band_bar
                };
                if aux_level > 0
                    && phase_idx as u128 * 1000
                        >= params.aux_permille as u128 * normal_len as u128
                {
                    let g = match greedy_distance.or(aux_g) {
                        Some(g) => g,
                        None => {
                            let g = greedy_total_distance(challenge)?;
                            aux_g = Some(g);
                            g
                        }
                    };
                    let quality = quality_of(g, incumbent.ind.cost as i32) as i64;
                    let stop_level = if challenge.num_nodes == 1000
                        && branch_idx == 0
                        && params.low_stop != 0
                    {
                        aux_level
                            .saturating_add(params.aux_offset)
                            .min(params.band_bar.saturating_add(params.low_stop))
                    } else {
                        aux_level.saturating_add(params.aux_offset)
                    };
                    if quality >= stop_level {
                        hit_b = true;
                        break;
                    }
                }
            }

            if params.decomp_adapt
                && phase_idx % (if params.adapt_window > 0 { params.adapt_window } else { 3 }) == 0
            {
                let current_cost = incumbent.ind.cost;
                let ratio =
                    (window_start_cost - current_cost) as f64 / window_start_cost.max(1) as f64;
                if ratio < 0.01 {
                    let num = if params.adapt_shrink_num > 0 { params.adapt_shrink_num } else { 4 };
                    let den = if params.adapt_shrink_den > 0 { params.adapt_shrink_den } else { 5 };
                    let floor = if params.adapt_floor > 0 { params.adapt_floor } else { 100 };
                    adaptive_target_size = (adaptive_target_size * num / den).max(floor);
                } else if ratio > 0.03 {
                    adaptive_target_size = (adaptive_target_size * 5 / 4).min(base_target_size);
                }
                window_start_cost = current_cost;
            }

            if band_added > 0 || post_added > 0 {
                continue;
            }

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

        minimum_branch_cost = minimum_branch_cost.min(incumbent.ind.cost);
        if relay_exhausted {
            break;
        }
        if params.aux_flag && hit_b {
            break;
        }
        if branch_idx >= triage_branches || fork_master.is_none() {
            break;
        }
        if branch_idx == 0 && params.triage_gate != 0 {
            let g = greedy_distance.expect("Triage snapshot requires greedy distance");
            debug_assert_eq!(
                incumbent.ind.cost,
                evaluate_total_distance(
                    challenge,
                    &Solution {
                        routes: incumbent.routes.clone()
                    },
                )? as i64,
            );
            let quality = quality_of(g, incumbent.ind.cost as i32) as i64;
            if quality < params.band_bar.saturating_sub(params.triage_gate_low)
                || quality >= params.band_bar.saturating_add(params.triage_gate_high)
            {
                break;
            }
        }
        if branch_idx > 0 {
            let g = greedy_distance.expect("Triage snapshot requires greedy distance");
            let best_cost = best_kept
                .as_ref()
                .map_or(incumbent.ind.cost, |best| best.ind.cost.min(incumbent.ind.cost));
            if quality_of(g, best_cost as i32) as i64
                >= params.band_bar.saturating_add(params.triage_gate_high)
            {
                break;
            }
        }
        if best_kept
            .as_ref()
            .map_or(true, |best| incumbent.ind.cost < best.ind.cost)
        {
            best_kept = Some(incumbent.clone());
        }
        save_below = best_kept.as_ref().unwrap().ind.cost;
        let snapshot = if triage_branches == 1 {
            fork_master.take().unwrap()
        } else {
            fork_master.as_ref().unwrap().clone()
        };
        branch_idx += 1;
        rng = snapshot.rng;
        if params.triage_reseed != 0 || branch_idx > 1 {
            rng = SmallRng::seed_from_u64(
                rng.gen::<u64>()
                    ^ (params.triage_reseed as u64)
                        .wrapping_add((branch_idx - 1) as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
        }
        ls = snapshot.ls;
        params = snapshot.params;
        incumbent = snapshot.incumbent;
        exploration_schedule = snapshot.exploration_schedule;
        carried = snapshot.carried;
        penalty_memory = snapshot.penalty_memory;
        adaptive_target_size = snapshot.adaptive_target_size;
        window_start_cost = snapshot.window_start_cost;
        greedy_distance = snapshot.greedy_distance;
        triage_sub_it_active = snapshot.triage_sub_it_active;
        phase_idx = snapshot.phase_idx;
        ot_rounds = snapshot.ot_rounds;
        normal_snapshot = snapshot.normal_snapshot;
        band_added = snapshot.band_added;
        band_stalled = snapshot.band_stalled;
        band_previous_cost = snapshot.band_previous_cost;
        post_added = snapshot.post_added;
        post_stalled = snapshot.post_stalled;
        post_previous_cost = snapshot.post_previous_cost;
        triage_phase = normal_len;
        if let Some(phases) = params.triage_band_phases {
            params.band_phases = phases;
            if phases == 0 {
                params.band_low = 0;
                params.band_high = 0;
            }
        }
        if let Some(phases) = params.triage_post_phases {
            params.post_phases = phases;
        }
        if params.triage_sub_it_permille > 0 {
            params.sub_it_permille = params.triage_sub_it_permille;
            params.sub_it_l4_permille = Some(params.triage_sub_it_permille);
            triage_sub_it_active = true;
        }
        let level = match params.band_level {
            2 | 3 | 4 => params.band_level,
            _ => 3,
        };
        let extra_phases = params.triage_extra_phases / triage_branches
            + usize::from(params.triage_extra_phases % triage_branches != 0);
        for _ in 0..extra_phases {
            exploration_schedule.push(level);
        }
    }

    if let Some(best) = best_kept {
        if best.ind.cost <= incumbent.ind.cost {
            incumbent = best;
        }
    }
    debug_assert_eq!(incumbent.ind.cost, minimum_branch_cost);

    if relay_exhausted {
        return Ok(Some(Solution {
            routes: incumbent.routes,
        }));
    }

    if params.route_elim > 0 {
        let mut attempts = 0usize;
        while attempts < params.route_elim {
            attempts += 1;
            if incumbent.routes.len() <= 1 {
                break;
            }
            let mut victim: Option<usize> = None;
            let mut victim_clients = usize::MAX;
            for (rid, route) in incumbent.routes.iter().enumerate() {
                let nb_clients = nb_clients_in_route(route);
                if nb_clients == 0 {
                    continue;
                }
                if nb_clients < victim_clients {
                    victim_clients = nb_clients;
                    victim = Some(rid);
                }
            }
            let Some(victim) = victim else {
                break;
            };
            let victim_len = incumbent.routes[victim].len();
            let ejected: Vec<usize> = incumbent.routes[victim][1..victim_len - 1].to_vec();
            let mut reduced: Vec<Vec<usize>> = Vec::with_capacity(incumbent.routes.len() - 1);
            for (rid, route) in incumbent.routes.iter().enumerate() {
                if rid != victim {
                    reduced.push(route.clone());
                }
            }
            let mut all_inserted = true;
            for &client in &ejected {
                let inserted = Sequence::singleton(data_arc.as_ref(), client);
                let mut best: Option<(usize, usize)> = None;
                let mut best_delta = i64::MAX;
                for (rid, route) in reduced.iter().enumerate() {
                    let len = route.len();
                    let mut head = vec![Sequence::default(); len];
                    head[0] = Sequence::singleton(data_arc.as_ref(), route[0]);
                    for p in 1..len {
                        let node = Sequence::singleton(data_arc.as_ref(), route[p]);
                        let prev = head[p - 1];
                        head[p] = Sequence::join2(data_arc.as_ref(), &prev, &node);
                    }
                    let mut tail = vec![Sequence::default(); len];
                    tail[len - 1] = Sequence::singleton(data_arc.as_ref(), route[len - 1]);
                    for p in (0..len - 1).rev() {
                        let node = Sequence::singleton(data_arc.as_ref(), route[p]);
                        let next = tail[p + 1];
                        tail[p] = Sequence::join2(data_arc.as_ref(), &node, &next);
                    }
                    let base_cost = head[len - 1].eval(data_arc.as_ref(), &params);
                    for pos in 1..len {
                        let head_seq = head[pos - 1];
                        let tail_seq = tail[pos];
                        let travel12 = data_arc
                            .dm(head_seq.last_node as usize, inserted.first_node as usize);
                        let travel23 = data_arc
                            .dm(inserted.last_node as usize, tail_seq.first_node as usize);
                        let tw = Sequence::tw3_with_travel(
                            &head_seq,
                            &inserted,
                            &tail_seq,
                            travel12,
                            travel23,
                        );
                        let load = head_seq.load + inserted.load + tail_seq.load;
                        if tw != 0 || load > data_arc.max_capacity {
                            continue;
                        }
                        let cand = Sequence::eval3(
                            data_arc.as_ref(),
                            &params,
                            &head_seq,
                            &inserted,
                            &tail_seq,
                        );
                        let delta = cand - base_cost;
                        if delta < best_delta {
                            best_delta = delta;
                            best = Some((rid, pos));
                        }
                    }
                }
                match best {
                    Some((rid, pos)) => reduced[rid].insert(pos, client),
                    None => {
                        all_inserted = false;
                        break;
                    }
                }
            }
            if !all_inserted {
                break;
            }
            let reduced_ind =
                Individual::new_from_routes(data_arc.as_ref(), &params, reduced.clone());
            if reduced_ind.load_excess != 0 || reduced_ind.tw_violation != 0 {
                break;
            }
            let polished_routes = ls.run_from_routes(&reduced, &[], params, &mut rng);
            let polished_ind =
                Individual::new_from_routes(data_arc.as_ref(), &params, polished_routes.clone());
            if polished_ind.load_excess != 0 || polished_ind.tw_violation != 0 {
                break;
            }
            if polished_ind.cost < incumbent.ind.cost {
                promote_if_improved(
                    challenge,
                    save_solution,
                    save_below,
                    &mut incumbent,
                    &polished_routes,
                    &polished_ind,
                    &mut greedy_distance,
                );
            } else {
                break;
            }
        }
    }

    let solution = Solution {
        routes: incumbent.routes,
    };
    Ok(Some(solution))
}
