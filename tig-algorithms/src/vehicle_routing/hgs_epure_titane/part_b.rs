use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
pub(crate) mod stopping {
use super::progress::Progress;
use super::baseline::baseline_distance;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::Challenge;

const SCALE: f64 = 1.0;
const BELOW: f64 = 0.004;
const BELOW_SENT: f64 = 0.003;
const BELOW_DEFAULT_TARGET_LATE: f64 = 0.003;
const LATE_TRACKS: [usize; 2] = [600, 800];
const ABOVE: f64 = 0.006;
const ABOVE_DEFAULT_TARGET: f64 = 0.0045;
const ABOVE_DEFAULT_TARGET_TRACKS: [(usize, f64); 1] = [(1000, 0.0021)];
const ABOVE_SENT: f64 = 0.0045;
const ABOVE_SENT_TRACKS: [usize; 3] = [600, 800, 1000];

const TARGETS: [(usize, i64); 5] = [(600, 204500), (700, 198100), (800, 193600), (900, 189600), (1000, 192000)];

#[rustfmt::skip]
const GAIN: [i64; 32] = [83450, 53889, 41417, 29114, 24146, 18989, 15405, 13879, 13508, 13297, 12344, 11160, 10457, 10297, 8711, 8585, 8280, 8062, 8062, 5208, 5150, 5150, 4578, 4578, 4218, 4197, 3669, 3362, 1943, 1943, 1434, 0];

#[rustfmt::skip]
const GAIN_P99: [(usize, [i64; 32]); 5] = [
    (600, [69021, 39098, 29264, 29264, 22321, 16344, 13957, 13957, 12114, 10781, 10781, 10174, 9302, 8004, 8004, 6997, 6119, 6119, 5244, 4055, 3785, 3785, 3414, 3140, 3140, 2193, 1930, 1169, 1169, 962, 579, 579]),
    (700, [69441, 38332, 38332, 23896, 18276, 18276, 14619, 12466, 12466, 10599, 10191, 10191, 9028, 7483, 7483, 6028, 5514, 5514, 4051, 3343, 3343, 3164, 2529, 2529, 2023, 1831, 1831, 1079, 942, 942, 547, 547]),
    (800, [75957, 45537, 30024, 22809, 18615, 14722, 12520, 10519, 9945, 8960, 8960, 7716, 7601, 6414, 5796, 5156, 4697, 4234, 4078, 3742, 2998, 2998, 2646, 2381, 1714, 1538, 1154, 671, 580, 510, 420, 420]),
    (900, [74319, 48188, 32337, 25117, 20620, 16982, 14216, 12784, 11061, 10107, 10107, 9039, 8038, 7381, 6867, 6171, 5886, 5293, 4598, 4297, 3961, 3961, 3420, 3173, 2381, 2088, 1597, 1435, 1119, 803, 528, 528]),
    (1000, [76638, 44447, 30298, 22243, 18352, 14909, 12783, 11583, 10547, 9646, 8643, 7927, 7614, 7042, 6437, 5578, 4752, 4190, 3832, 3745, 3170, 2863, 2434, 1828, 1495, 1493, 1356, 1143, 925, 542, 374, 374]),
];

pub(crate) fn gain_p99(num_nodes: usize) -> &'static [i64; 32] {
    &GAIN_P99
        .iter()
        .min_by_key(|t| (t.0 as i64 - num_nodes as i64).abs())
        .expect("Gain table must not be empty")
        .1
}

pub(crate) fn line_quality(
    num_nodes: usize,
    hyperparameters: &Option<Map<String, Value>>,
) -> i64 {
    if let Some(target) = hyperparameters
        .as_ref()
        .and_then(|m| m.get("target_quality"))
        .and_then(Value::as_f64)
    {
        return (if target > 1.0 { target } else { target * 1e6 }).round() as i64;
    }
    TARGETS
        .iter()
        .min_by_key(|t| (t.0 as i64 - num_nodes as i64).abs())
        .expect("Target table must not be empty")
        .1 - 1500
}

pub struct Stopping {
    progress: Progress,
    baseline: f64,
    stop_cost: i64,
    giveup_quality: f64,
    giveup_quality_late: f64,
    gain: &'static [i64; 32],
}

impl Stopping {
    pub fn new(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Option<Self> {
        let hp = |key: &str| hyperparameters.as_ref().and_then(|m| m.get(key)).and_then(Value::as_f64);
        if hp("early_stop") == Some(0.0) {
            return None;
        }
        let &(track, target_ppm) = TARGETS
            .iter()
            .min_by_key(|t| (t.0 as i64 - challenge.num_nodes as i64).abs())?;
        let target = hp("target_quality").map_or(target_ppm as f64 / 1e6, |t| if t > 1.0 { t / 1e6 } else { t });
        let default_search = hyperparameters
            .as_ref()
            .map_or(true, |m| m.keys().all(|k| k == "target_quality" || k == "early_stop"));
        let gain = if default_search {
            &GAIN_P99.iter().min_by_key(|t| (t.0 as i64 - challenge.num_nodes as i64).abs())?.1
        } else {
            &GAIN
        };
        let sent = default_search && hp("target_quality").is_some();
        let above = if !default_search {
            ABOVE
        } else if !sent {
            ABOVE_DEFAULT_TARGET_TRACKS
                .iter()
                .find(|t| t.0 == track && (t.0 != 1000 || challenge.num_nodes == 1000))
                .map_or(ABOVE_DEFAULT_TARGET, |t| t.1)
        } else if ABOVE_SENT_TRACKS.contains(&track) {
            ABOVE_SENT
        } else {
            ABOVE
        };
        let baseline = baseline_distance(challenge) as f64;
        let mut stop_cost = (baseline * (1.0 - target - above)).floor() as i64;
        if challenge.num_nodes == 800 && default_search && !sent {
            let line_cost = (baseline
                * (1.0 - (line_quality(challenge.num_nodes, hyperparameters) + 2000) as f64 / 1e6))
                .floor() as i64;
            stop_cost = stop_cost.max(line_cost);
        }
        Some(Self {
            progress: Progress::new(challenge.num_nodes, baseline, default_search),
            baseline,
            stop_cost,
            giveup_quality: target - if sent { BELOW_SENT } else { BELOW },
            giveup_quality_late: target - if sent {
                BELOW_SENT
            } else if default_search && LATE_TRACKS.contains(&track) {
                BELOW_DEFAULT_TARGET_LATE
            } else {
                BELOW
            },
            gain,
        })
    }

    pub fn settled(&mut self, cost: i64, done: usize, total: usize) -> bool {
        if cost <= self.stop_cost {
            return true;
        }
        if done >= total {
            return false;
        }
        let original_gain = self.gain[done * self.gain.len() / total];
        let gain = self.progress.bound(cost, done, total).map_or(original_gain, |g| g.min(original_gain)) as f64 / 1e6;
        let quality = 1.0 - cost as f64 / self.baseline;
        let giveup_quality = if done * 4 >= total { self.giveup_quality_late } else { self.giveup_quality };
        quality + SCALE * gain < giveup_quality
    }

    pub fn abandoned(&self, cost: i64, done: usize, total: usize) -> bool {
        if done >= total {
            return false;
        }
        let original_gain = self.gain[done * self.gain.len() / total];
        let gain = original_gain as f64 / 1e6;
        let quality = 1.0 - cost as f64 / self.baseline;
        let giveup_quality = if done * 4 >= total { self.giveup_quality_late } else { self.giveup_quality };
        quality + SCALE * gain < giveup_quality
    }
}
}
mod progress {
const G600: [i64; 32] = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 10725, 1000000, 9932, 8354, 1000000, 7741, 6989, 5729, 1000000, 5266, 4350, 1000000, 3974, 3637, 2989, 1000000, 2453, 1670, 1000000, 1059, 812, 610, 1000000, 408, 199, 1000000];
const G700: [i64; 32] = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 9837, 1000000, 10025, 8826, 1000000, 7124, 6193, 1000000, 4629, 3477, 1000000, 3199, 2657, 1000000, 2284, 1870, 1000000, 1270, 932, 1000000, 661, 485, 1000000, 248, 1000000];

pub struct Progress {
    costs: [i64; 3],
    baseline: f64,
    gain: Option<&'static [i64; 32]>,
}

impl Progress {
    pub fn new(nodes: usize, baseline: f64, enabled: bool) -> Self {
        let gain = if enabled {
            match nodes {
                600 => Some(&G600),
                700 => Some(&G700),
                800 => None,
                _ => None,
            }
        } else {
            None
        };
        Self { costs: [0; 3], baseline, gain }
    }

    pub fn bound(&mut self, cost: i64, done: usize, total: usize) -> Option<i64> {
        let gain = self.gain?;
        let old = self.costs[(done + 1) % 3];
        self.costs[done % 3] = cost;
        if done < 2 || total == 0 || old < cost || self.baseline <= 0.0
            || (old - cost) as f64 * 1000.0 > self.baseline {
            return None;
        }
        Some(gain[(done * gain.len() / total).min(gain.len() - 1)])
    }
}
}
mod baseline {
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::{Challenge, Solution};

pub fn baseline_distance(challenge: &Challenge) -> i64 {
    let routes = std::cell::RefCell::new(Vec::new());
    let _ = solve_challenge(challenge, &|s: &Solution| { *routes.borrow_mut() = s.routes.clone(); Ok(()) }, &None);
    let dm = &challenge.distance_matrix;
    routes.into_inner().iter().map(|r| r.windows(2).map(|w| dm[w[0]][w[1]] as i64).sum::<i64>()).sum()
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..challenge.num_nodes).collect();
    nodes.sort_by(|&a, &b| challenge.distance_matrix[0][a].cmp(&challenge.distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; challenge.num_nodes];
    remaining[0] = false;

    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        remaining[node] = false;
        let mut route = vec![0, node, 0];
        let mut route_demand = challenge.demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .enumerate()
                .filter(|(n, &flag)| {
                    flag && route_demand + challenge.demands[*n] <= challenge.max_capacity
                })
                .map(|(n, _)| n)
                .collect(),
            &challenge.distance_matrix,
            challenge.service_time,
            &challenge.ready_times,
            &challenge.due_times,
        ) {
            remaining[best_node] = false;
            route_demand += challenge.demands[best_node];
            route.insert(best_pos, best_node);
        }

        routes.push(route);
    }

    let _ = save_solution(&Solution { routes });
    Ok(())
}

fn is_feasible(
    route: &Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    mut curr_node: usize,
    mut curr_time: i32,
    start_pos: usize,
) -> bool {
    let mut valid = true;
    for pos in start_pos..route.len() {
        let next_node = route[pos];
        curr_time += distance_matrix[curr_node][next_node];
        if curr_time > due_times[route[pos]] {
            valid = false;
            break;
        }
        curr_time = curr_time.max(ready_times[next_node]) + service_time;
        curr_node = next_node;
    }
    valid
}

fn find_best_insertion(
    route: &Vec<usize>,
    remaining_nodes: Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Option<(usize, usize)> {
    let alpha1 = 1;
    let alpha2 = 0;
    let lambda = 1;

    let mut best_c2 = None;
    let mut best = None;
    for insert_node in remaining_nodes {
        let mut best_c1 = None;

        let mut curr_time = 0;
        let mut curr_node = 0;
        for pos in 1..route.len() {
            let next_node = route[pos];
            let new_arrival_time =
                ready_times[insert_node].max(curr_time + distance_matrix[curr_node][insert_node]);
            if new_arrival_time > due_times[insert_node] {
                continue;
            }
            let old_arrival_time =
                ready_times[next_node].max(curr_time + distance_matrix[curr_node][next_node]);

            let c11 = distance_matrix[curr_node][insert_node]
                + distance_matrix[insert_node][next_node]
                - distance_matrix[curr_node][next_node];

            let c12 = new_arrival_time - old_arrival_time;

            let c1 = -(alpha1 * c11 + alpha2 * c12);
            let c2 = lambda * distance_matrix[0][insert_node] + c1;

            if best_c1.is_none_or(|x| c1 > x)
                && best_c2.is_none_or(|x| c2 > x)
                && is_feasible(
                    route,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                    insert_node,
                    new_arrival_time + service_time,
                    pos,
                )
            {
                best_c1 = Some(c1);
                best_c2 = Some(c2);
                best = Some((insert_node, pos));
            }

            curr_time = ready_times[next_node]
                .max(curr_time + distance_matrix[curr_node][next_node])
                + service_time;
            curr_node = next_node;
        }
    }
    best
}
}
mod compression {
use super::problem::{NodeData, Problem};
use super::sequence::Sequence;

pub struct ProblemCompression {
    pub compact: Problem,
    pub chains: Vec<Vec<usize>>,
}

impl ProblemCompression {
    pub fn from_chains(data: &Problem, chains: Vec<Vec<usize>>) -> Option<Self> {
        let reuse = super::super::COMPACT_REUSE.with(|enabled| enabled.get());
        let mut compact_chains: Vec<Vec<usize>> = Vec::with_capacity(chains.len());
        for chain in chains {
            debug_assert!(!chain.is_empty(), "Consensus chain should never be empty");
            let chain_seq = Self::sequence_on_chain(data, &chain);
            if chain_seq.tw == 0 && chain_seq.earliest_start() <= chain_seq.tau_plus {
                compact_chains.push(chain);
            } else {
                for id in chain {
                    compact_chains.push(vec![id]);
                }
            }
        }

        let compact_clients = compact_chains.len();
        debug_assert!(compact_clients <= data.nb_nodes - 1);
        if compact_clients == data.nb_nodes - 1 {
            return None;
        }

        let mut node_data: Vec<NodeData> = Vec::with_capacity(compact_clients + 1);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(compact_clients + 1);
        let mut first_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut last_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut fixed_distance: i32 = 0;

        node_data.push(data.node_data[0]);
        node_positions.push(data.node_positions[0]);

        for chain in &compact_chains {
            if reuse && chain.len() == 1 {
                let id = chain[0];
                let compact_id = node_data.len();
                node_data.push(data.node_data[id]);
                node_positions.push(data.node_positions[id]);
                first_orig[compact_id] = id;
                last_orig[compact_id] = id;
                continue;
            }
            let chain_seq = Self::sequence_on_chain(data, chain);
            let demand_sum = chain_seq.load;
            let mut sum_x: i64 = 0;
            let mut sum_y: i64 = 0;
            for &id in chain {
                sum_x += data.node_positions[id].0 as i64;
                sum_y += data.node_positions[id].1 as i64;
            }
            let chain_len = chain.len() as i64;
            let barycenter_x = (sum_x / chain_len) as i32;
            let barycenter_y = (sum_y / chain_len) as i32;

            fixed_distance += chain_seq.distance;

            node_data.push(NodeData {
                start_tw: chain_seq.earliest_start(),
                end_tw: chain_seq.tau_plus,
                service_time: chain_seq.duration(),
                demand: demand_sum,
            });
            node_positions.push((barycenter_x, barycenter_y));
            first_orig[node_data.len() - 1] = chain[0];
            last_orig[node_data.len() - 1] = chain[chain.len() - 1];
        }

        let nb_nodes = compact_clients + 1;
        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        let distance_matrix_transposed;
        if reuse {
            let fill_matrix = |source: &[i32],
                               rows: &[usize],
                               columns: &[usize],
                               target: &mut [i32]| {
                let mut runs = Vec::with_capacity(columns.len());
                let mut start = 0;
                while start < columns.len() {
                    let mut end = start + 1;
                    while end < columns.len() && columns[end] == columns[end - 1] + 1 {
                        end += 1;
                    }
                    runs.push((start, end));
                    start = end;
                }

                for (i, &row) in rows.iter().enumerate() {
                    let source_row = row * data.nb_nodes;
                    let target_row = i * nb_nodes;
                    for &(start, end) in &runs {
                        let source_start = source_row + columns[start];
                        let target_start = target_row + start;
                        let len = end - start;
                        if len > 1 {
                            target[target_start..target_start + len]
                                .copy_from_slice(&source[source_start..source_start + len]);
                        } else {
                            target[target_start] = source[source_start];
                        }
                    }
                }
            };

            fill_matrix(
                &data.distance_matrix,
                &last_orig,
                &first_orig,
                &mut distance_matrix,
            );
            let mut transposed = vec![0i32; nb_nodes * nb_nodes];
            fill_matrix(
                &data.distance_matrix_transposed,
                &first_orig,
                &last_orig,
                &mut transposed,
            );
            distance_matrix_transposed = transposed;
        } else {
            for i in 0..nb_nodes {
                let from_node = if i == 0 { 0 } else { last_orig[i] };
                for j in 0..nb_nodes {
                    let to_node = if j == 0 { 0 } else { first_orig[j] };
                    distance_matrix[i * nb_nodes + j] = data.dm(from_node, to_node);
                }
            }
            distance_matrix_transposed =
                Problem::transpose_distances(&distance_matrix, nb_nodes);
        }

        let total_demand = node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand as i64)
            .sum::<i64>();
        let lb_vehicles =
            ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
        debug_assert!(
            lb_vehicles <= data.nb_vehicles,
            "lb_vehicles exceeds available vehicles"
        );
        debug_assert!(
            lb_vehicles <= compact_clients,
            "lb_vehicles exceeds number of clients"
        );

        let compact = Problem {
            seed: data.seed,
            nb_nodes,
            nb_vehicles: data.nb_vehicles,
            lb_vehicles,
            fixed_distance_offset: data.fixed_distance_offset + fixed_distance as i64,
            max_capacity: data.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_positions,
            node_data,
        };

        Some(Self {
            compact,
            chains: compact_chains,
        })
    }

    #[inline]
    fn sequence_on_chain(data: &Problem, chain: &[usize]) -> Sequence {
        let mut seq = Sequence::singleton(data, chain[0]);
        for &id in chain.iter().skip(1) {
            let node = Sequence::singleton(data, id);
            seq = Sequence::join2(data, &seq, &node);
        }
        seq
    }
}
}
mod constructive {
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};

#[inline(always)]
fn copy_at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

const SHUFFLE_WINDOW: usize = 10;

#[derive(Default)]
struct RouteTimes {
    depart: Vec<i32>,
    edge: Vec<i32>,
    late: Vec<i32>,
}

pub struct Constructive;

impl Constructive {
    pub fn build_routes(data: &Problem, rng: &mut SmallRng, randomize: bool) -> Vec<Vec<usize>> {
        let mut routes = Vec::new();
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        let n = nodes.len();
        nodes.sort_by(|&a, &b| data.dm(0, a).cmp(&data.dm(0, b)));

        if randomize {
            for i in 0..(n - 1) {
                nodes.swap(i, rng.gen_range(i + 1..=(i + SHUFFLE_WINDOW).min(n - 1)));
            }
        }

        let mut available = vec![true; data.nb_nodes];
        available[0] = false;
        let mut times = RouteTimes::default();

        while let Some(node) = nodes.pop() {
            if !copy_at(&available, node) {
                continue;
            }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.nd(node).demand;

            while let Some((best_node, best_pos)) = Self::find_best_insertion(
                &route,
                &nodes,
                &available,
                route_demand,
                data,
                &mut times,
            ) {
                available[best_node] = false;
                route_demand += data.nd(best_node).demand;
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        routes
    }

    fn find_best_insertion(
        route: &[usize],
        nodes: &[usize],
        available: &[bool],
        route_demand: i32,
        data: &Problem,
        times: &mut RouteTimes,
    ) -> Option<(usize, usize)> {
        let len = route.len();
        let last = len - 1;
        times.depart.clear();
        times.depart.resize(len, 0);
        times.edge.clear();
        times.edge.resize(len, 0);
        times.late.clear();
        times.late.resize(len, 0);
        let depart = &mut times.depart;
        let edge = &mut times.edge;
        let late = &mut times.late;

        let mut curr_time = 0;
        let mut curr_node = 0;
        for pos in 1..len {
            let next_node = copy_at(route, pos);
            let next_nd = data.nd(next_node);
            let travel = data.dm(curr_node, next_node);
            curr_time = next_nd.start_tw.max(curr_time + travel) + next_nd.service_time;
            unsafe {
                *edge.get_unchecked_mut(pos) = travel;
                *depart.get_unchecked_mut(pos) = curr_time;
            }
            curr_node = next_node;
        }
        unsafe {
            *late.get_unchecked_mut(last) = data.nd(copy_at(route, last)).end_tw;
        }
        for pos in (1..last).rev() {
            let nd = data.nd(copy_at(route, pos));
            let bound = copy_at(late, pos + 1) - copy_at(edge, pos + 1) - nd.service_time;
            unsafe {
                *late.get_unchecked_mut(pos) = nd.end_tw.min(bound);
            }
        }

        let max_capacity = data.max_capacity;
        let mut best_c2 = i32::MIN;
        let mut best_node = 0usize;
        let mut best_pos = 0usize;
        for &insert_node in nodes.iter() {
            let insert_nd = data.nd(insert_node);

            let skip = !copy_at(available, insert_node)
                | (route_demand + insert_nd.demand > max_capacity);
            if skip {
                continue;
            }

            let from_insert = data.distance_row(insert_node);
            let to_insert = data.distance_column(insert_node);
            let start_tw = insert_nd.start_tw;
            let end_tw = insert_nd.end_tw;
            let leave = insert_nd.service_time;
            let depot_trip = copy_at(to_insert, 0);
            let mut curr_node = 0;

            for pos in 1..len {
                let next_node = copy_at(route, pos);
                let d_in = copy_at(to_insert, curr_node);
                let arrival = start_tw.max(copy_at(depart, pos - 1) + d_in);
                if arrival > end_tw {
                    break;
                }
                let d_out = copy_at(from_insert, next_node);
                let c2 = depot_trip - (d_in + d_out - copy_at(edge, pos));
                let feasible = arrival + leave + d_out <= copy_at(late, pos);
                let better = (c2 > best_c2) & feasible;
                best_c2 = if better { c2 } else { best_c2 };
                best_node = if better { insert_node } else { best_node };
                best_pos = if better { pos } else { best_pos };
                curr_node = next_node;
            }
        }
        if best_c2 == i32::MIN {
            None
        } else {
            Some((best_node, best_pos))
        }
    }
}
}
#[path = "part_b_genetic.rs"]
mod genetic;
mod individual {
use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;

#[derive(Clone, Debug)]
pub struct Individual {
    pub routes: Vec<Vec<usize>>,
    pub nb_routes: usize,
    pub distance: i32,
    pub tw_violation: i32,
    pub load_excess: i32,
    pub cost: i64,
    pub pred: Vec<usize>,
    pub succ: Vec<usize>,
    pub adj: Vec<u32>,
}

impl Individual {
    pub fn new_from_routes(data: &Problem, params: &Params, routes: Vec<Vec<usize>>) -> Self {
        let (distance, tw_violation, load_excess) = Self::evaluate_routes(data, &routes);
        Self::new_from_evaluated_routes(data, params, routes, distance, tw_violation, load_excess)
    }

    pub fn new_from_evaluated_routes(
        data: &Problem,
        params: &Params,
        routes: Vec<Vec<usize>>,
        distance: i32,
        tw_violation: i32,
        load_excess: i32,
    ) -> Self {
        let cost = Self::compute_penalized_cost(distance, tw_violation, load_excess, params);
        let (pred, succ, adj, nb_routes) = Self::build_pred_succ_and_count(data, &routes);
        Self {
            routes,
            nb_routes,
            distance,
            tw_violation,
            load_excess,
            cost,
            pred,
            succ,
            adj,
        }
    }

    pub fn evaluate_routes(data: &Problem, routes: &[Vec<usize>]) -> (i32, i32, i32) {
        let mut total_distance: i32 = 0;
        let mut total_tw_violation: i32 = 0;
        let mut total_load_excess: i32 = 0;
        for route in routes {
            if route.is_empty() {
                continue;
            }
            let mut prefix = Sequence::singleton(data, route[0]);
            for position in 1..route.len() {
                let node = Sequence::singleton(data, route[position]);
                prefix = Sequence::join2(data, &prefix, &node);
            }
            total_distance += prefix.distance;
            total_tw_violation += prefix.tw;
            let excess = (prefix.load - data.max_capacity).max(0);
            total_load_excess += excess;
        }
        (total_distance, total_tw_violation, total_load_excess)
    }

    #[inline]
    pub fn compute_penalized_cost(
        distance: i32,
        tw_violation: i32,
        load_excess: i32,
        params: &Params,
    ) -> i64 {
        (distance as i64)
            + (params.penalty_tw as i64) * (tw_violation as i64)
            + (params.penalty_capa as i64) * (load_excess as i64)
    }

    #[inline]
    pub fn recompute_cost(&mut self, params: &Params) {
        self.cost = Self::compute_penalized_cost(
            self.distance,
            self.tw_violation,
            self.load_excess,
            params,
        );
    }

    fn build_pred_succ_and_count(
        data: &Problem,
        routes: &[Vec<usize>],
    ) -> (Vec<usize>, Vec<usize>, Vec<u32>, usize) {
        let nb_nodes = data.nb_nodes;
        debug_assert!(nb_nodes < 65536);
        let mut pred = vec![0usize; nb_nodes];
        let mut succ = vec![0usize; nb_nodes];
        let mut adj = vec![0u32; nb_nodes];
        let mut seen = vec![false; nb_nodes];
        let mut nb_routes: usize = 0;

        for route in routes {
            if route.len() > 2 {
                nb_routes += 1;
            }
            for window in route.windows(3) {
                let client = window[1];
                debug_assert!(
                    client < nb_nodes && !seen[client],
                    "Client {} appears more than once in routes",
                    client
                );
                unsafe {
                    *seen.get_unchecked_mut(client) = true;
                    *pred.get_unchecked_mut(client) = window[0];
                    *succ.get_unchecked_mut(client) = window[2];
                    *adj.get_unchecked_mut(client) =
                        window[0] as u32 | (window[2] as u32) << 16;
                }
            }
        }
        for client in 1..nb_nodes {
            debug_assert!(seen[client], "Client {} is missing from routes", client);
        }
        debug_assert!((0..nb_nodes).all(|c| {
            adj[c] == (pred[c] as u32 | (succ[c] as u32) << 16)
        }));
        (pred, succ, adj, nb_routes)
    }
}
}
mod loader_tig {
use super::problem::{NodeData, Problem};
use tig_challenges::vehicle_routing::*;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Problem {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut node_data: Vec<NodeData> = (0..nb_nodes)
            .map(|i| NodeData {
                start_tw: challenge.ready_times[i],
                end_tw: challenge.due_times[i],
                service_time: if i == 0 { 0 } else { challenge.service_time },
                demand: challenge.demands[i],
            })
            .collect();

        let depot_start = node_data[0].start_tw;
        let depot_end = node_data[0].end_tw;
        for i in 1..nb_nodes {
            let d0i = challenge.distance_matrix[0][i];
            let di0 = challenge.distance_matrix[i][0];
            let nd = &mut node_data[i];
            nd.start_tw = nd.start_tw.max(depot_start + d0i);
            nd.end_tw = nd.end_tw.min(depot_end - nd.service_time - di0);
            debug_assert!(
                nd.start_tw <= nd.end_tw,
                "Node {} has empty tightened TW: [{}, {}]",
                i,
                nd.start_tw,
                nd.end_tw
            );
        }

        let total_demand: i64 = challenge.demands.iter().map(|&d| d as i64).sum();
        let lb_vehicles = ((total_demand + challenge.max_capacity as i64 - 1)
            / challenge.max_capacity as i64) as usize;
        let distance_matrix: Vec<i32> = challenge
            .distance_matrix
            .iter()
            .flatten()
            .copied()
            .collect();
        let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);
        Problem {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            fixed_distance_offset: 0,
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_data,
        }
    }
}
}
#[path = "part_b_local_search.rs"]
mod local_search;
mod params {
use super::problem::Problem;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const DEFAULT_EXPLORATION_LEVEL: usize = 4;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Params {
    pub exploration_level: usize,
    pub allow_swap3: bool,
    pub granularity: usize,
    pub granularity2: usize,
    pub swapstar_capa_filter: f64,
    pub penalty_tw: usize,
    pub penalty_capa: usize,
    pub target_ratio: f64,
    pub max_it_noimprov: usize,
    pub max_it_total: usize,
    pub nb_it_adapt_penalties: usize,
    pub nb_it_compression: usize,
    pub mu: usize,
    pub mu_start: usize,
    pub lambda: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
    pub crossover_srex_percent: usize,
    pub max_cli_srex: usize,
    pub max_credit_deterioration: usize,
    pub selection_weight_feasible: usize,
    pub factor_split: f64,
    pub decomp_target_size: usize,
    pub decomp_nb_phases: usize,
    #[serde(default)]
    pub stop_distance: i64,
    #[serde(default)]
    pub giveup_at: f64,
    #[serde(default)]
    pub giveup_distance: i64,
    #[serde(default)]
    pub fuel_use: f64,
    #[serde(default)]
    pub ot_max: i64,
    #[serde(default)]
    pub ot_cap: f64,
    #[serde(default)]
    pub ot_window: i64,
    #[serde(default = "no_phase")]
    pub l1_permille: usize,
    #[serde(default = "one_thousand")]
    pub l2_permille: usize,
    #[serde(default = "one_thousand")]
    pub l3_permille: usize,
    #[serde(default = "six_hundred")]
    pub l4_permille: usize,
    #[serde(default = "thirty_two_hundred")]
    pub sub_it_permille: usize,
    #[serde(default = "one_thousand")]
    pub sub_mu_permille: usize,
    #[serde(default = "eight")]
    pub inh_k: usize,
    #[serde(default = "four")]
    pub inh_min_level: usize,
    #[serde(default = "default_sub_it_l4_permille")]
    pub sub_it_l4_permille: Option<usize>,
}

fn one_thousand() -> usize {
    1000
}

fn four() -> usize {
    4
}

fn no_phase() -> usize {
    0
}

fn six_hundred() -> usize {
    600
}

fn thirty_two_hundred() -> usize {
    3200
}

fn eight() -> usize {
    8
}

fn default_sub_it_l4_permille() -> Option<usize> {
    Some(1400)
}

struct Preset {
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    max_it_noimprov: usize,
    max_it_total: usize,
    nb_it_adapt_penalties: usize,
    nb_it_compression: usize,
    mu: usize,
    mu_start: usize,
    lambda: usize,
    nb_close: usize,
    nb_elite: usize,
    max_cli_srex: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
}

#[rustfmt::skip]
const PRESETS_BY_EXPLORATION_LEVEL: [Preset; 7] = [
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start:  1, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start: 20, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:   40, max_it_total:   150,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  3, mu_start:  6, lambda:  3,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  100, max_it_total:   400,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  7, mu_start: 10, lambda:  3,
             nb_close: 2, nb_elite: 2, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  200, max_it_total:  2000,
             nb_it_adapt_penalties: 50, nb_it_compression:  75, mu: 15, mu_start: 20, lambda:  5,
             nb_close: 2, nb_elite: 3, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 2000, max_it_total: 20000,
             nb_it_adapt_penalties: 50, nb_it_compression: 100, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 3 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 5000, max_it_total: 50000,
             nb_it_adapt_penalties: 50, nb_it_compression: 150, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 8 },
];

struct TrackProfile {
    nb_nodes: usize,
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    swapstar_capa_filter: f64,
    max_credit_deterioration: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
    l4_permille: usize,
}

#[rustfmt::skip]
const TRACK_PROFILES: [TrackProfile; 5] = [
    TrackProfile { nb_nodes:  600, allow_swap3: true, granularity: 35, granularity2: 45,
                   swapstar_capa_filter: 0.6, max_credit_deterioration: 25,
                   decomp_target_size: 200, decomp_nb_phases:  9, l4_permille: 600 },
    TrackProfile { nb_nodes:  700, allow_swap3: true, granularity: 30, granularity2: 45,
                   swapstar_capa_filter: 0.4, max_credit_deterioration: 30,
                   decomp_target_size: 233, decomp_nb_phases:  8, l4_permille: 600 },
    TrackProfile { nb_nodes:  800, allow_swap3: true, granularity: 25, granularity2: 35,
                   swapstar_capa_filter: 0.3, max_credit_deterioration: 10,
                   decomp_target_size: 200, decomp_nb_phases: 11, l4_permille: 400 },
    TrackProfile { nb_nodes:  900, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 150, decomp_nb_phases: 12, l4_permille: 400 },
    TrackProfile { nb_nodes: 1000, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 200, decomp_nb_phases: 10, l4_permille: 600 },
];

impl Params {
    pub(super) fn preset(exploration_level: usize, data: &Problem) -> Self {
        let max_dist = data.distance_matrix.iter().copied().max().unwrap_or(1) as f64;
        let max_demand = data
            .node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand)
            .max()
            .unwrap_or(1) as f64;
        let penalty_capa = (max_dist / max_demand).clamp(1.0, 500.0).round() as usize;

        let preset = match PRESETS_BY_EXPLORATION_LEVEL.get(exploration_level) {
            Some(preset) => preset,
            None => return Self::defaults(data),
        };

        Self {
            exploration_level,
            allow_swap3: preset.allow_swap3,
            granularity: preset.granularity,
            granularity2: preset.granularity2,
            swapstar_capa_filter: 0.3,
            penalty_tw: 10,
            penalty_capa,
            target_ratio: 0.4,
            max_it_noimprov: preset.max_it_noimprov,
            max_it_total: preset.max_it_total,
            nb_it_adapt_penalties: preset.nb_it_adapt_penalties,
            nb_it_compression: preset.nb_it_compression,
            mu: preset.mu,
            mu_start: preset.mu_start,
            lambda: preset.lambda,
            nb_close: preset.nb_close,
            nb_elite: preset.nb_elite,
            crossover_srex_percent: 75,
            max_cli_srex: preset.max_cli_srex,
            max_credit_deterioration: 10,
            selection_weight_feasible: 2,
            factor_split: 1.3,
            decomp_target_size: preset.decomp_target_size,
            decomp_nb_phases: preset.decomp_nb_phases,
            stop_distance: 0,
            giveup_at: 0.0,
            giveup_distance: 0,
            fuel_use: 0.0,
            ot_max: 0,
            ot_cap: 0.0,
            ot_window: 0,
            l1_permille: no_phase(),
            l2_permille: one_thousand(),
            l3_permille: one_thousand(),
            l4_permille: six_hundred(),
            sub_it_permille: thirty_two_hundred(),
            sub_mu_permille: one_thousand(),
            inh_k: eight(),
            inh_min_level: four(),
            sub_it_l4_permille: default_sub_it_l4_permille(),
        }
    }

    pub fn defaults(data: &Problem) -> Self {
        let mut params = Self::preset(DEFAULT_EXPLORATION_LEVEL, data);

        if let Some(profile) = TRACK_PROFILES
            .iter()
            .find(|profile| profile.nb_nodes == data.nb_nodes)
        {
            params.allow_swap3 = profile.allow_swap3;
            params.granularity = profile.granularity;
            params.granularity2 = profile.granularity2;
            params.swapstar_capa_filter = profile.swapstar_capa_filter;
            params.max_credit_deterioration = profile.max_credit_deterioration;
            params.decomp_target_size = profile.decomp_target_size;
            params.decomp_nb_phases = profile.decomp_nb_phases;
            params.l4_permille = profile.l4_permille;
        }

        if data.nb_nodes == 1000 {
            params.sub_it_permille = 3840;
        }

        params
    }

    fn with_overrides(base: Self, hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {
        let map = match hyperparameters {
            Some(map) if !map.is_empty() => map,
            _ => return base,
        };
        let mut merged_params = serde_json::to_value(base).expect("Params serializable");
        if let Value::Object(ref mut obj) = merged_params {
            for (k, v) in map {
                if k != "exploration_level" {
                    obj.insert(k.clone(), v.clone());
                }
            }
            for (key, default) in [
                ("inh_k", Value::from(eight())),
                ("inh_min_level", Value::from(four())),
                ("sub_it_l4_permille", Value::from(default_sub_it_l4_permille())),
            ] {
                let valid = match obj.get(key) {
                    Some(Value::Null) => key == "sub_it_l4_permille",
                    Some(value) => value
                        .as_u64()
                        .map_or(false, |u| usize::try_from(u).is_ok()),
                    None => false,
                };
                if !valid {
                    obj.insert(key.to_string(), default);
                }
            }
        }
        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(data))
    }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {
        let requested_level = requested_exploration_level(hyperparameters);

        let base_params = match requested_level {
            Some(level) if level != DEFAULT_EXPLORATION_LEVEL => Self::preset(level, data),
            _ => Self::defaults(data),
        };

        Self::with_overrides(base_params, hyperparameters, data)
    }

    pub(super) fn initialize_at_level(
        hyperparameters: &Option<Map<String, Value>>,
        exploration_level: usize,
        data: &Problem,
    ) -> Self {
        Self::with_overrides(Self::preset(exploration_level, data), hyperparameters, data)
    }
}

fn requested_exploration_level(hyperparameters: &Option<Map<String, Value>>) -> Option<usize> {
    hyperparameters
        .as_ref()
        .and_then(|map| map.get("exploration_level"))
        .and_then(|value| match value {
            Value::Number(n) => n.as_u64().map(|u| u as usize),
            Value::String(s) => s.parse::<usize>().ok(),
            _ => None,
        })
}
}
mod population {
use super::individual::Individual;
use super::params::Params;
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    pub prox: Vec<Vec<(f64, usize)>>,
    pub biased_fitness: Vec<f64>,
    pub order_cost: Vec<usize>,
}

pub struct Population {
    pub data: Arc<Problem>,
    pub feasible: Subpopulation,
    pub infeasible: Subpopulation,
    cap_window: VecDeque<bool>,
    tw_window: VecDeque<bool>,
    since_last_adapt: usize,
    consensus_ref_succ: Vec<usize>,
    consensus_stable: Vec<bool>,
    consensus_ref_pred: Vec<usize>,
    consensus_stable_pred: Vec<bool>,
    consensus_started: bool,
    pub consensus_enabled: bool,
}

pub struct ConsensusEpoch {
    pub chains: Vec<Vec<usize>>,
    pub complete_routes: Vec<Vec<usize>>,
}

impl Population {
    pub fn new(data: Arc<Problem>) -> Self {
        let nb_nodes = data.nb_nodes;
        Self {
            data,
            feasible: Subpopulation::default(),
            infeasible: Subpopulation::default(),
            cap_window: VecDeque::new(),
            tw_window: VecDeque::new(),
            since_last_adapt: 0,
            consensus_ref_succ: vec![0; nb_nodes],
            consensus_stable: vec![false; nb_nodes],
            consensus_ref_pred: vec![0; nb_nodes],
            consensus_stable_pred: vec![false; nb_nodes],
            consensus_started: false,
            consensus_enabled: false,
        }
    }

    pub fn survivors_selection(sub: &mut Subpopulation, params: &Params) {
        while sub.indivs.len() > params.mu {
            let idx = Self::worst_index_biased_with_clone_priority(sub);
            Self::remove_at_index(sub, idx);
            Self::order_cost_rebuild(sub);
            Self::update_biased_fitnesses(sub, params);
        }
    }

    pub fn add(&mut self, ind: Individual, params: &Params) {
        if self.cap_window.capacity() == 0 {
            let capacity = params.nb_it_adapt_penalties + 1;
            self.cap_window.reserve(capacity);
            self.tw_window.reserve(capacity);
        }

        let is_feasible = ind.load_excess == 0 && ind.tw_violation == 0;
        if is_feasible && self.consensus_enabled {
            self.observe_consensus_candidate(&ind);
        }
        debug_assert!(
            ind.nb_routes <= self.data.nb_vehicles,
            "Individual has too many routes"
        );
        {
            let sub = if is_feasible {
                &mut self.feasible
            } else {
                &mut self.infeasible
            };

            if sub.indivs.capacity() == 0 {
                let capacity = params.mu + params.lambda + 1;
                sub.indivs.reserve(capacity);
                sub.prox.reserve(capacity);
                sub.biased_fitness.reserve(capacity);
                sub.order_cost.reserve(capacity);
            }

            let new_idx = sub.indivs.len();
            sub.indivs.push(ind);
            Self::prox_add(sub, self.data.as_ref(), new_idx);

            Self::order_cost_rebuild(sub);

            Self::update_biased_fitnesses(sub, params);

            if sub.indivs.len() > params.mu + params.lambda {
                Self::survivors_selection(sub, params);
            }
        }
    }

    pub fn reset_consensus_epoch(&mut self) {
        self.consensus_ref_succ.fill(0);
        self.consensus_stable.fill(false);
        self.consensus_ref_pred.fill(0);
        self.consensus_stable_pred.fill(false);
        self.consensus_started = false;
    }

    pub fn copy_tracking_state_from(&mut self, other: &Population) {
        self.cap_window = other.cap_window.clone();
        self.tw_window = other.tw_window.clone();
        self.since_last_adapt = other.since_last_adapt;
    }

    pub fn consensus_epoch(&mut self, extra_individual: Option<&Individual>) -> ConsensusEpoch {
        let nb_nodes = self.data.nb_nodes;
        if nb_nodes <= 1 || !self.consensus_started {
            return ConsensusEpoch {
                chains: Vec::new(),
                complete_routes: Vec::new(),
            };
        }

        for i in 1..nb_nodes {
            if self.consensus_stable[i] {
                let expected_succ = self.consensus_ref_succ[i];
                for ind in self
                    .feasible
                    .indivs
                    .iter()
                    .chain(self.infeasible.indivs.iter())
                {
                    if ind.succ[i] != expected_succ {
                        self.consensus_stable[i] = false;
                        break;
                    }
                }
                if self.consensus_stable[i] {
                    if let Some(ind) = extra_individual {
                        if ind.succ[i] != expected_succ {
                            self.consensus_stable[i] = false;
                        }
                    }
                }
            }
            if self.consensus_stable_pred[i] {
                let expected_pred = self.consensus_ref_pred[i];
                for ind in self
                    .feasible
                    .indivs
                    .iter()
                    .chain(self.infeasible.indivs.iter())
                {
                    if ind.pred[i] != expected_pred {
                        self.consensus_stable_pred[i] = false;
                        break;
                    }
                }
                if self.consensus_stable_pred[i] {
                    if let Some(ind) = extra_individual {
                        if ind.pred[i] != expected_pred {
                            self.consensus_stable_pred[i] = false;
                        }
                    }
                }
            }
        }

        let mut succ = vec![0usize; nb_nodes];
        let mut pred = vec![0usize; nb_nodes];
        for i in 1..nb_nodes {
            let j = self.consensus_ref_succ[i];
            if self.consensus_stable[i]
                && j != 0
                && self.consensus_stable_pred[j]
                && self.consensus_ref_pred[j] == i
            {
                succ[i] = j;
                pred[j] = i;
            }
        }

        let chains = Self::chains_from_successors(&succ, &pred, nb_nodes);
        let mut kept: Vec<Vec<usize>> = Vec::new();
        let mut complete_routes: Vec<Vec<usize>> = Vec::new();
        for chain in chains {
            let first = chain[0];
            let last = chain[chain.len() - 1];
            if self.consensus_stable_pred[first]
                && self.consensus_ref_pred[first] == 0
                && self.consensus_stable[last]
                && self.consensus_ref_succ[last] == 0
            {
                complete_routes.push(chain);
            } else {
                kept.push(chain);
            }
        }

        ConsensusEpoch {
            chains: kept,
            complete_routes,
        }
    }

    fn observe_consensus_candidate(&mut self, ind: &Individual) {
        if !self.consensus_started {
            for i in 1..self.data.nb_nodes {
                self.consensus_ref_succ[i] = ind.succ[i];
                self.consensus_stable[i] = true;
                self.consensus_ref_pred[i] = ind.pred[i];
                self.consensus_stable_pred[i] = true;
            }
            self.consensus_started = true;
        } else {
            for i in 1..self.data.nb_nodes {
                if self.consensus_stable[i] && ind.succ[i] != self.consensus_ref_succ[i] {
                    self.consensus_stable[i] = false;
                }
                if self.consensus_stable_pred[i] && ind.pred[i] != self.consensus_ref_pred[i] {
                    self.consensus_stable_pred[i] = false;
                }
            }
        }
    }

    fn chains_from_successors(succ: &[usize], pred: &[usize], nb_nodes: usize) -> Vec<Vec<usize>> {
        if nb_nodes <= 1 {
            return Vec::new();
        }

        let mut visited = vec![false; nb_nodes];
        let mut chains: Vec<Vec<usize>> = Vec::with_capacity(nb_nodes - 1);

        for i in 1..nb_nodes {
            if pred[i] != 0 || visited[i] {
                continue;
            }
            let mut current = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[current] {
                    break;
                }
                visited[current] = true;
                chain.push(current);
                let next = succ[current];
                if next == 0 || pred[next] != current {
                    break;
                }
                current = next;
            }
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        for i in 1..nb_nodes {
            if visited[i] {
                continue;
            }
            let mut current = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[current] {
                    break;
                }
                visited[current] = true;
                chain.push(current);
                let next = succ[current];
                if next == 0 {
                    break;
                }
                current = next;
            }
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        debug_assert_eq!(
            chains.iter().map(|c| c.len()).sum::<usize>(),
            nb_nodes - 1,
            "Consensus chains should partition all clients"
        );
        chains
    }

    pub fn record_and_adapt(&mut self, cap_feasible: bool, tw_feasible: bool, params: &mut Params) {
        let period = params.nb_it_adapt_penalties;
        self.cap_window.push_back(cap_feasible);
        self.tw_window.push_back(tw_feasible);
        if self.cap_window.len() > period {
            self.cap_window.pop_front();
        }
        if self.tw_window.len() > period {
            self.tw_window.pop_front();
        }
        self.since_last_adapt += 1;

        if self.since_last_adapt == period {
            debug_assert!(self.cap_window.len() == period && self.tw_window.len() == period);
            let cap_ok = self
                .cap_window
                .iter()
                .rev()
                .take(period)
                .filter(|&&b| b)
                .count();
            let tw_ok = self
                .tw_window
                .iter()
                .rev()
                .take(period)
                .filter(|&&b| b)
                .count();
            let frac_cap = (cap_ok as f64) / (period as f64);
            let frac_tw = (tw_ok as f64) / (period as f64);

            let block_penalty_decrease = if !self.infeasible.indivs.is_empty() {
                let best_infeas_routes =
                    self.infeasible.indivs[self.infeasible.order_cost[0]].nb_routes;
                best_infeas_routes < self.data.lb_vehicles
            } else {
                false
            };

            if frac_cap < params.target_ratio {
                params.penalty_capa =
                    (((params.penalty_capa as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else if !block_penalty_decrease {
                params.penalty_capa =
                    (((params.penalty_capa as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }
            if frac_tw < params.target_ratio {
                params.penalty_tw =
                    (((params.penalty_tw as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else {
                params.penalty_tw =
                    (((params.penalty_tw as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }

            self.since_last_adapt = 0;
            self.recompute_costs(params);
        }
    }

    pub fn recompute_costs(&mut self, params: &Params) {
        for ind in self.feasible.indivs.iter_mut() {
            ind.recompute_cost(params);
        }
        for ind in self.infeasible.indivs.iter_mut() {
            ind.recompute_cost(params);
        }

        Self::order_cost_rebuild(&mut self.feasible);
        Self::order_cost_rebuild(&mut self.infeasible);

        Self::update_biased_fitnesses(&mut self.feasible, params);
        Self::update_biased_fitnesses(&mut self.infeasible, params);
    }

    pub fn best_feasible(&self) -> Option<&Individual> {
        if !self.feasible.indivs.is_empty() {
            return Some(&self.feasible.indivs[self.feasible.order_cost[0]]);
        }
        None
    }

    pub fn get_binary_tournament<'b>(
        &'b self,
        rng: &mut SmallRng,
        params: &Params,
    ) -> &'b Individual {
        let feas_n = self.feasible.indivs.len();
        let inf_n = self.infeasible.indivs.len();
        let total = feas_n + inf_n;
        debug_assert!(
            total >= 2,
            "Population should contain at least two individuals for tournament"
        );
        let feasible_weight = params.selection_weight_feasible;
        let weighted_total = feasible_weight * feas_n + inf_n;

        let pick = |rng: &mut SmallRng| -> (bool, usize, f64) {
            let ticket = rng.gen_range(0..weighted_total);
            let feasible_tickets = feasible_weight * feas_n;
            if ticket < feasible_tickets {
                let i = ticket / feasible_weight;
                (true, i, self.feasible.biased_fitness[i])
            } else {
                let i = ticket - feasible_tickets;
                (false, i, self.infeasible.biased_fitness[i])
            }
        };

        let (feasible1, index1, fitness1) = pick(rng);
        let (feasible2, index2, fitness2) = pick(rng);

        if fitness1 <= fitness2 {
            if feasible1 {
                &self.feasible.indivs[index1]
            } else {
                &self.infeasible.indivs[index1]
            }
        } else {
            if feasible2 {
                &self.feasible.indivs[index2]
            } else {
                &self.infeasible.indivs[index2]
            }
        }
    }

    pub fn best_metric(&self) -> BestMetric {
        let offset = self.data.fixed_distance_offset;
        if !self.feasible.indivs.is_empty() {
            let mut best_distance = i64::MAX;
            for ind in &self.feasible.indivs {
                let distance = (ind.distance as i64) + offset;
                if distance < best_distance {
                    best_distance = distance;
                }
            }
            return BestMetric {
                feasible: true,
                distance: best_distance,
                infeas_sum: 0,
            };
        }
        let mut best_sum = i32::MAX;
        let mut best_dist = i64::MAX;
        for ind in &self.infeasible.indivs {
            let violation = ind.load_excess + ind.tw_violation;
            let distance = (ind.distance as i64) + offset;
            if violation < best_sum || (violation == best_sum && distance < best_dist) {
                best_sum = violation;
                best_dist = distance;
            }
        }
        BestMetric {
            feasible: false,
            distance: best_dist,
            infeas_sum: best_sum,
        }
    }

    fn worst_index_biased_with_clone_priority(sub: &Subpopulation) -> usize {
        const CLONE_EPS: f64 = 1e-6;
        let mut worst_idx = 0usize;
        let mut worst_is_clone = (sub.prox[0][0].0 <= CLONE_EPS) as u8;
        let mut worst_fit = sub.biased_fitness[0];

        for i in 1..sub.indivs.len() {
            let is_clone = (sub.prox[i][0].0 <= CLONE_EPS) as u8;
            let fitness = sub.biased_fitness[i];
            if is_clone > worst_is_clone || (is_clone == worst_is_clone && fitness > worst_fit) {
                worst_is_clone = is_clone;
                worst_fit = fitness;
                worst_idx = i;
            }
        }
        worst_idx
    }

    fn prox_add(sub: &mut Subpopulation, data: &Problem, new_idx: usize) {
        let count = sub.indivs.len();
        debug_assert_eq!(sub.prox.len(), new_idx);
        sub.prox.push(Vec::with_capacity(count.saturating_sub(1)));

        let sampled = super::super::SAMPLED_DIVERSITY.with(|enabled| enabled.get());
        let mut panel = [0usize; 64];
        let panel_len = if sampled {
            let n_clients = data.nb_nodes - 1;
            let k = n_clients.min(panel.len());
            for j in 0..k {
                panel[j] = 1 + j * n_clients / k;
            }
            k
        } else {
            0
        };

        for i in 0..new_idx {
            let d = if sampled && panel_len > 0 {
                let a = &sub.indivs[i];
                let b = &sub.indivs[new_idx];
                let mut differences = 0usize;
                for &node in &panel[..panel_len] {
                    let same_adj = (a.pred[node] == b.pred[node])
                        & (a.succ[node] == b.succ[node]);
                    differences += (!same_adj) as usize;
                }
                if differences == 0 {
                    Self::hamming_distance(data, a, b)
                } else {
                    (differences as f64) / (panel_len as f64)
                }
            } else {
                Self::hamming_distance(data, &sub.indivs[i], &sub.indivs[new_idx])
            };

            let row = &mut sub.prox[i];
            let insert_pos = row.partition_point(|(dd, _)| *dd <= d);
            row.insert(insert_pos, (d, new_idx));
            sub.prox[new_idx].push((d, i));
        }
        sub.prox[new_idx].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    fn remove_at_index(sub: &mut Subpopulation, idx: usize) {
        let count = sub.indivs.len();
        let last = count - 1;
        sub.indivs.swap_remove(idx);
        sub.biased_fitness.swap_remove(idx);
        if idx != last {
            sub.prox.swap(idx, last);
        }
        sub.prox.pop();

        for i in 0..sub.prox.len() {
            let row = &mut sub.prox[i];
            let mut write = 0usize;
            let len = row.len();
            for read in 0..len {
                let (d, mut j) = row[read];
                if j == idx {
                    continue;
                }
                if idx != last && j == last {
                    j = idx;
                }
                row[write] = (d, j);
                write += 1;
            }
            row.truncate(write);
        }

        debug_assert_eq!(sub.indivs.len(), sub.prox.len());
        debug_assert!(sub
            .prox
            .iter()
            .all(|row| row.len() == sub.indivs.len().saturating_sub(1)));
    }

    fn order_cost_rebuild(sub: &mut Subpopulation) {
        sub.order_cost.clear();
        sub.order_cost.extend(0..sub.indivs.len());
        sub.order_cost.sort_unstable_by_key(|&i| sub.indivs[i].cost);
    }

    fn update_biased_fitnesses(sub: &mut Subpopulation, params: &Params) {
        let n = sub.indivs.len();
        if n == 0 {
            return;
        }
        sub.biased_fitness.resize(n, 0.0);
        if n == 1 {
            sub.biased_fitness[0] = 0.0;
            return;
        }

        let nb_close = params.nb_close.min(n - 1);

        let mut avg_closest = vec![0.0; n];
        for i in 0..n {
            let neighbors = &sub.prox[i];
            debug_assert!(neighbors.len() == n - 1);
            let mut sum = 0.0;
            for k in 0..nb_close {
                sum += neighbors[k].0;
            }
            avg_closest[i] = sum / (nb_close as f64);
        }

        let mut diversity_order: Vec<(f64, usize)> = (0..n).map(|i| (-avg_closest[i], i)).collect();
        diversity_order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let denom = (n - 1) as f64;
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in diversity_order.iter().enumerate() {
            div_rank[idx] = (pos as f64) / denom;
        }

        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in sub.order_cost.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let fit_rank: Vec<f64> = cost_pos.iter().map(|&p| (p as f64) / denom).collect();

        let scale = 1.0 - (params.nb_elite as f64) / (n as f64);
        for i in 0..n {
            sub.biased_fitness[i] = fit_rank[i] + scale * div_rank[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn adj_diff_avx2(a: &[u32], b: &[u32]) -> usize {
        use std::arch::x86_64::*;
        assert_eq!(a.len(), b.len());
        let mut differences = 0usize;
        let mut i = 0usize;
        unsafe {
            while a.len() - i >= 8 {
                let x = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let y = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let equal = _mm256_cmpeq_epi32(x, y);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u32;
                differences += 8 - mask.count_ones() as usize;
                i += 8;
            }
        }
        differences + a[i..].iter().zip(&b[i..]).filter(|(x, y)| x != y).count()
    }

    fn hamming_distance(data: &Problem, indiv_a: &Individual, indiv_b: &Individual) -> f64 {
        let n_clients = data.nb_nodes - 1;
        let a = &indiv_a.adj[1..data.nb_nodes];
        let b = &indiv_b.adj[1..data.nb_nodes];
        let scalar = || a.iter().zip(b).filter(|(x, y)| x != y).count();
        #[cfg(target_arch = "x86_64")]
        let differences = if std::is_x86_feature_detected!("avx2") {
            unsafe { Self::adj_diff_avx2(a, b) }
        } else {
            scalar()
        };
        #[cfg(not(target_arch = "x86_64"))]
        let differences = scalar();
        debug_assert_eq!(differences, scalar());
        #[cfg(debug_assertions)]
        {
            for ind in [indiv_a, indiv_b] {
                debug_assert!((0..data.nb_nodes).all(|c| {
                    ind.adj[c] == (ind.pred[c] as u32 | (ind.succ[c] as u32) << 16)
                }));
            }
            let original = (1..data.nb_nodes)
                .filter(|&c| {
                    !((indiv_a.pred[c] == indiv_b.pred[c])
                        & (indiv_a.succ[c] == indiv_b.succ[c]))
                })
                .count();
            debug_assert_eq!(differences, original);
        }
        (differences as f64) / (n_clients as f64)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BestMetric {
    pub feasible: bool,
    pub distance: i64,
    pub infeas_sum: i32,
}

impl BestMetric {
    #[inline]
    pub fn better_than(self, other: BestMetric) -> bool {
        if self.feasible && !other.feasible {
            return true;
        }
        if !self.feasible && other.feasible {
            return false;
        }
        if self.feasible {
            self.distance < other.distance
        } else {
            if self.infeas_sum != other.infeas_sum {
                self.infeas_sum < other.infeas_sum
            } else {
                self.distance < other.distance
            }
        }
    }
}
}
mod pred_queue {
pub struct IndexDeque {
    pub data: Vec<usize>,
    pub head: usize,
}

impl IndexDeque {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            head: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head >= self.data.len()
    }

    #[inline]
    pub fn push_back(&mut self, v: usize) {
        self.data.push(v);
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            self.data.last().copied()
        }
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.data[self.head])
        }
    }

    #[inline]
    pub fn pop_back(&mut self) {
        if self.is_empty() {
            return;
        }
        self.data.pop();
        if self.head >= self.data.len() {
            self.clear();
        }
    }

    #[inline]
    pub fn pop_front(&mut self) {
        if self.is_empty() {
            return;
        }
        self.head += 1;
        if self.head >= self.data.len() {
            self.clear();
        }
    }
}

pub struct PredQueue {
    pub prev: Vec<usize>,
    pub next: Vec<usize>,
    pub nil: usize,
    pub size: usize,
    pub feas: usize,
    pub no_warp: usize,
}

impl PredQueue {
    pub fn new(n: usize) -> Self {
        Self {
            prev: vec![n; n + 1],
            next: vec![n; n + 1],
            nil: n,
            size: 0,
            feas: n,
            no_warp: n,
        }
    }

    #[inline(always)]
    fn link(&self, id: usize) -> (usize, usize) {
        debug_assert!(id <= self.nil);
        unsafe {
            (
                *self.prev.get_unchecked(id),
                *self.next.get_unchecked(id),
            )
        }
    }

    #[inline(always)]
    pub fn head(&self) -> usize {
        self.link(self.nil).1
    }

    #[inline(always)]
    pub fn tail(&self) -> usize {
        self.link(self.nil).0
    }

    #[inline(always)]
    pub fn succ(&self, id: usize) -> usize {
        self.link(id).1
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        let h = self.head();
        if h != self.nil {
            Some(h)
        } else {
            None
        }
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        let t = self.tail();
        if t != self.nil {
            Some(t)
        } else {
            None
        }
    }

    #[inline]
    pub fn front2(&self) -> Option<usize> {
        let n = self.succ(self.head());
        if n != self.nil {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    pub fn insert_back(&mut self, id: usize) {
        let nil = self.nil;
        let t = self.tail();
        unsafe {
            *self.next.get_unchecked_mut(t) = id;
            *self.prev.get_unchecked_mut(id) = t;
            *self.next.get_unchecked_mut(id) = nil;
            *self.prev.get_unchecked_mut(nil) = id;
        }
        self.size += 1;
    }

    #[inline]
    pub fn remove_node(&mut self, id: usize) {
        let (p, n) = self.link(id);
        unsafe {
            *self.next.get_unchecked_mut(p) = n;
            *self.prev.get_unchecked_mut(n) = p;
        }
        self.feas = if self.feas == id { n } else { self.feas };
        self.no_warp = if self.no_warp == id { n } else { self.no_warp };
        self.size -= 1;
    }

    #[inline]
    pub fn remove_front(&mut self) {
        let h = self.head();
        if h != self.nil {
            self.remove_node(h);
        }
    }

    #[inline]
    pub fn remove_front2(&mut self) {
        if let Some(h2) = self.front2() {
            self.remove_node(h2);
        }
    }

    #[inline]
    pub fn feas_prev(&self) -> Option<usize> {
        if self.feas == self.nil {
            return None;
        }
        let p = self.link(self.feas).0;
        if p != self.nil {
            Some(p)
        } else {
            None
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        let nil = self.nil;
        unsafe {
            *self.next.get_unchecked_mut(nil) = nil;
            *self.prev.get_unchecked_mut(nil) = nil;
        }
        self.size = 0;
        self.feas = nil;
        self.no_warp = nil;
    }
}
}
mod problem {
#[derive(Copy, Clone)]
pub struct NodeData {
    pub start_tw: i32,
    pub end_tw: i32,
    pub service_time: i32,
    pub demand: i32,
}

pub struct Problem {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub fixed_distance_offset: i64,
    pub max_capacity: i32,
    pub distance_matrix: Vec<i32>,
    pub distance_matrix_transposed: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub node_data: Vec<NodeData>,
}

impl Problem {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        debug_assert!(i < self.nb_nodes && j < self.nb_nodes);
        let idx = i * self.nb_nodes + j;
        debug_assert!(idx < self.distance_matrix.len());
        unsafe { *self.distance_matrix.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn distance_row(&self, i: usize) -> &[i32] {
        debug_assert!(i < self.nb_nodes);
        let start = i * self.nb_nodes;
        unsafe {
            self.distance_matrix
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    #[inline(always)]
    pub fn distance_column(&self, j: usize) -> &[i32] {
        debug_assert!(j < self.nb_nodes);
        let start = j * self.nb_nodes;
        unsafe {
            self.distance_matrix_transposed
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    pub fn transpose_distances(distance_matrix: &[i32], nb_nodes: usize) -> Vec<i32> {
        let mut transposed = vec![0; distance_matrix.len()];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                transposed[j * nb_nodes + i] = distance_matrix[i * nb_nodes + j];
            }
        }
        transposed
    }

    #[inline(always)]
    pub fn nd(&self, i: usize) -> &NodeData {
        debug_assert!(i < self.nb_nodes);
        unsafe { self.node_data.get_unchecked(i) }
    }
}
}
#[path = "part_b_reverse_mode.rs"]
mod reverse_mode;
mod sequence {
use super::params::Params;
use super::problem::Problem;
use std::cmp::{max, min};

#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    pub earliest_end: i32,
    pub tau_plus: i32,
    pub duration_net: i32,
    pub tw: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: u16,
    pub last_node: u16,
}

impl Sequence {
    #[inline(always)]
    pub fn earliest_start(&self) -> i32 {
        self.earliest_end - self.duration_net
    }

    #[inline(always)]
    pub fn duration(&self) -> i32 {
        self.duration_net + self.tw
    }

    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        debug_assert!(u16::try_from(node).is_ok());
        let nd = data.nd(node);
        self.earliest_end = nd.start_tw + nd.service_time;
        self.tau_plus = nd.end_tw;
        self.duration_net = nd.service_time;
        self.tw = 0;
        self.load = nd.demand;
        self.distance = 0;
        self.first_node = node as u16;
        self.last_node = node as u16;
    }

    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node as usize, s2.first_node as usize);
        Self::join2_with_travel(s1, s2, travel)
    }

    #[inline(always)]
    pub fn join2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let mut out = Self::join_tw_with_travel(s1, s2, travel);
        out.distance = s1.distance + s2.distance + travel;
        out.load = s1.load + s2.load;
        out
    }

    #[inline(always)]
    pub fn join_tw_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let arrival = s1.earliest_end + travel;
        let extra = max(arrival - s2.tau_plus, 0);
        let shifted = s1.duration_net + travel;
        Sequence {
            earliest_end: max(s2.earliest_end, arrival + s2.duration_net) - extra,
            tau_plus: min(s2.tau_plus - shifted + extra, s1.tau_plus),
            duration_net: max(shifted + s2.duration_net, s2.earliest_end - s1.tau_plus) - extra,
            tw: s1.tw + s2.tw + extra,
            load: 0,
            distance: 0,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    #[inline(always)]
    pub fn singleton(data: &Problem, node: usize) -> Sequence {
        let mut s = Sequence::default();
        s.initialize(data, node);
        s
    }

    #[inline(always)]
    pub fn eval(&self, data: &Problem, params: &Params) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let load_excess = (self.load - data.max_capacity).max(0) as i64;
        (self.distance as i64) + load_excess * pcap + (self.tw as i64) * ptw
    }

    #[inline(always)]
    pub fn tw2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> i32 {
        s1.tw + s2.tw + max(s1.earliest_end + travel - s2.tau_plus, 0)
    }

    #[inline(always)]
    pub fn tw3_with_travel(
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        travel12: i32,
        travel23: i32,
    ) -> i32 {
        let arrival = s1.earliest_end + travel12;
        let extra12 = max(arrival - s2.tau_plus, 0);
        let end12 = max(s2.earliest_end, arrival + s2.duration_net);
        s1.tw + s2.tw + s3.tw + max(extra12, end12 + travel23 - s3.tau_plus)
    }

    #[inline(always)]
    pub fn tw5_with_travel(
        s0: &Sequence,
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        s4: &Sequence,
        travel01: i32,
        travel12: i32,
        travel23: i32,
        travel34: i32,
    ) -> i32 {
        let arrival1 = s0.earliest_end + travel01;
        let extra1 = max(arrival1 - s1.tau_plus, 0);
        let end1 = max(s1.earliest_end, arrival1 + s1.duration_net) - extra1;

        let arrival2 = end1 + travel12;
        let extra2 = max(arrival2 - s2.tau_plus, 0);
        let end2 = max(s2.earliest_end, arrival2 + s2.duration_net) - extra2;

        let arrival3 = end2 + travel23;
        let extra3 = max(arrival3 - s3.tau_plus, 0);
        let end3 = max(s3.earliest_end, arrival3 + s3.duration_net) - extra3;

        s0.tw + s1.tw + s2.tw + s3.tw + s4.tw
            + extra1
            + extra2
            + extra3
            + max(end3 + travel34 - s4.tau_plus, 0)
    }

    #[inline(always)]
    pub fn eval3(
        data: &Problem,
        params: &Params,
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
    ) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let travel12 = data.dm(s1.last_node as usize, s2.first_node as usize);
        let travel23 = data.dm(s2.last_node as usize, s3.first_node as usize);
        let distance = s1.distance + s2.distance + s3.distance + travel12 + travel23;
        let tw_viol = Self::tw3_with_travel(s1, s2, s3, travel12, travel23);
        let load = s1.load + s2.load + s3.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }
}
}
mod solver {
use super::stopping::Stopping;
use super::genetic::Genetic;
use super::loader_tig::TigLoader;
use super::params::Params;
use super::problem::Problem;
use super::reverse_mode;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    fn solve(
        data: Problem,
        params: Params,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        stopping: Option<Stopping>,
    ) -> Result<Option<Solution>> {
        if params.decomp_nb_phases > 0 {
            return reverse_mode::solve_reversed_mode(data, params, hyperparameters, save_solution, stopping);
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga
            .run(&mut rng, save_solution, None)
            .map(|individual| Solution { routes: individual.routes }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let data = TigLoader::load(challenge);
        let params = Params::initialize(hyperparameters, &data);
        let stopping = Stopping::new(challenge, hyperparameters);

        match Self::solve(data, params, hyperparameters, save_solution, stopping) {
            Ok(Some(solution)) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}
}
pub use solver::Solver;
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

#[allow(dead_code)]
pub fn help() {
    println!("Send {{\"target_quality\": T}}, T = the track's qualifying quality (199543 on chain = 0.199543).");
    println!("Update T when it is more than 0.003 from the line; missing = the README default.");
    println!("{{\"early_stop\": 0}} turns stopping off and gives the hgs_epure output.");
}
