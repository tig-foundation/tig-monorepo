use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};

/// Unchecked read of a solver-owned table whose length is fixed by the problem size.
#[inline(always)]
fn copy_at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

const SHUFFLE_WINDOW: usize = 10;

/// Departure time at each position, the arc reaching it and the latest
/// arrival that keeps the rest of the route feasible.
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

        // Availability bitmap: true = not yet routed. Index 0 is the depot, never routed.
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
