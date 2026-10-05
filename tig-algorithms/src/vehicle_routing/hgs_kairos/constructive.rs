use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};

/// Unchecked read of a solver-owned table whose length is fixed by the problem size.
#[inline(always)]
fn at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

const SHUFFLE_WINDOW: usize = 10;

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

        while let Some(node) = nodes.pop() {
            if !at(&available, node) {
                continue;
            }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.nd(node).demand;

            while let Some((best_node, best_pos)) =
                Self::find_best_insertion(&route, &nodes, &available, route_demand, data)
            {
                available[best_node] = false;
                route_demand += data.nd(best_node).demand;
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        routes
    }

    fn find_best_insertion(
        route: &Vec<usize>,
        nodes: &Vec<usize>,
        available: &Vec<bool>,
        route_demand: i32,
        data: &Problem,
    ) -> Option<(usize, usize)> {
        let mut best_c2 = i32::MIN;
        let mut best = None;
        for &insert_node in nodes.iter() {
            let insert_nd = data.nd(insert_node);

            let skip = !at(available, insert_node)
                | (route_demand + insert_nd.demand > data.max_capacity);
            if skip {
                continue;
            }

            let mut curr_time = 0;
            let mut curr_node = 0;

            for pos in 1..route.len() {
                let next_node = at(route, pos);
                let new_arrival_time_insert_node = insert_nd
                    .start_tw
                    .max(curr_time + data.dm(curr_node, insert_node));
                if new_arrival_time_insert_node > insert_nd.end_tw {
                    break;
                }

                // c11: extra distance from inserting `insert_node` between `curr_node` and
                // `next_node`; c2: that extra distance netted against a dedicated depot trip.
                let c11 = data.dm(curr_node, insert_node) + data.dm(insert_node, next_node)
                    - data.dm(curr_node, next_node);
                let c2 = data.dm(0, insert_node) - c11;

                let c2_is_better = c2 > best_c2;

                if c2_is_better
                    && Self::is_feasible(
                        route,
                        insert_node,
                        new_arrival_time_insert_node + insert_nd.service_time,
                        pos,
                        data,
                    )
                {
                    best_c2 = c2;
                    best = Some((insert_node, pos));
                }

                let next_nd = data.nd(next_node);
                curr_time = next_nd
                    .start_tw
                    .max(curr_time + data.dm(curr_node, next_node))
                    + next_nd.service_time;
                curr_node = next_node;
            }
        }
        best
    }

    fn is_feasible(
        route: &Vec<usize>,
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
        data: &Problem,
    ) -> bool {
        for pos in start_pos..route.len() {
            let next_node = at(route, pos);
            let next_nd = data.nd(next_node);
            curr_time += data.dm(curr_node, next_node);
            if curr_time > next_nd.end_tw {
                return false;
            }
            curr_time = curr_time.max(next_nd.start_tw) + next_nd.service_time;
            curr_node = next_node;
        }
        true
    }
}
