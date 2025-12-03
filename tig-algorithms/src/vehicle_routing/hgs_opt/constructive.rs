use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};

pub struct Constructive;

impl Constructive {
    pub fn build_routes(data: &Problem, rng: &mut SmallRng, randomize: bool) -> Vec<Vec<usize>> {
        let mut routes = Vec::with_capacity(data.nb_vehicles);
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        let n = nodes.len();
        nodes.sort_by(|&a, &b| data.dm(0,a).cmp(&data.dm(0,b)));

        if randomize {
            for i in 0..(n - 1) { nodes.swap(i, rng.gen_range(i + 1..=(i + 5).min(n - 1))); }
        }

        // Availability bitmap: true = not yet routed
        let mut available = vec![true; data.nb_nodes];
        available[0] = false; // depot

        while let Some(node) = nodes.pop() {
            if !available[node] { continue; }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.demands[node];

            while let Some((best_node, best_pos)) =
                Self::find_best_insertion(&route, &nodes, &available, route_demand,data)
            {
                available[best_node] = false;
                route_demand += data.demands[best_node];
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
    ) -> Option<(usize, usize)> {
        let mut best_c2 = None;
        let mut best = None;
        for &insert_node in nodes.iter() {

            if !available[insert_node] || route_demand + data.demands[insert_node] > data.max_capacity {
                continue;
            }

            let mut curr_time = 0;
            let mut curr_node = 0;

            for pos in 1..route.len() {
                let next_node = route[pos];
                let new_arrival_time_insert_node = data.start_tw[insert_node]
                    .max(curr_time + data.dm(curr_node,insert_node));
                if new_arrival_time_insert_node > data.end_tw[insert_node] {
                    break;
                }

                // Extra distance caused by insertion
                let c11 = data.dm(curr_node,insert_node)
                    + data.dm(insert_node,next_node)
                    - data.dm(curr_node,next_node);

                // Gain of distance compared to a direct trip
                let c2 = data.dm(0,insert_node) - c11;

                let c2_is_better = match best_c2 {
                    None => true,
                    Some(x) => c2 > x,
                };

                if c2_is_better
                    && Self::is_feasible(
                        route,
                        insert_node,
                        new_arrival_time_insert_node + data.service_times[insert_node],
                        pos,
                        data,
                    )
                {
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                curr_time = data.start_tw[next_node]
                    .max(curr_time + data.dm(curr_node,next_node))
                    + data.service_times[next_node];
                curr_node = next_node;
            }
        }
        best
    }

    fn is_feasible(
        route: &[usize],
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
        data: &Problem,
    ) -> bool {
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            curr_time += data.dm(curr_node,next_node);
            if curr_time > data.end_tw[route[pos]] {
                return false;
            }
            curr_time = curr_time.max(data.start_tw[next_node]) + data.service_times[next_node];
            curr_node = next_node;
        }
        true
    }
}
