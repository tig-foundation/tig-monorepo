use super::instance::Instance;
use rand::rngs::SmallRng;
use rand::Rng;

pub struct Builder;

impl Builder {
    pub fn build_routes(data: &Instance, rng: &mut SmallRng, randomize: bool) -> Vec<Vec<usize>> {
        let mut routes = Vec::new();
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        let n = nodes.len();
        nodes.sort_by(|&a, &b| data.dm(0, a).cmp(&data.dm(0, b)));

        if randomize {
            let window = if n < 1000 { 10 } else { 5 };
            for i in 0..(n - 1) {
                nodes.swap(i, rng.gen_range(i + 1..=(i + window).min(n - 1)));
            }
        }

        let mut available = vec![true; data.nb_nodes];
        available[0] = false;

        while let Some(node) = nodes.pop() {
            if !available[node] { continue; }
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
        route: &Vec<usize>, nodes: &Vec<usize>, available: &Vec<bool>,
        route_demand: i32, data: &Instance,
    ) -> Option<(usize, usize)> {
        let mut best_c2 = None;
        let mut best = None;
        for &insert_node in nodes.iter() {
            let nd_ins = data.nd(insert_node);
            if !available[insert_node] || route_demand + nd_ins.demand > data.max_capacity {
                continue;
            }

            let mut curr_time = 0;
            let mut curr_node = 0;

            for pos in 1..route.len() {
                let next_node = route[pos];
                let arrival = nd_ins.start_tw.max(curr_time + data.dm(curr_node, insert_node));
                if arrival > nd_ins.end_tw { break; }

                let c11 = data.dm(curr_node, insert_node) + data.dm(insert_node, next_node) - data.dm(curr_node, next_node);
                let c2 = data.dm(0, insert_node) - c11;

                if match best_c2 { None => true, Some(x) => c2 > x }
                    && Self::is_feasible(route, insert_node, arrival + nd_ins.service_time, pos, data)
                {
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                let nd_next = data.nd(next_node);
                curr_time = nd_next.start_tw.max(curr_time + data.dm(curr_node, next_node)) + nd_next.service_time;
                curr_node = next_node;
            }
        }
        best
    }

    fn is_feasible(route: &Vec<usize>, mut curr_node: usize, mut curr_time: i32, start_pos: usize, data: &Instance) -> bool {
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            let nd = data.nd(next_node);
            curr_time += data.dm(curr_node, next_node);
            if curr_time > nd.end_tw { return false; }
            curr_time = curr_time.max(nd.start_tw) + nd.service_time;
            curr_node = next_node;
        }
        true
    }
}
