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
        let (pred, succ, nb_routes) = Self::build_pred_succ_and_count(data, &routes);
        Self {
            routes,
            nb_routes,
            distance,
            tw_violation,
            load_excess,
            cost,
            pred,
            succ,
        }
    }

    pub fn evaluate_routes(data: &Problem, routes: &Vec<Vec<usize>>) -> (i32, i32, i32) {
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
        routes: &Vec<Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let nb_nodes = data.nb_nodes;
        let mut pred = vec![0usize; nb_nodes];
        let mut succ = vec![0usize; nb_nodes];
        let mut seen = vec![false; nb_nodes];
        let mut nb_routes: usize = 0;

        for route in routes {
            // Route layout is [0, c1, ..., ck, 0], so a route serving a client has len >= 3.
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
                }
            }
        }
        for client in 1..nb_nodes {
            debug_assert!(seen[client], "Client {} is missing from routes", client);
        }
        (pred, succ, nb_routes)
    }
}
