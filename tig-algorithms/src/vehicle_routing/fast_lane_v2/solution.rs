use super::instance::Instance;
use super::config::Config;
use super::route_eval::RouteEval;

#[derive(Clone)]
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
    pub fn new_from_routes(data: &Instance, params: &Config, routes: Vec<Vec<usize>>) -> Self {
        let (distance, tw_violation, load_excess) = Self::evaluate_routes(data, &routes);
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

    pub fn evaluate_routes(data: &Instance, routes: &Vec<Vec<usize>>) -> (i32, i32, i32) {
        let mut dist: i32 = 0;
        let mut tw: i32 = 0;
        let mut loadx: i32 = 0;
        for r in routes {
            if r.is_empty() {
                continue;
            }
            let mut acc = RouteEval::singleton(data, r[0]);
            for idx in 1..r.len() {
                let next = RouteEval::singleton(data, r[idx]);
                acc = RouteEval::join2(data, &acc, &next);
            }
            dist += acc.distance;
            tw += acc.tw;
            let ex = (acc.load - data.max_capacity).max(0);
            loadx += ex;
        }
        (dist, tw, loadx)
    }

    #[inline]
    pub fn compute_penalized_cost(distance: i32, tw_violation: i32, load_excess: i32, params: &Config) -> i64 {
        (distance as i64)
            + (params.penalty_tw as i64) * (tw_violation as i64)
            + (params.penalty_capa as i64) * (load_excess as i64)
    }

    #[inline]
    pub fn recompute_cost(&mut self, params: &Config) {
        self.cost = Self::compute_penalized_cost(self.distance, self.tw_violation, self.load_excess, params);
    }

    fn build_pred_succ_and_count(data: &Instance, routes: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>, usize) {
        let n_all = data.nb_nodes;
        let mut pred = vec![0usize; n_all];
        let mut succ = vec![0usize; n_all];
        let mut nb_routes: usize = 0;

        for r in routes {
            if r.len() > 2 {
                nb_routes += 1;
            }
            if r.len() < 2 {
                continue;
            }
            for p in 1..r.len() - 1 {
                let id = r[p];
                pred[id] = r[p - 1];
                succ[id] = r[p + 1];
            }
        }
        (pred, succ, nb_routes)
    }
}
