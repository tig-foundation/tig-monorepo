use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;

#[derive(Clone, Debug)]
pub struct Individual {
    pub(crate) packed_adjacency:Vec<u32>,
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

    /// Construct an individual from route metrics already maintained by local
    /// search, avoiding a second full sequence evaluation of every route.
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
        // Every ID is below nb_nodes. Packed equality is exactly the pair
        // equality used by population diversity, including reversed pairs.
        #[cfg(target_arch="x86_64")]
        let packed_enabled=data.nb_nodes<=65536 && std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch="x86_64"))]
        let packed_enabled=false;
        let packed_adjacency=if packed_enabled {
            pred.iter().zip(succ.iter()).map(|(&p,&s)|p as u32|((s as u32)<<16)).collect()
        }else{Vec::new()};

        Self {
            packed_adjacency,
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
        let mut dist: i32 = 0;
        let mut tw: i32 = 0;
        let mut loadx: i32 = 0;
        for r in routes {
            if r.is_empty() {
                continue;
            }
            let mut acc = Sequence::singleton(data, r[0]);
            for idx in 1..r.len() {
                let next = Sequence::singleton(data, r[idx]);
                acc = Sequence::join2(data, &acc, &next);
            }
            dist += acc.distance;
            tw += acc.tw;
            let ex = (acc.load - data.max_capacity).max(0);
            loadx += ex;
        }
        (dist, tw, loadx)
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

    /// Build predecessor/successor arrays and count non-empty routes.
    fn build_pred_succ_and_count(
        data: &Problem,
        routes: &Vec<Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let n_all = data.nb_nodes;
        let mut pred = vec![0usize; n_all];
        let mut succ = vec![0usize; n_all];
        let mut seen = vec![false; n_all];
        let mut nb_routes: usize = 0;

        for r in routes {
            // A non-empty route contains at least one client: [0, c1, ..., 0] has len >= 3
            if r.len() > 2 {
                nb_routes += 1;
            }
            if r.len() < 2 {
                continue;
            } // defensive
            for adjacent in r.windows(3) {
                let id = adjacent[1];
                debug_assert!(
                    id < n_all && !seen[id],
                    "Client {} appears more than once in routes",
                    id
                );
                unsafe {
                    *seen.get_unchecked_mut(id) = true;
                    *pred.get_unchecked_mut(id) = adjacent[0];
                    *succ.get_unchecked_mut(id) = adjacent[2];
                }
            }
        }
        for id in 1..n_all {
            debug_assert!(seen[id], "Client {} is missing from routes", id);
        }
        (pred, succ, nb_routes)
    }
}
