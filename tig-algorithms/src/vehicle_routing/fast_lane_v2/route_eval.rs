use super::instance::Instance;
use super::config::Config;
use std::cmp::{max, min};

#[derive(Copy, Clone, Default)]
pub struct RouteEval {
    pub tau_minus: i32,
    pub tau_plus: i32,
    pub tmin: i32,
    pub tw: i32,
    pub total_service_duration: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: usize,
    pub last_node: usize,
}

impl RouteEval {
    #[inline(always)]
    pub fn initialize(&mut self, data: &Instance, node: usize) {
        let st = data.start_tw[node];
        let et = data.end_tw[node];
        let svc = data.service_times[node];
        let ld = data.demands[node];
        self.tau_minus = st;
        self.tau_plus = et;
        self.tmin = svc;
        self.tw = 0;
        self.total_service_duration = svc;
        self.load = ld;
        self.distance = 0;
        self.first_node = node;
        self.last_node = node;
    }

    #[inline(always)]
    pub fn join2(data: &Instance, s1: &RouteEval, s2: &RouteEval) -> RouteEval {
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = travel + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw = s1.tw + s2.tw + twij;
        let tmin = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus = min(s2.tau_plus - temp + twij, s1.tau_plus);
        let load = s1.load + s2.load;

        RouteEval {
            tau_minus,
            tau_plus,
            tmin,
            tw,
            total_service_duration: s1.total_service_duration + s2.total_service_duration,
            load,
            distance,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    #[inline(always)]
    pub fn singleton(data: &Instance, node: usize) -> RouteEval {
        let mut s = RouteEval::default();
        s.initialize(data, node);
        s
    }

    #[inline(always)]
    pub fn eval(&self, data: &Instance, params: &Config) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let load_excess = (self.load - data.max_capacity).max(0) as i64;
        (self.distance as i64) + load_excess * pcap + (self.tw as i64) * ptw
    }

    #[inline(always)]
    pub fn eval2(data: &Instance, params: &Config, s1: &RouteEval, s2: &RouteEval) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = s1.tmin - s1.tw + travel;
        let tw_viol = s1.tw + s2.tw + max(s1.tau_minus - s2.tau_plus + temp, 0);
        let load = s1.load + s2.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[inline(always)]
    pub fn eval3(data: &Instance, params: &Config, s1: &RouteEval, s2: &RouteEval, s3: &RouteEval) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;

        let travel12 = data.dm(s1.last_node, s2.first_node);
        let distance12 = s1.distance + s2.distance + travel12;
        let temp = travel12 + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw_viol12 = s1.tw + s2.tw + twij;
        let tmin12 = temp + s1.tw + s2.tmin + wtij;
        let tau_m12 = max(s2.tau_minus - temp - wtij, s1.tau_minus);

        let travel23 = data.dm(s2.last_node, s3.first_node);
        let distance = distance12 + s3.distance + travel23;
        let temp2 = travel23 + tmin12 - tw_viol12;

        let tw_viol = tw_viol12 + s3.tw + max(tau_m12 - s3.tau_plus + temp2, 0);
        let load = s1.load + s2.load + s3.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[inline(always)]
    pub fn eval_n(data: &Instance, params: &Config, chain: &[RouteEval]) -> i64 {
        let mut agg = chain[0];
        for s in &chain[1..chain.len() - 1] {
            agg = RouteEval::join2(data, &agg, s);
        }
        let last = &chain[chain.len() - 1];
        RouteEval::eval2(data, params, &agg, last)
    }
}
