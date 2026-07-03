use super::problem::Problem;
use super::params::Params;
use std::cmp::{max, min};

#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    /// earliest arrival
    pub tau_minus: i32,
    /// latest arrival
    pub tau_plus: i32,
    /// min travel time
    pub tmin: i32,
    /// time window violation
    pub tw: i32,
    /// total load
    pub load: i32,
    /// total distance
    pub distance: i32,
    /// first node in sequence
    pub first_node: usize,
    /// last node in sequence
    pub last_node: usize,
}

impl Sequence {

    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        let nd = data.nd(node);
        let st = nd.start_tw;
        let et = nd.end_tw;
        let svc = nd.service_time;
        let ld = nd.demand;
        self.tau_minus = st;
        self.tau_plus  = et;
        self.tmin      = svc;
        self.tw        = 0;
        self.load      = ld;
        self.distance  = 0;
        self.first_node = node;
        self.last_node  = node;
    }

    // Compose (concatenate) two sequences s1 ∘ s2
    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = travel + s1.tmin - s1.tw;

        let wtij      = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij      = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw        = s1.tw + s2.tw + twij;
        let tmin      = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus  = min(s2.tau_plus  - temp + twij, s1.tau_plus);
        let load      = s1.load + s2.load;

        Sequence {
            tau_minus, tau_plus, tmin, tw,
            load, distance,
            first_node: s1.first_node,
            last_node:  s2.last_node,
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
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;
        let load_excess = (self.load - data.max_capacity).max(0) as i64;
        (self.distance as i64) + load_excess * pcap + (self.tw as i64) * ptw
    }

    // s1 ∘ s2 evaluated directly (no materialized join)
    #[inline(always)]
    pub fn eval2(data: &Problem, params: &Params, s1: &Sequence, s2: &Sequence) -> i64 {
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;
        let travel   = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp     = s1.tmin - s1.tw + travel;
        let tw_viol  = s1.tw + s2.tw + max(s1.tau_minus - s2.tau_plus + temp, 0);
        let load     = s1.load + s2.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    // s1 ∘ s2 ∘ s3 evaluated directly (kept for convenience)
    #[inline(always)]
    pub fn eval3(data: &Problem, params: &Params, s1: &Sequence, s2: &Sequence, s3: &Sequence) -> i64 {
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;

        let travel12   = data.dm(s1.last_node, s2.first_node);
        let distance12 = s1.distance + s2.distance + travel12;
        let temp   = travel12 + s1.tmin - s1.tw;

        let wtij       = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij       = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw_viol12  = s1.tw + s2.tw + twij;
        let tmin12     = temp + s1.tw + s2.tmin + wtij;
        let tau_m12    = max(s2.tau_minus - temp - wtij, s1.tau_minus);

        let travel23   = data.dm(s2.last_node, s3.first_node);
        let distance   = distance12 + s3.distance + travel23;
        let temp2      = travel23 + tmin12 - tw_viol12;

        let tw_viol    = tw_viol12 + s3.tw + max(tau_m12 - s3.tau_plus + temp2, 0);
        let load       = s1.load + s2.load + s3.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn eval4(data: &Problem, params: &Params, s0: &Sequence, s1: &Sequence, s2: &Sequence, s3: &Sequence) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;

        // Join s0 and s1
        let travel01 = data.dm(s0.last_node, s1.first_node);
        let distance01 = s0.distance + s1.distance + travel01;
        let temp01 = travel01 + s0.tmin - s0.tw;
        let wt01 = max(s1.tau_minus - temp01 - s0.tau_plus, 0);
        let tw01 = max(temp01 + s0.tau_minus - s1.tau_plus, 0);
        let tw_viol01 = s0.tw + s1.tw + tw01;
        let tmin01 = temp01 + s0.tw + s1.tmin + wt01;
        let tau_m01 = max(s1.tau_minus - temp01 - wt01, s0.tau_minus);
        let tau_p01 = min(s1.tau_plus - temp01 + tw01, s0.tau_plus);

        // Join (s0,s1) with s2
        let travel12 = data.dm(s1.last_node, s2.first_node);
        let distance012 = distance01 + s2.distance + travel12;
        let temp12 = travel12 + tmin01 - tw_viol01;
        let wt12 = max(s2.tau_minus - temp12 - tau_p01, 0);
        let tw12 = max(temp12 + tau_m01 - s2.tau_plus, 0);
        let tw_viol012 = tw_viol01 + s2.tw + tw12;
        let tmin012 = temp12 + tw_viol01 + s2.tmin + wt12;
        let tau_m012 = max(s2.tau_minus - temp12 - wt12, tau_m01);

        // Evaluate (s0,s1,s2) followed by s3 (eval2-style)
        let travel23 = data.dm(s2.last_node, s3.first_node);
        let distance = distance012 + s3.distance + travel23;
        let temp23 = tmin012 - tw_viol012 + travel23;
        let tw_viol = tw_viol012 + s3.tw + max(tau_m012 - s3.tau_plus + temp23, 0);
        let load = s0.load + s1.load + s2.load + s3.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[inline(always)]
    pub fn eval5(data: &Problem, params: &Params, s0: &Sequence, s1: &Sequence, s2: &Sequence, s3: &Sequence, s4: &Sequence) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;

        // Join s0 and s1
        let travel01 = data.dm(s0.last_node, s1.first_node);
        let distance01 = s0.distance + s1.distance + travel01;
        let temp01 = travel01 + s0.tmin - s0.tw;
        let wt01 = max(s1.tau_minus - temp01 - s0.tau_plus, 0);
        let tw01 = max(temp01 + s0.tau_minus - s1.tau_plus, 0);
        let tw_viol01 = s0.tw + s1.tw + tw01;
        let tmin01 = temp01 + s0.tw + s1.tmin + wt01;
        let tau_m01 = max(s1.tau_minus - temp01 - wt01, s0.tau_minus);
        let tau_p01 = min(s1.tau_plus - temp01 + tw01, s0.tau_plus);

        // Join (s0,s1) with s2
        let travel12 = data.dm(s1.last_node, s2.first_node);
        let distance012 = distance01 + s2.distance + travel12;
        let temp12 = travel12 + tmin01 - tw_viol01;
        let wt12 = max(s2.tau_minus - temp12 - tau_p01, 0);
        let tw12 = max(temp12 + tau_m01 - s2.tau_plus, 0);
        let tw_viol012 = tw_viol01 + s2.tw + tw12;
        let tmin012 = temp12 + tw_viol01 + s2.tmin + wt12;
        let tau_m012 = max(s2.tau_minus - temp12 - wt12, tau_m01);
        let tau_p012 = min(s2.tau_plus - temp12 + tw12, tau_p01);

        // Join (s0,s1,s2) with s3
        let travel23 = data.dm(s2.last_node, s3.first_node);
        let distance0123 = distance012 + s3.distance + travel23;
        let temp23 = travel23 + tmin012 - tw_viol012;
        let wt23 = max(s3.tau_minus - temp23 - tau_p012, 0);
        let tw23 = max(temp23 + tau_m012 - s3.tau_plus, 0);
        let tw_viol0123 = tw_viol012 + s3.tw + tw23;
        let tmin0123 = temp23 + tw_viol012 + s3.tmin + wt23;
        let tau_m0123 = max(s3.tau_minus - temp23 - wt23, tau_m012);

        // Evaluate (s0,s1,s2,s3) followed by s4 (eval2-style)
        let travel34 = data.dm(s3.last_node, s4.first_node);
        let distance = distance0123 + s4.distance + travel34;
        let temp34 = tmin0123 - tw_viol0123 + travel34;
        let tw_viol = tw_viol0123 + s4.tw + max(tau_m0123 - s4.tau_plus + temp34, 0);
        let load = s0.load + s1.load + s2.load + s3.load + s4.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }
}
