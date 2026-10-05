use super::params::Params;
use super::problem::Problem;
use std::cmp::{max, min};

/// Concatenable summary of a node range: earliest and latest feasible arrival
/// at its first node, minimum duration, accumulated time-window violation,
/// then load, distance and the two endpoints.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    pub tau_minus: i32,
    pub tau_plus: i32,
    pub tmin: i32,
    pub tw: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: u16,
    pub last_node: u16,
}

impl Sequence {
    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        debug_assert!(u16::try_from(node).is_ok());
        let nd = data.nd(node);
        let st = nd.start_tw;
        let et = nd.end_tw;
        let svc = nd.service_time;
        let ld = nd.demand;
        self.tau_minus = st;
        self.tau_plus = et;
        self.tmin = svc;
        self.tw = 0;
        self.load = ld;
        self.distance = 0;
        self.first_node = node as u16;
        self.last_node = node as u16;
    }

    /// Concatenation s1 then s2.
    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node as usize, s2.first_node as usize);
        Self::join2_with_travel(s1, s2, travel)
    }

    #[inline(always)]
    pub fn join2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let distance = s1.distance + s2.distance + travel;
        let temp = travel + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw = s1.tw + s2.tw + twij;
        let tmin = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus = min(s2.tau_plus - temp + twij, s1.tau_plus);
        let load = s1.load + s2.load;

        Sequence {
            tau_minus,
            tau_plus,
            tmin,
            tw,
            load,
            distance,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    /// Composes the temporal state and the endpoints only: `load` and
    /// `distance` come out as zero and must not be read.
    #[inline(always)]
    pub fn join_tw_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let temp = travel + s1.tmin - s1.tw;
        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw = s1.tw + s2.tw + twij;
        let tmin = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus = min(s2.tau_plus - temp + twij, s1.tau_plus);

        Sequence {
            tau_minus,
            tau_plus,
            tmin,
            tw,
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

    /// Time-window violation of s1 then s2, caller-supplied connecting arc.
    #[inline(always)]
    pub fn tw2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> i32 {
        let temp = s1.tmin - s1.tw + travel;
        s1.tw + s2.tw + max(s1.tau_minus - s2.tau_plus + temp, 0)
    }

    /// Time-window violation of s1 then s2 then s3, caller-supplied arcs.
    #[inline(always)]
    pub fn tw3_with_travel(
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        travel12: i32,
        travel23: i32,
    ) -> i32 {
        let temp = travel12 + s1.tmin - s1.tw;
        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw12 = s1.tw + s2.tw + twij;
        let tmin12 = temp + s1.tw + s2.tmin + wtij;
        let tau_m12 = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let temp2 = travel23 + tmin12 - tw12;
        tw12 + s3.tw + max(tau_m12 - s3.tau_plus + temp2, 0)
    }

    /// Time-window violation of s0 then s1 then s2 then s3 then s4.
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
        let temp01 = travel01 + s0.tmin - s0.tw;
        let wt01 = max(s1.tau_minus - temp01 - s0.tau_plus, 0);
        let tw01 = max(temp01 + s0.tau_minus - s1.tau_plus, 0);
        let tw_viol01 = s0.tw + s1.tw + tw01;
        let tmin01 = temp01 + s0.tw + s1.tmin + wt01;
        let tau_m01 = max(s1.tau_minus - temp01 - wt01, s0.tau_minus);
        let tau_p01 = min(s1.tau_plus - temp01 + tw01, s0.tau_plus);

        let temp12 = travel12 + tmin01 - tw_viol01;
        let wt12 = max(s2.tau_minus - temp12 - tau_p01, 0);
        let tw12 = max(temp12 + tau_m01 - s2.tau_plus, 0);
        let tw_viol012 = tw_viol01 + s2.tw + tw12;
        let tmin012 = temp12 + tw_viol01 + s2.tmin + wt12;
        let tau_m012 = max(s2.tau_minus - temp12 - wt12, tau_m01);
        let tau_p012 = min(s2.tau_plus - temp12 + tw12, tau_p01);

        let temp23 = travel23 + tmin012 - tw_viol012;
        let wt23 = max(s3.tau_minus - temp23 - tau_p012, 0);
        let tw23 = max(temp23 + tau_m012 - s3.tau_plus, 0);
        let tw_viol0123 = tw_viol012 + s3.tw + tw23;
        let tmin0123 = temp23 + tw_viol012 + s3.tmin + wt23;
        let tau_m0123 = max(s3.tau_minus - temp23 - wt23, tau_m012);

        let temp34 = tmin0123 - tw_viol0123 + travel34;
        tw_viol0123 + s4.tw + max(tau_m0123 - s4.tau_plus + temp34, 0)
    }

    /// Penalized cost of s1 then s2 then s3, without materializing the join.
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
        let distance12 = s1.distance + s2.distance + travel12;
        let temp = travel12 + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw_viol12 = s1.tw + s2.tw + twij;
        let tmin12 = temp + s1.tw + s2.tmin + wtij;
        let tau_m12 = max(s2.tau_minus - temp - wtij, s1.tau_minus);

        let travel23 = data.dm(s2.last_node as usize, s3.first_node as usize);
        let distance = distance12 + s3.distance + travel23;
        let temp2 = travel23 + tmin12 - tw_viol12;

        let tw_viol = tw_viol12 + s3.tw + max(tau_m12 - s3.tau_plus + temp2, 0);
        let load = s1.load + s2.load + s3.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }
}
