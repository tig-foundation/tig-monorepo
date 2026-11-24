use super::params::Params;
use super::problem::Problem;
use std::cmp::{max, min};

/// Temporal summary, load, distance and endpoints of a node sequence.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    pub earliest_end: i32,
    pub tau_plus: i32,
    pub duration_net: i32,
    pub tw: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: u16,
    pub last_node: u16,
}

impl Sequence {
    #[inline(always)]
    pub fn earliest_start(&self) -> i32 {
        self.earliest_end - self.duration_net
    }

    #[inline(always)]
    pub fn duration(&self) -> i32 {
        self.duration_net + self.tw
    }

    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        debug_assert!(u16::try_from(node).is_ok());
        let nd = data.nd(node);
        self.earliest_end = nd.start_tw + nd.service_time;
        self.tau_plus = nd.end_tw;
        self.duration_net = nd.service_time;
        self.tw = 0;
        self.load = nd.demand;
        self.distance = 0;
        self.first_node = node as u16;
        self.last_node = node as u16;
    }

    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node as usize, s2.first_node as usize);
        Self::join2_with_travel(s1, s2, travel)
    }

    #[inline(always)]
    pub fn join2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let mut out = Self::join_tw_with_travel(s1, s2, travel);
        out.distance = s1.distance + s2.distance + travel;
        out.load = s1.load + s2.load;
        out
    }

    /// Returns zero load and distance; only temporal fields and endpoints are valid.
    #[inline(always)]
    pub fn join_tw_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let arrival = s1.earliest_end + travel;
        let extra = max(arrival - s2.tau_plus, 0);
        let shifted = s1.duration_net + travel;
        Sequence {
            earliest_end: max(s2.earliest_end, arrival + s2.duration_net) - extra,
            tau_plus: min(s2.tau_plus - shifted + extra, s1.tau_plus),
            duration_net: max(shifted + s2.duration_net, s2.earliest_end - s1.tau_plus) - extra,
            tw: s1.tw + s2.tw + extra,
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

    #[inline(always)]
    pub fn tw2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> i32 {
        s1.tw + s2.tw + max(s1.earliest_end + travel - s2.tau_plus, 0)
    }

    #[inline(always)]
    pub fn tw3_with_travel(
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        travel12: i32,
        travel23: i32,
    ) -> i32 {
        let arrival = s1.earliest_end + travel12;
        let extra12 = max(arrival - s2.tau_plus, 0);
        let end12 = max(s2.earliest_end, arrival + s2.duration_net);
        s1.tw + s2.tw + s3.tw + max(extra12, end12 + travel23 - s3.tau_plus)
    }

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
        let arrival1 = s0.earliest_end + travel01;
        let extra1 = max(arrival1 - s1.tau_plus, 0);
        let end1 = max(s1.earliest_end, arrival1 + s1.duration_net) - extra1;

        let arrival2 = end1 + travel12;
        let extra2 = max(arrival2 - s2.tau_plus, 0);
        let end2 = max(s2.earliest_end, arrival2 + s2.duration_net) - extra2;

        let arrival3 = end2 + travel23;
        let extra3 = max(arrival3 - s3.tau_plus, 0);
        let end3 = max(s3.earliest_end, arrival3 + s3.duration_net) - extra3;

        s0.tw + s1.tw + s2.tw + s3.tw + s4.tw
            + extra1
            + extra2
            + extra3
            + max(end3 + travel34 - s4.tau_plus, 0)
    }

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
        let travel23 = data.dm(s2.last_node as usize, s3.first_node as usize);
        let distance = s1.distance + s2.distance + s3.distance + travel12 + travel23;
        let tw_viol = Self::tw3_with_travel(s1, s2, s3, travel12, travel23);
        let load = s1.load + s2.load + s3.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }
}
