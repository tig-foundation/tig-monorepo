use super::params::Params;
use super::problem::Problem;
use std::cmp::{max, min};

/// Exact reparameterization of the Vidal sequence summary.
/// E = tau_minus + tmin - tw; D = tmin - tw; L = tau_plus.
/// The original fields are recovered by tau_minus = E-D, tmin = D+tw.
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
    pub fn earliest_start(&self) -> i32 { self.earliest_end - self.duration_net }
    #[inline(always)]
    pub fn duration(&self) -> i32 { self.duration_net + self.tw }
    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        debug_assert!(u16::try_from(node).is_ok());
        let nd = data.nd(node);
        *self = Self {
            earliest_end: nd.start_tw + nd.service_time,
            tau_plus: nd.end_tw,
            duration_net: nd.service_time,
            tw: 0,
            load: nd.demand,
            distance: 0,
            first_node: node as u16,
            last_node: node as u16,
        };
    }
    #[inline(always)]
    pub fn singleton(data: &Problem, node: usize) -> Self {
        let mut s=Self::default(); s.initialize(data,node); s
    }
    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Self, s2: &Self) -> Self {
        Self::join2_with_travel(s1,s2,data.dm(s1.last_node as usize,s2.first_node as usize))
    }
    #[inline(always)]
    pub fn join2_with_travel(s1: &Self, s2: &Self, travel: i32) -> Self {
        let mut out=Self::join_tw_with_travel(s1,s2,travel);
        out.distance=s1.distance+s2.distance+travel;
        out.load=s1.load+s2.load;
        out
    }
    #[inline(always)]
    pub fn join_tw(data: &Problem, s1: &Self, s2: &Self) -> Self {
        Self::join_tw_with_travel(s1,s2,data.dm(s1.last_node as usize,s2.first_node as usize))
    }
    #[inline(always)]
    pub fn join_tw_with_travel(s1: &Self, s2: &Self, travel: i32) -> Self {
        let arrival=s1.earliest_end+travel;
        let extra=max(arrival-s2.tau_plus,0);
        let shifted=s1.duration_net+travel;
        Self {
            earliest_end: max(s2.earliest_end,arrival+s2.duration_net)-extra,
            duration_net: max(shifted+s2.duration_net,s2.earliest_end-s1.tau_plus)-extra,
            tau_plus: min(s2.tau_plus-shifted+extra,s1.tau_plus),
            tw: s1.tw+s2.tw+extra,
            load: 0, distance: 0,
            first_node: s1.first_node, last_node: s2.last_node,
        }
    }
    #[inline(always)]
    pub fn eval(&self, data: &Problem, params: &Params) -> i64 {
        self.distance as i64 + (self.load-data.max_capacity).max(0) as i64 * params.penalty_capa as i64
            + self.tw as i64 * params.penalty_tw as i64
    }
    #[inline(always)]
    pub fn tw2_with_travel(s1: &Self, s2: &Self, travel: i32) -> i32 {
        s1.tw+s2.tw+max(s1.earliest_end+travel-s2.tau_plus,0)
    }
    #[inline(always)]
    pub fn tw3_with_travel(s1: &Self, s2: &Self, s3: &Self, t12: i32, t23: i32) -> i32 {
        let arrival=s1.earliest_end+t12;
        let first_extra=max(arrival-s2.tau_plus,0);
        let last_extra=max(s2.earliest_end,arrival+s2.duration_net)+t23-s3.tau_plus;
        s1.tw+s2.tw+s3.tw+max(first_extra,last_extra)
    }
    #[inline(always)]
    pub fn tw5_with_travel(s0: &Self, s1: &Self, s2: &Self, s3: &Self, s4: &Self,
                          t01: i32, t12: i32, t23: i32, t34: i32) -> i32 {
        let mut end=s0.earliest_end;
        let mut warp=s0.tw;
        macro_rules! step {($s:expr,$t:expr)=>{{
            let arrival=end+$t;
            let extra=max(arrival-$s.tau_plus,0);
            warp += $s.tw+extra;
            end=max($s.earliest_end,arrival+$s.duration_net)-extra;
        }}}
        step!(s1,t01);step!(s2,t12);step!(s3,t23);
        warp+s4.tw+max(end+t34-s4.tau_plus,0)
    }
    #[inline(always)]
    pub fn eval2(data: &Problem, params: &Params, s1: &Self, s2: &Self) -> i64 {
        let travel=data.dm(s1.last_node as usize,s2.first_node as usize);
        let distance=s1.distance+s2.distance+travel;
        let load=s1.load+s2.load;
        distance as i64 + (load-data.max_capacity).max(0) as i64 * params.penalty_capa as i64
            + Self::tw2_with_travel(s1,s2,travel) as i64 * params.penalty_tw as i64
    }
    #[inline(always)]
    pub fn eval3(data: &Problem, params: &Params, s1: &Self, s2: &Self, s3: &Self) -> i64 {
        let t12=data.dm(s1.last_node as usize,s2.first_node as usize);
        let t23=data.dm(s2.last_node as usize,s3.first_node as usize);
        let distance=s1.distance+s2.distance+t12+s3.distance+t23;
        let load=s1.load+s2.load+s3.load;
        distance as i64 + (load-data.max_capacity).max(0) as i64 * params.penalty_capa as i64
            + Self::tw3_with_travel(s1,s2,s3,t12,t23) as i64 * params.penalty_tw as i64
    }
    #[inline(always)]
    pub fn eval4(data: &Problem, params: &Params, s0: &Self, s1: &Self, s2: &Self, s3: &Self) -> i64 {
        Self::eval3(data,params,&Self::join2(data,s0,s1),s2,s3)
    }
    #[inline(always)]
    pub fn eval5(data: &Problem, params: &Params, s0: &Self, s1: &Self, s2: &Self, s3: &Self, s4: &Self) -> i64 {
        let t01=data.dm(s0.last_node as usize,s1.first_node as usize);
        let t12=data.dm(s1.last_node as usize,s2.first_node as usize);
        let t23=data.dm(s2.last_node as usize,s3.first_node as usize);
        let t34=data.dm(s3.last_node as usize,s4.first_node as usize);
        let distance=s0.distance+s1.distance+t01+s2.distance+t12+s3.distance+t23+s4.distance+t34;
        let load=s0.load+s1.load+s2.load+s3.load+s4.load;
        distance as i64 + (load-data.max_capacity).max(0) as i64 * params.penalty_capa as i64
            + Self::tw5_with_travel(s0,s1,s2,s3,s4,t01,t12,t23,t34) as i64 * params.penalty_tw as i64
    }
}
