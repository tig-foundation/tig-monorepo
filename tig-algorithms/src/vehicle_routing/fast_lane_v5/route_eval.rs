use super::instance::Instance;
use super::config::Config;
use std::cmp::{max, min};

#[derive(Copy, Clone, Default)]
pub struct RouteEval {
    pub tau_minus: i32,
    pub tau_plus: i32,
    pub tmin: i32,
    pub tw: i32,
    pub end_min: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: usize,
    pub last_node: usize,
}

#[inline(always)]
fn score_route(distance: i32, load: i32, tw: i32, data: &Instance, params: &Config) -> i64 {
    let load_excess = (load - data.max_capacity).max(0) as i64;
    (distance as i64)
        + load_excess * (params.penalty_capa as i64)
        + (tw as i64) * (params.penalty_tw as i64)
}

#[inline(always)]
fn compose_tw(s1: &RouteEval, s2: &RouteEval, travel: i32) -> (i32, i32, i32, i32, i32) {
    let temp = travel + s1.tmin - s1.tw;
    let wt = max(s2.tau_minus - temp - s1.tau_plus, 0);
    let tw_add = max(temp + s1.tau_minus - s2.tau_plus, 0);
    let tau_minus = max(s2.tau_minus - temp - wt, s1.tau_minus);
    let tau_plus = min(s2.tau_plus - temp + tw_add, s1.tau_plus);
    let tmin = temp + s1.tw + s2.tmin + wt;
    let tw = s1.tw + s2.tw + tw_add;
    let base = tmin - tw;
    (tau_minus, tau_plus, tmin, tw, base + tau_minus)
}

#[inline(always)]
fn advance_bounds(end_min: i32, seg: &RouteEval, travel: i32) -> (i32, i32) {
    let arrival_min = end_min + travel;
    let shift_min = seg.end_min - seg.tau_minus;
    let next_min = min(max(arrival_min, seg.tau_minus), seg.tau_plus) + shift_min;
    let tw_add = seg.tw + max(arrival_min - seg.tau_plus, 0);
    (next_min, tw_add)
}

macro_rules! eval_chain_fn {
    (
        $name:ident,
        ($first:ident $(, $rest:ident)+),
        [$(($travel_mid:ident, $from_mid:ident, $to_mid:ident, $mid:ident)),*],
        ($travel_last:ident, $from_last:ident, $to_last:ident, $last:ident)
    ) => {
        #[inline(always)]
        pub fn $name(data: &Instance, params: &Config, $first: &RouteEval, $($rest: &RouteEval),+) -> i64 {
            $(let $travel_mid = data.dm($from_mid.last_node, $to_mid.first_node);)*
            let $travel_last = data.dm($from_last.last_node, $to_last.first_node);
            let distance = $first.distance $(+ $rest.distance)+ $(+ $travel_mid)* + $travel_last;
            let load = $first.load $(+ $rest.load)+;

            let mut tw = $first.tw;
            let mut end_min = $first.end_min;

            $(
                let (next_min, tw_add) = advance_bounds(end_min, $mid, $travel_mid);
                tw += tw_add;
                end_min = next_min;
            )*

            let tw_viol = tw + $last.tw + max(end_min + $travel_last - $last.tau_plus, 0);
            score_route(distance, load, tw_viol, data, params)
        }
    };
}

impl RouteEval {
    #[inline(always)]
    pub fn join2(data: &Instance, s1: &RouteEval, s2: &RouteEval) -> RouteEval {
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let load = s1.load + s2.load;
        let (tau_minus, tau_plus, tmin, tw, end_min) = compose_tw(s1, s2, travel);

        RouteEval {
            tau_minus,
            tau_plus,
            tmin,
            tw,
            end_min,
            load,
            distance,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    #[inline(always)]
    pub fn singleton(data: &Instance, node: usize) -> RouteEval {
        let nd = data.nd(node);
        RouteEval {
            tau_minus: nd.start_tw,
            tau_plus: nd.end_tw,
            tmin: nd.service_time,
            tw: 0,
            end_min: nd.service_time + nd.start_tw,
            load: nd.demand,
            distance: 0,
            first_node: node,
            last_node: node,
        }
    }

    #[inline(always)]
    pub fn eval(&self, data: &Instance, params: &Config) -> i64 {
        score_route(self.distance, self.load, self.tw, data, params)
    }

    #[inline(always)]
    pub fn eval2(data: &Instance, params: &Config, s1: &RouteEval, s2: &RouteEval) -> i64 {
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let load = s1.load + s2.load;
        let tw_viol = s1.tw + s2.tw + max(s1.end_min + travel - s2.tau_plus, 0);
        score_route(distance, load, tw_viol, data, params)
    }

    eval_chain_fn!(
        eval3,
        (s1, s2, s3),
        [(travel12, s1, s2, s2)],
        (travel23, s2, s3, s3)
    );

    eval_chain_fn!(
        eval4,
        (s0, s1, s2, s3),
        [(travel01, s0, s1, s1), (travel12, s1, s2, s2)],
        (travel23, s2, s3, s3)
    );

    eval_chain_fn!(
        eval5,
        (s0, s1, s2, s3, s4),
        [(travel01, s0, s1, s1), (travel12, s1, s2, s2), (travel23, s2, s3, s3)],
        (travel34, s3, s4, s4)
    );
}