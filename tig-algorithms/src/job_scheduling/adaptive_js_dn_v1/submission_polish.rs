//! Deterministic, bounded critical-block polish. The incumbent is immutable
//! unless the official validator confirms a strictly smaller makespan.
use tig_challenges::job_scheduling::{Challenge, Solution};

const NONE: usize = usize::MAX;

struct Graph {
    offsets: Vec<usize>,
    duration: Vec<u32>,
    machine: Vec<usize>,
    job: Vec<usize>,
    jp: Vec<usize>,
    jn: Vec<usize>,
    mp: Vec<usize>,
    mn: Vec<usize>,
}

struct Scratch {
    head: Vec<u32>,
    tail: Vec<u32>,
    degree: Vec<u8>,
    order: Vec<usize>,
}

#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
struct MachineMove {
    rank: usize,
    estimate: u32,
    node: usize,
    machine: usize,
    duration: u32,
    previous: usize,
    next: usize,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self { head: vec![0;n], tail: vec![0;n], degree: vec![0;n], order: Vec::with_capacity(n) }
    }
}

impl Graph {
    fn build(c: &Challenge, s: &Solution) -> Option<Self> {
        let mut offsets = Vec::with_capacity(c.num_jobs + 1);
        let mut duration = Vec::new();
        let mut machine = Vec::new();
        let mut job = Vec::new();
        let mut jp = Vec::new();
        let mut jn = Vec::new();
        let mut machines = vec![Vec::new(); c.num_machines];
        let mut j = 0;
        for (product, &count) in c.jobs_per_product.iter().enumerate() {
            let processing = &c.product_processing_times[product];
            for _ in 0..count {
                offsets.push(duration.len());
                let row = s.job_schedule.get(j)?;
                if row.len() != processing.len() { return None; }
                for (k, &(m, start)) in row.iter().enumerate() {
                    let id = duration.len();
                    duration.push(*processing[k].get(&m)?);
                    machine.push(m);
                    job.push(j);
                    jp.push(if k == 0 { NONE } else { id - 1 });
                    jn.push(if k + 1 == row.len() { NONE } else { id + 1 });
                    machines.get_mut(m)?.push((start,id));
                }
                j += 1;
            }
        }
        offsets.push(duration.len());
        if j != c.num_jobs || j != s.job_schedule.len() { return None; }
        let mut mp = vec![NONE; duration.len()];
        let mut mn = mp.clone();
        for row in &mut machines {
            row.sort_unstable();
            for pair in row.windows(2) {
                mn[pair[0].1] = pair[1].1;
                mp[pair[1].1] = pair[0].1;
            }
        }
        Some(Self { offsets,duration,machine,job,jp,jn,mp,mn })
    }

    fn evaluate(&self, w: &mut Scratch, cutoff: u32) -> Option<u32> {
        w.head.fill(0);
        w.order.clear();
        for v in 0..self.duration.len() {
            let d = u8::from(self.jp[v] != NONE) + u8::from(self.mp[v] != NONE);
            w.degree[v] = d;
            if d == 0 { w.order.push(v); }
        }
        let mut cursor = 0;
        let mut makespan = 0;
        while cursor < w.order.len() {
            let v = w.order[cursor];
            cursor += 1;
            let finish = w.head[v].checked_add(self.duration[v])?;
            // Exact lower bound: unfinished nonnegative operations cannot undo
            // a completion time already at or above the incumbent.
            if finish >= cutoff { return None; }
            makespan = makespan.max(finish);
            for next in [self.jn[v], self.mn[v]] {
                if next != NONE {
                    w.head[next] = w.head[next].max(finish);
                    w.degree[next] -= 1;
                    if w.degree[next] == 0 { w.order.push(next); }
                }
            }
        }
        if w.order.len() != self.duration.len() { return None; }
        for &v in w.order.iter().rev() {
            let mut after = 0;
            for next in [self.jn[v],self.mn[v]] {
                if next != NONE { after = after.max(w.tail[next]); }
            }
            w.tail[v] = self.duration[v].checked_add(after)?;
        }
        Some(makespan)
    }

    fn swap_adjacent(&mut self, u: usize, v: usize) {
        debug_assert_eq!(self.mn[u],v);
        let p = self.mp[u];
        let n = self.mn[v];
        if p != NONE { self.mn[p] = v; }
        if n != NONE { self.mp[n] = u; }
        self.mp[v] = p;
        self.mn[v] = u;
        self.mp[u] = v;
        self.mn[u] = n;
    }

    fn solution(&self, w: &Scratch) -> Solution {
        Solution { job_schedule: self.offsets.windows(2).map(|range|
            (range[0]..range[1]).map(|v| (self.machine[v],w.head[v])).collect()).collect() }
    }

    fn relocate(&mut self,u: usize,machine: usize,duration: u32,p: usize,n: usize) {
        let old_p=self.mp[u];
        let old_n=self.mn[u];
        if old_p!=NONE { self.mn[old_p]=old_n; }
        if old_n!=NONE { self.mp[old_n]=old_p; }
        self.mp[u]=p;
        self.mn[u]=n;
        if p!=NONE { self.mn[p]=u; }
        if n!=NONE { self.mp[n]=u; }
        self.machine[u]=machine;
        self.duration[u]=duration;
    }
}

pub fn polish(c: &Challenge, incumbent: &Solution, max_trials: usize) -> Option<Solution> {
    if max_trials == 0 { return None; }
    let original = c.evaluate_makespan(incumbent).ok()?;
    let mut graph = Graph::build(c,incumbent)?;
    let mut current = Scratch::new(graph.duration.len());
    let mut trial = Scratch::new(graph.duration.len());
    let mut best = graph.evaluate(&mut current,u32::MAX)?;
    if best > original { return None; }
    let mut attempts = 0;
    let order_limit=max_trials/2+max_trials%2;
    loop {
        let mut improved = false;
        for u in 0..graph.duration.len() {
            let v = graph.mn[u];
            if v == NONE { continue; }
            if current.head[u].checked_add(current.tail[u])? != best
                || current.head[v].checked_add(current.tail[v])? != best
                || current.head[u].checked_add(graph.duration[u])? != current.head[v] { continue; }
            let p = graph.mp[u];
            let n = graph.mn[v];
            let starts_block = p == NONE || current.head[p].checked_add(graph.duration[p])? != current.head[u]
                || current.head[p].checked_add(current.tail[p])? != best;
            let ends_block = n == NONE || current.head[v].checked_add(graph.duration[v])? != current.head[n]
                || current.head[n].checked_add(current.tail[n])? != best;
            if (starts_block || ends_block) && graph.job[u] != graph.job[v] {
                if attempts >= order_limit { break; }
                attempts += 1;
                graph.swap_adjacent(u,v);
                if let Some(value) = graph.evaluate(&mut trial,best) {
                    best = value;
                    __tig_safe_swap(&mut current,&mut trial);
                    improved = true;
                    break;
                }
                graph.swap_adjacent(v,u);
            }
            // N7-style boundary relocation across two consecutive critical
            // machine arcs. Intermediate graphs are never accepted or saved.
            let w = graph.mn[v];
            if w == NONE || current.head[w].checked_add(current.tail[w])? != best
                || current.head[v].checked_add(graph.duration[v])? != current.head[w] { continue; }
            let after = graph.mn[w];
            let triple_end = after == NONE
                || current.head[w].checked_add(graph.duration[w])? != current.head[after]
                || current.head[after].checked_add(current.tail[after])? != best;
            if !starts_block && !triple_end { continue; }
            for direction in 0..2 {
                let moving_job = if direction == 0 { graph.job[u] } else { graph.job[w] };
                let crossed_job = if direction == 0 { graph.job[w] } else { graph.job[u] };
                if moving_job == graph.job[v] || moving_job == crossed_job { continue; }
                if attempts >= order_limit { break; }
                attempts += 1;
                if direction == 0 {
                    graph.swap_adjacent(u,v);
                    graph.swap_adjacent(u,w);
                } else {
                    graph.swap_adjacent(v,w);
                    graph.swap_adjacent(u,w);
                }
                if let Some(value) = graph.evaluate(&mut trial,best) {
                    best = value;
                    __tig_safe_swap(&mut current,&mut trial);
                    improved = true;
                    break;
                }
                graph.swap_adjacent(w,u);
                if direction == 0 { graph.swap_adjacent(v,u); }
                else { graph.swap_adjacent(w,v); }
            }
            if improved || attempts >= order_limit { break; }
        }
        if !improved || attempts >= order_limit { break; }
    }
    // Critical eligible-machine insertions may retime other operations. The
    // local estimate orders candidates only; it never rejects an improvement.
    // Full DAG evaluation and strict acceptance remain authoritative.
    let products: Vec<usize>=c.jobs_per_product.iter().enumerate()
        .flat_map(|(p,&count)|std::iter::repeat(p).take(count)).collect();
    let mut machine_heads=vec![NONE;c.num_machines];
    let mut options=Vec::with_capacity(c.num_machines);
    let mut moves=Vec::with_capacity(graph.duration.len().saturating_mul(4));
    let idle_limit=max_trials-order_limit;
    let mut idle_attempts=0;
    let mut accepted_rounds=0;
    while idle_attempts<idle_limit && accepted_rounds<4 {
        machine_heads.fill(NONE);
        for v in 0..graph.duration.len() {
            if graph.mp[v]==NONE { machine_heads[graph.machine[v]]=v; }
        }
        moves.clear();
        for u in 0..graph.duration.len() {
            if current.head[u].checked_add(current.tail[u])?!=best { continue; }
            let job=graph.job[u];
            let op=u-graph.offsets[job];
            options.clear();
            options.extend(c.product_processing_times[products[job]][op].iter().map(|(&m,&d)|(m,d)));
            options.sort_unstable();
            let old_machine=graph.machine[u];
            let release=if graph.jp[u]==NONE {0} else {
                current.head[graph.jp[u]].checked_add(graph.duration[graph.jp[u]])?
            };
            let job_tail=if graph.jn[u]==NONE {0} else {current.tail[graph.jn[u]]};
            for &(machine,duration) in &options {
                if machine==old_machine || machine>=c.num_machines { continue; }
                let mut p=NONE;
                let mut n=machine_heads[machine];
                loop {
                    let earliest=if p==NONE {release} else {
                        release.max(current.head[p].checked_add(graph.duration[p])?)
                    };
                    let machine_tail=if n==NONE {0} else {current.tail[n]};
                    let estimate=earliest.saturating_add(duration).saturating_add(job_tail.max(machine_tail));
                    moves.push(MachineMove {rank:0,estimate,node:u,machine,duration,previous:p,next:n});
                    if n==NONE {break;}
                    p=n;
                    n=graph.mn[n];
                }
            }
        }
        // Round-robin ranked placements across critical operations prevents a
        // single long machine sequence from consuming the entire trial budget.
        moves.sort_unstable_by_key(|m|(m.node,m.estimate,m.machine,m.previous,m.next));
        let mut previous_node=NONE;
        let mut rank=0;
        for m in &mut moves {
            if m.node!=previous_node {previous_node=m.node;rank=0;} else {rank+=1;}
            m.rank=rank;
        }
        moves.sort_unstable();
        let mut improved=false;
        for m in &moves {
            if idle_attempts>=idle_limit {break;}
            idle_attempts+=1;
            let u=m.node;
            let old_machine=graph.machine[u];
            let old_duration=graph.duration[u];
            let old_p=graph.mp[u];
            let old_n=graph.mn[u];
            graph.relocate(u,m.machine,m.duration,m.previous,m.next);
            if let Some(value)=graph.evaluate(&mut trial,best) {
                best=value;
                __tig_safe_swap(&mut current,&mut trial);
                improved=true;
                accepted_rounds+=1;
                break;
            }
            graph.relocate(u,old_machine,old_duration,old_p,old_n);
        }
        if !improved {break;}
    }
    if best >= original { return None; }
    let candidate = graph.solution(&current);
    match c.evaluate_makespan(&candidate) {
        Ok(value) if value < original => Some(candidate),
        _ => None,
    }
}

#[inline(always)]
fn __tig_safe_swap<T: Clone>(a: &mut T, b: &mut T) { let tmp = a.clone(); *a = b.clone(); *b = tmp; }
