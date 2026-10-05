//! Study settings of the etna_highcap_fast build, read only from the hyperparameter key "v1lab".
//!
//! Without that key (null hyperparameters included) every setting keeps its default, ember_etna's
//! except the fjsp_high work cap (200e9; ember_etna: 65e9): `solve_challenge` (mod.rs) then runs
//! ember_etna's sequence of operations. A setting left out also keeps its default.
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};

/// Study settings. `None` / `false` / `0` = ember_etna behaviour, except `cap_high` (see below).
#[derive(Clone, Debug, Default)]
pub struct Lab {
    /// Fuel the hybrid construction and tabu search may spend together (ember_etna: 200e9).
    pub cap_hybrid: Option<u64>,
    /// Work cap of the fjsp_high pool search (default 200e9 when `None`; ember_etna: 65e9).
    pub cap_high: Option<u64>,
    /// Work cap of the fjsp_medium pool search (ember_etna: 200e9).
    pub cap_medium: Option<u64>,
    /// Iterations of each full job-shop tabu search (ember_etna: 300 000).
    pub js_iters: Option<usize>,
    /// Pooled schedules the first job-shop tabu searches start from (ember_etna: 4).
    pub js_starts: Option<usize>,
    /// Extra shifting-bottleneck seeds run after ember_etna's last job-shop save (0 to 4).
    pub js_extra_sbp: usize,
    /// Restarts of the hybrid alternative-start construction (ember_etna: 1000).
    pub hyb_alt_iters: Option<usize>,
    /// flow_shop / job_shop: stop the engine once the saved makespan reaches the lower bound.
    pub stop_lb: bool,
    /// Hybrid: the full task_tree_j t47 pipeline replaces the alternative-start construction.
    pub y2: bool,
    /// Hybrid: the tabu budget is recomputed after the alternative-start construction.
    pub y2_tabu_cap: bool,
}

pub const CAP_MIN: u64 = 10_000_000_000;
pub const CAP_MAX: u64 = 4_500_000_000_000;

/// Non-negative integer in `lo..=hi`, given as a JSON integer or as an integral float (`3e11`).
fn num_u64(v: &Value, lo: u64, hi: u64) -> Option<u64> {
    let x = v.as_u64().or_else(|| {
        v.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0 && *f <= 9.0e15)
            .map(|f| f as u64)
    })?;
    if x < lo || x > hi {
        None
    } else {
        Some(x)
    }
}

/// `true`, or the integer 1.
fn flag(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(x) => x.as_u64() == Some(1),
        None => false,
    }
}

/// `None` when the hyperparameters are null or have no "v1lab" key: the caller then runs the
/// defaults (ember_etna's, fjsp_high cap 200e9). A "v1lab" value that is not an object gives the defaults. A
/// field of the wrong type or out of range keeps its default. Never panics.
pub fn parse(hp: &Option<Map<String, Value>>) -> Option<Lab> {
    let v = hp.as_ref()?.get("v1lab")?;
    let mut lab = Lab::default();
    let Some(o) = v.as_object() else { return Some(lab) };
    if let Some(c) = o.get("caps").and_then(|c| c.as_object()) {
        lab.cap_hybrid = c.get("hybrid").and_then(|x| num_u64(x, CAP_MIN, CAP_MAX));
        lab.cap_high = c.get("fjsp_high").and_then(|x| num_u64(x, CAP_MIN, CAP_MAX));
        lab.cap_medium = c.get("fjsp_medium").and_then(|x| num_u64(x, CAP_MIN, CAP_MAX));
    }
    lab.js_iters = o.get("js_iters").and_then(|x| num_u64(x, 1_000, 3_000_000)).map(|x| x as usize);
    lab.js_starts = o.get("js_starts").and_then(|x| num_u64(x, 1, 15)).map(|x| x as usize);
    lab.js_extra_sbp = o.get("js_extra_sbp").and_then(|x| num_u64(x, 0, 4)).unwrap_or(0) as usize;
    lab.hyb_alt_iters = o.get("hyb_alt_iters").and_then(|x| num_u64(x, 100, 20_000)).map(|x| x as usize);
    lab.stop_lb = flag(o.get("stop_lb"));
    lab.y2 = flag(o.get("y2"));
    lab.y2_tabu_cap = flag(o.get("y2_tabu_cap"));
    Some(lab)
}

/// Entries kept per list of the trace; later ones are only counted.
const MAX_ENTRIES: usize = 20_000;

/// In-memory run log of the laboratory build, which wrote it out at the end of the run. This build
/// has no output for it and creates it off (`solve_challenge`, mod.rs): it then reads no fuel and
/// records nothing.
pub struct Trace {
    on: bool,
    /// Fuel when created, read only when on; its one reader was the output of the laboratory build.
    #[allow(dead_code)]
    f0: u64,
    track: Cell<&'static str>,
    lb: Cell<Option<u32>>,
    lbhit: Cell<Option<u64>>,
    saves: Cell<u64>,
    last: Cell<u64>,
    dropped: Cell<u64>,
    steps: RefCell<Vec<(u64, u32)>>,
    marks: RefCell<Vec<(&'static str, u64, u64, u32)>>,
}

impl Trace {
    /// Reads the fuel only when on.
    pub fn new(on: bool) -> Self {
        Trace {
            on,
            f0: if on { super::fuel_left() } else { 0 },
            track: Cell::new("?"),
            lb: Cell::new(None),
            lbhit: Cell::new(None),
            saves: Cell::new(0),
            last: Cell::new(0),
            dropped: Cell::new(0),
            steps: RefCell::new(Vec::new()),
            marks: RefCell::new(Vec::new()),
        }
    }

    pub fn set_track(&self, t: &'static str) {
        self.track.set(t);
    }

    pub fn set_lb(&self, lb: Option<u32>) {
        self.lb.set(lb);
    }

    /// A save the host accepted, of makespan `mk`.
    pub fn save(&self, mk: u32) {
        if !self.on {
            return;
        }
        let f = super::fuel_left();
        self.saves.set(self.saves.get() + 1);
        self.last.set(f);
        if self.lbhit.get().is_none() && self.lb.get().map_or(false, |lb| mk <= lb) {
            self.lbhit.set(Some(f));
        }
        let mut st = self.steps.borrow_mut();
        if st.last().map_or(true, |&(_, m)| mk < m) {
            if st.len() < MAX_ENTRIES {
                st.push((f, mk));
            } else {
                self.dropped.set(self.dropped.get() + 1);
            }
        }
    }

    /// A phase boundary `id` with a phase-specific value, and the best saved makespan.
    pub fn mark(&self, id: &'static str, extra: u64, best: u32) {
        if !self.on {
            return;
        }
        let f = super::fuel_left();
        let mut m = self.marks.borrow_mut();
        if m.len() < MAX_ENTRIES {
            m.push((id, extra, f, best));
        } else {
            self.dropped.set(self.dropped.get() + 1);
        }
    }
}

// BEGIN_BOUNDS (v1/tests/bounds extracts this block verbatim)
/// Lower bound when every operation has a single eligible machine: the longest job, and per
/// machine the preemptive one-machine bound with job heads and tails (Jackson's preemptive
/// schedule). Integer arithmetic only.
pub fn lb_rigid(job_ops: &[Vec<Vec<(usize, u32)>>], num_machines: usize) -> u32 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    // (head, duration, tail, operation id) per machine.
    let mut per_m: Vec<Vec<(u64, u64, u64, usize)>> = vec![Vec::new(); num_machines];
    let mut lb = 0u64;
    let mut id = 0usize;
    for ops in job_ops {
        let total: u64 = ops.iter().map(|o| o.first().map_or(0, |&(_, p)| p as u64)).sum();
        lb = lb.max(total);
        let mut head = 0u64;
        for o in ops {
            if let Some(&(m, p)) = o.first() {
                let p = p as u64;
                let tail = total.saturating_sub(head + p);
                if m < num_machines {
                    per_m[m].push((head, p, tail, id));
                }
                head += p;
            }
            id += 1;
        }
    }
    // Max-heap on the tail; ties by smaller operation id.
    let mut heap: BinaryHeap<(u64, Reverse<usize>, u64)> = BinaryHeap::new();
    for ops in per_m.iter_mut() {
        ops.sort_unstable_by_key(|&(r, _, _, i)| (r, i));
        heap.clear();
        let mut t = 0u64;
        let mut i = 0usize;
        loop {
            if heap.is_empty() {
                if i >= ops.len() {
                    break;
                }
                t = t.max(ops[i].0);
            }
            while i < ops.len() && ops[i].0 <= t {
                heap.push((ops[i].2, Reverse(ops[i].3), ops[i].1));
                i += 1;
            }
            let Some((q, k, rem)) = heap.pop() else { break };
            let next = if i < ops.len() { ops[i].0 } else { u64::MAX };
            if t.saturating_add(rem) <= next {
                t += rem;
                lb = lb.max(t + q);
            } else {
                heap.push((q, k, rem - (next - t)));
                t = next;
            }
        }
    }
    lb.min(u32::MAX as u64) as u32
}

/// Lower bound of a flexible schedule from the shortest eligible durations: the longest job, and
/// for every distinct eligible-machine set S, the operations whose eligible machines all lie in S
/// spread over |S| machines after their smallest head and before their smallest tail.
pub fn lb_flex(job_ops: &[Vec<Vec<(usize, u32)>>], num_machines: usize) -> u32 {
    let words = (num_machines + 63) / 64;
    if words == 0 {
        return 0;
    }
    // (eligible set, shortest duration, head, tail) per operation.
    let mut ops: Vec<(Vec<u64>, u64, u64, u64)> = Vec::new();
    let mut lb = 0u64;
    for job in job_ops {
        let pmin: Vec<u64> = job
            .iter()
            .map(|o| o.iter().map(|&(_, t)| t as u64).min().unwrap_or(0))
            .collect();
        let total: u64 = pmin.iter().sum();
        lb = lb.max(total);
        let mut head = 0u64;
        for (o, &p) in job.iter().zip(pmin.iter()) {
            let mut set = vec![0u64; words];
            for &(m, _) in o {
                if m < num_machines {
                    set[m / 64] |= 1u64 << (m % 64);
                }
            }
            ops.push((set, p, head, total.saturating_sub(head + p)));
            head += p;
        }
    }
    let mut sets: Vec<Vec<u64>> = ops.iter().map(|o| o.0.clone()).collect();
    sets.sort_unstable();
    sets.dedup();
    for s in &sets {
        let size: u64 = s.iter().map(|w| w.count_ones() as u64).sum();
        if size == 0 {
            continue;
        }
        let (mut sum, mut rmin, mut qmin, mut any) = (0u64, u64::MAX, u64::MAX, false);
        for (set, p, r, q) in &ops {
            if set.iter().zip(s.iter()).all(|(a, b)| a & !b == 0) {
                sum += *p;
                rmin = rmin.min(*r);
                qmin = qmin.min(*q);
                any = true;
            }
        }
        if any {
            lb = lb.max(rmin + (sum + size - 1) / size + qmin);
        }
    }
    lb.min(u32::MAX as u64) as u32
}
// END_BOUNDS
