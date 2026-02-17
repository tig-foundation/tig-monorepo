// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("FJSP solver: machine-first construction + N5 tabu search");
    println!();
    println!("Optional hyperparameters (JSON). If omitted, per-track presets are used.");
    println!();
    println!("  restarts          Number of random construction restarts");
    println!("                    flow_shop=1500, job_shop=1500, hybrid_flow=3000, fjsp_medium=3000, fjsp_high=5000");
    println!("  top_starts        Number of best constructions kept for tabu phase");
    println!("                    flow=8, job=10, hybrid=12, fjsp_medium=12, fjsp_high=15");
    println!("  tabu_iters        Tabu search iterations per start");
    println!("                    flow=2500, job=2000, hybrid=1500, fjsp_medium=1500, fjsp_high=1500");
    println!("  final_iters       Tabu search iterations on global best at the end");
    println!("                    flow=8000, job=6000, hybrid=4000, fjsp_medium=4000, fjsp_high=4000");
    println!("  perturb_strength  Number of random swaps per perturbation");
    println!("                    default=4 (all tracks)");
    println!("  no_improve_limit  Max iterations without improvement before restart");
    println!("                    flow=120, job=120, hybrid=100, fjsp_medium=100, fjsp_high=80");
    println!();
    println!("Example: {{\"restarts\":2000,\"tabu_iters\":3000}}");
}

fn hp_f(hp: &Option<Map<String, Value>>, key: &str, default: f64) -> f64 {
    hp.as_ref().and_then(|m| m.get(key)).and_then(|v| v.as_f64()).unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq)]
enum Track { FlowShop, JobShop, HybridFlow, FjspMedium, FjspHigh }

fn detect_track(pd: &PD) -> Track {
    let total_flex: usize = pd.op_elig.iter().map(|e| e.len()).sum();
    let avg_flex = total_flex as f64 / pd.n as f64;
    if avg_flex > 5.0 { return Track::FjspHigh; }
    let has_flex = avg_flex > 1.5;
    let is_flow = if pd.jn[0] >= 2 {
        let machines: Vec<usize> = (0..pd.jn[0]).map(|k| {
            pd.op_elig[pd.fi(0, k)].iter().map(|&(m, _)| m).min().unwrap()
        }).collect();
        let mut uniq = machines.clone(); uniq.sort(); uniq.dedup();
        uniq.len() as f64 / machines.len() as f64 > 0.7
    } else { true };
    match (has_flex, is_flow) {
        (false, true) => Track::FlowShop,
        (false, false) => Track::JobShop,
        (true, true) => Track::HybridFlow,
        (true, false) => Track::FjspMedium,
    }
}

struct PD {
    nj: usize, nm: usize, n: usize,
    op_job: Vec<usize>, op_kidx: Vec<usize>,
    op_elig: Vec<Vec<(usize, u32)>>,
    jo: Vec<usize>, jn: Vec<usize>,
    op_work: Vec<f64>,
}

impl PD {
    fn new(c: &Challenge) -> Self {
        let nj = c.num_jobs; let nm = c.num_machines;
        let mut jp = Vec::with_capacity(nj);
        for (p, &cnt) in c.jobs_per_product.iter().enumerate() {
            for _ in 0..cnt { jp.push(p); }
        }
        let mut jo = Vec::with_capacity(nj);
        let mut jn = Vec::with_capacity(nj);
        let mut n = 0usize;
        for j in 0..nj {
            jo.push(n);
            let ops = c.product_processing_times[jp[j]].len();
            jn.push(ops); n += ops;
        }
        let mut op_job = vec![]; let mut op_kidx = vec![]; let mut op_elig = vec![];
        let mut op_work = vec![];
        for j in 0..nj {
            for k in 0..jn[j] {
                op_job.push(j); op_kidx.push(k);
                let mut el: Vec<(usize, u32)> = c.product_processing_times[jp[j]][k]
                    .iter().map(|(&m, &t)| (m, t)).collect();
                el.sort_unstable_by_key(|&(m, _)| m);
                let avg = el.iter().map(|&(_, t)| t as f64).sum::<f64>() / el.len() as f64;
                let min = el.iter().map(|&(_, t)| t).min().unwrap_or(0) as f64;
                op_work.push(avg * 0.7 + min * 0.3);
                op_elig.push(el);
            }
        }
        PD { nj, nm, n, op_job, op_kidx, op_elig, jo, jn, op_work }
    }
    #[inline] fn fi(&self, j: usize, k: usize) -> usize { self.jo[j] + k }
    #[inline] fn jnext(&self, o: usize) -> Option<usize> {
        if self.op_kidx[o] + 1 < self.jn[self.op_job[o]] { Some(o + 1) } else { None }
    }
}

#[derive(Clone)]
struct Sched {
    a: Vec<usize>, p: Vec<u32>, ms: Vec<Vec<usize>>,
    st: Vec<u32>, et: Vec<u32>, mk: u32,
}

impl Sched {
    fn new(pd: &PD) -> Self {
        Sched { a: vec![0;pd.n], p: vec![0;pd.n], ms: vec![vec![];pd.nm],
                 st: vec![0;pd.n], et: vec![0;pd.n], mk: 0 }
    }
    fn compute(&mut self, pd: &PD) -> bool {
        let n = pd.n;
        let mut mn = vec![usize::MAX; n];
        let mut id = vec![0u8; n];
        for m in 0..pd.nm {
            let seq = &self.ms[m];
            for i in 0..seq.len() {
                if i > 0 { id[seq[i]] += 1; }
                if i + 1 < seq.len() { mn[seq[i]] = seq[i + 1]; }
            }
        }
        for j in 0..pd.nj { for k in 1..pd.jn[j] { id[pd.fi(j,k)] += 1; } }
        for o in 0..n { self.st[o] = 0; self.et[o] = 0; }
        let mut q = Vec::with_capacity(n);
        for o in 0..n { if id[o] == 0 { self.et[o] = self.p[o]; q.push(o); } }
        let mut qi = 0;
        while qi < q.len() {
            let o = q[qi]; qi += 1; let e = self.et[o];
            if let Some(ns) = pd.jnext(o) {
                if e > self.st[ns] { self.st[ns] = e; self.et[ns] = e + self.p[ns]; }
                id[ns] -= 1; if id[ns] == 0 { q.push(ns); }
            }
            if mn[o] != usize::MAX {
                let ns = mn[o];
                if e > self.st[ns] { self.st[ns] = e; self.et[ns] = e + self.p[ns]; }
                id[ns] -= 1; if id[ns] == 0 { q.push(ns); }
            }
        }
        self.mk = self.et.iter().copied().max().unwrap_or(0);
        q.len() == n
    }
    fn to_sol(&self, pd: &PD) -> Solution {
        let mut js = Vec::with_capacity(pd.nj);
        for j in 0..pd.nj {
            let mut ops = Vec::with_capacity(pd.jn[j]);
            for k in 0..pd.jn[j] { let o = pd.fi(j,k); ops.push((self.a[o], self.st[o])); }
            js.push(ops);
        }
        Solution { job_schedule: js }
    }
    fn copy_from(&mut self, o: &Sched, pd: &PD) {
        self.a.copy_from_slice(&o.a); self.p.copy_from_slice(&o.p);
        for m in 0..pd.nm { self.ms[m].clear(); self.ms[m].extend_from_slice(&o.ms[m]); }
        self.st.copy_from_slice(&o.st); self.et.copy_from_slice(&o.et); self.mk = o.mk;
    }
}

fn construct(pd: &PD, rule: usize, rng: &mut SmallRng, top_k: usize) -> Sched {
    let mut s = Sched::new(pd);
    let mut m_avail = vec![0u32; pd.nm];
    let mut j_ready = vec![0u32; pd.nj];
    let mut next_op = vec![0usize; pd.nj];
    let mut j_work: Vec<f64> = (0..pd.nj).map(|j| {
        (0..pd.jn[j]).map(|k| pd.op_work[pd.fi(j, k)]).sum()
    }).collect();

    let use_random = top_k > 1;
    let mut remaining = pd.n;
    let mut time = 0u32;

    while remaining > 0 {
        let mut avail_m: Vec<usize> = (0..pd.nm)
            .filter(|&m| m_avail[m] <= time)
            .collect();
        if use_random {
            avail_m.shuffle(rng);
        }

        let mut scheduled_any = false;
        for &machine in &avail_m {
            let mut candidates: Vec<(usize, f64, u32, u32, usize)> = Vec::new();

            for j in 0..pd.nj {
                if next_op[j] >= pd.jn[j] { continue; }
                if j_ready[j] > time { continue; }

                let o = pd.fi(j, next_op[j]);
                let proc_time = match pd.op_elig[o].iter().find(|&&(m, _)| m == machine) {
                    Some(&(_, pt)) => pt,
                    None => continue,
                };

                let earliest_end = pd.op_elig[o].iter().map(|&(m, pt)| {
                    time.max(m_avail[m]) + pt
                }).min().unwrap();
                let machine_end = time.max(m_avail[machine]) + proc_time;
                if machine_end != earliest_end {
                    continue;
                }

                let flexibility = pd.op_elig[o].len();
                let priority = match rule % 5 {
                    0 => j_work[j],
                    1 => (pd.jn[j] - next_op[j]) as f64,
                    2 => -(flexibility as f64),
                    3 => -(proc_time as f64),
                    _ => proc_time as f64,
                };

                candidates.push((j, priority, machine_end, proc_time, flexibility));
            }

            if candidates.is_empty() { continue; }

            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.2.cmp(&b.2))
                    .then(a.3.cmp(&b.3))
                    .then(a.4.cmp(&b.4))
                    .then(a.0.cmp(&b.0))
            });

            let pick = if use_random && candidates.len() > 1 {
                rng.gen_range(0..top_k.min(candidates.len()))
            } else {
                0
            };

            let (j, _, _, _, _) = candidates[pick];
            let o = pd.fi(j, next_op[j]);
            let proc_time = pd.op_elig[o].iter().find(|&&(m, _)| m == machine).unwrap().1;
            let start = time.max(m_avail[machine]);
            let end = start + proc_time;

            s.a[o] = machine;
            s.p[o] = proc_time;
            s.ms[machine].push(o);
            m_avail[machine] = end;
            j_ready[j] = end;
            j_work[j] -= pd.op_work[o];
            if j_work[j] < 0.0 { j_work[j] = 0.0; }
            next_op[j] += 1;
            remaining -= 1;
            scheduled_any = true;
        }

        if remaining == 0 { break; }

        let mut next_time = u32::MAX;
        for &t in &m_avail {
            if t > time && t < next_time { next_time = t; }
        }
        for j in 0..pd.nj {
            if next_op[j] < pd.jn[j] && j_ready[j] > time && j_ready[j] < next_time {
                next_time = j_ready[j];
            }
        }

        if next_time == u32::MAX {
            if !scheduled_any { break; }
            break;
        }
        time = next_time;
    }

    s.compute(&pd);
    s
}

struct CB { m: usize, ops: Vec<usize>, pos: Vec<usize> }

fn find_blocks(s: &Sched, pd: &PD) -> Vec<CB> {
    let n = pd.n;
    let mut tail = vec![0u32; n];
    let mut mn = vec![usize::MAX; n];
    for m in 0..pd.nm {
        let seq = &s.ms[m];
        for i in 0..seq.len().saturating_sub(1) { mn[seq[i]] = seq[i+1]; }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a,&b| s.et[b].cmp(&s.et[a]));
    for &o in &order {
        let mut t = 0u32;
        if let Some(jn) = pd.jnext(o) { t = t.max(s.p[jn]+tail[jn]); }
        if mn[o] != usize::MAX { let nx=mn[o]; t = t.max(s.p[nx]+tail[nx]); }
        tail[o] = t;
    }
    let mk = s.mk;
    let mut blocks = Vec::new();
    for m in 0..pd.nm {
        let seq = &s.ms[m]; let mut i = 0;
        while i < seq.len() {
            let o = seq[i];
            if s.st[o]+s.p[o]+tail[o] == mk {
                let mut bo = vec![o]; let mut bp = vec![i]; let mut j = i+1;
                while j < seq.len() {
                    let oj = seq[j];
                    if s.st[oj]+s.p[oj]+tail[oj]==mk && s.et[seq[j-1]]==s.st[oj] {
                        bo.push(oj); bp.push(j); j+=1;
                    } else { break; }
                }
                if bo.len() >= 2 { blocks.push(CB { m, ops: bo, pos: bp }); }
                i = j;
            } else { i += 1; }
        }
    }
    blocks
}

fn perturb_swap(pd: &PD, s: &mut Sched, rng: &mut SmallRng, n: usize) -> bool {
    let mut done = 0;
    for _ in 0..n*8 {
        if done >= n { break; }
        let m = rng.gen_range(0..pd.nm);
        if s.ms[m].len() >= 2 {
            let i = rng.gen_range(0..s.ms[m].len()-1);
            let (o1,o2) = (s.ms[m][i], s.ms[m][i+1]);
            if pd.op_job[o1] != pd.op_job[o2] { s.ms[m].swap(i,i+1); done+=1; }
        }
    }
    if done > 0 { s.compute(pd) } else { true }
}

fn tabu_n5(
    pd: &PD, init: &Sched, save: &dyn Fn(&Solution) -> Result<()>,
    max_iter: usize, rng: &mut SmallRng, gbest: &mut u32,
    pstr: usize, noi_limit: usize,
) -> Result<Sched> {
    let mut cur = init.clone();
    let mut best = init.clone();
    let mut best_mk = init.mk;
    let mut tabu = vec![0usize; pd.n];
    let tenure = 7 + (pd.n as f64).sqrt() as usize;
    let mut trial = init.clone();
    let mut noi = 0usize;

    for iter in 0..max_iter {
        let blocks = find_blocks(&cur, pd);
        let mut bmk = u32::MAX;
        let mut bm=0; let mut bp1=0; let mut bp2=0;
        let mut found = false;

        for block in &blocks {
            let m = block.m; let len = block.ops.len();
            {
                let (p1,p2) = (block.pos[0], block.pos[1]);
                let (o1,o2) = (block.ops[0], block.ops[1]);
                let tb = tabu[o1] > iter || tabu[o2] > iter;
                trial.copy_from(&cur, pd);
                trial.ms[m].swap(p1,p2);
                if trial.compute(pd) && trial.mk < bmk && (!tb || trial.mk < best_mk) {
                    bmk=trial.mk; bm=m; bp1=p1; bp2=p2; found=true;
                }
            }
            if len > 2 {
                let (p1,p2) = (block.pos[len-2], block.pos[len-1]);
                let (o1,o2) = (block.ops[len-2], block.ops[len-1]);
                let tb = tabu[o1] > iter || tabu[o2] > iter;
                trial.copy_from(&cur, pd);
                trial.ms[m].swap(p1,p2);
                if trial.compute(pd) && trial.mk < bmk && (!tb || trial.mk < best_mk) {
                    bmk=trial.mk; bm=m; bp1=p1; bp2=p2; found=true;
                }
            }
        }

        if found {
            let o1=cur.ms[bm][bp1]; let o2=cur.ms[bm][bp2];
            cur.ms[bm].swap(bp1,bp2); cur.compute(pd);
            tabu[o1]=iter+tenure; tabu[o2]=iter+tenure;
            if cur.mk < best_mk {
                best_mk=cur.mk; best.copy_from(&cur, pd);
                if best_mk < *gbest { *gbest=best_mk; save(&best.to_sol(pd))?; }
                noi=0;
            } else { noi+=1; }
        } else { noi+=1; }

        if noi > noi_limit {
            cur.copy_from(&best, pd);
            let ok = perturb_swap(pd, &mut cur, rng, pstr);
            if !ok { cur.copy_from(&best, pd); }
            noi=0;
            for t in tabu.iter_mut() { *t=0; }
        }
    }
    Ok(best)
}

struct Cfg {
    restarts: usize,
    top_starts: usize,
    tabu_iters: usize,
    final_iters: usize,
    pstr: usize,
    noi: usize,
}

fn config_for(track: Track) -> Cfg {
    match track {
        Track::FlowShop => Cfg {
            restarts: 1500, top_starts: 8, tabu_iters: 2500, final_iters: 8000,
            pstr: 4, noi: 120,
        },
        Track::JobShop => Cfg {
            restarts: 1500, top_starts: 10, tabu_iters: 2000, final_iters: 6000,
            pstr: 4, noi: 120,
        },
        Track::HybridFlow => Cfg {
            restarts: 3000, top_starts: 12, tabu_iters: 1500, final_iters: 4000,
            pstr: 4, noi: 100,
        },
        Track::FjspMedium => Cfg {
            restarts: 3000, top_starts: 12, tabu_iters: 1500, final_iters: 4000,
            pstr: 4, noi: 100,
        },
        Track::FjspHigh => Cfg {
            restarts: 5000, top_starts: 15, tabu_iters: 1500, final_iters: 4000,
            pstr: 4, noi: 80,
        },
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pd = PD::new(challenge);
    let mut rng = SmallRng::from_seed(challenge.seed);
    let track = detect_track(&pd);
    let dc = config_for(track);

    let restarts = hp_f(hyperparameters, "restarts", dc.restarts as f64) as usize;
    let top_starts = hp_f(hyperparameters, "top_starts", dc.top_starts as f64) as usize;
    let tabu_iters = hp_f(hyperparameters, "tabu_iters", dc.tabu_iters as f64) as usize;
    let final_iters = hp_f(hyperparameters, "final_iters", dc.final_iters as f64) as usize;
    let pstr = hp_f(hyperparameters, "perturb_strength", dc.pstr as f64) as usize;
    let noi = hp_f(hyperparameters, "no_improve_limit", dc.noi as f64) as usize;

    let mut best_mk = u32::MAX;
    let mut best_s = Sched::new(&pd);
    let mut tops: Vec<(u32, Sched)> = Vec::new();

    for rule in 0..5 {
        let s = construct(&pd, rule, &mut rng, 1);
        if s.mk < best_mk { best_mk = s.mk; best_s.copy_from(&s, &pd); }
        tops.push((s.mk, s));
    }

    for _ in 0..restarts {
        let rule = rng.gen_range(0..5);
        let tk = rng.gen_range(2..=5);
        let s = construct(&pd, rule, &mut rng, tk);
        if s.mk < best_mk { best_mk = s.mk; best_s.copy_from(&s, &pd); }
        tops.push((s.mk, s));
    }
    save_solution(&best_s.to_sol(&pd))?;

    tops.sort_by_key(|&(mk, _)| mk);
    tops.truncate(top_starts);

    let mut gbest = best_mk;
    for i in 0..tops.len() {
        let iters = if i < 3 { tabu_iters } else { tabu_iters / 2 };
        let r = tabu_n5(&pd, &tops[i].1, save_solution, iters, &mut rng, &mut gbest, pstr, noi)?;
        if r.mk < best_s.mk { best_s.copy_from(&r, &pd); }
    }

    let r = tabu_n5(&pd, &best_s, save_solution, final_iters, &mut rng, &mut gbest, pstr, noi)?;
    if r.mk < best_s.mk { best_s.copy_from(&r, &pd); }

    save_solution(&best_s.to_sol(&pd))?;
    Ok(())
}
