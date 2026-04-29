use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

const NONE: usize = usize::MAX;
const INF32: u32 = u32::MAX / 4;

struct PD {
    nj: usize,
    nm: usize,
    n: usize,
    job_product: Vec<usize>,
    job_ops_len: Vec<usize>,
    job_offset: Vec<usize>,
    op_elig: Vec<Vec<(usize, u32)>>,
    op_min_pt: Vec<u32>,
    op_flex: Vec<usize>,
    flex_avg: f64,
    machine_load0: Vec<f64>,
    op_pt_on: Vec<Vec<u32>>,
    op_avg_pt: Vec<f64>,
}

impl PD {
    fn new(c: &Challenge) -> Self {
        let nj = c.num_jobs;
        let nm = c.num_machines;

        let mut job_product = Vec::with_capacity(nj);
        for (p, &cnt) in c.jobs_per_product.iter().enumerate() {
            for _ in 0..cnt { job_product.push(p); }
        }

        let mut job_ops_len = Vec::with_capacity(nj);
        let mut job_offset = Vec::with_capacity(nj);
        let mut n = 0usize;
        for j in 0..nj {
            job_offset.push(n);
            let ops = c.product_processing_times[job_product[j]].len();
            job_ops_len.push(ops);
            n += ops;
        }

        let mut op_elig = Vec::with_capacity(n);
        let mut op_min_pt = Vec::with_capacity(n);
        let mut op_flex = Vec::with_capacity(n);
        let mut total_flex = 0usize;
        let mut machine_load0 = vec![0.0f64; nm];

        for j in 0..nj {
            let p = job_product[j];
            for op in &c.product_processing_times[p] {
                let mut el: Vec<(usize, u32)> = op.iter().map(|(&m, &t)| (m, t)).collect();
                el.sort_unstable_by_key(|&(_, t)| t);
                let min_pt = el.first().map(|&(_, t)| t).unwrap_or(INF32);
                let flex = el.len();
                total_flex += flex;
                if flex > 0 && min_pt < INF32 {
                    let delta = min_pt as f64 / flex as f64;
                    for &(m, _) in &el { machine_load0[m] += delta; }
                }
                op_min_pt.push(min_pt);
                op_flex.push(flex);
                op_elig.push(el);
            }
        }

        let mut op_pt_on = vec![vec![0u32; nm]; n];
        let mut op_avg_pt = vec![0.0f64; n];
        for o in 0..n {
            for &(m, pt) in &op_elig[o] { op_pt_on[o][m] = pt; }
            if !op_elig[o].is_empty() {
                let sum: u32 = op_elig[o].iter().map(|&(_, t)| t).sum();
                op_avg_pt[o] = sum as f64 / op_elig[o].len() as f64;
            }
        }

        let flex_avg = if n > 0 { total_flex as f64 / n as f64 } else { 1.0 };
        PD { nj, nm, n, job_product, job_ops_len, job_offset, op_elig, op_min_pt, op_flex, flex_avg, machine_load0, op_pt_on, op_avg_pt }
    }

    #[inline] fn op_id(&self, j: usize, k: usize) -> usize { self.job_offset[j] + k }
}

#[derive(Clone)]
struct DG {
    n: usize,
    nm: usize,
    nj: usize,
    node_pt: Vec<u32>,
    node_machine: Vec<usize>,
    node_job: Vec<usize>,
    node_op: Vec<usize>,
    job_offset: Vec<usize>,
    job_ops_len: Vec<usize>,
    machine_seq: Vec<Vec<usize>>,
}

struct Buf {
    indeg: Vec<u16>,
    start: Vec<u32>,
    pred: Vec<usize>,
    msucc: Vec<usize>,
    stack: Vec<usize>,
    tail: Vec<u32>,
    crit: Vec<bool>,
    mpred: Vec<usize>,
    jpred: Vec<usize>,
    saved_crit: Vec<bool>,
    saved_start: Vec<u32>,
}

impl Buf {
    fn new(n: usize) -> Self {
        Buf {
            indeg: vec![0; n], start: vec![0; n], pred: vec![NONE; n],
            msucc: vec![NONE; n], stack: Vec::with_capacity(n),
            tail: vec![0; n], crit: vec![false; n],
            mpred: vec![NONE; n], jpred: vec![NONE; n],
            saved_crit: vec![false; n], saved_start: vec![0; n],
        }
    }
}

impl DG {
    fn from_solution(pd: &PD, c: &Challenge, sol: &Solution) -> Result<Self> {
        let n = pd.n;
        let nm = pd.nm;
        let nj = pd.nj;
        let mut node_pt = vec![0u32; n];
        let mut node_machine = vec![0usize; n];
        let mut node_job = vec![0usize; n];
        let mut node_op = vec![0usize; n];
        let mut machine_ops: Vec<Vec<(u32, usize)>> = vec![vec![]; nm];

        for j in 0..nj {
            let p = pd.job_product[j];
            for k in 0..pd.job_ops_len[j] {
                let o = pd.op_id(j, k);
                let (m, st) = sol.job_schedule[j][k];
                let pt = c.product_processing_times[p][k].get(&m).copied().unwrap_or(1);
                node_pt[o] = pt; node_machine[o] = m; node_job[o] = j; node_op[o] = k;
                machine_ops[m].push((st, o));
            }
        }
        let mut machine_seq = vec![vec![]; nm];
        for m in 0..nm {
            machine_ops[m].sort_unstable_by_key(|&(st, _)| st);
            machine_seq[m] = machine_ops[m].iter().map(|&(_, o)| o).collect();
        }
        Ok(DG { n, nm, nj, node_pt, node_machine, node_job, node_op,
                 job_offset: pd.job_offset.clone(), job_ops_len: pd.job_ops_len.clone(), machine_seq })
    }

    fn eval(&self, buf: &mut Buf) -> Option<(u32, usize)> {
        let n = self.n;
        for i in 0..n { buf.indeg[i] = 0; buf.start[i] = 0; buf.pred[i] = NONE; buf.msucc[i] = NONE; }

        for j in 0..self.nj {
            let base = self.job_offset[j];
            for k in 1..self.job_ops_len[j] { buf.indeg[base + k] += 1; }
        }
        for m in 0..self.nm {
            let seq = &self.machine_seq[m];
            for i in 1..seq.len() { buf.indeg[seq[i]] += 1; }
            for i in 0..seq.len().saturating_sub(1) { buf.msucc[seq[i]] = seq[i + 1]; }
        }

        buf.stack.clear();
        for o in 0..n { if buf.indeg[o] == 0 { buf.stack.push(o); } }

        let mut qi = 0;
        while qi < buf.stack.len() {
            let o = buf.stack[qi]; qi += 1;
            let end = buf.start[o] + self.node_pt[o];
            let j = self.node_job[o];
            let k = self.node_op[o];
            if k + 1 < self.job_ops_len[j] {
                let ns = self.job_offset[j] + k + 1;
                if end > buf.start[ns] { buf.start[ns] = end; buf.pred[ns] = o; }
                buf.indeg[ns] -= 1;
                if buf.indeg[ns] == 0 { buf.stack.push(ns); }
            }
            let ms = buf.msucc[o];
            if ms != NONE {
                if end > buf.start[ms] { buf.start[ms] = end; buf.pred[ms] = o; }
                buf.indeg[ms] -= 1;
                if buf.indeg[ms] == 0 { buf.stack.push(ms); }
            }
        }
        if qi != n { return None; }

        let mut mk = 0u32;
        let mut mk_node = 0;
        for o in 0..n {
            let end = buf.start[o] + self.node_pt[o];
            if end > mk { mk = end; mk_node = o; }
        }
        Some((mk, mk_node))
    }

    fn compute_tails_and_preds(&self, buf: &mut Buf) {
        let n = self.n;
        for i in 0..n { buf.tail[i] = 0; buf.mpred[i] = NONE; buf.jpred[i] = NONE; }

        for seq in &self.machine_seq {
            for i in 1..seq.len() { buf.mpred[seq[i]] = seq[i - 1]; }
        }
        for j in 0..self.nj {
            let base = self.job_offset[j];
            for k in 1..self.job_ops_len[j] { buf.jpred[base + k] = base + k - 1; }
        }

        for &o in buf.stack.iter().rev() {
            let contrib = self.node_pt[o] + buf.tail[o];
            let j = self.node_job[o];
            let k = self.node_op[o];
            if k > 0 {
                let jp = self.job_offset[j] + k - 1;
                if contrib > buf.tail[jp] { buf.tail[jp] = contrib; }
            }
            if buf.mpred[o] != NONE {
                let mp = buf.mpred[o];
                if contrib > buf.tail[mp] { buf.tail[mp] = contrib; }
            }
        }
    }

    fn mark_critical(&self, buf: &mut Buf, mk: u32, mk_node: usize) {
        for o in 0..self.n {
            buf.crit[o] = buf.start[o] + self.node_pt[o] + buf.tail[o] == mk;
        }
        let mut u = mk_node;
        while u != NONE { buf.crit[u] = true; u = buf.pred[u]; }
    }

    fn to_solution(&self, pd: &PD, buf: &Buf) -> Solution {
        let mut js = Vec::with_capacity(pd.nj);
        for j in 0..pd.nj {
            let mut ops = Vec::with_capacity(pd.job_ops_len[j]);
            for k in 0..pd.job_ops_len[j] {
                let o = pd.op_id(j, k);
                ops.push((self.node_machine[o], buf.start[o]));
            }
            js.push(ops);
        }
        Solution { job_schedule: js }
    }
}

#[inline]
fn estimate_insert_mk(
    op: usize, old_m: usize, old_idx: usize,
    target_m: usize, ins_pos: usize,
    new_pt: u32, buf: &Buf, dg: &DG,
) -> u32 {
    let j = dg.node_job[op];
    let k = dg.node_op[op];
    let jp = if k > 0 { dg.job_offset[j] + k - 1 } else { NONE };
    let js = if k + 1 < dg.job_ops_len[j] { dg.job_offset[j] + k + 1 } else { NONE };
    let r_from_job = if jp != NONE { buf.start[jp] + dg.node_pt[jp] } else { 0 };
    let q_from_job = if js != NONE { dg.node_pt[js] + buf.tail[js] } else { 0 };

    let (mp, ms, old_mp, old_ms);

    if old_m == target_m {
        let seq = &dg.machine_seq[old_m];
        let slen = seq.len();
        let cleaned_len = slen - 1;
        mp = if ins_pos == 0 { NONE } else {
            let ci = ins_pos - 1;
            if ci < old_idx { seq[ci] } else { seq[ci + 1] }
        };
        ms = if ins_pos >= cleaned_len { NONE } else {
            let ci = ins_pos;
            if ci < old_idx { seq[ci] } else { seq[ci + 1] }
        };
        old_mp = if old_idx > 0 { seq[old_idx - 1] } else { NONE };
        old_ms = if old_idx + 1 < slen { seq[old_idx + 1] } else { NONE };
    } else {
        let tseq = &dg.machine_seq[target_m];
        mp = if ins_pos > 0 { tseq[ins_pos - 1] } else { NONE };
        ms = if ins_pos < tseq.len() { tseq[ins_pos] } else { NONE };
        let oseq = &dg.machine_seq[old_m];
        old_mp = if old_idx > 0 { oseq[old_idx - 1] } else { NONE };
        old_ms = if old_idx + 1 < oseq.len() { oseq[old_idx + 1] } else { NONE };
    }

    let r_mach = if mp != NONE { buf.start[mp] + dg.node_pt[mp] } else { 0 };
    let new_r_op = r_from_job.max(r_mach);
    let q_mach = if ms != NONE { dg.node_pt[ms] + buf.tail[ms] } else { 0 };
    let new_q_op = q_from_job.max(q_mach);
    let path_op = new_r_op + new_pt + new_q_op;

    let path_gap = if old_ms != NONE {
        let jj = dg.node_job[old_ms];
        let kk = dg.node_op[old_ms];
        let jpp = if kk > 0 { dg.job_offset[jj] + kk - 1 } else { NONE };
        let r_j = if jpp != NONE { buf.start[jpp] + dg.node_pt[jpp] } else { 0 };
        let r_m = if old_mp != NONE { buf.start[old_mp] + dg.node_pt[old_mp] } else { 0 };
        r_j.max(r_m) + dg.node_pt[old_ms] + buf.tail[old_ms]
    } else { 0 };

    let path_ms = if ms != NONE {
        let jj = dg.node_job[ms];
        let kk = dg.node_op[ms];
        let jpp = if kk > 0 { dg.job_offset[jj] + kk - 1 } else { NONE };
        let r_j = if jpp != NONE { buf.start[jpp] + dg.node_pt[jpp] } else { 0 };
        r_j.max(new_r_op + new_pt) + dg.node_pt[ms] + buf.tail[ms]
    } else { 0 };

    path_op.max(path_gap).max(path_ms)
}

struct CB { m: usize, s: usize, e: usize }

fn find_blocks(dg: &DG, buf: &Buf) -> Vec<CB> {
    let mut blocks = Vec::new();
    for m in 0..dg.nm {
        let seq = &dg.machine_seq[m];
        let mut i = 0;
        while i < seq.len() {
            if !buf.crit[seq[i]] { i += 1; continue; }
            let bs = i;
            let mut be = i;
            while be + 1 < seq.len() {
                let x = seq[be]; let y = seq[be + 1];
                if !buf.crit[y] { break; }
                if buf.start[y] != buf.start[x] + dg.node_pt[x] { break; }
                be += 1;
            }
            if be > bs { blocks.push(CB { m, s: bs, e: be }); }
            i = be + 1;
        }
    }
    blocks
}

#[inline]
fn estimate_swap_mk(m: usize, pos: usize, buf: &Buf, dg: &DG) -> u32 {
    let seq = &dg.machine_seq[m];
    let o1 = seq[pos];
    let o2 = seq[pos + 1];

    let mp = if pos > 0 { seq[pos - 1] } else { NONE };
    let ms = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE };

    let end_mp = if mp != NONE { buf.start[mp] + dg.node_pt[mp] } else { 0 };

    let j2 = dg.node_job[o2]; let k2 = dg.node_op[o2];
    let jp2 = if k2 > 0 { dg.job_offset[j2] + k2 - 1 } else { NONE };
    let end_jp2 = if jp2 != NONE { buf.start[jp2] + dg.node_pt[jp2] } else { 0 };
    let r_o2 = end_mp.max(end_jp2);

    let j1 = dg.node_job[o1]; let k1 = dg.node_op[o1];
    let jp1 = if k1 > 0 { dg.job_offset[j1] + k1 - 1 } else { NONE };
    let end_jp1 = if jp1 != NONE { buf.start[jp1] + dg.node_pt[jp1] } else { 0 };
    let r_o1 = (r_o2 + dg.node_pt[o2]).max(end_jp1);

    let js1 = if k1 + 1 < dg.job_ops_len[j1] { dg.job_offset[j1] + k1 + 1 } else { NONE };
    let q_j1 = if js1 != NONE { dg.node_pt[js1] + buf.tail[js1] } else { 0 };
    let q_ms = if ms != NONE { dg.node_pt[ms] + buf.tail[ms] } else { 0 };
    let q_o1 = q_j1.max(q_ms);

    let js2 = if k2 + 1 < dg.job_ops_len[j2] { dg.job_offset[j2] + k2 + 1 } else { NONE };
    let q_j2 = if js2 != NONE { dg.node_pt[js2] + buf.tail[js2] } else { 0 };
    let q_o2 = q_j2.max(dg.node_pt[o1] + q_o1);

    let path1 = r_o2 + dg.node_pt[o2] + q_o2;
    let path2 = r_o1 + dg.node_pt[o1] + q_o1;

    let path3 = if ms != NONE {
        let jms = dg.node_job[ms]; let kms = dg.node_op[ms];
        let jpms = if kms > 0 { dg.job_offset[jms] + kms - 1 } else { NONE };
        let end_jpms = if jpms != NONE { buf.start[jpms] + dg.node_pt[jpms] } else { 0 };
        end_jpms.max(r_o1 + dg.node_pt[o1]) + dg.node_pt[ms] + buf.tail[ms]
    } else { 0 };

    path1.max(path2).max(path3)
}

struct Cand {
    kind: u8,
    op: usize,
    m: usize,
    old_idx: usize,
    ins_pos: usize,
    new_m: usize,
    new_pt: u32,
    est: u32,
}

fn hybrid_search(
    dg: &mut DG, pd: &PD, buf: &mut Buf,
    max_iter: usize, high_flex: bool, pseed: &mut u64,
    save: &dyn Fn(&Solution) -> Result<()>,
    global_best: &mut u32,
    perturb_str: usize, noi_limit: usize,
    verify_k: usize,
) -> Result<(u32, DG)> {
    let Some((mk, mk_node)) = dg.eval(buf) else { return Ok((u32::MAX, dg.clone())); };
    dg.compute_tails_and_preds(buf);
    dg.mark_critical(buf, mk, mk_node);

    let mut best_mk = mk;

    let mut best_dg = dg.clone();
    let mut tabu = vec![0usize; pd.n];
    let tenure = if high_flex { 4 + (pd.n as f64).sqrt() as usize / 2 }
                 else { 7 + (pd.n as f64).sqrt() as usize };
    let mut no_improve = 0usize;
    let mut cands: Vec<Cand> = Vec::with_capacity(256);

    for iter in 0..max_iter {
        buf.saved_crit.copy_from_slice(&buf.crit);
        buf.saved_start.copy_from_slice(&buf.start);

        let blocks = find_blocks(dg, buf);
        cands.clear();

        for block in &blocks {
            let m = block.m;
            let seq = &dg.machine_seq[m];
            let bs = block.s;
            let be = block.e;
            if be <= bs { continue; }

            let est = estimate_swap_mk(m, bs, buf, dg);
            cands.push(Cand { kind: 0, op: 0, m, old_idx: bs, ins_pos: bs + 1, new_m: m, new_pt: 0, est });

            if be - bs > 1 {
                let est = estimate_swap_mk(m, be - 1, buf, dg);
                cands.push(Cand { kind: 0, op: 0, m, old_idx: be - 1, ins_pos: be, new_m: m, new_pt: 0, est });
            }

            let first_op = seq[bs];
            let last_op = seq[be];

            for target in (bs + 1)..=be {
                let est = estimate_insert_mk(first_op, m, bs, m, target, dg.node_pt[first_op], buf, dg);
                cands.push(Cand { kind: 1, op: first_op, m, old_idx: bs, ins_pos: target, new_m: m, new_pt: dg.node_pt[first_op], est });
            }

            for target in bs..be {
                let est = estimate_insert_mk(last_op, m, be, m, target, dg.node_pt[last_op], buf, dg);
                cands.push(Cand { kind: 1, op: last_op, m, old_idx: be, ins_pos: target, new_m: m, new_pt: dg.node_pt[last_op], est });
            }

            for pos in (bs + 1)..be {
                let op = seq[pos];
                let pt = dg.node_pt[op];
                let est1 = estimate_insert_mk(op, m, pos, m, bs, pt, buf, dg);
                cands.push(Cand { kind: 1, op, m, old_idx: pos, ins_pos: bs, new_m: m, new_pt: pt, est: est1 });
                let est2 = estimate_insert_mk(op, m, pos, m, be, pt, buf, dg);
                cands.push(Cand { kind: 1, op, m, old_idx: pos, ins_pos: be, new_m: m, new_pt: pt, est: est2 });
            }
        }

        if high_flex {
            for o in 0..pd.n {
                if !buf.saved_crit[o] || pd.op_flex[o] < 2 { continue; }
                let old_m = dg.node_machine[o];
                let old_idx = match dg.machine_seq[old_m].iter().position(|&x| x == o) {
                    Some(i) => i, None => continue,
                };

                for &(new_m, new_pt) in &pd.op_elig[o] {
                    if new_m == old_m { continue; }
                    let tseq = &dg.machine_seq[new_m];
                    for ip in 0..=tseq.len() {
                        let est = estimate_insert_mk(o, old_m, old_idx, new_m, ip, new_pt, buf, dg);
                        cands.push(Cand { kind: 2, op: o, m: old_m, old_idx, ins_pos: ip, new_m, new_pt, est });
                    }
                }
            }
        }

        if cands.is_empty() {
            no_improve += 1;
            if no_improve > noi_limit {
                dg.clone_from(&best_dg);
                perturb(dg, pd, buf, perturb_str, pseed);
                if let Some((mk, mkn)) = dg.eval(buf) {
                    dg.compute_tails_and_preds(buf);
                    dg.mark_critical(buf, mk, mkn);
                }
                no_improve = 0;
                for t in tabu.iter_mut() { *t = 0; }
            }
            continue;
        }

        let k = verify_k.min(cands.len());
        if k > 0 && cands.len() > k {
            cands.select_nth_unstable_by_key(k - 1, |c| c.est);
        }

        let mut bmk = u32::MAX;
        let mut best_ci: Option<usize> = None;

        for ci in 0..k {
            let c = &cands[ci];
            let (is_tabu, mk_val) = match c.kind {
                0 => {
                    let o1 = dg.machine_seq[c.m][c.old_idx];
                    let o2 = dg.machine_seq[c.m][c.ins_pos];
                    let tb = tabu[o1] > iter || tabu[o2] > iter;
                    dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                    let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                    dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                    (tb, mk)
                },
                1 => {
                    let tb = tabu[c.op] > iter;
                    dg.machine_seq[c.m].remove(c.old_idx);
                    let ins = c.ins_pos.min(dg.machine_seq[c.m].len());
                    dg.machine_seq[c.m].insert(ins, c.op);
                    let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                    dg.machine_seq[c.m].remove(ins);
                    dg.machine_seq[c.m].insert(c.old_idx, c.op);
                    (tb, mk)
                },
                _ => {
                    let tb = tabu[c.op] > iter;
                    let saved_pt = dg.node_pt[c.op];
                    let saved_machine = dg.node_machine[c.op];
                    dg.machine_seq[c.m].remove(c.old_idx);
                    dg.node_machine[c.op] = c.new_m;
                    dg.node_pt[c.op] = c.new_pt;
                    let ins = c.ins_pos.min(dg.machine_seq[c.new_m].len());
                    dg.machine_seq[c.new_m].insert(ins, c.op);
                    let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                    dg.machine_seq[c.new_m].remove(ins);
                    dg.node_machine[c.op] = saved_machine;
                    dg.node_pt[c.op] = saved_pt;
                    dg.machine_seq[c.m].insert(c.old_idx, c.op);
                    (tb, mk)
                },
            };

            if mk_val < bmk && (!is_tabu || mk_val < best_mk) {
                bmk = mk_val;
                best_ci = Some(ci);
            }
        }

        if let Some(ci) = best_ci {
            let c = &cands[ci];
            match c.kind {
                0 => {
                    let o1 = dg.machine_seq[c.m][c.old_idx];
                    let o2 = dg.machine_seq[c.m][c.ins_pos];
                    dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                    tabu[o1] = iter + tenure;
                    tabu[o2] = iter + tenure;
                },
                1 => {
                    dg.machine_seq[c.m].remove(c.old_idx);
                    let ins = c.ins_pos.min(dg.machine_seq[c.m].len());
                    dg.machine_seq[c.m].insert(ins, c.op);
                    tabu[c.op] = iter + tenure;
                },
                _ => {
                    dg.machine_seq[c.m].remove(c.old_idx);
                    dg.node_machine[c.op] = c.new_m;
                    dg.node_pt[c.op] = c.new_pt;
                    let ins = c.ins_pos.min(dg.machine_seq[c.new_m].len());
                    dg.machine_seq[c.new_m].insert(ins, c.op);
                    tabu[c.op] = iter + tenure;
                },
            }

            if let Some((mk, mkn)) = dg.eval(buf) {
                dg.compute_tails_and_preds(buf);
                dg.mark_critical(buf, mk, mkn);
                if mk < best_mk {
                    best_mk = mk;
                    best_dg.clone_from(dg);
                    if mk < *global_best {
                        *global_best = mk;
                        let s = dg.to_solution(pd, buf);
                        save(&s)?;
                    }
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                dg.clone_from(&best_dg);
                if let Some((mk, mkn)) = dg.eval(buf) {

                    dg.compute_tails_and_preds(buf);
                    dg.mark_critical(buf, mk, mkn);
                }
                no_improve += 1;
            }
        } else {
            no_improve += 1;
        }

        if no_improve > noi_limit {
            dg.clone_from(&best_dg);
            perturb(dg, pd, buf, perturb_str, pseed);
            if let Some((mk, mkn)) = dg.eval(buf) {
                dg.compute_tails_and_preds(buf);
                dg.mark_critical(buf, mk, mkn);
            }
            no_improve = 0;
            for t in tabu.iter_mut() { *t = 0; }
        }
    }

    Ok((best_mk, best_dg))
}

fn perturb(dg: &mut DG, pd: &PD, buf: &mut Buf, strength: usize, pseed: &mut u64) {
    let Some((mk, mkn)) = dg.eval(buf) else { return; };
    dg.compute_tails_and_preds(buf);
    dg.mark_critical(buf, mk, mkn);
    let blocks = find_blocks(dg, buf);
    if blocks.is_empty() { return; }

    let crit_flex: Vec<usize> = (0..pd.n)
        .filter(|&o| buf.crit[o] && pd.op_flex[o] >= 2)
        .collect();

    for _ in 0..strength {
        *pseed ^= (*pseed).wrapping_shl(13);
        *pseed ^= (*pseed).wrapping_shr(7);
        *pseed ^= (*pseed).wrapping_shl(17);

        if !crit_flex.is_empty() && (*pseed % 3) == 0 {
            let ci = (*pseed as usize / 3) % crit_flex.len();
            let node = crit_flex[ci];
            let old_m = dg.node_machine[node];
            *pseed ^= (*pseed).wrapping_shl(13);
            *pseed ^= (*pseed).wrapping_shr(7);
            *pseed ^= (*pseed).wrapping_shl(17);
            let ei = (*pseed as usize) % pd.op_elig[node].len();
            let (new_m, new_pt) = pd.op_elig[node][ei];
            if new_m != old_m {
                if let Some(from_pos) = dg.machine_seq[old_m].iter().position(|&x| x == node) {
                    dg.machine_seq[old_m].remove(from_pos);
                    dg.node_machine[node] = new_m;
                    dg.node_pt[node] = new_pt;
                    let j = dg.node_job[node];
                    let k = dg.node_op[node];
                    let jpred_end = if k > 0 {
                        let jp = dg.job_offset[j] + k - 1;
                        buf.start[jp] + dg.node_pt[jp]
                    } else { 0 };
                    let near = dg.machine_seq[new_m]
                        .partition_point(|&x| buf.start[x] < jpred_end);
                    let ip = near.min(dg.machine_seq[new_m].len());
                    dg.machine_seq[new_m].insert(ip, node);
                }
            }
        } else {
            let bi = (*pseed as usize) % blocks.len();
            let block = &blocks[bi];
            let blen = block.e - block.s;
            if blen < 1 { continue; }
            *pseed ^= (*pseed).wrapping_shl(13);
            *pseed ^= (*pseed).wrapping_shr(7);
            *pseed ^= (*pseed).wrapping_shl(17);
            let pos = block.s + (*pseed as usize) % blen;
            dg.machine_seq[block.m].swap(pos, pos + 1);
        }
    }
}

fn construct_mc(pd: &PD, rule: usize, rng: &mut SmallRng, top_k: usize) -> (Solution, u32) {
    let nj = pd.nj;
    let nm = pd.nm;
    let mut job_next = vec![0usize; nj];
    let mut job_ready = vec![0u32; nj];
    let mut m_avail = vec![0u32; nm];
    let mut schedule: Vec<Vec<(usize, u32)>> = pd.job_ops_len.iter().map(|&l| Vec::with_capacity(l)).collect();
    let mut remaining = pd.n;
    let mut makespan = 0u32;

    let mut job_rem_work: Vec<f64> = (0..nj).map(|j| {
        (0..pd.job_ops_len[j]).map(|k| {
            let o = pd.op_id(j, k);
            pd.op_avg_pt[o] * 0.7 + pd.op_min_pt[o] as f64 * 0.3
        }).sum()
    }).collect();

    let mut time = 0u32;
    let use_random = top_k > 1;
    let mut cands: Vec<(f64, usize, u32, usize)> = Vec::with_capacity(nj);
    let mut m_order: Vec<usize> = (0..nm).collect();

    while remaining > 0 {
        let mut scheduled_any = false;


        if use_random {
            for i in (1..nm).rev() {
                let j = rng.gen_range(0..=i);
                m_order.swap(i, j);
            }
        }

        for &m in &m_order {
            if m_avail[m] > time { continue; }

            cands.clear();

            for j in 0..nj {
                let k = job_next[j];
                if k >= pd.job_ops_len[j] { continue; }
                if job_ready[j] > time { continue; }

                let o = pd.op_id(j, k);
                let pt_on_m = pd.op_pt_on[o][m];
                if pt_on_m == 0 { continue; }


                let end_on_m = time + pt_on_m;
                let mut earliest_end = u32::MAX;
                for &(em, ept) in &pd.op_elig[o] {
                    let eend = time.max(m_avail[em]) + ept;
                    if eend < earliest_end { earliest_end = eend; }
                }
                if end_on_m != earliest_end { continue; }

                let ops_rem = (pd.job_ops_len[j] - k) as f64;
                let priority = match rule % 5 {
                    0 => job_rem_work[j],
                    1 => ops_rem,
                    2 => -(pd.op_flex[o] as f64),
                    3 => -(pt_on_m as f64),
                    _ => pt_on_m as f64,
                };

                cands.push((priority, j, pt_on_m, pd.op_flex[o]));
            }

            if cands.is_empty() { continue; }

            let (j, pt) = if !use_random || cands.len() <= 1 {

                let mut bi = 0usize;
                for ci in 1..cands.len() {
                    let (cp, cj, cpt, cflex) = cands[ci];
                    let (bp, bj, bpt, bflex) = cands[bi];
                    let eps = 1e-9;
                    if cp > bp + eps
                        || ((cp - bp).abs() <= eps && cpt < bpt)
                        || ((cp - bp).abs() <= eps && cpt == bpt && cflex < bflex)
                        || ((cp - bp).abs() <= eps && cpt == bpt && cflex == bflex && cj < bj) {
                        bi = ci;
                    }
                }
                (cands[bi].1, cands[bi].2)
            } else {
                cands.sort_unstable_by(|a, b| {
                    let ord = b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);
                    if ord != std::cmp::Ordering::Equal { return ord; }
                    let ord = a.2.cmp(&b.2);
                    if ord != std::cmp::Ordering::Equal { return ord; }
                    let ord = a.3.cmp(&b.3);
                    if ord != std::cmp::Ordering::Equal { return ord; }
                    a.1.cmp(&b.1)
                });
                let pick = rng.gen_range(0..top_k.min(cands.len()));
                (cands[pick].1, cands[pick].2)
            };

            let start = time;
            let end = start + pt;
            schedule[j].push((m, start));
            m_avail[m] = end;
            job_ready[j] = end;
            if end > makespan { makespan = end; }
            let oid = pd.op_id(j, job_next[j]);
            job_rem_work[j] -= pd.op_avg_pt[oid] * 0.7 + pd.op_min_pt[oid] as f64 * 0.3;
            if job_rem_work[j] < 0.0 { job_rem_work[j] = 0.0; }
            job_next[j] += 1;
            remaining -= 1;
            scheduled_any = true;
        }

        if remaining == 0 { break; }

        let mut next_time = u32::MAX;
        for mm in 0..nm { if m_avail[mm] > time && m_avail[mm] < next_time { next_time = m_avail[mm]; } }
        for j in 0..nj {
            if job_next[j] < pd.job_ops_len[j] && job_ready[j] > time && job_ready[j] < next_time {
                next_time = job_ready[j];
            }
        }
        if next_time == u32::MAX || next_time == time {
            if !scheduled_any { break; }
            continue;
        }
        time = next_time;
    }

    let mk = if remaining > 0 { INF32 } else { makespan };
    (Solution { job_schedule: schedule }, mk)
}

fn construct(pd: &PD, rule: usize, rng: &mut SmallRng, top_k: usize) -> (Solution, u32) {
    let nj = pd.nj;
    let nm = pd.nm;
    let mut job_next_op = vec![0usize; nj];
    let mut job_ready = vec![0u32; nj];
    let mut m_avail = vec![0u32; nm];
    let mut schedule: Vec<Vec<(usize, u32)>> = pd.job_ops_len.iter().map(|&l| Vec::with_capacity(l)).collect();
    let mut remaining = pd.n;
    let mut makespan = 0u32;

    let mut job_rem_work: Vec<f64> = (0..nj).map(|j| {
        (0..pd.job_ops_len[j]).map(|k| pd.op_min_pt[pd.op_id(j, k)] as f64).sum()
    }).collect();
    let avg_load = pd.machine_load0.iter().sum::<f64>() / nm.max(1) as f64;

    let mut cands: Vec<(f64, usize, usize, u32)> = Vec::with_capacity(nj);

    while remaining > 0 {
        cands.clear();
        for j in 0..nj {
            let k = job_next_op[j];
            if k >= pd.job_ops_len[j] { continue; }
            let o = pd.op_id(j, k);
            let jr = job_ready[j];

            let mut best_end = u32::MAX;
            let mut best_m = 0usize;
            let mut best_pt = 0u32;
            let mut second_end = u32::MAX;

            for &(m, pt) in &pd.op_elig[o] {
                let start = jr.max(m_avail[m]);
                let end = start + pt;
                if end < best_end {
                    second_end = best_end;
                    best_end = end; best_m = m; best_pt = pt;
                } else if end < second_end {
                    second_end = end;
                }
            }
            if best_end == u32::MAX { continue; }

            let ops_rem = (pd.job_ops_len[j] - k) as f64;
            let flex_inv = 1.0 / (pd.op_flex[o] as f64).max(1.0);
            let rem_work_n = job_rem_work[j] / avg_load.max(1.0);
            let end_n = best_end as f64 / avg_load.max(1.0);

            let score = match rule % 7 {
                0 => rem_work_n + 0.15 * ops_rem - 0.7 * end_n,
                1 => ops_rem - 0.5 * end_n,
                2 => flex_inv + 0.4 * rem_work_n - 0.3 * end_n,
                3 => -(best_end as f64),
                4 => -(best_pt as f64) + 0.3 * rem_work_n,
                5 => {
                    let regret = if second_end < INF32 {
                        (second_end - best_end) as f64 / avg_load.max(1.0)
                    } else {
                        0.0
                    };
                    regret + 0.5 * rem_work_n - 0.3 * end_n
                }
                _ => rem_work_n + 0.3 * flex_inv
                    - 0.5 * (pd.machine_load0[best_m] / avg_load.max(1.0))
                    - 0.4 * end_n,
            };

            cands.push((score, j, best_m, best_pt));
        }

        if cands.is_empty() { break; }

        cands.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let pick = if top_k > 1 && cands.len() > 1 { rng.gen_range(0..top_k.min(cands.len())) } else { 0 };
        let (_, j, m, pt) = cands[pick];

        let start = job_ready[j].max(m_avail[m]);
        let end = start + pt;
        schedule[j].push((m, start));
        m_avail[m] = end;
        job_ready[j] = end;
        if end > makespan { makespan = end; }
        let oid = pd.op_id(j, job_next_op[j]);
        job_rem_work[j] -= pd.op_min_pt[oid] as f64;
        if job_rem_work[j] < 0.0 { job_rem_work[j] = 0.0; }
        job_next_op[j] += 1;
        remaining -= 1;
    }

    let mk = if remaining > 0 { INF32 } else { makespan };
    (Solution { job_schedule: schedule }, mk)
}

fn insert_top(top: &mut Vec<(Solution, u32)>, sol: Solution, mk: u32, max_size: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    if pos < max_size {
        top.insert(pos, (sol, mk));
        if top.len() > max_size { top.truncate(max_size); }
    }
}

pub fn solve_our(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let pd = PD::new(challenge);
    let mut rng = SmallRng::from_seed(challenge.seed);
    let high_flex = pd.flex_avg > 1.5;


    let max_ops = *pd.job_ops_len.iter().max().unwrap_or(&0);
    let min_ops = *pd.job_ops_len.iter().min().unwrap_or(&0);
    let uniform_routing = max_ops == min_ops;
    let (num_restarts, keep_top, search_iters, final_iters, perturb_str, noi_limit, verify_k) =
        if pd.flex_avg > 5.0 {
            (2000, 5, 5000, 12000, 5, 60, 25)
        } else if high_flex && !uniform_routing {
            // FjspMedium — reduced for speed, still good quality
            (2000, 6, 5000, 12000, 4, 80, 20)
        } else if high_flex {
            // HybridFlowShop path (rarely used — most HFS goes to dedicated solver)
            (1500, 8, 3000, 8000, 4, 100, 15)
        } else {
            (1500, 8, 4000, 10000, 4, 100, 12)
        };


    let mut top: Vec<(Solution, u32)> = Vec::new();
    let mut best_mk = u32::MAX;

    for rule in 0..5 {
        let (sol, mk) = construct_mc(&pd, rule, &mut rng, 1);
        if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
        insert_top(&mut top, sol, mk, keep_top);
    }
    for rule in 0..7 {
        let (sol, mk) = construct(&pd, rule, &mut rng, 1);
        if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
        insert_top(&mut top, sol, mk, keep_top);
    }
    for i in 0..num_restarts {
        let k = rng.gen_range(2..=5);
        let (sol, mk) = if i % 2 == 0 {
            construct_mc(&pd, rng.gen_range(0..5), &mut rng, k)
        } else {
            construct(&pd, rng.gen_range(0..7), &mut rng, k)
        };
        if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
        insert_top(&mut top, sol, mk, keep_top);
    }


    let mut buf = Buf::new(pd.n);
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (best_mk as u64).wrapping_shl(16) ^ (pd.n as u64);
    let mut global_best = best_mk;
    let mut best_dg: Option<DG> = None;

    for i in 0..top.len() {
        let mut dg = match DG::from_solution(&pd, challenge, &top[i].0) {
            Ok(d) => d, Err(_) => continue,
        };
        let iters = if i < 3 { search_iters } else { search_iters / 2 };
        let (mk, rdg) = hybrid_search(
            &mut dg, &pd, &mut buf, iters, high_flex, &mut pseed,
            save_solution, &mut global_best, perturb_str, noi_limit, verify_k,
        )?;
        if best_dg.is_none() || mk < best_mk {
            best_mk = mk;
            best_dg = Some(rdg);
        }
    }


    if let Some(ref bdg) = best_dg {
        let mut dg = bdg.clone();
        let (mk3, dg3) = hybrid_search(
            &mut dg, &pd, &mut buf, final_iters, high_flex, &mut pseed,
            save_solution, &mut global_best, perturb_str, noi_limit, verify_k,
        )?;
        if mk3 < best_mk { best_mk = mk3; best_dg = Some(dg3); }
    }

    Ok(())
}
