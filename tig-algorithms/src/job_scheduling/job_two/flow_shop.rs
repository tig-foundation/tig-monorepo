use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::infra::{
    flow_makespan, reentrant_makespan, push_top_solutions,
    critical_block_move_local_search,
};

fn johnson_order_from_ab(a: &[u32], b: &[u32]) -> Vec<usize> {
    let n = a.len().min(b.len());
    let mut front: Vec<(u32, usize)> = Vec::with_capacity(n);
    let mut back: Vec<(u32, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        if a[j] <= b[j] {
            front.push((a[j], j));
        } else {
            back.push((b[j], j));
        }
    }
    front.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    back.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
    let mut ord = Vec::with_capacity(n);
    for &(_, j) in &front {
        ord.push(j);
    }
    for &(_, j) in &back {
        ord.push(j);
    }
    ord
}

fn palmer_order(pt: &[Vec<u32>]) -> Vec<usize> {
    let n = pt.len();
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if m == 0 {
        return (0..n).collect();
    }
    let mm = m as i64;
    let mut jobs: Vec<(i64, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let row = &pt[j];
        let mut s: i64 = 0;
        for k in 0..m {
            let w = mm - 2 * (k as i64) - 1;
            s += w * (row[k] as i64);
        }
        jobs.push((s, j));
    }
    jobs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    jobs.into_iter().map(|x| x.1).collect()
}

fn gupta_order(pt: &[Vec<u32>]) -> Vec<usize> {
    let n = pt.len();
    if n == 0 {
        return vec![];
    }
    let m = pt[0].len();
    if m < 2 {
        return (0..n).collect();
    }

    let mut g1: Vec<(u32, usize)> = Vec::with_capacity(n);
    let mut g2: Vec<(u32, usize)> = Vec::with_capacity(n);

    for j in 0..n {
        let row = &pt[j];
        let mut u = u32::MAX;
        for k in 0..(m - 1) {
            u = u.min(row[k].saturating_add(row[k + 1]));
        }
        if row[0] < row[m - 1] {
            g1.push((u, j));
        } else {
            g2.push((u, j));
        }
    }

    g1.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    g2.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut ord = Vec::with_capacity(n);
    ord.extend(g1.into_iter().map(|x| x.1));
    ord.extend(g2.into_iter().map(|x| x.1));
    ord
}

fn rz_order(pt: &[Vec<u32>]) -> Vec<usize> {
    let n = pt.len();
    if n == 0 {
        return vec![];
    }
    let m = pt[0].len();
    if m == 0 {
        return (0..n).collect();
    }
    let mid = m / 2;
    let mut v: Vec<(i64, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let row = &pt[j];
        let mut a: i64 = 0;
        let mut b: i64 = 0;
        for k in 0..mid {
            a += row[k] as i64;
        }
        for k in mid..m {
            b += row[k] as i64;
        }
        v.push((a - b, j));
    }
    v.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
    v.into_iter().map(|x| x.1).collect()
}

fn cds_orders(pt: &[Vec<u32>]) -> Vec<Vec<usize>> {
    let n = pt.len();
    if n == 0 {
        return vec![];
    }
    let m = pt[0].len();
    if m <= 1 {
        return vec![(0..n).collect()];
    }
    let mut totals = vec![0u32; n];
    let mut prefix = vec![vec![0u32; m + 1]; n];
    for j in 0..n {
        let row = &pt[j];
        let mut s = 0u32;
        prefix[j][0] = 0;
        for k in 0..m {
            s = s.saturating_add(row[k]);
            prefix[j][k + 1] = s;
        }
        totals[j] = s;
    }

    let mut res: Vec<Vec<usize>> = Vec::with_capacity(m - 1);
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];
    for k in 1..m {
        for j in 0..n {
            a[j] = prefix[j][k];
            b[j] = totals[j].saturating_sub(prefix[j][k]);
        }
        res.push(johnson_order_from_ab(&a, &b));
    }
    res
}

fn route_is_unique(route: &[usize], num_machines: usize) -> bool {
    if route.is_empty() {
        return false;
    }
    let mut seen = vec![false; num_machines.max(1)];
    for &m in route {
        if m >= seen.len() {
            return false;
        }
        if seen[m] {
            return false;
        }
        seen[m] = true;
    }
    true
}

#[derive(Default, Clone)]
struct TailInsBuf {
    f: Vec<u32>,
    b: Vec<u32>,
    e: Vec<u32>,
    comp: Vec<u32>,
}

impl TailInsBuf {
    fn ensure(&mut self, len: usize, m: usize) {
        let need = (len + 1) * m;
        if self.f.len() < need {
            self.f.resize(need, 0);
        }
        if self.b.len() < need {
            self.b.resize(need, 0);
        }
        if self.e.len() < m {
            self.e.resize(m, 0);
        }
        if self.comp.len() < m {
            self.comp.resize(m, 0);
        }
    }
}

fn taillard_best_insert_pos(
    seq: &[usize],
    job: usize,
    pt: &[Vec<u32>],
    m: usize,
    buf: &mut TailInsBuf,
) -> (usize, u32) {
    let l = seq.len();
    if m == 0 {
        return (0, 0);
    }
    buf.ensure(l, m);
    let f = &mut buf.f;
    let b = &mut buf.b;
    let e = &mut buf.e;

    for k in 0..m {
        f[k] = 0;
    }
    for t in 1..=l {
        let jj = seq[t - 1];
        let row = &pt[jj];
        let base = t * m;
        let prev = (t - 1) * m;

        f[base] = f[prev].saturating_add(row[0]);
        for k in 1..m {
            f[base + k] = f[base + k - 1].max(f[prev + k]).saturating_add(row[k]);
        }
    }

    let base_l = l * m;
    for k in 0..m {
        b[base_l + k] = 0;
    }
    for t in (0..l).rev() {
        let jj = seq[t];
        let row = &pt[jj];
        let base = t * m;
        let next = (t + 1) * m;

        b[base + (m - 1)] = b[next + (m - 1)].saturating_add(row[m - 1]);
        if m >= 2 {
            for kk in 0..(m - 1) {
                let k = (m - 2) - kk;
                b[base + k] = b[base + k + 1].max(b[next + k]).saturating_add(row[k]);
            }
        }
    }

    let prow = &pt[job];
    let mut best_pos = 0usize;
    let mut best_mk = u32::MAX;
    let mut best_last = u32::MAX;

    for pos in 0..=l {
        let fb = pos * m;
        e[0] = f[fb].saturating_add(prow[0]);
        for k in 1..m {
            e[k] = e[k - 1].max(f[fb + k]).saturating_add(prow[k]);
        }
        let mut mk = 0u32;
        for k in 0..m {
            mk = mk.max(e[k].saturating_add(b[fb + k]));
        }
        let last = e[m - 1];
        if mk < best_mk
            || (mk == best_mk
                && (last < best_last || (last == best_last && pos < best_pos)))
        {
            best_mk = mk;
            best_pos = pos;
            best_last = last;
        }
    }
    (best_pos, best_mk)
}

fn taillard_topk_insert_positions(
    seq: &[usize],
    job: usize,
    pt: &[Vec<u32>],
    m: usize,
    buf: &mut TailInsBuf,
    k: usize,
) -> Vec<(usize, u32)> {
    let l = seq.len();
    if m == 0 {
        return vec![(0, 0)];
    }
    let kk = k.max(1).min(l + 1);
    buf.ensure(l, m);
    let f = &mut buf.f;
    let b = &mut buf.b;
    let e = &mut buf.e;

    for mm in 0..m {
        f[mm] = 0;
    }
    for t in 1..=l {
        let jj = seq[t - 1];
        let row = &pt[jj];
        let base = t * m;
        let prev = (t - 1) * m;

        f[base] = f[prev].saturating_add(row[0]);
        for mm in 1..m {
            f[base + mm] = f[base + mm - 1].max(f[prev + mm]).saturating_add(row[mm]);
        }
    }

    let base_l = l * m;
    for mm in 0..m {
        b[base_l + mm] = 0;
    }
    for t in (0..l).rev() {
        let jj = seq[t];
        let row = &pt[jj];
        let base = t * m;
        let next = (t + 1) * m;

        b[base + (m - 1)] = b[next + (m - 1)].saturating_add(row[m - 1]);
        if m >= 2 {
            for kk2 in 0..(m - 1) {
                let mm = (m - 2) - kk2;
                b[base + mm] = b[base + mm + 1].max(b[next + mm]).saturating_add(row[mm]);
            }
        }
    }

    let prow = &pt[job];
    let mut cands: Vec<(u32, u32, usize)> = Vec::with_capacity(l + 1);
    for pos in 0..=l {
        let fb = pos * m;
        e[0] = f[fb].saturating_add(prow[0]);
        for mm in 1..m {
            e[mm] = e[mm - 1].max(f[fb + mm]).saturating_add(prow[mm]);
        }
        let mut mk = 0u32;
        for mm in 0..m {
            mk = mk.max(e[mm].saturating_add(b[fb + mm]));
        }
        cands.push((mk, e[m - 1], pos));
    }
    cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
    cands.truncate(kk);
    cands.into_iter().map(|x| (x.2, x.0)).collect()
}

fn improve_perm_seq_taillard(
    seq: &mut Vec<usize>,
    pt: &[Vec<u32>],
    rounds: usize,
    buf: &mut TailInsBuf,
) {
    let n = seq.len();
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if n <= 2 || m == 0 || rounds == 0 {
        return;
    }
    buf.ensure(n, m);
    let mut cur_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);

    let mut jobs = Vec::with_capacity(n);
    for r in 0..rounds {
        let mut improved_any = false;
        jobs.clear();
        jobs.extend_from_slice(seq);
        if r & 1 == 1 {
            jobs.reverse();
        }

        for &job in &jobs {
            let Some(i0) = seq.iter().position(|&x| x == job) else {
                continue;
            };
            let j = seq.remove(i0);
            let (pos, mk) = taillard_best_insert_pos(seq, j, pt, m, buf);
            if mk < cur_mk {
                seq.insert(pos, j);
                cur_mk = mk;
                improved_any = true;
            } else {
                seq.insert(i0, j);
            }
        }

        if !improved_any {
            break;
        }
    }
}

fn best_pair_swap_window_once(
    seq: &mut Vec<usize>,
    pt: &[Vec<u32>],
    m: usize,
    cur_mk: u32,
    win: usize,
    buf: &mut TailInsBuf,
) -> Option<u32> {
    let n = seq.len();
    if n <= 2 || m == 0 {
        return None;
    }
    let w = win.max(2).min(n);
    let mut best_mk = cur_mk;
    let mut bi = usize::MAX;
    let mut bj = usize::MAX;

    for i in 0..(n - 1) {
        let jmax = (i + w).min(n - 1);
        for j in (i + 1)..=jmax {
            seq.swap(i, j);
            let mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
            if mk < best_mk {
                best_mk = mk;
                bi = i;
                bj = j;
            }
            seq.swap(i, j);
        }
    }

    if bi != usize::MAX {
        seq.swap(bi, bj);
        return Some(best_mk);
    }
    None
}

fn steepest_relocate_once(
    seq: &mut Vec<usize>,
    pt: &[Vec<u32>],
    m: usize,
    cur_mk: u32,
    buf: &mut TailInsBuf,
) -> Option<u32> {
    let n = seq.len();
    if n <= 2 || m == 0 {
        return None;
    }
    let mut best_mk = cur_mk;
    let mut best_from = usize::MAX;
    let mut best_pos = usize::MAX;

    for from in 0..n {
        let job = seq.remove(from);
        let (pos, mk) = taillard_best_insert_pos(seq, job, pt, m, buf);
        seq.insert(from, job);
        if mk < best_mk && pos != from {
            best_mk = mk;
            best_from = from;
            best_pos = pos;
        }
    }

    if best_from != usize::MAX {
        let job = seq.remove(best_from);
        seq.insert(best_pos, job);
        return Some(best_mk);
    }
    None
}

fn perm_vnd_improve(seq: &mut Vec<usize>, pt: &[Vec<u32>], max_steps: usize, buf: &mut TailInsBuf) -> u32 {
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if seq.len() <= 1 || m == 0 {
        return 0;
    }
    buf.ensure(seq.len(), m);
    let mut cur_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
    let mut steps = 0usize;

    while steps < max_steps {
        if let Some(mk2) = steepest_relocate_once(seq, pt, m, cur_mk, buf) {
            cur_mk = mk2;
            steps += 1;
            continue;
        }
        if let Some(mk2) = best_pair_swap_window_once(seq, pt, m, cur_mk, 10, buf) {
            cur_mk = mk2;
            steps += 1;
            continue;
        }
        break;
    }
    cur_mk
}

fn adjacent_swap_hillclimb(
    seq: &mut Vec<usize>,
    pt: &[Vec<u32>],
    passes: usize,
    buf: &mut TailInsBuf,
) -> u32 {
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if seq.len() <= 1 || m == 0 {
        return 0;
    }
    buf.ensure(seq.len(), m);
    let mut best_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);

    for _ in 0..passes {
        let mut improved = false;
        for i in 0..(seq.len() - 1) {
            seq.swap(i, i + 1);
            let mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
            if mk <= best_mk {
                best_mk = mk;
                improved = true;
            } else {
                seq.swap(i, i + 1);
            }
        }
        if !improved {
            break;
        }
    }
    best_mk
}

thread_local! {
    static TL_TAILLARD: RefCell<TailInsBuf> = RefCell::new(TailInsBuf::default());
}

fn neh_build_seq(
    order: &[usize],
    route: &[usize],
    pt: &[Vec<u32>],
    num_machines: usize,
) -> Vec<usize> {
    let unique = route_is_unique(route, num_machines);
    if unique {
        let m = route.len();
        if m == 0 {
            return vec![];
        }
        return TL_TAILLARD.with(|cell| {
            let mut buf = cell.borrow_mut();
            let mut seq: Vec<usize> = Vec::with_capacity(order.len());
            for &j in order {
                if seq.is_empty() {
                    seq.push(j);
                    continue;
                }
                let (pos, _mk) = taillard_best_insert_pos(&seq, j, pt, m, &mut buf);
                seq.insert(pos, j);
            }
            seq
        });
    }

    let mut seq: Vec<usize> = Vec::with_capacity(order.len());
    let mut tmp: Vec<usize> = Vec::with_capacity(order.len());
    let mut mready = vec![0u32; num_machines];
    for &j in order {
        if seq.is_empty() {
            seq.push(j);
            continue;
        }
        let mut best_mk = u32::MAX;
        let mut best_pos = 0usize;
        for pos in 0..=seq.len() {
            tmp.clear();
            tmp.extend_from_slice(&seq[..pos]);
            tmp.push(j);
            tmp.extend_from_slice(&seq[pos..]);
            let mk = reentrant_makespan(&tmp, route, pt, &mut mready);
            if mk < best_mk {
                best_mk = mk;
                best_pos = pos;
            }
        }
        seq.insert(best_pos, j);
    }
    seq
}

fn neh_build_seq_beam_unique(
    order: &[usize],
    pt: &[Vec<u32>],
    m: usize,
    beam_w: usize,
    topk_pos: usize,
) -> Vec<usize> {
    if m == 0 || order.is_empty() {
        return vec![];
    }
    let bw = beam_w.max(1).min(16);

    TL_TAILLARD.with(|cell| {
        let mut buf = cell.borrow_mut();
        let mut beam: Vec<(Vec<usize>, u32)> = Vec::with_capacity(bw);

        let j0 = order[0];
        let s0 = vec![j0];
        buf.ensure(1, m);
        let mk0 = flow_makespan(&s0, pt, &mut buf.comp[..m]);
        beam.push((s0, mk0));

        for &j in &order[1..] {
            let mut next: Vec<(Vec<usize>, u32)> = Vec::with_capacity(bw * topk_pos.max(1));
            for (seq, _mk_prev) in beam.iter() {
                let poss = taillard_topk_insert_positions(seq, j, pt, m, &mut buf, topk_pos);
                for (pos, mk) in poss {
                    let mut ns = seq.clone();
                    ns.insert(pos, j);
                    next.push((ns, mk));
                }
            }

            next.sort_unstable_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    let la = a.0.len().min(b.0.len());
                    for i in 0..la {
                        let ca = a.0[i];
                        let cb = b.0[i];
                        if ca != cb {
                            return ca.cmp(&cb);
                        }
                    }
                    a.0.len().cmp(&b.0.len())
                })
            });

            beam.clear();
            for (seq, mk) in next.into_iter() {
                if beam.len() >= bw {
                    break;
                }
                let mut dup = false;
                for (s2, _) in beam.iter() {
                    if *s2 == seq {
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    beam.push((seq, mk));
                }
            }
            if beam.is_empty() {
                beam.push((vec![j], 0));
            }
        }

        beam.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        beam.first().map(|x| x.0.clone()).unwrap_or_default()
    })
}

fn fs_improve_reentrant_seq(
    seq: &mut Vec<usize>,
    route: &[usize],
    pt: &[Vec<u32>],
    num_machines: usize,
) {
    if seq.len() <= 2 || route.is_empty() {
        return;
    }
    let unique = route_is_unique(route, num_machines);
    if unique {
        TL_TAILLARD.with(|cell| {
            let mut buf = cell.borrow_mut();
            improve_perm_seq_taillard(seq, pt, 10, &mut buf);
            let _ = perm_vnd_improve(seq, pt, 18, &mut buf);
            let _ = adjacent_swap_hillclimb(seq, pt, 1, &mut buf);
        });
        return;
    }

    let mut mready = vec![0u32; num_machines];
    let mut cur_mk = reentrant_makespan(seq, route, pt, &mut mready);

    for _round in 0..8usize {
        let mut improved_any = false;
        for i0 in 0..seq.len() {
            let j = seq.remove(i0);
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=seq.len() {
                seq.insert(pos, j);
                let mk = reentrant_makespan(seq, route, pt, &mut mready);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
                seq.remove(pos);
            }
            seq.insert(best_pos, j);
            if best_mk < cur_mk {
                cur_mk = best_mk;
                improved_any = true;
            }
        }
        if !improved_any {
            break;
        }
    }
}

fn build_perm_solution_from_seq(
    seq: &[usize],
    route: &[usize],
    pt: &[Vec<u32>],
    num_jobs: usize,
    num_machines: usize,
) -> Solution {
    let ops = route.len();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs];
    let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs {
            continue;
        }
        let row = &pt[j];
        let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines {
                break;
            }
            let p = row[op_idx];
            let st = prev_end.max(machine_ready[m]);
            job_schedule[j].push((m, st));
            let end = st.saturating_add(p);
            machine_ready[m] = end;
            prev_end = end;
        }
    }
    Solution { job_schedule }
}

fn order_from_solution_first_op_start(sol: &Solution, num_jobs: usize) -> Vec<usize> {
    let mut v: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let st = sol
            .job_schedule
            .get(j)
            .and_then(|ops| ops.first())
            .map(|x| x.1);
        if let Some(t) = st {
            v.push((t, j));
        }
    }
    v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut seen = vec![false; num_jobs];
    let mut ord: Vec<usize> = Vec::with_capacity(num_jobs);
    for &(_, j) in &v {
        if j < num_jobs && !seen[j] {
            seen[j] = true;
            ord.push(j);
        }
    }
    for j in 0..num_jobs {
        if !seen[j] {
            ord.push(j);
        }
    }
    ord
}

fn neh_best_sequence(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<Vec<usize>> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let ops = route.len();
    if ops == 0 || pt.len() != num_jobs {
        return Err(anyhow!("Invalid flow data"));
    }

    let mut lpt: Vec<usize> = (0..num_jobs).collect();
    lpt.sort_unstable_by(|&a, &b| {
        let sa: u32 = pt[a].iter().copied().sum();
        let sb: u32 = pt[b].iter().copied().sum();
        sb.cmp(&sa).then_with(|| a.cmp(&b))
    });

    let mut candidates: Vec<Vec<usize>> = Vec::new();
    candidates.push(lpt.clone());
    let mut lpt_r = lpt.clone();
    lpt_r.reverse();
    candidates.push(lpt_r);

    let p = palmer_order(pt);
    if p.len() == num_jobs {
        candidates.push(p.clone());
        let mut pr = p;
        pr.reverse();
        candidates.push(pr);
    }

    let g = gupta_order(pt);
    if g.len() == num_jobs {
        candidates.push(g);
    }

    let rz = rz_order(pt);
    if rz.len() == num_jobs {
        candidates.push(rz);
    }

    for o in cds_orders(pt) {
        if o.len() == num_jobs {
            candidates.push(o);
        }
    }

    let unique = route_is_unique(route, num_machines);
    let mut best_seq: Vec<usize> = (0..num_jobs).collect();
    let mut best_mk: u32 = u32::MAX;

    if unique {
        TL_TAILLARD.with(|cell| {
            let mut buf = cell.borrow_mut();
            let m = ops;

            for (idx, ord) in candidates.iter().enumerate() {
                if ord.len() != num_jobs {
                    continue;
                }

                let mut seq = neh_build_seq(ord, route, pt, num_machines);
                improve_perm_seq_taillard(&mut seq, pt, 8, &mut buf);
                let _ = perm_vnd_improve(&mut seq, pt, 14, &mut buf);
                buf.ensure(seq.len(), m);
                let mk = flow_makespan(&seq, pt, &mut buf.comp[..m]);
                if mk < best_mk {
                    best_mk = mk;
                    best_seq = seq;
                }

                if idx < 4 {
                    let mut seq2 = neh_build_seq_beam_unique(ord, pt, m, 6, 3);
                    improve_perm_seq_taillard(&mut seq2, pt, 6, &mut buf);
                    let _ = perm_vnd_improve(&mut seq2, pt, 12, &mut buf);
                    buf.ensure(seq2.len(), m);
                    let mk2 = flow_makespan(&seq2, pt, &mut buf.comp[..m]);
                    if mk2 < best_mk {
                        best_mk = mk2;
                        best_seq = seq2;
                    }
                }
            }

            if !best_seq.is_empty() {
                let _ = adjacent_swap_hillclimb(&mut best_seq, pt, 1, &mut buf);
                let _ = perm_vnd_improve(&mut best_seq, pt, 12, &mut buf);
            }
        });
        return Ok(best_seq);
    }

    let mut mready = vec![0u32; num_machines];
    for ord in candidates.iter() {
        if ord.len() != num_jobs {
            continue;
        }
        let mut seq = neh_build_seq(ord, route, pt, num_machines);
        fs_improve_reentrant_seq(&mut seq, route, pt, num_machines);
        let mk = reentrant_makespan(&seq, route, pt, &mut mready);
        if mk < best_mk {
            best_mk = mk;
            best_seq = seq;
        }
    }
    Ok(best_seq)
}

fn neh_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<Solution> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let best_seq = neh_best_sequence(pre, num_jobs, num_machines)?;
    Ok(build_perm_solution_from_seq(
        &best_seq,
        route,
        pt,
        num_jobs,
        num_machines,
    ))
}

fn iterated_greedy_search(
    init: &[usize],
    pt: &[Vec<u32>],
    iters: usize,
    d: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = init.len();
    if n <= 2 {
        return init.to_vec();
    }
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if m == 0 {
        return init.to_vec();
    }

    let mut totals: Vec<u32> = Vec::with_capacity(pt.len());
    for row in pt {
        totals.push(row.iter().copied().sum());
    }

    let mut buf = TailInsBuf::default();
    buf.ensure(n, m);

    let mut cur = init.to_vec();
    let mut best = cur.clone();

    let mut cur_mk = flow_makespan(&cur, pt, &mut buf.comp[..m]);
    let mut best_mk = cur_mk;

    let mut temp = (cur_mk as f64) * 0.07 + 1.0;

    let dd = d.clamp(3, n.saturating_sub(1));
    let its = iters.max(1);

    let mut idxs: Vec<usize> = Vec::with_capacity(dd);
    let mut removed: Vec<usize> = Vec::with_capacity(dd);

    let mut no_improve = 0usize;

    for it in 0..its {
        idxs.clear();
        if dd < n && (rng.gen::<u32>() & 3) != 0 {
            let start = rng.gen_range(0..=(n - dd));
            for x in start..(start + dd) {
                idxs.push(x);
            }
        } else {
            while idxs.len() < dd {
                let x = rng.gen_range(0..n);
                if !idxs.iter().any(|&y| y == x) {
                    idxs.push(x);
                }
            }
            idxs.sort_unstable();
        }

        removed.clear();
        let mut partial = cur.clone();
        for &ix in idxs.iter().rev() {
            if ix < partial.len() {
                removed.push(partial.remove(ix));
            }
        }

        removed.sort_unstable_by(|a, b| {
            let ta = totals.get(*a).copied().unwrap_or(0);
            let tb = totals.get(*b).copied().unwrap_or(0);
            tb.cmp(&ta).then_with(|| a.cmp(b))
        });

        for &j in &removed {
            let (pos, _mk) = taillard_best_insert_pos(&partial, j, pt, m, &mut buf);
            partial.insert(pos, j);
        }

        let mut cand = partial;

        if (it & 7) == 0 && n >= 2 {
            let i = rng.gen_range(0..(n - 1));
            cand.swap(i, i + 1);
        }

        improve_perm_seq_taillard(&mut cand, pt, 3, &mut buf);
        let cur_mk_for_swap = flow_makespan(&cand, pt, &mut buf.comp[..m]);
        let _ = best_pair_swap_window_once(&mut cand, pt, m, cur_mk_for_swap, 8, &mut buf);

        let cand_mk = flow_makespan(&cand, pt, &mut buf.comp[..m]);

        if cand_mk < best_mk {
            best_mk = cand_mk;
            best = cand.clone();
            no_improve = 0;
        } else {
            no_improve = no_improve.saturating_add(1);
        }

        if cand_mk <= cur_mk {
            cur = cand;
            cur_mk = cand_mk;
        } else {
            let delta = (cand_mk - cur_mk) as f64;
            let prob = (-delta / temp).exp();
            if rng.gen::<f64>() < prob {
                cur = cand;
                cur_mk = cand_mk;
            }
        }

        if no_improve > 220 {
            cur = best.clone();
            cur_mk = best_mk;
            temp = (best_mk as f64) * 0.045 + 1.0;
            no_improve = 0;
        }

        temp = (temp * 0.9982).max(1.0);
    }

    best
}

fn strict_compute_next_time(
    m: usize,
    machine_avail: &[u32],
    future: &[BinaryHeap<Reverse<(u32, usize, usize)>>],
    avail: &[BinaryHeap<Reverse<(usize, usize)>>],
) -> Option<u32> {
    if !avail[m].is_empty() {
        return Some(machine_avail[m]);
    }
    if let Some(Reverse((release, _, _))) = future[m].peek().copied() {
        return Some(machine_avail[m].max(release));
    }
    None
}

#[derive(Default)]
struct StrictBuf {
    job_next_op: Vec<usize>,
    job_ready: Vec<u32>,
    machine_avail: Vec<u32>,
    future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>,
    avail: Vec<BinaryHeap<Reverse<(usize, usize)>>>,
    next_time: Vec<Option<u32>>,
    machine_events: BinaryHeap<Reverse<(u32, usize)>>,
}

impl StrictBuf {
    fn ensure(&mut self, num_jobs: usize, num_machines: usize) {
        if self.job_next_op.len() < num_jobs {
            self.job_next_op.resize(num_jobs, 0);
        }
        if self.job_ready.len() < num_jobs {
            self.job_ready.resize(num_jobs, 0);
        }
        if self.machine_avail.len() < num_machines {
            self.machine_avail.resize(num_machines, 0);
        }
        if self.future.len() < num_machines {
            while self.future.len() < num_machines {
                self.future.push(BinaryHeap::new());
            }
        } else if self.future.len() > num_machines {
            self.future.truncate(num_machines);
        }
        if self.avail.len() < num_machines {
            while self.avail.len() < num_machines {
                self.avail.push(BinaryHeap::new());
            }
        } else if self.avail.len() > num_machines {
            self.avail.truncate(num_machines);
        }
        if self.next_time.len() < num_machines {
            self.next_time.resize(num_machines, None);
        }
    }
}

thread_local! {
    static TL_STRICT: RefCell<StrictBuf> = RefCell::new(StrictBuf::default());
}

fn strict_makespan(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<u32> {
    let route = pre
        .flow_route
        .as_ref()
        .ok_or_else(|| anyhow!("flow_route missing"))?;
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let pt_by_job = pre.flow_pt_by_job.as_ref();

    TL_STRICT.with(|cell| {
        let mut sb = cell.borrow_mut();
        sb.ensure(num_jobs, num_machines);

        for x in sb.job_next_op.iter_mut().take(num_jobs) {
            *x = 0;
        }
        for x in sb.job_ready.iter_mut().take(num_jobs) {
            *x = 0;
        }
        for x in sb.machine_avail.iter_mut().take(num_machines) {
            *x = 0;
        }
        for m in 0..num_machines {
            sb.future[m].clear();
            sb.avail[m].clear();
            sb.next_time[m] = None;
        }
        sb.machine_events.clear();

        let mut remaining_ops = pre.total_ops;

        for job in 0..num_jobs {
            if pre.job_ops_len[job] == 0 {
                continue;
            }
            let m = route[0];
            sb.future[m].push(Reverse((0u32, rank[job], job)));
        }

        for m in 0..num_machines {
            let t = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
            sb.next_time[m] = t;
            if let Some(tt) = t {
                sb.machine_events.push(Reverse((tt, m)));
            }
        }

        let mut makespan = 0u32;

        while remaining_ops > 0 {
            let Reverse((t, m)) = sb
                .machine_events
                .pop()
                .ok_or_else(|| anyhow!("stalled in strict makespan"))?;

            if sb.next_time[m] != Some(t) {
                continue;
            }
            if sb.machine_avail[m] > t {
                continue;
            }

            while let Some(Reverse((release, _, _job))) = sb.future[m].peek().copied() {
                if release > t {
                    break;
                }
                let Reverse((_, pr, job)) = sb.future[m].pop().unwrap();
                sb.avail[m].push(Reverse((pr, job)));
            }

            let Some(Reverse((_, job))) = sb.avail[m].pop() else {
                let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m] = nt;
                if let Some(tt) = nt {
                    sb.machine_events.push(Reverse((tt, m)));
                }
                continue;
            };

            let op_idx = sb.job_next_op[job];
            if op_idx >= pre.job_ops_len[job] {
                return Err(anyhow!("job popped but already complete"));
            }
            if route[op_idx] != m {
                return Err(anyhow!("route mismatch in strict makespan"));
            }

            let start = t.max(sb.job_ready[job]).max(sb.machine_avail[m]);
            if start != t {
                sb.avail[m].push(Reverse((rank[job], job)));
                let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m] = nt;
                if let Some(tt) = nt {
                    sb.machine_events.push(Reverse((tt, m)));
                }
                continue;
            }

            let ptv = if let Some(pt) = pt_by_job {
                pt.get(job)
                    .and_then(|row| row.get(op_idx))
                    .copied()
                    .unwrap_or(0)
            } else {
                let product = pre.job_products[job];
                *challenge.product_processing_times[product][op_idx]
                    .get(&m)
                    .ok_or_else(|| anyhow!("missing pt in strict makespan"))?
            };

            let end = start.saturating_add(ptv);

            sb.job_next_op[job] += 1;
            sb.job_ready[job] = end;
            sb.machine_avail[m] = end;
            remaining_ops -= 1;
            makespan = makespan.max(end);

            if sb.job_next_op[job] < pre.job_ops_len[job] {
                let next_op = sb.job_next_op[job];
                let m2 = route[next_op];
                sb.future[m2].push(Reverse((end, rank[job], job)));

                let nt2 = strict_compute_next_time(m2, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m2] = nt2;
                if let Some(tt) = nt2 {
                    sb.machine_events.push(Reverse((tt, m2)));
                }
            }

            let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
            sb.next_time[m] = nt;
            if let Some(tt) = nt {
                sb.machine_events.push(Reverse((tt, m)));
            }
        }

        Ok(makespan)
    })
}

fn strict_simulate(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
    let route = pre
        .flow_route
        .as_ref()
        .ok_or_else(|| anyhow!("flow_route missing"))?;
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let pt_by_job = pre.flow_pt_by_job.as_ref();

    TL_STRICT.with(|cell| {
        let mut sb = cell.borrow_mut();
        sb.ensure(num_jobs, num_machines);

        for x in sb.job_next_op.iter_mut().take(num_jobs) {
            *x = 0;
        }
        for x in sb.job_ready.iter_mut().take(num_jobs) {
            *x = 0;
        }
        for x in sb.machine_avail.iter_mut().take(num_machines) {
            *x = 0;
        }
        for m in 0..num_machines {
            sb.future[m].clear();
            sb.avail[m].clear();
            sb.next_time[m] = None;
        }
        sb.machine_events.clear();

        let mut job_schedule: Vec<Vec<(usize, u32)>> = pre
            .job_ops_len
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect();

        let mut remaining_ops = pre.total_ops;

        for job in 0..num_jobs {
            if pre.job_ops_len[job] == 0 {
                continue;
            }
            let m = route[0];
            sb.future[m].push(Reverse((0u32, rank[job], job)));
        }

        for m in 0..num_machines {
            let t = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
            sb.next_time[m] = t;
            if let Some(tt) = t {
                sb.machine_events.push(Reverse((tt, m)));
            }
        }

        let mut makespan = 0u32;

        while remaining_ops > 0 {
            let Reverse((t, m)) = sb
                .machine_events
                .pop()
                .ok_or_else(|| anyhow!("stalled in strict simulate"))?;

            if sb.next_time[m] != Some(t) {
                continue;
            }
            if sb.machine_avail[m] > t {
                continue;
            }

            while let Some(Reverse((release, _, _job))) = sb.future[m].peek().copied() {
                if release > t {
                    break;
                }
                let Reverse((_, pr, job)) = sb.future[m].pop().unwrap();
                sb.avail[m].push(Reverse((pr, job)));
            }

            let Some(Reverse((_, job))) = sb.avail[m].pop() else {
                let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m] = nt;
                if let Some(tt) = nt {
                    sb.machine_events.push(Reverse((tt, m)));
                }
                continue;
            };

            let op_idx = sb.job_next_op[job];
            if op_idx >= pre.job_ops_len[job] {
                return Err(anyhow!("job popped but already complete"));
            }
            if route[op_idx] != m {
                return Err(anyhow!("route mismatch in strict simulate"));
            }

            let start = t.max(sb.job_ready[job]).max(sb.machine_avail[m]);
            if start != t {
                sb.avail[m].push(Reverse((rank[job], job)));
                let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m] = nt;
                if let Some(tt) = nt {
                    sb.machine_events.push(Reverse((tt, m)));
                }
                continue;
            }

            let ptv = if let Some(pt) = pt_by_job {
                pt.get(job)
                    .and_then(|row| row.get(op_idx))
                    .copied()
                    .unwrap_or(0)
            } else {
                let product = pre.job_products[job];
                *challenge.product_processing_times[product][op_idx]
                    .get(&m)
                    .ok_or_else(|| anyhow!("missing pt in strict simulate"))?
            };

            let end = start.saturating_add(ptv);

            job_schedule[job].push((m, start));
            sb.job_next_op[job] += 1;
            sb.job_ready[job] = end;
            sb.machine_avail[m] = end;
            remaining_ops -= 1;
            makespan = makespan.max(end);

            if sb.job_next_op[job] < pre.job_ops_len[job] {
                let next_op = sb.job_next_op[job];
                let m2 = route[next_op];
                sb.future[m2].push(Reverse((end, rank[job], job)));

                let nt2 = strict_compute_next_time(m2, &sb.machine_avail, &sb.future, &sb.avail);
                sb.next_time[m2] = nt2;
                if let Some(tt) = nt2 {
                    sb.machine_events.push(Reverse((tt, m2)));
                }
            }

            let nt = strict_compute_next_time(m, &sb.machine_avail, &sb.future, &sb.avail);
            sb.next_time[m] = nt;
            if let Some(tt) = nt {
                sb.machine_events.push(Reverse((tt, m)));
            }
        }

        Ok((Solution { job_schedule }, makespan))
    })
}

fn strict_best_by_order_search(
    challenge: &Challenge,
    pre: &Pre,
    passes: usize,
) -> Result<(Solution, u32)> {
    if pre.flow_route.is_none() || pre.flex_avg > 1.25 {
        return Err(anyhow!("not strict-like"));
    }
    let n = challenge.num_jobs;
    let route = pre.flow_route.as_ref().unwrap();
    let ops = route.len();
    if ops == 0 {
        return Err(anyhow!("empty route"));
    }

    let pt_stage: Vec<Vec<u32>> = if let Some(pt) = pre.flow_pt_by_job.as_ref() {
        pt.clone()
    } else {
        let mut tmp = vec![vec![0u32; pre.max_ops.max(1)]; n];
        for j in 0..n {
            let p = pre.job_products[j];
            let len = pre.job_ops_len[j];
            for k in 0..len {
                let op = &pre.product_ops[p][k];
                tmp[j][k] = op.machines.first().map(|x| x.1).unwrap_or(0);
            }
            tmp[j].truncate(len);
        }
        tmp
    };

    let mut cand_orders: Vec<Vec<usize>> = Vec::new();

    let mut lpt: Vec<usize> = (0..n).collect();
    lpt.sort_unstable_by(|&a, &b| {
        let sa: u32 = pt_stage[a].iter().copied().sum();
        let sb: u32 = pt_stage[b].iter().copied().sum();
        sb.cmp(&sa).then_with(|| a.cmp(&b))
    });
    cand_orders.push(lpt.clone());

    let mut lpt_r = lpt.clone();
    lpt_r.reverse();
    cand_orders.push(lpt_r);

    let mut spt: Vec<usize> = (0..n).collect();
    spt.sort_unstable_by(|&a, &b| {
        let sa: u32 = pt_stage[a].iter().copied().sum();
        let sb: u32 = pt_stage[b].iter().copied().sum();
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    cand_orders.push(spt);

    let p = palmer_order(&pt_stage);
    if p.len() == n {
        cand_orders.push(p.clone());
        let mut pr = p;
        pr.reverse();
        cand_orders.push(pr);
    }

    let g = gupta_order(&pt_stage);
    if g.len() == n {
        cand_orders.push(g);
    }

    let rz = rz_order(&pt_stage);
    if rz.len() == n {
        cand_orders.push(rz);
    }

    for o in cds_orders(&pt_stage) {
        if o.len() == n {
            cand_orders.push(o);
        }
    }

    if route_is_unique(route, challenge.num_machines) && ops == pt_stage[0].len() {
        let neh1 = neh_build_seq_beam_unique(&lpt, &pt_stage, ops, 6, 3);
        if neh1.len() == n {
            cand_orders.push(neh1.clone());
            let mut r = neh1;
            r.reverse();
            cand_orders.push(r);
        }
    }

    let mut seed = challenge.seed;
    seed[0] ^= 0x3C;
    let mut rng = SmallRng::from_seed(seed);
    for _ in 0..7usize {
        let mut r: Vec<usize> = (0..n).collect();
        r.shuffle(&mut rng);
        cand_orders.push(r);
    }

    let mut rank = vec![0usize; n];
    let mut best_mk = u32::MAX;
    let mut best_order: Vec<usize> = (0..n).collect();

    for ord in cand_orders.iter() {
        if ord.len() != n {
            continue;
        }
        for (pos, &j) in ord.iter().enumerate() {
            rank[j] = pos;
        }
        let mk = strict_makespan(challenge, pre, &rank)?;
        if mk < best_mk {
            best_mk = mk;
            best_order.clone_from(ord);
        }
    }

    let ls_passes = passes.max(2).min(10);
    let w = 12usize.min(n.saturating_sub(1));

    let mut cand_order = vec![0usize; n];

    for _ in 0..ls_passes {
        let mut improved = false;

        for i in 0..n {
            let job = best_order[i];
            let start = i.saturating_sub(w);
            let end = (i + w).min(n - 1);

            let mut best_pos = i;
            let mut best_local_mk = best_mk;

            for pos in start..=end {
                if pos == i {
                    continue;
                }
                if pos < i {
                    cand_order[..pos].copy_from_slice(&best_order[..pos]);
                    cand_order[pos] = job;
                    cand_order[pos + 1..=i].copy_from_slice(&best_order[pos..i]);
                    cand_order[i + 1..].copy_from_slice(&best_order[i + 1..]);
                } else {
                    cand_order[..i].copy_from_slice(&best_order[..i]);
                    cand_order[i..pos].copy_from_slice(&best_order[i + 1..=pos]);
                    cand_order[pos] = job;
                    cand_order[pos + 1..].copy_from_slice(&best_order[pos + 1..]);
                }

                for (p2, &jj) in cand_order.iter().enumerate() {
                    rank[jj] = p2;
                }
                let mk = strict_makespan(challenge, pre, &rank)?;
                if mk < best_local_mk {
                    best_local_mk = mk;
                    best_pos = pos;
                    if mk + 1 < best_mk {
                        break;
                    }
                }
            }

            if best_local_mk < best_mk {
                best_mk = best_local_mk;
                if best_pos < i {
                    best_order[best_pos..=i].rotate_right(1);
                } else if best_pos > i {
                    best_order[i..=best_pos].rotate_left(1);
                }
                improved = true;
            }
        }

        for (pos, &j) in best_order.iter().enumerate() {
            rank[j] = pos;
        }
        for i in 0..(n.saturating_sub(1)) {
            best_order.swap(i, i + 1);
            rank[best_order[i]] = i;
            rank[best_order[i + 1]] = i + 1;
            let mk = strict_makespan(challenge, pre, &rank)?;
            if mk < best_mk {
                best_mk = mk;
                improved = true;
            } else {
                best_order.swap(i, i + 1);
                rank[best_order[i]] = i;
                rank[best_order[i + 1]] = i + 1;
            }
        }

        if !improved {
            break;
        }
    }

    let mut seed2 = challenge.seed;
    seed2[0] ^= 0xA5;
    let mut rng2 = SmallRng::from_seed(seed2);

    let mut cur = best_order.clone();
    for (pos, &j) in cur.iter().enumerate() {
        rank[j] = pos;
    }
    let mut cur_mk = strict_makespan(challenge, pre, &rank)?;
    if cur_mk < best_mk {
        best_mk = cur_mk;
        best_order = cur.clone();
    }

    let iters = (900usize + ls_passes * 450).min(2800);
    let mut temp = (cur_mk as f64) * 0.03 + 1.0;

    for _ in 0..iters {
        let do_insert = (rng2.gen::<u32>() & 1) == 0;

        if do_insert {
            let from = rng2.gen_range(0..n);
            let mut to = rng2.gen_range(0..n);
            if to == from {
                to = (to + 1) % n;
            }
            let job = cur.remove(from);
            cur.insert(to, job);
        } else {
            let i = rng2.gen_range(0..n);
            let mut j = rng2.gen_range(0..n);
            if j == i {
                j = (j + 1) % n;
            }
            cur.swap(i, j);
        }

        for (pos, &j) in cur.iter().enumerate() {
            rank[j] = pos;
        }
        let mk = strict_makespan(challenge, pre, &rank)?;

        let accept = if mk <= cur_mk {
            true
        } else {
            let delta = (mk - cur_mk) as f64;
            rng2.gen::<f64>() < (-delta / temp).exp()
        };

        if accept {
            cur_mk = mk;
            if mk < best_mk {
                best_mk = mk;
                best_order = cur.clone();
            }
        } else {
            cur.clone_from(&best_order);
            cur_mk = best_mk;
        }

        temp = (temp * 0.9991).max(1.0);
    }

    for (pos, &j) in best_order.iter().enumerate() {
        rank[j] = pos;
    }
    let (best_sol, mk2) = strict_simulate(challenge, pre, &rank)?;
    let best_mk2 = if mk2 != best_mk { mk2 } else { best_mk };
    Ok((best_sol, best_mk2))
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    greedy_sol: Solution,
    greedy_mk: u32,
    effort: &EffortConfig,
) -> Result<()> {
    let mut best_sol = greedy_sol;
    let mut best_mk = greedy_mk;
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &best_sol, best_mk, 10);

    let mut strict_sol: Option<(Solution, u32)> = None;

    let strict_depth = (effort.flow_shop_iters / 650).max(4);
    if pre.flow_route.is_some() && pre.flex_avg <= 1.25 {
        if let Ok((sol, mk)) = strict_best_by_order_search(challenge, pre, strict_depth) {
            strict_sol = Some((sol.clone(), mk));
            if mk <= best_mk {
                best_mk = mk;
                best_sol = sol;
                save_solution(&best_sol)?;
                push_top_solutions(&mut top_solutions, &best_sol, best_mk, 10);
            } else {
                push_top_solutions(&mut top_solutions, &sol, mk, 10);
            }
        }
    }

    if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
        if let Ok(neh_seq) = neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines) {
            let perm_sol = build_perm_solution_from_seq(
                &neh_seq,
                route,
                pt,
                challenge.num_jobs,
                challenge.num_machines,
            );
            if let Ok(mk) = challenge.evaluate_makespan(&perm_sol) {
                if mk <= best_mk {
                    best_mk = mk;
                    best_sol = perm_sol.clone();
                    save_solution(&best_sol)?;
                }
                push_top_solutions(&mut top_solutions, &perm_sol, mk, 10);
            }

            let unique = route_is_unique(route, challenge.num_machines);
            if unique && !neh_seq.is_empty() && route.len() == pt[neh_seq[0]].len() {
                let mut starts: Vec<Vec<usize>> = Vec::new();
                starts.push(neh_seq.clone());
                if let Some((s, _)) = &strict_sol {
                    starts.push(order_from_solution_first_op_start(s, challenge.num_jobs));
                }
                starts.push(order_from_solution_first_op_start(&best_sol, challenge.num_jobs));

                let mut uniq: Vec<Vec<usize>> = Vec::new();
                for ord in starts {
                    if ord.len() != challenge.num_jobs {
                        continue;
                    }
                    let mut ok = true;
                    for u in &uniq {
                        if *u == ord {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        uniq.push(ord);
                    }
                    if uniq.len() >= 4 {
                        break;
                    }
                }

                let mut seed = challenge.seed;
                seed[0] ^= 0x6B;
                let mut rng = SmallRng::from_seed(seed);

                let num_restarts = (effort.flow_shop_iters / 2600).max(1);
                let iters_per_restart = effort.flow_shop_iters / num_restarts;
                let per = (iters_per_restart / uniq.len().max(1)).max(650);
                let d = (iters_per_restart / 370).clamp(7, 15);

                let mut best_ig_seq = uniq.first().cloned().unwrap_or_else(|| neh_seq.clone());
                TL_TAILLARD.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    let m = route.len();

                    buf.ensure(best_ig_seq.len(), m);
                    let mut best_ig_mk = flow_makespan(&best_ig_seq, pt, &mut buf.comp[..m]);

                    let taillard_passes = (iters_per_restart / 650).max(4);
                    let vnd_passes = (iters_per_restart / 185).max(14);

                    for start_seq in uniq.iter() {
                        let mut cand_seq = iterated_greedy_search(start_seq, pt, per, d, &mut rng);
                        improve_perm_seq_taillard(&mut cand_seq, pt, taillard_passes, &mut buf);
                        let _ = perm_vnd_improve(&mut cand_seq, pt, vnd_passes, &mut buf);
                        buf.ensure(cand_seq.len(), m);
                        let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                        if mk < best_ig_mk {
                            best_ig_mk = mk;
                            best_ig_seq = cand_seq.clone();
                        }
                    }

                    for _restart in 1..num_restarts {
                        let mut perturbed = best_ig_seq.clone();
                        let n = perturbed.len();
                        let num_swaps = (n / 8).max(2).min(n / 2);
                        for _ in 0..num_swaps {
                            let i = rng.gen_range(0..n);
                            let j = rng.gen_range(0..n);
                            perturbed.swap(i, j);
                        }

                        let mut cand_seq = iterated_greedy_search(&perturbed, pt, per, d, &mut rng);
                        improve_perm_seq_taillard(&mut cand_seq, pt, taillard_passes, &mut buf);
                        let _ = perm_vnd_improve(&mut cand_seq, pt, vnd_passes, &mut buf);
                        buf.ensure(cand_seq.len(), m);
                        let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                        if mk < best_ig_mk {
                            best_ig_mk = mk;
                            best_ig_seq = cand_seq;
                        }
                    }

                    let final_taillard = (effort.flow_shop_iters / 430).max(6);
                    let swap_passes = (effort.flow_shop_iters / 2600).max(1);
                    improve_perm_seq_taillard(&mut best_ig_seq, pt, final_taillard, &mut buf);
                    let _ = perm_vnd_improve(&mut best_ig_seq, pt, vnd_passes, &mut buf);
                    let _ = adjacent_swap_hillclimb(&mut best_ig_seq, pt, swap_passes, &mut buf);
                });

                let ig_perm_sol = build_perm_solution_from_seq(
                    &best_ig_seq,
                    route,
                    pt,
                    challenge.num_jobs,
                    challenge.num_machines,
                );
                if let Ok(mk) = challenge.evaluate_makespan(&ig_perm_sol) {
                    if mk <= best_mk {
                        best_mk = mk;
                        best_sol = ig_perm_sol.clone();
                        save_solution(&best_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &ig_perm_sol, mk, 10);
                }
            }
        } else if let Ok(sol) = neh_solution(pre, challenge.num_jobs, challenge.num_machines) {
            if let Ok(mk) = challenge.evaluate_makespan(&sol) {
                if mk <= best_mk {
                    best_mk = mk;
                    best_sol = sol.clone();
                    save_solution(&best_sol)?;
                }
                push_top_solutions(&mut top_solutions, &sol, mk, 10);
            }
        }
    }

    let ls_runs = (effort.flow_shop_iters / 850).clamp(3, 12).min(top_solutions.len());
    let ls_iters = (effort.flow_shop_iters / 160).max(16);
    let ls_cands = (effort.flow_shop_iters / 85).max(30);
    for i in 0..ls_runs {
        let base_sol = &top_solutions[i].0;
        if let Ok(Some((sol2, _mk2))) =
            critical_block_move_local_search(pre, challenge, base_sol, ls_iters, ls_cands)
        {
            if let Ok(mk_eval) = challenge.evaluate_makespan(&sol2) {
                if mk_eval <= best_mk {
                    best_mk = mk_eval;
                    best_sol = sol2.clone();
                    save_solution(&best_sol)?;
                }
                push_top_solutions(&mut top_solutions, &sol2, mk_eval, 10);
            }
        }
    }

    Ok(())
}
