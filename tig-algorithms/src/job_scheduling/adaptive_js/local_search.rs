use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::helpers::*;

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs {
        job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j];
    }
    let n = job_offsets[num_jobs];
    if n == 0 {
        return Err(anyhow!("No operations"));
    }

    let mut node_machine = vec![0usize; n];
    let mut node_pt = vec![0u32; n];
    let mut node_job = vec![0usize; n];
    let mut node_op = vec![0usize; n];

    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected {
            return Err(anyhow!("Invalid solution: job {} ops len mismatch", job));
        }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx;
            let (m, st) = sol.job_schedule[job][op_idx];
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
            if m >= num_machines {
                return Err(anyhow!("Invalid solution: machine out of range"));
            }
            node_machine[id] = m;
            node_pt[id] = pt;
            node_job[id] = job;
            node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }

    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut seq = Vec::with_capacity(per_machine[m].len());
        for &(_st, id) in &per_machine[m] {
            seq.push(id);
        }
        machine_seq.push(seq);
    }

    let mut job_succ = vec![NONE_USIZE; n];
    let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job];
        let base = job_offsets[job];
        for k in 0..len {
            let id = base + k;
            if k + 1 < len {
                job_succ[id] = id + 1;
                indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1);
            }
        }
    }

    Ok(DisjSchedule {
        n,
        num_jobs,
        num_machines,
        job_offsets,
        job_succ,
        indeg_job,
        node_machine,
        node_pt,
        node_job,
        node_op,
        machine_seq,
    })
}

pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;

    buf.indeg.clone_from_slice(&ds.indeg_job);
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.machine_succ.fill(NONE_USIZE);
    buf.stack.clear();

    for seq in &ds.machine_seq {
        if seq.len() <= 1 {
            continue;
        }
        for i in 0..(seq.len() - 1) {
            let u = seq[i];
            let v = seq[i + 1];
            buf.machine_succ[u] = v;
            buf.indeg[v] = buf.indeg[v].saturating_add(1);
        }
    }

    for i in 0..n {
        if buf.indeg[i] == 0 {
            buf.stack.push(i);
        }
    }

    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;

    while let Some(u) = buf.stack.pop() {
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk {
            mk = end_u;
            mk_node = u;
        }

        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] = buf.indeg[js].saturating_sub(1);
            if buf.indeg[js] == 0 {
                buf.stack.push(js);
            }
        }

        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
            if buf.indeg[ms] == 0 {
                buf.stack.push(ms);
            }
        }
    }

    if processed != n {
        return None;
    }
    Some((mk, mk_node))
}

#[inline]
pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    if seq.is_empty() || from >= seq.len() {
        return from.min(seq.len().saturating_sub(1));
    }
    let x = seq.remove(from);
    let t = to_after_removal.min(seq.len());
    seq.insert(t, x);
    t
}

#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() {
        return false;
    }
    seq.swap(i, i + 1);
    true
}

#[inline]
pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
    for (i, &id) in seq.iter().enumerate() {
        if start[id] >= desired_start {
            return i;
        }
    }
    seq.len()
}

#[inline]
pub fn apply_reroute(
    ds: &mut DisjSchedule,
    m_from: usize,
    idx_from: usize,
    m_to: usize,
    idx_to: usize,
    new_pt: u32,
) -> Option<(usize, u32, usize)> {
    if m_from >= ds.num_machines || m_to >= ds.num_machines {
        return None;
    }
    if idx_from >= ds.machine_seq[m_from].len() {
        return None;
    }
    let node = ds.machine_seq[m_from].remove(idx_from);
    let old_pt = ds.node_pt[node];
    ds.node_machine[node] = m_to;
    ds.node_pt[node] = new_pt;
    let ins = idx_to.min(ds.machine_seq[m_to].len());
    ds.machine_seq[m_to].insert(ins, node);
    Some((node, old_pt, ins))
}

#[inline]
pub fn undo_reroute(
    ds: &mut DisjSchedule,
    m_from: usize,
    idx_from: usize,
    m_to: usize,
    ins_idx: usize,
    node: usize,
    old_pt: u32,
) -> bool {
    if m_from >= ds.num_machines || m_to >= ds.num_machines {
        return false;
    }
    if ins_idx >= ds.machine_seq[m_to].len() {
        return false;
    }
    let x = ds.machine_seq[m_to].remove(ins_idx);
    if x != node {
        let len_now = ds.machine_seq[m_to].len();
        let back_pos = ins_idx.min(len_now);
        ds.machine_seq[m_to].insert(back_pos, x);
        return false;
    }
    let ins_back = idx_from.min(ds.machine_seq[m_from].len());
    ds.machine_seq[m_from].insert(ins_back, node);
    ds.node_machine[node] = m_from;
    ds.node_pt[node] = old_pt;
    true
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j];
        let mut v = Vec::with_capacity(len);
        let base = ds.job_offsets[j];
        for k in 0..len {
            let id = base + k;
            v.push((ds.node_machine[id], start[id]));
        }
        job_schedule.push(v);
    }
    Ok(Solution { job_schedule })
}

fn descent_phase(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    crit: &mut Vec<bool>,
    pre: &Pre,
    cur_eval: &mut (u32, usize),
    max_iters: usize,
    top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0;
    let mut improved = false;

    for _iter in 0..max_iters {
        crit.fill(false);
        let mut u = cur_eval.1;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));

        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.len() <= 1 {
                continue;
            }

            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i];
                if !crit[a] {
                    i += 1;
                    continue;
                }

                let bstart = i;
                let mut bend = i;
                while bend + 1 < seq.len() {
                    let x = seq[bend];
                    let y = seq[bend + 1];
                    if !crit[y] {
                        break;
                    }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }

                if bend > bstart {
                    let max_shift = bend - bstart;
                    let mut shifts: [usize; 3] = [1, 2, max_shift];
                    for sh in shifts.iter_mut() {
                        if *sh > max_shift {
                            *sh = 0;
                        }
                    }

                    for &sh in &shifts {
                        if sh == 0 {
                            continue;
                        }

                        {
                            let from = bstart;
                            let to_after = bstart + sh;
                            if from < seq.len() && to_after <= seq.len() {
                                let tgt_idx = (bstart + sh).min(seq.len() - 1);
                                let score = buf.start[seq[tgt_idx]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score },
                                    top_cands,
                                );
                            }
                        }
                        {
                            let from = bend;
                            let to_after = bend - sh;
                            let score = buf.start[seq[bend]];
                            push_top_k_move(
                                &mut cands,
                                MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score },
                                top_cands,
                            );
                        }
                    }

                    {
                        if bstart > 0 {
                            let score = buf.start[seq[bstart]];
                            push_top_k_move(
                                &mut cands,
                                MoveCand { kind: 2, m_from: m, from: bstart - 1, m_to: m, to: 0, new_pt: 0, score },
                                top_cands,
                            );
                        }
                        if bend + 1 < seq.len() {
                            let score = buf.start[seq[bend]];
                            push_top_k_move(
                                &mut cands,
                                MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score },
                                top_cands,
                            );
                        }
                        if bstart + 1 <= bend {
                            let score = buf.start[seq[bstart + 1]];
                            push_top_k_move(
                                &mut cands,
                                MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score },
                                top_cands,
                            );
                            if bend >= 1 && bend - 1 >= bstart {
                                let score2 = buf.start[seq[bend]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 2, m_from: m, from: bend - 1, m_to: m, to: 0, new_pt: 0, score: score2 },
                                    top_cands,
                                );
                            }
                        }
                    }

                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() {
                            continue;
                        }
                        let node = seq[idx];
                        if !crit[node] {
                            continue;
                        }

                        let job = ds.node_job[node];
                        let op_idx = ds.node_op[node];
                        let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];

                        if op.flex < 2 || op.machines.len() < 2 {
                            continue;
                        }

                        let old_m = ds.node_machine[node];
                        let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);

                        let best2 = best_two_by_pt(op);
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF {
                                continue;
                            }
                            let w_to = pre.machine_weight[m_to].max(1e-9);

                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) {
                                continue;
                            }

                            let desired = buf.start[node];
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() {
                                    continue;
                                }

                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired
                                    .saturating_add(old_pt)
                                    .saturating_add(diffw)
                                    .saturating_add(difpt.saturating_mul(2));

                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score },
                                    top_cands,
                                );
                            }
                        }
                    }
                }

                i = bend + 1;
            }
        }

        if cands.is_empty() {
            break;
        }

        let mut best_cand: Option<MoveCand> = None;
        let mut best_mk = cur_mk;

        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from;
                if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                    continue;
                }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) {
                    if mk2 < best_mk {
                        best_mk = mk2;
                        best_cand = Some(*cand);
                    }
                }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from;
                if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() {
                    continue;
                }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) {
                    continue;
                }
                if let Some((mk2, _)) = eval_disj(ds, buf) {
                    if mk2 < best_mk {
                        best_mk = mk2;
                        best_cand = Some(*cand);
                    }
                }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from;
                let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines {
                    continue;
                }
                if cand.from >= ds.machine_seq[m_from].len() {
                    continue;
                }
                let node = ds.machine_seq[m_from][cand.from];
                if ds.node_machine[node] != m_from {
                    continue;
                }

                let applied = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt);
                if let Some((node2, old_pt, ins_idx)) = applied {
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }

        let Some(bc) = best_cand else { break };

        let mut accepted = false;

        if bc.kind == 0 {
            let m = bc.m_from;
            let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(next_eval) = eval_disj(ds, buf) {
                if next_eval.0 < cur_mk {
                    *cur_eval = next_eval;
                    cur_mk = cur_eval.0;
                    improved = true;
                    accepted = true;
                } else {
                    let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from);
                }
            } else {
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from);
            }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(next_eval) = eval_disj(ds, buf) {
                        if next_eval.0 < cur_mk {
                            *cur_eval = next_eval;
                            cur_mk = cur_eval.0;
                            improved = true;
                            accepted = true;
                        } else {
                            let _ = apply_swap(&mut ds.machine_seq[m], bc.from);
                        }
                    } else {
                        let _ = apply_swap(&mut ds.machine_seq[m], bc.from);
                    }
                }
            }
        } else {
            let applied = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt);
            if let Some((node2, old_pt, ins_idx)) = applied {
                if let Some(next_eval) = eval_disj(ds, buf) {
                    if next_eval.0 < cur_mk {
                        *cur_eval = next_eval;
                        cur_mk = cur_eval.0;
                        improved = true;
                        accepted = true;
                    } else {
                        let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt);
                    }
                } else {
                    let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt);
                }
            }
        }

        if !accepted {
            break;
        }
    }

    improved
}

pub fn critical_block_move_local_search(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iters: usize,
    top_cands: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let mut crit = vec![false; ds.n];

    let mut cur_eval = match eval_disj(&ds, &mut buf) {
        Some(x) => x,
        None => return Ok(None),
    };
    let initial_mk = cur_eval.0;

    descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);

    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };

    let mut global_best_mk = mk_after;
    let mut global_best_ds = ds.clone();

    let perturb_cycles = 3usize;

    // Add solution-specific entropy from machine assignments to avoid identical
    // perturbation sequences for different solutions with the same makespan
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }

    let mut pseed: u64 = (challenge.seed[0] as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (ds.n as u64)
        ^ sol_hash;

    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.len() <= 1 {
                continue;
            }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < seq.len() {
                    let x = seq[bend];
                    let y = seq[bend + 1];
                    if !crit[y] {
                        break;
                    }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    blocks.push((m, bstart, bend));
                }
                i = bend + 1;
            }
        }

        if blocks.is_empty() {
            break;
        }

        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len();
            let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart;
            if block_len == 0 {
                continue;
            }
            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() {
                ds.machine_seq[m].swap(swap_pos, swap_pos + 1);
            }
        }

        match eval_disj(&ds, &mut buf) {
            Some(x) => cur_eval = x,
            None => continue,
        }

        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);

        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) {
            if mk_now < global_best_mk {
                global_best_mk = mk_now;
                global_best_ds = ds.clone();
            }
        }
    }

    if global_best_mk >= initial_mk {
        return Ok(None);
    }

    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}
