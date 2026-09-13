use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>(); let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_greedy_rule(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let priority = match rule {
                        GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                        GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                        GreedyRule::LongestProc => pt as f64,
                    };
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let cand = GCandidate {
                        job: j,
                        priority: match rule {
                            GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                            GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                            GreedyRule::LongestProc => pt as f64,
                        },
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let n = job_offsets[num_jobs];
    if n == 0 { return Err(anyhow!("No operations")); }
    let mut node_machine = vec![0usize; n]; let mut node_pt = vec![0u32; n]; let mut node_job = vec![0usize; n]; let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution: job {} ops len mismatch", job)); }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx; let (m, st) = sol.job_schedule[job][op_idx];
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
            if m >= num_machines { return Err(anyhow!("Invalid solution: machine out of range")); }
            node_machine[id] = m; node_pt[id] = pt; node_job[id] = job; node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        machine_seq.push(per_machine[m].iter().map(|&(_, id)| id).collect());
    }
    let mut job_succ = vec![NONE_USIZE; n]; let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job]; let base = job_offsets[job];
        for k in 0..len { let id = base + k; if k + 1 < len { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
    }
    Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, node_machine, node_pt, node_job, node_op, machine_seq })
}

#[inline]
pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
    for &(m, pt) in &op.machines { if m == machine { return Some(pt); } }
    None
}

pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.indeg.clone_from_slice(&ds.indeg_job);
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.machine_succ.fill(NONE_USIZE);
    buf.stack.clear();
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            buf.indeg[v] = buf.indeg[v].saturating_add(1);
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
    for i in 0..n {
        if buf.indeg[i] == 0 { buf.stack.push(i); }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while let Some(u) = buf.stack.pop() {
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] = buf.indeg[js].saturating_sub(1);
            if buf.indeg[js] == 0 { buf.stack.push(js); }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
            if buf.indeg[ms] == 0 { buf.stack.push(ms); }
        }
    }
    if processed != n { return None; }
    Some((mk, mk_node))
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j]; let mut v = Vec::with_capacity(len); let base = ds.job_offsets[j];
        for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
        job_schedule.push(v);
    }
    Ok(Solution { job_schedule })
}

pub fn critical_block_move_local_search_ex(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(swap_pos, swap_pos + 1); }
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

fn descent_phase(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    let max_shift = bend - bstart;
                    let mut shifts: [usize; 3] = [1, 2, max_shift];
                    for sh in shifts.iter_mut() { if *sh > max_shift { *sh = 0; } }
                    for &sh in &shifts {
                        if sh == 0 { continue; }
                        { let from = bstart; let to_after = bstart + sh; if from < seq.len() && to_after <= seq.len() { let tgt_idx = (bstart+sh).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); } }
                        { let from = bend; let to_after = bend - sh; push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let best2 = best_two_by_pt(op);
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let desired = buf.start[node];
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

#[inline]
pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len { return from.min(len.saturating_sub(1)); }
    let t = to_after_removal.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() { return false; } seq.swap(i, i + 1); true
}

#[inline]
pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = seq.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if start[seq[mid]] < desired_start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
pub fn apply_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, idx_to: usize, new_pt: u32) -> Option<(usize, u32, usize)> {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || idx_from >= ds.machine_seq[m_from].len() { return None; }
    let node = ds.machine_seq[m_from].remove(idx_from); let old_pt = ds.node_pt[node];
    ds.node_machine[node] = m_to; ds.node_pt[node] = new_pt;
    let ins = idx_to.min(ds.machine_seq[m_to].len()); ds.machine_seq[m_to].insert(ins, node);
    Some((node, old_pt, ins))
}

#[inline]
pub fn undo_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, ins_idx: usize, node: usize, old_pt: u32) -> bool {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || ins_idx >= ds.machine_seq[m_to].len() { return false; }
    let x = ds.machine_seq[m_to].remove(ins_idx);
    if x != node { let len_now = ds.machine_seq[m_to].len(); ds.machine_seq[m_to].insert(ins_idx.min(len_now), x); return false; }
    let ins_from = idx_from.min(ds.machine_seq[m_from].len());
    ds.machine_seq[m_from].insert(ins_from, node);
    ds.node_machine[node] = m_from; ds.node_pt[node] = old_pt; true
}

#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut r = [(NONE_USIZE, INF); 2];
    for &(m, pt) in &op.machines {
        if pt < r[0].1 { r[1] = r[0]; r[0] = (m, pt); }
        else if pt < r[1].1 { r[1] = (m, pt); }
    }
    r
}

pub fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in bias calc"))?;
            end_j = end_j.max(st.saturating_add(pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }

    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 1.2 * pre.high_flex + 0.6 * pre.jobshopness;
    Ok(completion.into_iter().map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0)).collect())
}

pub fn machine_penalty_from_solution(pre: &Pre, sol: &Solution, num_machines: usize) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in machine penalty"))?;
            let end = st.saturating_add(pt);
            if end > m_end[m] { m_end[m] = end; }
            m_sum[m] = m_sum[m].saturating_add(pt as u64);
            makespan = makespan.max(end);
        }
    }

    let mk = (makespan as f64).max(1.0);
    let total: u64 = m_sum.iter().copied().sum();
    let avg = ((total as f64) / (num_machines as f64).max(1.0)).max(1.0);

    let use_load = pre.high_flex > 0.35 || pre.jobshopness > 0.45;
    let w_load = if use_load {
        (0.20 + 0.30 * pre.high_flex + 0.12 * pre.jobshopness).clamp(0.18, 0.58)
    } else {
        0.0
    };
    let w_end = 1.0 - w_load;
    let exp = 2.0 + 1.2 * pre.high_flex + 0.55 * pre.jobshopness;

    let mut mp = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        let endn = (m_end[m] as f64 / mk).clamp(0.0, 1.0);
        let loadr = ((m_sum[m] as f64) / avg).max(0.0);
        let loadn = (loadr / (loadr + 1.0)).clamp(0.0, 1.0);
        let mix = (w_end * endn + w_load * loadn).clamp(0.0, 1.0);
        mp[m] = mix.powf(exp).clamp(0.0, 1.0);
    }
    Ok(mp)
}

pub fn route_pref_from_solution_lite(pre: &Pre, sol: &Solution, challenge: &Challenge) -> Result<RoutePrefLite> {
    let nm = challenge.num_machines;
    let np = challenge.product_processing_times.len();

    let mut counts: Vec<Vec<u16>> = Vec::with_capacity(np);
    let mut ops_len: Vec<usize> = Vec::with_capacity(np);
    for p in 0..np {
        let ol = challenge.product_processing_times[p].len();
        ops_len.push(ol);
        counts.push(vec![0u16; ol.saturating_mul(nm)]);
    }

    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        let ol = ops_len[product];
        for (op_idx, &(m, _st)) in sol.job_schedule[job].iter().enumerate() {
            if op_idx >= ol || m >= nm { continue; }
            let idx = op_idx * nm + m;
            counts[product][idx] = counts[product][idx].saturating_add(1);
        }
    }

    let mut rp: RoutePrefLite = Vec::with_capacity(np);
    for p in 0..np {
        let ol = ops_len[p];
        let denom_u32 = (challenge.jobs_per_product[p].max(1) as u32).max(1);
        let mut v: Vec<OpRoute> = Vec::with_capacity(ol);

        for op_idx in 0..ol {
            let base = op_idx * nm;
            let mut best_m = 0usize;
            let mut best_c = 0u16;
            let mut second_m = 0usize;
            let mut second_c = 0u16;

            for m in 0..nm {
                let c = counts[p][base + m];
                if c > best_c {
                    second_c = best_c; second_m = best_m;
                    best_c = c; best_m = m;
                } else if c > second_c && m != best_m {
                    second_c = c; second_m = m;
                }
            }

            let best_w = (((best_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;
            let second_w = (((second_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;

            v.push(OpRoute { best_m: best_m.min(255) as u8, best_w, second_m: second_m.min(255) as u8, second_w });
        }

        rp.push(v);
    }

    Ok(rp)
}

pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
}

pub fn neh_reentrant_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let ops = route.len(); if ops == 0 || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
    let mut jobs: Vec<usize> = (0..num_jobs).collect();
    jobs.sort_unstable_by(|&a, &b| { let sa: u32 = pt[a].iter().copied().sum(); let sb: u32 = pt[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) });
    let mut seq: Vec<usize> = Vec::with_capacity(num_jobs); let mut mready = vec![0u32; num_machines];
    for &j in &jobs {
        if seq.is_empty() { seq.push(j); continue; }
        let mut best_mk = u32::MAX; let mut best_pos = 0usize; let mut tmp = seq.clone();
        for pos in 0..=seq.len() {
            tmp.clear(); tmp.extend_from_slice(&seq[..pos]); tmp.push(j); tmp.extend_from_slice(&seq[pos..]);
            let mk = reentrant_makespan_local(&tmp, route, pt, &mut mready);
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        seq.insert(best_pos, j);
    }
    let mk = reentrant_makespan_local(&seq, route, pt, &mut mready);
    let sol = build_perm_solution_from_seq_local(&seq, route, pt, num_jobs, num_machines);
    Ok((sol, mk))
}

fn reentrant_makespan_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0); let mut mk = 0u32;
    for &j in seq { let row = &pt[j]; let mut prev = 0u32; for (op_idx, &m) in route.iter().enumerate() { let p = row[op_idx]; let st = prev.max(mready[m]); let end = st.saturating_add(p); mready[m] = end; prev = end; } if prev > mk { mk = prev; } }
    mk
}

fn build_perm_solution_from_seq_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], num_jobs: usize, num_machines: usize) -> Solution {
    let ops = route.len(); let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs]; let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs { continue; } let row = &pt[j]; let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines { break; }
            let p = row[op_idx]; let st = prev_end.max(machine_ready[m]); job_schedule[j].push((m, st)); let end = st.saturating_add(p); machine_ready[m] = end; prev_end = end;
        }
    }
    Solution { job_schedule }
}

#[inline]
pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
    let mut best = INF; let mut second = INF; let mut cnt_best = 0usize; let mut cnt_best_idle = 0usize;
    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best { second = best; best = end; cnt_best = 1; cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 }; }
        else if end == best { cnt_best += 1; if machine_avail[m] <= time { cnt_best_idle += 1; } }
        else if end < second { second = end; }
    }
    if cnt_best > 1 { second = best; }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].base_score < c.base_score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let min_s = top.last().unwrap().score; let n = top.len().min(8);
    let mut w = [0.0f64; 8]; let mut sum = 0.0f64;
    for i in 0..n { let d = (top[i].score - min_s) + 1e-9; let wi = d * d; w[i] = wi; sum += wi; }
    if !(sum > 0.0) { return top[rng.gen_range(0..top.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n { r -= w[i]; if r <= 0.0 { return top[i]; } }
    top[n - 1]
}

const K_6C17: f64 = 0.3;

#[derive(Clone, Copy)]
enum J7 { R0, R1, R2, R3, R4 }

#[derive(Clone, Copy)]
struct J8 {
    job: usize,
    priority: f64,
    machine_end: u32,
    proc_time: u32,
    flexibility: usize,
}

struct J9 { makespan: u32 }

struct JA {
    makespan: u32,
    rule: J7,
    random_top_k: usize,
    seed: u64,
}

struct JB {
    n0: Vec<usize>,
    n1: Vec<u32>,
    n2: Vec<u32>,
    n3: Vec<f64>,
    n4: Vec<usize>,
    n5: Vec<J8>,
}

impl JB {
    fn new(pre: &Pre, base: &[f64], nm: usize) -> Self {
        Self {
            n0: vec![0usize; pre.job_ops_len.len()],
            n1: vec![0u32; pre.job_ops_len.len()],
            n2: vec![0u32; nm],
            n3: base.to_vec(),
            n4: Vec::with_capacity(nm),
            n5: Vec::with_capacity(5),
        }
    }

    #[inline]
    fn reset(&mut self, base: &[f64]) {
        self.n0.fill(0);
        self.n1.fill(0);
        self.n2.fill(0);
        self.n3.copy_from_slice(base);
        self.n4.clear();
        self.n5.clear();
    }
}

#[inline]
fn h_09a4(candidate: &J8, best: &J8, eps: f64) -> bool {
    if candidate.priority > best.priority + eps { return true; }
    if (candidate.priority - best.priority).abs() <= eps {
        if candidate.machine_end < best.machine_end { return true; }
        if candidate.machine_end == best.machine_end {
            if candidate.proc_time < best.proc_time { return true; }
            if candidate.proc_time == best.proc_time {
                if candidate.flexibility < best.flexibility { return true; }
                if candidate.flexibility == best.flexibility && candidate.job < best.job { return true; }
            }
        }
    }
    false
}

#[inline]
fn h_09aa(a: &J8, b: &J8) -> std::cmp::Ordering {
    let ord = b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal);
    if ord != std::cmp::Ordering::Equal { return ord; }
    a.machine_end.cmp(&b.machine_end)
        .then(a.proc_time.cmp(&b.proc_time))
        .then(a.flexibility.cmp(&b.flexibility))
        .then(a.job.cmp(&b.job))
}

#[inline]
fn h_09ab(v: &mut Vec<J8>, c: J8, k: usize) {
    let mut lo = 0usize;
    let mut hi = v.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if h_09aa(&c, &v[mid]) == std::cmp::Ordering::Less { hi = mid; }
        else { lo = mid + 1; }
    }
    if lo < k {
        v.insert(lo, c);
        if v.len() > k { v.pop(); }
    }
}

#[inline]
fn h_09ac(time: u32, ma: &[u32], op: &OpInfo, target_m: usize) -> Option<(u32, u32)> {
    let mut earliest = u32::MAX;
    let mut pt = None;
    for &(m, p) in &op.machines {
        let e = time.max(ma[m]) + p;
        if e < earliest { earliest = e; }
        if m == target_m { pt = Some(p); }
    }
    pt.map(|p| (earliest, p))
}

fn h_09a5(
    challenge: &Challenge,
    pre: &Pre,
    product_work_times: &[Vec<f64>],
    job_total_work: &[f64],
    rule: J7,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
    w: &mut JB,
) -> Result<J9> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    w.reset(job_total_work);

    let mut remaining_ops = pre.total_ops;
    let mut time = 0u32;
    let eps = 1e-9_f64;
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        w.n4.clear();
        for m in 0..num_machines {
            if w.n2[m] <= time { w.n4.push(m); }
        }
        if use_random {
            use rand::seq::SliceRandom;
            w.n4.shuffle(rng.as_mut().unwrap());
        }

        let mut scheduled_any = false;
        for ix in 0..w.n4.len() {
            let machine = w.n4[ix];
            let mut best_candidate: Option<J8> = None;

            if use_random {
                w.n5.clear();
                for job in 0..num_jobs {
                    let op_idx = w.n0[job];
                    if op_idx >= pre.job_ops_len[job] || w.n1[job] > time { continue; }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 { continue; }
                    let Some((earliest_end, proc_time)) = h_09ac(time, &w.n2, op, machine) else { continue; };
                    let machine_end = time.max(w.n2[machine]) + proc_time;
                    if machine_end != earliest_end { continue; }
                    let priority = match rule {
                        J7::R0 => w.n3[job],
                        J7::R1 => (pre.job_ops_len[job] - op_idx) as f64,
                        J7::R2 => -(op.flex as f64),
                        J7::R3 => -(proc_time as f64),
                        J7::R4 => proc_time as f64,
                    };
                    h_09ab(&mut w.n5, J8 { job, priority, machine_end, proc_time, flexibility: op.flex }, random_top_k);
                }
                if !w.n5.is_empty() {
                    let k = random_top_k.min(w.n5.len());
                    let pick = rng.as_mut().unwrap().gen_range(0..k);
                    best_candidate = Some(w.n5[pick]);
                }
            } else {
                for job in 0..num_jobs {
                    let op_idx = w.n0[job];
                    if op_idx >= pre.job_ops_len[job] || w.n1[job] > time { continue; }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 { continue; }
                    let Some((earliest_end, proc_time)) = h_09ac(time, &w.n2, op, machine) else { continue; };
                    let machine_end = time.max(w.n2[machine]) + proc_time;
                    if machine_end != earliest_end { continue; }
                    let priority = match rule {
                        J7::R0 => w.n3[job],
                        J7::R1 => (pre.job_ops_len[job] - op_idx) as f64,
                        J7::R2 => -(op.flex as f64),
                        J7::R3 => -(proc_time as f64),
                        J7::R4 => proc_time as f64,
                    };
                    let c = J8 { job, priority, machine_end, proc_time, flexibility: op.flex };
                    if best_candidate.as_ref().map_or(true, |b| h_09a4(&c, b, eps)) { best_candidate = Some(c); }
                }
            }

            if let Some(c) = best_candidate {
                let job = c.job;
                let product = pre.job_products[job];
                let op_idx = w.n0[job];
                let end_time = time.max(w.n2[machine]) + c.proc_time;
                w.n0[job] += 1;
                w.n1[job] = end_time;
                w.n2[machine] = end_time;
                w.n3[job] -= product_work_times[product][op_idx];
                if w.n3[job] < 0.0 { w.n3[job] = 0.0; }
                remaining_ops -= 1;
                scheduled_any = true;
            }
        }

        if remaining_ops == 0 { break; }
        let mut next_time: Option<u32> = None;
        for &x in &w.n2 {
            if x > time { next_time = Some(next_time.map_or(x, |b| b.min(x))); }
        }
        for job in 0..num_jobs {
            if w.n0[job] < pre.job_ops_len[job] && w.n1[job] > time {
                let x = w.n1[job];
                next_time = Some(next_time.map_or(x, |b| b.min(x)));
            }
        }
        time = next_time.ok_or_else(|| {
            if scheduled_any { anyhow!("No next event time found while operations remain unscheduled") }
            else { anyhow!("No schedulable operations remain; dispatching rules stalled") }
        })?;
    }

    Ok(J9 { makespan: w.n1.iter().copied().max().unwrap_or(0) })
}

fn h_09a6(challenge: &Challenge, pre: &Pre) -> Result<u32> {
    let random_restarts = 200usize;
    let top_k = 2usize;
    let local_search_tries = 4usize;

    let mut product_work_times = Vec::with_capacity(pre.product_ops.len());
    for ops in &pre.product_ops {
        let mut x = Vec::with_capacity(ops.len());
        for op in ops {
            let min = if op.flex == 0 || op.machines.is_empty() { 0.0 } else { op.min_pt as f64 };
            x.push(op.avg_pt * (1.0 - K_6C17) + min * K_6C17);
        }
        product_work_times.push(x);
    }

    let mut job_total_work = Vec::with_capacity(pre.job_products.len());
    for &product in &pre.job_products {
        job_total_work.push(product_work_times[product].iter().sum());
    }
    let mut w = JB::new(pre, &job_total_work, challenge.num_machines);
    let rules = [J7::R0, J7::R1, J7::R2, J7::R3, J7::R4];

    let mut best_result: Option<J9> = None;
    for rule in rules.iter().copied() {
        let result = h_09a5(challenge, pre, &product_work_times, &job_total_work, rule, None, None, &mut w)?;
        if best_result.as_ref().map_or(true, |b| result.makespan < b.makespan) { best_result = Some(result); }
    }
    let mut best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    let mut top_restarts: Vec<JA> = Vec::with_capacity(top_k);
    let mut rng = SmallRng::from_seed(challenge.seed);

    for _ in 0..random_restarts {
        let seed = rng.r#gen::<u64>();
        let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5);
        let mut local_rng = SmallRng::seed_from_u64(seed);
        let result = h_09a5(challenge, pre, &product_work_times, &job_total_work, rule, Some(random_top_k), Some(&mut local_rng), &mut w)?;
        let makespan = result.makespan;
        if makespan < best_result.makespan { best_result = result; }
        let a = JA { makespan, rule, random_top_k, seed };
        let pos = top_restarts.iter().position(|x| makespan < x.makespan).unwrap_or(top_restarts.len());
        if pos < top_k {
            top_restarts.insert(pos, a);
            if top_restarts.len() > top_k { top_restarts.pop(); }
        }
    }

    for restart in &top_restarts {
        for attempt in 0..local_search_tries {
            let mut local_rng = SmallRng::seed_from_u64(restart.seed.wrapping_add(attempt as u64 + 1));
            let local_k = match attempt % 3 {
                0 => restart.random_top_k,
                1 => restart.random_top_k.saturating_sub(1),
                _ => restart.random_top_k.saturating_add(1),
            }.max(2);
            let result = h_09a5(challenge, pre, &product_work_times, &job_total_work, restart.rule, Some(local_k), Some(&mut local_rng), &mut w)?;
            if result.makespan < best_result.makespan { best_result = result; }
        }
    }
    Ok(best_result.makespan)
}

pub(crate) fn p_6d3e(challenge: &Challenge, pre: &Pre) -> Result<u32> {
    h_09a6(challenge, pre)
}
