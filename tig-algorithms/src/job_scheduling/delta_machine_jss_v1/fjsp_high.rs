use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::infra::*;

static FH_NONCE_COUNTER: AtomicUsize = AtomicUsize::new(0);
static FH_ELITE_POOL: Mutex<Vec<(Solution, u64)>> = Mutex::new(Vec::new());
const FH_MAX_POOL_SIZE: usize = 8;
const FH_COLD_START_NONCES: usize = 2;

const PERM_2: [[usize; 2]; 2] = [[0, 1], [1, 0]];
const PERM_3: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const PERM_4: [[usize; 4]; 24] = [
    [0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
    [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,0,2],[1,3,2,0],
    [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
    [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0],
];

#[inline(always)]
fn local_rule_idx(r: Rule) -> usize {
    match r {
        Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3,
        Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8,
        Rule::MostFlex=>9,
    }
}

// ============================================================================
// ELITE DIVERSITY POOL
// ============================================================================

fn solution_machine_signature(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Vec<u16> {
    let mut sig: Vec<u16> = Vec::with_capacity(pre.total_ops);
    for job in 0..challenge.num_jobs {
        let lim = pre.job_ops_len[job].min(sol.job_schedule[job].len());
        for op_idx in 0..pre.job_ops_len[job] {
            sig.push(if op_idx < lim { sol.job_schedule[job][op_idx].0 as u16 } else { u16::MAX });
        }
    }
    sig
}

#[inline(always)]
fn hamming_distance_sig(a: &[u16], b: &[u16]) -> u32 {
    let mut d = 0u32;
    for i in 0..a.len().min(b.len()) { if a[i] != b[i] { d += 1; } }
    d
}

fn push_top_solutions_diverse(
    pre: &Pre, challenge: &Challenge, pool: &mut Vec<(Solution, u32, Vec<u16>)>, sol: &Solution, mk: u32, cap: usize,
) {
    let sig = solution_machine_signature(pre, challenge, sol);
    pool.push((sol.clone(), mk, sig));
    while pool.len() > cap {
        let len = pool.len();
        let mut min_nn = vec![u32::MAX; len];
        for i in 0..len { for j in (i+1)..len {
            let d = hamming_distance_sig(&pool[i].2, &pool[j].2);
            if d < min_nn[i] { min_nn[i] = d; }
            if d < min_nn[j] { min_nn[j] = d; }
        }}
        let mut worst_mk = 0u32;
        for i in 0..len { if pool[i].1 > worst_mk { worst_mk = pool[i].1; } }
        let mut drop_idx = 0; let mut drop_min_nn = u32::MAX;
        for i in 0..len { if pool[i].1 == worst_mk && min_nn[i] < drop_min_nn { drop_idx = i; drop_min_nn = min_nn[i]; } }
        pool.swap_remove(drop_idx);
    }
    pool.sort_unstable_by_key(|x| x.1);
}

fn consensus_learning_from_elites(
    pre: &Pre, challenge: &Challenge, elites: &[(Solution, u32, Vec<u16>)],
) -> Result<(Vec<f64>, Vec<f64>, RoutePrefLite)> {
    if elites.is_empty() { return Err(anyhow!("empty")); }
    let denom = elites.len() as f64;
    let mut jb_sum = vec![0.0f64; challenge.num_jobs];
    for (sol, _, _) in elites { let jb = job_bias_from_solution(pre, sol)?; for j in 0..challenge.num_jobs { jb_sum[j] += jb[j]; } }
    for j in 0..challenge.num_jobs { jb_sum[j] /= denom; }

    let mut bn_cnt = vec![0u32; challenge.num_machines];
    let mut m_end = vec![0u32; challenge.num_machines];
    for (sol, mk, _) in elites {
        m_end.fill(0);
        for job in 0..challenge.num_jobs {
            let prod = pre.job_products[job]; if prod >= pre.product_ops.len() { continue; }
            let sched = &sol.job_schedule[job]; let lim = sched.len().min(pre.product_ops[prod].len());
            for op_idx in 0..lim {
                let (m, st) = sched[op_idx]; if m >= challenge.num_machines { continue; }
                let pt = pre.product_ops[prod][op_idx].machines.iter().find(|&&(mm,_)| mm==m).map(|&(_,p)|p).unwrap_or(0);
                let end = st.saturating_add(pt); if end > m_end[m] { m_end[m] = end; }
            }
        }
        for m in 0..challenge.num_machines { if m_end[m] == *mk { bn_cnt[m] += 1; } }
    }
    let mut mp = vec![0.0f64; challenge.num_machines];
    for m in 0..challenge.num_machines { mp[m] = (bn_cnt[m] as f64) / denom; }

    // Learn route_pref from ALL elites (majority vote per product×op)
    let mut rp = route_pref_from_solution_lite(pre, &elites[0].0, challenge)?;
    let num_products = pre.product_ops.len();
    let mut counts: Vec<Vec<Vec<u32>>> = pre.product_ops.iter()
        .map(|ops| ops.iter().map(|_| vec![0u32; challenge.num_machines]).collect())
        .collect();
    for (sol, _, _) in elites {
        for job in 0..challenge.num_jobs {
            let product = pre.job_products[job];
            if product >= num_products { continue; }
            let sched = &sol.job_schedule[job];
            let ops_len = counts[product].len();
            for (op_idx, (m, _)) in sched.iter().enumerate() {
                if op_idx >= ops_len { break; }
                if *m < challenge.num_machines { counts[product][op_idx][*m] = counts[product][op_idx][*m].saturating_add(1); }
            }
        }
    }
    for product in 0..num_products {
        let ops_len = counts[product].len();
        if product >= rp.len() { break; }
        for op_idx in 0..ops_len {
            if op_idx >= rp[product].len() { break; }
            let row = &counts[product][op_idx];
            let total: u32 = row.iter().sum();
            if total == 0 { continue; }
            let mut best_m = 0usize; let mut best_c = 0u32;
            let mut second_m = 0usize; let mut second_c = 0u32;
            for (m, &c) in row.iter().enumerate() {
                if c > best_c { second_m = best_m; second_c = best_c; best_m = m; best_c = c; }
                else if m != best_m && c > second_c { second_m = m; second_c = c; }
            }
            if second_c == 0 { second_m = best_m; }
            let total64 = total as u64;
            rp[product][op_idx].best_m = best_m.min(255) as u8;
            rp[product][op_idx].best_w = ((best_c as u64) * 255 / total64) as u8;
            rp[product][op_idx].second_m = second_m.min(255) as u8;
            rp[product][op_idx].second_w = ((second_c as u64) * 255 / total64) as u8;
        }
    }
    Ok((jb_sum, mp, rp))
}

// PULSAR functions (schrage_pass, slack_ejection_schrage) are in infra.rs

// ============================================================================
// WINDOW EXHAUSTIVE — Local permutation search on bottleneck machine (kept as complement)
// ============================================================================
fn cp_window_exhaustive(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    let init_mk = cur_mk; let n = ds.n;

    for _ in 0..max_iters {
        let mut on_cp = vec![false; n];
        for nd in 0..n { if buf.start[nd].saturating_add(ds.node_pt[nd]) == cur_mk { on_cp[nd] = true; } }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&a| std::cmp::Reverse(buf.start[a]));
        for &nd in &order {
            if !on_cp[nd] { continue; }
            let seq = &ds.machine_seq[ds.node_machine[nd]];
            if let Some(pos) = seq.iter().position(|&x| x == nd) {
                if pos > 0 { let p = seq[pos-1]; if buf.start[p].saturating_add(ds.node_pt[p]) == buf.start[nd] { on_cp[p] = true; } }
            }
        }

        let mut mc = vec![0usize; ds.num_machines];
        for nd in 0..n { if on_cp[nd] { mc[ds.node_machine[nd]] += 1; } }
        let bn_m = mc.iter().enumerate().max_by_key(|&(_,&c)| c).map(|(m,_)| m).unwrap_or(0);
        let mut found = false;

        'ws: for &wsize in &[4usize, 3, 2] {
            let slen = ds.machine_seq[bn_m].len(); if slen < wsize { continue; }
            for start in 0..=(slen - wsize) {
                if !ds.machine_seq[bn_m][start..start+wsize].iter().any(|&nd| on_cp[nd]) { continue; }
                let mut orig = [0usize; 4];
                for i in 0..wsize { orig[i] = ds.machine_seq[bn_m][start+i]; }
                let mut best_pmk = cur_mk; let mut best_p: Option<&[usize]> = None;
                let pc = if wsize==4 { 24 } else if wsize==3 { 6 } else { 2 };
                for pi in 0..pc {
                    let p = if wsize==4 { &PERM_4[pi][..] } else if wsize==3 { &PERM_3[pi][..] } else { &PERM_2[pi][..] };
                    for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[p[i]]; }
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) { if tmk < best_pmk { best_pmk = tmk; best_p = Some(p); } }
                }
                if let Some(p) = best_p {
                    for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[p[i]]; }
                    cur_mk = best_pmk; let _ = eval_disj(&ds, &mut buf); found = true; break 'ws;
                } else { for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[i]; } }
            }
        }
        if !found { break; }
    }
    if cur_mk < init_mk { Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk))) } else { Ok(None) }
}

// ============================================================================
// ITERATIVE CP DESCENT — smart insertion + pair-wise CP moves
// ============================================================================
fn iterative_cp_descent(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    let initial_mk = current_mk;
    let n = ds.n;

    let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
    for nd in 0..n {
        let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
        if op_idx >= job_op_to_node[job].len() { job_op_to_node[job].resize(op_idx + 1, usize::MAX); }
        job_op_to_node[job][op_idx] = nd;
    }

    for _iter in 0..max_iters {
        let mut finish = vec![0u32; n];
        for nd in 0..n { finish[nd] = buf.start[nd].saturating_add(ds.node_pt[nd]); }
        let makespan = current_mk;

        let mut on_cp = vec![false; n];
        for nd in 0..n { if finish[nd] == makespan { on_cp[nd] = true; } }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));

        for &nd in &order {
            if !on_cp[nd] { continue; }
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            if op_idx > 0 {
                let pred_op = op_idx - 1;
                if pred_op < job_op_to_node[job].len() {
                    let pred = job_op_to_node[job][pred_op];
                    if pred != usize::MAX && finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                }
            }
            let machine = ds.node_machine[nd];
            let seq = &ds.machine_seq[machine];
            if let Some(pos) = seq.iter().position(|&x| x == nd) {
                if pos > 0 { let pred = seq[pos - 1]; if finish[pred] == buf.start[nd] { on_cp[pred] = true; } }
            }
        }

        let mut cp_flex: Vec<usize> = (0..n).filter(|&nd| {
            if !on_cp[nd] { return false; }
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            let product = pre.job_products[job];
            op_idx < pre.product_ops[product].len() && pre.product_ops[product][op_idx].machines.len() > 1
        }).collect();
        cp_flex.sort_by_key(|&nd| buf.start[nd]);
        if cp_flex.is_empty() { break; }

        let mut iter_improved = false;

        // Single-node CP moves with smart insertion
        for &nd in &cp_flex {
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            let product = pre.job_products[job];
            if op_idx >= pre.product_ops[product].len() { continue; }
            let op_info = &pre.product_ops[product][op_idx];
            let cur_m = ds.node_machine[nd]; let cur_pt = ds.node_pt[nd];
            let mut best_m = cur_m; let mut best_pt = cur_pt;
            let mut best_mk = current_mk; let mut best_ins = 0usize;

            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_m { continue; }
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) { Some(p)=>p, None=>continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                let _ = eval_disj(&ds, &mut buf);
                let cur_start = buf.start[nd];
                let tlen = ds.machine_seq[new_m].len();
                let mut spos = tlen;
                for (ki, &nd2) in ds.machine_seq[new_m].iter().enumerate() {
                    if buf.start[nd2] >= cur_start { spos = ki; break; }
                }
                let mut positions: Vec<usize> = Vec::with_capacity(4);
                for &p in &[spos, spos.saturating_sub(1), (spos+1).min(tlen), tlen] {
                    if p <= tlen && !positions.contains(&p) { positions.push(p); }
                }
                for &pos in &positions {
                    ds.machine_seq[new_m].insert(pos, nd);
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                        if tmk < best_mk { best_mk = tmk; best_m = new_m; best_pt = new_pt; best_ins = pos; }
                    }
                    ds.machine_seq[new_m].remove(pos);
                }
                ds.machine_seq[cur_m].insert(old_pos, nd);
                ds.node_machine[nd] = cur_m; ds.node_pt[nd] = cur_pt;
            }
            if best_m != cur_m {
                let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
                ds.machine_seq[cur_m].remove(old_pos);
                let ins = best_ins.min(ds.machine_seq[best_m].len());
                ds.machine_seq[best_m].insert(ins, nd);
                ds.node_machine[nd] = best_m; ds.node_pt[nd] = best_pt;
                current_mk = best_mk;
                let _ = eval_disj(&ds, &mut buf);
                iter_improved = true;
            }
        }

        // Pair-wise CP moves: try coordinated moves of consecutive CP node pairs
        if cp_flex.len() >= 2 {
            let pairs: Vec<(usize, usize)> = cp_flex.windows(2).map(|w| (w[0], w[1])).collect();

            let cand_positions = |ds: &DisjSchedule, buf: &EvalBuf, m: usize, nd: usize| -> Vec<usize> {
                let cur_start = buf.start[nd];
                let tlen = ds.machine_seq[m].len();
                let mut spos = tlen;
                for (ki, &nx) in ds.machine_seq[m].iter().enumerate() {
                    if buf.start[nx] >= cur_start { spos = ki; break; }
                }
                let mut positions: Vec<usize> = Vec::with_capacity(3);
                for &p in &[spos, spos.saturating_sub(1), tlen] {
                    if p <= tlen && !positions.contains(&p) { positions.push(p); }
                }
                positions
            };

            for (nd1, nd2) in pairs {
                let job1 = ds.node_job[nd1]; let op1 = ds.node_op[nd1];
                let job2 = ds.node_job[nd2]; let op2 = ds.node_op[nd2];
                let prod1 = pre.job_products[job1]; let prod2 = pre.job_products[job2];
                if op1 >= pre.product_ops[prod1].len() || op2 >= pre.product_ops[prod2].len() { continue; }
                let op_info1 = &pre.product_ops[prod1][op1];
                let op_info2 = &pre.product_ops[prod2][op2];
                if op_info1.machines.len() <= 1 && op_info2.machines.len() <= 1 { continue; }

                let cur_m1 = ds.node_machine[nd1]; let cur_pt1 = ds.node_pt[nd1];
                let cur_m2 = ds.node_machine[nd2]; let cur_pt2 = ds.node_pt[nd2];
                let mut opts1: Vec<(usize, u32)> = Vec::new();
                opts1.push((cur_m1, cur_pt1));
                opts1.extend(op_info1.machines.iter().copied().filter(|&(m, _)| m != cur_m1).take(4));
                let mut opts2: Vec<(usize, u32)> = Vec::new();
                opts2.push((cur_m2, cur_pt2));
                opts2.extend(op_info2.machines.iter().copied().filter(|&(m, _)| m != cur_m2).take(4));

                let pos1 = match ds.machine_seq[cur_m1].iter().position(|&x| x == nd1) { Some(p)=>p, None=>continue };
                let pos2 = match ds.machine_seq[cur_m2].iter().position(|&x| x == nd2) { Some(p)=>p, None=>continue };

                // Remove both nodes
                if cur_m1 == cur_m2 {
                    if pos1 > pos2 { ds.machine_seq[cur_m1].remove(pos1); ds.machine_seq[cur_m2].remove(pos2); }
                    else { ds.machine_seq[cur_m2].remove(pos2); ds.machine_seq[cur_m1].remove(pos1); }
                } else {
                    ds.machine_seq[cur_m1].remove(pos1); ds.machine_seq[cur_m2].remove(pos2);
                }

                let mut best_mk_pair = current_mk;
                let mut best_config: Option<(usize, u32, usize, usize, u32, usize, u8)> = None;

                for &(m1, pt1) in &opts1 {
                    for &(m2, pt2) in &opts2 {
                        if m1 != m2 {
                            ds.node_machine[nd1] = m1; ds.node_pt[nd1] = pt1;
                            ds.node_machine[nd2] = m2; ds.node_pt[nd2] = pt2;
                            let p1s = cand_positions(&ds, &buf, m1, nd1);
                            let p2s = cand_positions(&ds, &buf, m2, nd2);
                            for &p1i in &p1s {
                                ds.machine_seq[m1].insert(p1i, nd1);
                                for &p2i in &p2s {
                                    ds.machine_seq[m2].insert(p2i, nd2);
                                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                        if tmk < best_mk_pair { best_mk_pair = tmk; best_config = Some((m1,pt1,p1i,m2,pt2,p2i,0)); }
                                    }
                                    ds.machine_seq[m2].remove(p2i);
                                }
                                ds.machine_seq[m1].remove(p1i);
                            }
                        } else {
                            let m = m1;
                            ds.node_machine[nd1] = m; ds.node_pt[nd1] = pt1;
                            ds.node_machine[nd2] = m; ds.node_pt[nd2] = pt2;
                            for order_flag in 0u8..=1u8 {
                                let (a, b) = if order_flag == 0 { (nd1, nd2) } else { (nd2, nd1) };
                                let pa = cand_positions(&ds, &buf, m, a);
                                for &pai in &pa {
                                    ds.machine_seq[m].insert(pai, a);
                                    let pb = cand_positions(&ds, &buf, m, b);
                                    for &pbi in &pb {
                                        ds.machine_seq[m].insert(pbi, b);
                                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                            if tmk < best_mk_pair {
                                                best_mk_pair = tmk;
                                                if order_flag == 0 { best_config = Some((m,pt1,pai,m,pt2,pbi,0)); }
                                                else { best_config = Some((m,pt1,pbi,m,pt2,pai,1)); }
                                            }
                                        }
                                        ds.machine_seq[m].remove(pbi);
                                    }
                                    ds.machine_seq[m].remove(pai);
                                }
                            }
                        }
                    }
                }

                if let Some((bm1, bpt1, bp1, bm2, bpt2, bp2, order_flag)) = best_config {
                    if best_mk_pair < current_mk {
                        ds.node_machine[nd1] = bm1; ds.node_pt[nd1] = bpt1;
                        ds.node_machine[nd2] = bm2; ds.node_pt[nd2] = bpt2;
                        if bm1 != bm2 {
                            let ins1 = bp1.min(ds.machine_seq[bm1].len());
                            ds.machine_seq[bm1].insert(ins1, nd1);
                            let ins2 = bp2.min(ds.machine_seq[bm2].len());
                            ds.machine_seq[bm2].insert(ins2, nd2);
                        } else {
                            let m = bm1;
                            if order_flag == 0 {
                                let ins1 = bp1.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins1, nd1);
                                let ins2 = bp2.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins2, nd2);
                            } else {
                                let ins2 = bp2.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins2, nd2);
                                let ins1 = bp1.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins1, nd1);
                            }
                        }
                        current_mk = best_mk_pair;
                        let _ = eval_disj(&ds, &mut buf);
                        iter_improved = true;
                        continue;
                    }
                }

                // Restore both nodes to original positions
                ds.node_machine[nd1] = cur_m1; ds.node_pt[nd1] = cur_pt1;
                ds.node_machine[nd2] = cur_m2; ds.node_pt[nd2] = cur_pt2;
                if cur_m1 != cur_m2 {
                    let ins1 = pos1.min(ds.machine_seq[cur_m1].len());
                    ds.machine_seq[cur_m1].insert(ins1, nd1);
                    let ins2 = pos2.min(ds.machine_seq[cur_m2].len());
                    ds.machine_seq[cur_m2].insert(ins2, nd2);
                } else {
                    let m = cur_m1;
                    if pos1 <= pos2 {
                        let ins1 = pos1.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins1, nd1);
                        let ins2 = pos2.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins2, nd2);
                    } else {
                        let ins2 = pos2.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins2, nd2);
                        let ins1 = pos1.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins1, nd1);
                    }
                }
                let _ = eval_disj(&ds, &mut buf);
            }
        }

        if !iter_improved { break; }
    }

    if current_mk >= initial_mk { return Ok(None); }
    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    if final_mk >= initial_mk { return Ok(None); }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk)))
}

pub fn solve(
    challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre, greedy_sol: Solution, greedy_mk: u32, effort: &EffortConfig,
) -> Result<()> {
    let solve_start = std::time::Instant::now();
    let nonce_id = FH_NONCE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut best_mk = greedy_mk; let mut best_sol = greedy_sol.clone();
    let mut pool: Vec<(Solution, u32, Vec<u16>)> = Vec::with_capacity(30);
    push_top_solutions_diverse(pre, challenge, &mut pool, &greedy_sol, greedy_mk, 15);

    // Cross-nonce warm-start: if a previous instance produced a strong solution,
    // seed this instance's per-instance diverse pool with it so the bandit/learning
    // sees the elite as a high-quality reference and benefits from its structure.
    if nonce_id >= FH_COLD_START_NONCES {
        let pool_snapshot = {
            let lp = FH_ELITE_POOL.lock().unwrap();
            lp.clone()
        };
        for (elite_sol, elite_mk_u64) in pool_snapshot.iter().take(3) {
            let elite_mk = *elite_mk_u64 as u32;
            push_top_solutions_diverse(pre, challenge, &mut pool, elite_sol, elite_mk, 15);
            if elite_mk < best_mk {
                best_mk = elite_mk;
                best_sol = elite_sol.clone();
                save_solution(&best_sol)?;
            }
        }
    }

    let rules = [Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath,
        Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc, Rule::FlexBalance, Rule::MostFlex];
    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(10);
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
        if mk < best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; }
        push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 15);
        ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0 = ranked[0].0; let r1 = ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2 = ranked.get(2).map(|x|x.0).unwrap_or(r1);
    let mut rb = vec![u32::MAX; 10]; let mut rt = vec![0u32; 10];
    for (rr, mk, _) in &ranked { let i = local_rule_idx(*rr); rb[i] = rb[i].min(*mk); rt[i] += 1; }

    let (jb0, mp0, rp0) = consensus_learning_from_elites(pre, challenge, &pool)?;
    let mut ljb = Some(jb0); let mut lmp = Some(mp0); let mut lrp = Some(rp0);
    let mut lu = 10usize;
    let tm: u32 = ((pre.avg_op_min * (0.9 + 0.9*pre.high_flex + 0.6*pre.jobshopness)).max(1.0)) as u32;
    let rw: f64 = (0.040 + 0.10*pre.high_flex + 0.08*pre.jobshopness).clamp(0.04, 0.22);
    let kh = if pre.flex_avg > 8.0 { 8 } else if pre.flex_avg > 6.5 { 7 } else { 6 };
    let mut stuck = 0usize;

    // PHASE 1: Construction 5000 restarts with INLINE Schrage + Window improvement
    // Key insight: improvements run INSIDE loop so they feed back into consensus learning
    // → better biases → better constructions → compound effect.
    let num_restarts = effort.fjsp_high_iters;
    for r in 0..num_restarts {
        let p = r as f64 / num_restarts as f64; let late = p >= 0.66;
        let (kn, kx) = if stuck>170 {(4,6usize.min(kh))} else if stuck>90 {(3,6usize.min(kh))} else if stuck>35 {(2,6usize.min(kh))} else {(2,4usize.min(kh))};
        let (kn, kx) = if p >= 0.80 { (1,1) } else if p >= 0.60 { let kx2=(kx/2).max(1); (kn.min(kx2), kx2) } else { (kn, kx) };
        let rule = if r < 35 { let u: f64 = rng.gen(); if u<0.12{Rule::FlexBalance} else if u<0.50{r0} else if u<0.75{r1} else if u<0.90{r2} else {rules[rng.gen_range(0..10)]} }
            else { choose_rule_bandit(&mut rng, &rules.to_vec(), &rb, &rt, best_mk, tm, stuck, false, late) };
        let k = if kx<=kn { kn } else { rng.gen_range(kn..=kx) };
        let lp = (0.08 + 0.22*pre.jobshopness + 0.18*pre.high_flex).clamp(0.05, 0.42) * (1.0 + 0.35*((stuck as f64)/120.0).clamp(0.0,1.0));
        let ul = ljb.is_some() && lmp.is_some() && rng.gen::<f64>() < lp && lrp.is_some();
        let tgt = if best_mk < u32::MAX/2 { Some(best_mk.saturating_add(tm)) } else { None };
        let (sol, mk) = if ul { construct_solution_conflict(challenge, pre, rule, k, tgt, &mut rng, ljb.as_deref(), lmp.as_deref(), lrp.as_ref(), rw)? }
            else { construct_solution_conflict(challenge, pre, rule, k, tgt, &mut rng, None, None, None, 0.0)? };
        let ri = local_rule_idx(rule); rt[ri] += 1; rb[ri] = rb[ri].min(mk);
        push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 15);
        if mk < best_mk {
            best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; stuck = 0;
            // INLINE improvement: Schrage resequencing + Window Exhaustive after each new best
            // (Schrage optimizes sequence within each CP machine, window exhaustive permutes windows)
            let mut cur_sol = sol.clone(); let mut cur_mk = mk;
            if let Ok(Some((s2, m2))) = schrage_pass(pre, challenge, &cur_sol, 2) {
                if m2 < cur_mk { cur_mk = m2; cur_sol = s2; }
            }
            if let Ok(Some((s3, m3))) = cp_window_exhaustive(pre, challenge, &cur_sol, 2) {
                if m3 < cur_mk { cur_mk = m3; cur_sol = s3; }
            }
            if cur_mk < best_mk { best_mk = cur_mk; best_sol = cur_sol.clone(); save_solution(&cur_sol)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &cur_sol, cur_mk, 15);
            // Update consensus from improved pool (improves next construction bias)
            if lu > 0 { if let Ok((j,m,rp)) = consensus_learning_from_elites(pre, challenge, &pool) { ljb=Some(j); lmp=Some(m); lrp=Some(rp); lu-=1; } }
        } else { stuck += 1; }
    }

    // PHASE 2: Refinement 10 elites × 10
    let rwl = (rw*1.40).clamp(rw, 0.40);
    let er = pool.len().min(10);
    for i in 0..er {
        let bs = &pool[i].0; let jb = job_bias_from_solution(pre, bs)?; let mp = machine_penalty_from_solution(pre, bs, challenge.num_machines)?;
        let rp = Some(route_pref_from_solution_lite(pre, bs, challenge)?);
        let tl = if best_mk < u32::MAX/2 { Some(best_mk.saturating_add(tm/2)) } else { None };
        for a in 0..10 {
            let rule = match a { 0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>r1 };
            let k = match a%4 { 0=>2,1=>3,2=>4,_=>2 }.min(kh);
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, tl, &mut rng, Some(&jb), Some(&mp), rp.as_ref(), rwl)?;
            if mk < best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 15);
        }
    }

    // PHASE 3: CBMLS-ex short bursts on 15 elites
    let lr = pool.len().min(15);
    for i in 0..lr {
        let bs = pool[i].0.clone();
        if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, &bs, 8, 128, 24)? {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 15);
        }
    }
    if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &best_sol) {
        if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
    }

    // PHASE 4: PULSAR — Schrage resequencing on top 12 elites
    // Analytically optimizes entire machine sequences (vs window exhaustive's 4-op limit)
    let sr = pool.len().min(12);
    for i in 0..sr {
        let bs = pool[i].0.clone();
        if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &bs, 10) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 15);
        }
    }

    // PHASE 5: Iterative CP Descent on top 12 elites (smart insertion + pair-wise moves)
    let ejr = pool.len().min(12);
    for i in 0..ejr {
        let bs = pool[i].0.clone();
        if let Ok(Some((s, m))) = iterative_cp_descent(pre, challenge, &bs, 8) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 15);
            // Greedy reassign after each CP descent improvement
            if let Ok(Some((s2, m2))) = greedy_reassign_pass(pre, challenge, &s) {
                if m2 < best_mk { best_mk = m2; best_sol = s2.clone(); save_solution(&s2)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s2, m2, 15);
            }
        }
    }

    // PHASE 6: Window Exhaustive on top 10 elites → CP descent after each improvement
    let wr = pool.len().min(10);
    for i in 0..wr {
        let bs = pool[i].0.clone();
        if let Ok(Some((s, m))) = cp_window_exhaustive(pre, challenge, &bs, 6) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 15);
            // CP descent after window improvement (compound effect)
            if let Ok(Some((s2, m2))) = iterative_cp_descent(pre, challenge, &s, 5) {
                if m2 < best_mk { best_mk = m2; best_sol = s2.clone(); save_solution(&s2)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s2, m2, 15);
            }
        }
    }

    // PHASE 7: CBMLS polish after machine reassignments (N1 sequencing pass)
    if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, &best_sol, 10, 80, 14)? {
        if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
    }

    // PHASE 8: Final greedy reassign polish
    for _ in 0..2 {
        if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &best_sol) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
        }
    }

    // PHASE 9: Time-bounded CBMLS loop (12s per nonce = 85% of 14112ms budget)
    let time_limit_ms = 12_000u128;
    let mut pool_idx = 0usize;
    while solve_start.elapsed().as_millis() < time_limit_ms {
        let base_candidate = if !pool.is_empty() {
            pool[pool_idx % pool.len()].0.clone()
        } else {
            best_sol.clone()
        };
        pool_idx = pool_idx.wrapping_add(1);
        if let Ok(Some((s, m))) = critical_block_move_local_search_ex(
            pre, challenge, &base_candidate, 500, 2, 3
        ) {
            if m < best_mk {
                best_mk = m;
                best_sol = s.clone();
                save_solution(&best_sol)?;
                push_top_solutions_diverse(pre, challenge, &mut pool, &best_sol, best_mk, 10);
            }
        }
    }

    // Persist best solution to cross-nonce elite pool for subsequent instances
    {
        let mk = best_mk as u64;
        let mut lp = FH_ELITE_POOL.lock().unwrap();
        if !lp.iter().any(|(_, m)| *m == mk) {
            lp.push((best_sol.clone(), mk));
            lp.sort_by_key(|(_, m)| *m);
            lp.truncate(FH_MAX_POOL_SIZE);
        }
    }

    Ok(())
}
