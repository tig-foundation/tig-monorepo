// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use std::collections::HashMap;

use super::types::*;
use super::infra::*;

// ============================================================================
// TABU SEARCH — integrated swap + reassign on disjunctive graph
// ============================================================================

/// O(1) estimate of makespan after swapping adjacent nodes u, v on same machine.
/// u is at pos, v is at pos+1. After swap: v first, then u.
#[inline]
fn estimate_swap_fm(
    u: usize, v: usize,
    heads: &[u32], tails: &[u32], pt: &[u32],
    job_pred: &[usize], job_succ: &[usize],
    machine_pred: &[usize], machine_succ: &[usize],
) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];

    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);

    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));

    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);

    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));

    (new_r_v.saturating_add(pt[v]).saturating_add(new_q_v))
        .max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
}

/// O(1) estimate of makespan after reassigning `node` to `new_machine` at `insert_pos`.
#[inline]
fn estimate_reassign_fm(
    ds: &DisjSchedule,
    heads: &[u32], tails: &[u32],
    node: usize, new_machine: usize, new_pt: u32, insert_pos: usize,
    job_pred: &[usize], machine_pred: &[usize], machine_succ: &[usize],
) -> u32 {
    let jp = job_pred[node];
    let js = ds.job_succ[node];
    let old_mp = machine_pred[node];
    let old_ms = machine_succ[node];

    let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };

    let new_seq = &ds.machine_seq[new_machine];
    let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
        let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
        heads[pred].saturating_add(ds.node_pt[pred])
    } else { 0 };

    let new_start = jp_end.max(new_mp_end);
    let new_end = new_start.saturating_add(new_pt);

    let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
    let new_ms_tail = if insert_pos < new_seq.len() {
        let succ = new_seq[insert_pos];
        ds.node_pt[succ].saturating_add(tails[succ])
    } else { 0 };

    let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));

    let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
        let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
        old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else { 0 };

    node_path.max(old_reconnect)
}

/// Find candidate insertion positions for `node` on `new_machine`.
fn find_insert_positions_fm(
    ds: &DisjSchedule,
    starts: &[u32],
    node: usize,
    new_machine: usize,
    job_pred: &[usize],
) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 { return vec![0]; }

    let jp = job_pred[node];
    let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let cur_start = starts[node];

    let mut pos_after_jp = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] > job_pred_end { pos_after_jp = i; break; }
    }
    let pos_after_jp = pos_after_jp.min(len);

    let mut pos_by_cur = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] >= cur_start { pos_by_cur = i; break; }
    }
    let pos_by_cur = pos_by_cur.min(len);

    let mut out: Vec<usize> = Vec::with_capacity(6);
    let push = |v: &mut Vec<usize>, p: usize| {
        if p <= len && !v.contains(&p) { v.push(p); }
    };
    push(&mut out, pos_after_jp);
    push(&mut out, pos_after_jp.saturating_sub(1));
    push(&mut out, pos_by_cur);
    push(&mut out, pos_by_cur.saturating_sub(1));
    push(&mut out, 0);
    push(&mut out, len);

    if out.is_empty() { out.push(len); }
    if out.len() > 6 { out.truncate(6); }
    out
}

enum MoveTypeFm {
    Swap { machine: usize, pos: usize },
    Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
}

/// Tabu search with integrated swap + reassign moves on disjunctive graph.
/// Combined neighborhood is key for FJSP medium quality.
fn tabu_search_fjsp_medium(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iterations: usize, tenure_base: usize,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    global_best: &mut u32,
) -> Result<Option<(Solution, u32)>> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, base_sol) else { return Ok(None) };
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;
    let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };

    let mut best_global_mk = initial_mk;
    let mut best_global_ds = ds.clone();

    let tenure = tenure_base.max(5);
    let tenure_delta = (tenure / 3).max(2);
    let max_no_improve = (max_iterations / 2).max(60);
    let kick_threshold = (max_no_improve * 2 / 3).max(50);

    let mut tabu_swap: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
    let mut tabu_reassign: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 4);

    // Precompute job_pred_node (constant - job chains don't change)
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end { job_pred_node[k] = k - 1; }
    }

    let mut no_improve = 0usize;
    let mut kicks_left = 3usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);

    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut crit = vec![false; n];

    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            if kicks_left == 0 { break; }
            ds = best_global_ds.clone();
            no_improve = 0;
            kicks_left -= 1;
            tabu_swap.clear();
            tabu_reassign.clear();
            continue;
        }

        // Periodic kick: perturb on CP swaps
        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false);
            let mut u = kick_mk_node;
            while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
            let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 { continue; }
                for i in 0..(ds.machine_seq[m].len() - 1) {
                    if crit[ds.machine_seq[m][i]] && crit[ds.machine_seq[m][i + 1]] {
                        kick_swaps.push((m, i));
                    }
                }
            }
            if !kick_swaps.is_empty() {
                for _ in 0..2 {
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let idx = (pseed as usize) % kick_swaps.len();
                    let (m, pos) = kick_swaps[idx];
                    if pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos, pos + 1); }
                }
            }
            kicks_left -= 1;
            continue;
        }

        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        if iter > 0 {
            if cur_mk < best_global_mk {
                best_global_mk = cur_mk;
                best_global_ds = ds.clone();
                no_improve = 0;
                if cur_mk < *global_best {
                    *global_best = cur_mk;
                    if let Ok(s) = disj_to_solution(pre, &ds, &buf.start) {
                        let _ = save_solution(&s);
                    }
                }
            } else {
                no_improve += 1;
            }
        }

        let tails = compute_tails_pulsar(&ds, &buf);

        machine_pred_node.fill(NONE_USIZE);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i - 1]; }
        }

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }

        let mut best_move: Option<MoveTypeFm> = None;
        let mut best_move_mk = u32::MAX;
        let mut fallback_move: Option<MoveTypeFm> = None;
        let mut fallback_mk = u32::MAX;

        // Swap moves on critical blocks (N1 neighborhood)
        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 { continue; }
            let mut i = 0;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] { i += 1; continue; }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < ds.machine_seq[m].len() {
                    let x = ds.machine_seq[m][bend];
                    let y = ds.machine_seq[m][bend + 1];
                    if !crit[y] { break; }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x { break; }
                    bend += 1;
                }
                if bend > bstart {
                    let block_len = bend - bstart + 1;
                    let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                    let num_swaps = if block_len >= 3 { 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos = swap_positions[si];
                        if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() { continue; }
                        let node_u = ds.machine_seq[m][pos];
                        let node_v = ds.machine_seq[m][pos + 1];
                        let est_mk = estimate_swap_fm(
                            node_u, node_v, &buf.start, &tails, &ds.node_pt,
                            &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                        );
                        let key = (node_u.min(node_v), node_u.max(node_v));
                        let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                        let aspiration = est_mk < best_global_mk;
                        if (!is_tabu || aspiration) && est_mk < best_move_mk {
                            best_move_mk = est_mk;
                            best_move = Some(MoveTypeFm::Swap { machine: m, pos });
                        }
                        if est_mk < fallback_mk {
                            fallback_mk = est_mk;
                            fallback_move = Some(MoveTypeFm::Swap { machine: m, pos });
                        }
                    }
                }
                i = bend + 1;
            }
        }

        // Reassign moves on critical nodes (every 3rd iteration)
        if iter % 3 == 0 {
            for node in 0..n {
                if !crit[node] { continue; }
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                if op_idx >= pre.product_ops[product].len() { continue; }
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let cur_machine = ds.node_machine[node];

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine { continue; }
                    let key = (node, new_m);
                    let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                    let positions = find_insert_positions_fm(&ds, &buf.start, node, new_m, &job_pred_node);
                    for insert_pos in positions {
                        let est_mk = estimate_reassign_fm(
                            &ds, &buf.start, &tails,
                            node, new_m, new_pt, insert_pos,
                            &job_pred_node, &machine_pred_node, &buf.machine_succ,
                        );
                        let aspiration = est_mk < best_global_mk;
                        if (!is_tabu || aspiration) && est_mk < best_move_mk {
                            best_move_mk = est_mk;
                            best_move = Some(MoveTypeFm::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                        if est_mk < fallback_mk {
                            fallback_mk = est_mk;
                            fallback_move = Some(MoveTypeFm::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                    }
                }
            }
        }

        match best_move.or(fallback_move) {
            Some(MoveTypeFm::Swap { machine: m, pos }) => {
                let node_a = ds.machine_seq[m][pos];
                let node_b = ds.machine_seq[m][pos + 1];
                ds.machine_seq[m].swap(pos, pos + 1);
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let progress = (iter as f64) / (max_iterations as f64);
                let late_bonus = if progress > 0.6 { ((progress - 0.6) * 10.0) as usize } else { 0 };
                let this_tenure = (tenure + offset + late_bonus).saturating_sub(tenure_delta);
                tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
            }
            Some(MoveTypeFm::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let this_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                tabu_reassign.insert((node, old_machine), iter + this_tenure);
            }
            None => break,
        }
    }

    // Final evaluation
    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    if final_mk < best_global_mk { best_global_mk = final_mk; best_global_ds = ds; }
    if best_global_mk >= initial_mk { return Ok(None); }

    let Some((_, _)) = eval_disj(&best_global_ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &best_global_ds, &buf.start)?;
    Ok(Some((sol, best_global_mk)))
}

/// CP-targeted ruin: reassigns flexible CP nodes to non-current machines
/// and swaps CP nodes in the temporal sequence, then decodes via forward scheduler.
fn targeted_cp_kick(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    d_reassign: usize,
    d_swap: usize,
    rng: &mut SmallRng,
) -> Result<Solution> {
    use rand::seq::SliceRandom;
    let ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let (mk, _mk_node) = match eval_disj(&ds, &mut buf) {
        Some(v) => v,
        None => return Ok(base_sol.clone()),
    };
    let tails = compute_tails_pulsar(&ds, &buf);

    // Build temporal sequence and machine assignments
    let mut m_assign: Vec<Vec<usize>> = vec![Vec::new(); challenge.num_jobs];
    let mut ops_by_start: Vec<(u32, usize, usize)> = Vec::with_capacity(ds.n);
    for j in 0..challenge.num_jobs {
        for (k, &(m, _)) in base_sol.job_schedule[j].iter().enumerate() {
            m_assign[j].push(m);
            let node = ds.job_offsets[j] + k;
            ops_by_start.push((buf.start[node], j, k));
        }
    }
    ops_by_start.sort_unstable_by_key(|x| x.0);
    let mut seq: Vec<usize> = ops_by_start.iter().map(|x| x.1).collect();

    // Identify flexible CP nodes
    let mut flex_cp: Vec<(usize, usize)> = Vec::new();
    let mut cp_indices: Vec<usize> = Vec::new();
    for (idx, &(st, j, k)) in ops_by_start.iter().enumerate() {
        let node = ds.job_offsets[j] + k;
        let pt = ds.node_pt[node];
        let tail = tails[node];
        if st + pt + tail == mk {
            cp_indices.push(idx);
            let prod = pre.job_products[j];
            if pre.product_ops[prod][k].machines.len() > 1 {
                flex_cp.push((j, k));
            }
        }
    }

    // Reassign flexible CP nodes to alternative machines
    flex_cp.shuffle(rng);
    let num_reassign = d_reassign.min(flex_cp.len());
    for i in 0..num_reassign {
        let (j, k) = flex_cp[i];
        let prod = pre.job_products[j];
        let cur_m = m_assign[j][k];
        let alts: Vec<(usize, u32)> = pre.product_ops[prod][k].machines.iter()
            .filter(|&&(m, _)| m != cur_m).copied().collect();
        if !alts.is_empty() {
            let &(new_m, _) = alts.choose(rng).unwrap();
            m_assign[j][k] = new_m;
        }
    }

    // Swap CP positions in sequence
    cp_indices.shuffle(rng);
    let mut swaps_done = 0;
    for &idx in &cp_indices {
        if swaps_done >= d_swap { break; }
        if idx + 1 < seq.len() && seq[idx] != seq[idx + 1] {
            seq.swap(idx, idx + 1);
            swaps_done += 1;
        }
    }

    // Forward decode with forced machine assignments
    let mut next_op = vec![0usize; challenge.num_jobs];
    let mut mready = vec![0u32; challenge.num_machines];
    let mut jready = vec![0u32; challenge.num_jobs];
    let mut new_job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::new(); challenge.num_jobs];

    for &j in &seq {
        let k = next_op[j];
        if k >= pre.job_ops_len[j] { continue; }
        next_op[j] += 1;
        let m = m_assign[j][k];
        let prod = pre.job_products[j];
        let pt = pt_from_op(&pre.product_ops[prod][k], m).unwrap_or(1);
        let st = jready[j].max(mready[m]);
        new_job_schedule[j].push((m, st));
        let end = st + pt;
        jready[j] = end;
        mready[m] = end;
    }

    Ok(Solution { job_schedule: new_job_schedule })
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    greedy_sol: Solution,
    greedy_mk: u32,
    effort: &EffortConfig,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut best_mk = greedy_mk;
    let mut top_solutions: Vec<(Solution, u32)> = Vec::with_capacity(25);
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 20);

    let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;
    let mut rules: Vec<Rule> = vec![
        Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath,
        Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc,
    ];
    if allow_flex_balance { rules.push(Rule::FlexBalance); }

    let target_margin = ((pre.avg_op_min * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness)).max(1.0)) as u32;
    let route_w_base = if pre.chaotic_like {
        0.0
    } else {
        (0.050 + 0.12 * pre.high_flex + 0.10 * pre.jobshopness + 0.08 / pre.flex_avg.max(1.0)).clamp(0.04, 0.28)
    };

    // Phase 1: Initial construction with all rules
    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
        if mk < best_mk { best_mk = mk; save_solution(&sol)?; }
        push_top_solutions(&mut top_solutions, &sol, mk, 20);
        ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);

    let r0 = ranked[0].0;
    let r1 = ranked.get(1).map(|x| x.0).unwrap_or(r0);
    let r2 = ranked.get(2).map(|x| x.0).unwrap_or(r1);

    let mut rule_best = vec![u32::MAX; 10];
    let mut rule_tries = vec![0u32; 10];
    for (rr, mk, _) in &ranked {
        let idx = rule_idx(*rr);
        rule_best[idx] = rule_best[idx].min(*mk);
        rule_tries[idx] = rule_tries[idx].saturating_add(1);
    }

    // Learn from initial top solutions
    let base_sol = &top_solutions[0].0;
    let mut learned_jb = Some(job_bias_from_solution(pre, base_sol)?);
    let mut learned_mp = Some(machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 {
        Some(route_pref_from_solution_lite(pre, base_sol, challenge)?)
    } else { None };
    let mut learn_updates_left = 12usize;

    // Phase 2: Main construction loop with bandit rule selection
    let num_restarts = effort.fjsp_medium_iters;
    let k_hi = if pre.flex_avg > 4.0 { 5 } else { 6 };
    let mut stuck = 0usize;

    for r in 0..num_restarts {
        let late = r >= (num_restarts * 2) / 3;
        let (k_min, k_max) = if stuck > 170 { (4, 6usize.min(k_hi)) }
            else if stuck > 90 { (3, 6usize.min(k_hi)) }
            else if stuck > 35 { (2, k_hi) }
            else { (2, k_hi.min(4)) };
        let rule = if r < 35 {
            let u: f64 = rng.gen();
            if allow_flex_balance && pre.high_flex > 0.82 && u < 0.10 { Rule::FlexBalance }
            else if u < 0.52 { r0 } else if u < 0.80 { r1 } else if u < 0.92 { r2 }
            else { rules[rng.gen_range(0..rules.len())] }
        } else {
            choose_rule_bandit(&mut rng, &rules, &rule_best, &rule_tries, best_mk, target_margin, stuck, pre.chaotic_like, late)
        };
        let k = if k_max <= k_min { k_min } else { rng.gen_range(k_min..=k_max) };
        let learn_base = if pre.chaotic_like { 0.0 } else {
            (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42)
        };
        let learn_boost = (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);
        let use_learn = learned_jb.is_some() && learned_mp.is_some()
            && rng.gen::<f64>() < learn_p
            && (route_w_base == 0.0 || learned_rp.is_some());
        let target = if best_mk < u32::MAX / 2 { Some(best_mk.saturating_add(target_margin)) } else { None };

        let (sol, mk) = if use_learn {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng,
                learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base)?
        } else {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0)?
        };

        let ridx = rule_idx(rule);
        rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
        rule_best[ridx] = rule_best[ridx].min(mk);

        if mk < best_mk { best_mk = mk; save_solution(&sol)?; stuck = 0; } else { stuck = stuck.saturating_add(1); }
        push_top_solutions(&mut top_solutions, &sol, mk, 20);

        // Periodic elite learning update
        if learn_updates_left > 0 && !pre.chaotic_like && !top_solutions.is_empty() {
            let refresh = (r > 0 && r % 35 == 0) || stuck == 90 || stuck == 170;
            if refresh {
                let rep_sol = &top_solutions[top_solutions.len().min(10) / 2].0;
                learned_jb = Some(job_bias_from_solution(pre, rep_sol)?);
                learned_mp = Some(machine_penalty_from_solution(pre, rep_sol, challenge.num_machines)?);
                if route_w_base > 0.0 {
                    learned_rp = Some(route_pref_from_solution_lite(pre, rep_sol, challenge)?);
                }
                learn_updates_left -= 1;
            }
        }
    }

    // Phase 3: Per-elite refinement
    let route_w_ls = if route_w_base > 0.0 { (route_w_base * 1.40).clamp(route_w_base, 0.40) } else { 0.0 };
    let refine_n = top_solutions.len().min(10);
    let mut refine_buf: Vec<(Solution, u32)> = Vec::new();
    for i in 0..refine_n {
        let base = top_solutions[i].0.clone();
        let jb = job_bias_from_solution(pre, &base)?;
        let mp = machine_penalty_from_solution(pre, &base, challenge.num_machines)?;
        let rp = if route_w_ls > 0.0 { Some(route_pref_from_solution_lite(pre, &base, challenge)?) } else { None };
        let target_ls = if best_mk < u32::MAX / 2 { Some(best_mk.saturating_add(target_margin / 2)) } else { None };
        for attempt in 0..8 {
            let rule = match attempt {
                0 => r0, 1 => Rule::Adaptive, 2 => Rule::BnHeavy, 3 => Rule::EndTight,
                4 => Rule::Regret, 5 => Rule::LeastFlex, 6 => Rule::MostWork, _ => r1,
            };
            let k = match attempt % 4 { 0 => 2, 1 => 3, 2 => 4, _ => 2 }.min(k_hi);
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng,
                Some(&jb), Some(&mp), rp.as_ref(), if rp.is_some() { route_w_ls } else { 0.0 })?;
            if mk < best_mk { best_mk = mk; save_solution(&sol)?; }
            refine_buf.push((sol, mk));
        }
    }
    for (sol, mk) in refine_buf { push_top_solutions(&mut top_solutions, &sol, mk, 20); }

    // Phase 4: Tabu search with combined swap + reassign on top solutions
    let ts_n = top_solutions.len().min(12);
    let ts_iters = (effort.fjsp_medium_iters * 3 / 4).max(60);
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 12);
    for i in 0..ts_n {
        let base_sol = top_solutions[i].0.clone();
        if let Some((sol, mk)) = tabu_search_fjsp_medium(
            pre, challenge, &base_sol, ts_iters, ts_tenure, save_solution, &mut best_mk,
        )? {
            push_top_solutions(&mut top_solutions, &sol, mk, 20);
        }
    }

    // Phase 5: CBMLS on top solutions
    let cb_passes = 6;
    let cb_iters = (pre.total_ops / 8).max(30).min(120);
    let cb_no_improve = cb_iters / 2;
    let cb_top_n = top_solutions.len().min(8);
    for ci in 0..cb_top_n {
        let base_sol = top_solutions[ci].0.clone();
        if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, &base_sol, cb_passes, cb_iters, cb_no_improve) {
            if cb_mk < best_mk { best_mk = cb_mk; save_solution(&cb_sol)?; }
            push_top_solutions(&mut top_solutions, &cb_sol, cb_mk, 20);
        }
    }
    if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, &top_solutions[0].0, 20) {
        if bmr_mk < best_mk { best_mk = bmr_mk; save_solution(&bmr_sol)?; }
        push_top_solutions(&mut top_solutions, &bmr_sol, bmr_mk, 20);
    }

    // Phase 6: ILS loop — perturbation + greedy_reassign + CBMLS
    let ils_rounds = if effort.fjsp_medium_iters > 300 { 30 } else { 20 };
    let ils_max_no_improve = (ils_rounds * 3) / 4 + 3;
    let mut ils_best_mk = best_mk;
    let mut ils_no_improve = 0usize;
    let mut op_weights = vec![10.0f64; 5]; // strategy 4 = targeted CP ejection

    for ils_r in 0..ils_rounds {
        if ils_no_improve >= ils_max_no_improve { break; }
        let mut ds = build_disj_from_solution(pre, challenge, &top_solutions[0].0)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { continue };
        let n = ds.n;
        let mut pseed: u64 = (ils_r as u64).wrapping_mul(0x517CC1B727220A95)
            .wrapping_add(ils_best_mk as u64)
            .wrapping_add(challenge.seed[0] as u64)
            .wrapping_add((ils_r as u64).wrapping_mul(0xDEADBEEF));
        let k_perturb = (3 + ils_r / 3).min(8);

        let total_weight: f64 = op_weights.iter().sum();
        let mut choice_val = rng.gen::<f64>() * total_weight;
        let mut strategy = 3usize;
        for (i, &weight) in op_weights.iter().enumerate() {
            if choice_val < weight { strategy = i; break; }
            choice_val -= weight;
        }

        if strategy == 0 {
            let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
            let mut u = mk_node;
            while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
            let mut perturbed = 0; let mut attempts = 0;
            while perturbed < k_perturb && attempts < crit_nodes.len() * 4 {
                attempts += 1;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                if crit_nodes.is_empty() { break; }
                let node = crit_nodes[(pseed as usize) % crit_nodes.len()];
                let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let cur_m = ds.node_machine[node];
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let (new_m, new_pt) = op_info.machines[(pseed as usize) % op_info.machines.len()];
                if new_m == cur_m { continue; }
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                let cur_start = buf.start[node];
                let mut ins_pos = ds.machine_seq[new_m].len();
                for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                ds.machine_seq[new_m].insert(ins_pos, node);
                perturbed += 1;
            }
        } else if strategy == 1 {
            let mut machine_loads = vec![0u32; ds.num_machines];
            for node in 0..n { let m = ds.node_machine[node]; machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[node]); }
            let worst_m = machine_loads.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
            if ds.machine_seq[worst_m].is_empty() { continue; }
            let mut perturbed = 0; let mut attempts = 0;
            while perturbed < k_perturb && attempts < ds.machine_seq[worst_m].len() * 4 {
                attempts += 1;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let cur_seq_len = ds.machine_seq[worst_m].len();
                if cur_seq_len == 0 { break; }
                let node = ds.machine_seq[worst_m][(pseed as usize) % cur_seq_len];
                let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let mut best_alt_m = worst_m; let mut best_alt_pt = ds.node_pt[node];
                for &(am, apt) in &op_info.machines { if am != worst_m && apt < best_alt_pt { best_alt_pt = apt; best_alt_m = am; } }
                if best_alt_m == worst_m { continue; }
                let old_pos = match ds.machine_seq[worst_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                ds.machine_seq[worst_m].remove(old_pos);
                ds.node_machine[node] = best_alt_m; ds.node_pt[node] = best_alt_pt;
                let cur_start = buf.start[node];
                let mut ins_pos = ds.machine_seq[best_alt_m].len();
                for (ki, &nd) in ds.machine_seq[best_alt_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                ds.machine_seq[best_alt_m].insert(ins_pos, node);
                perturbed += 1;
            }
        } else if strategy == 2 {
            let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
            let mut crit_machines: Vec<usize> = Vec::with_capacity(16);
            let mut u = mk_node;
            while u != NONE_USIZE {
                crit_nodes.push(u);
                let m = ds.node_machine[u];
                if !crit_machines.contains(&m) { crit_machines.push(m); }
                u = buf.best_pred[u];
            }
            let k_re = k_perturb / 2;
            let mut perturbed = 0; let mut attempts = 0;
            while perturbed < k_re && attempts < crit_nodes.len() * 3 {
                attempts += 1;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                if crit_nodes.is_empty() { break; }
                let node = crit_nodes[(pseed as usize) % crit_nodes.len()];
                let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let cur_m = ds.node_machine[node];
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let (new_m, new_pt) = op_info.machines[(pseed as usize) % op_info.machines.len()];
                if new_m == cur_m { continue; }
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                let cur_start = buf.start[node];
                let mut ins_pos = ds.machine_seq[new_m].len();
                for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                ds.machine_seq[new_m].insert(ins_pos, node);
                perturbed += 1;
            }
            let k_sw = k_perturb - k_re;
            let mut swapped = 0;
            for _ in 0..(k_sw * 4) {
                if swapped >= k_sw || crit_machines.is_empty() { break; }
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let m = crit_machines[(pseed as usize) % crit_machines.len()];
                if ds.machine_seq[m].len() < 2 { continue; }
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let pos = (pseed as usize) % (ds.machine_seq[m].len() - 1);
                ds.machine_seq[m].swap(pos, pos + 1);
                swapped += 1;
            }
        } else if strategy == 3 {
            let mut swapped = 0; let mut attempts = 0;
            while swapped < k_perturb && attempts < 100 {
                attempts += 1;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let m = (pseed as usize) % ds.num_machines;
                if ds.machine_seq[m].len() < 2 { continue; }
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let pos = (pseed as usize) % (ds.machine_seq[m].len() - 1);
                ds.machine_seq[m].swap(pos, pos + 1);
                swapped += 1;
            }
        } else {
            // strategy 4: targeted CP ejection from bottleneck machine (Gemini i4)
            let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
            let mut u = mk_node;
            while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
            if crit_nodes.is_empty() { ils_no_improve += 1; continue; }
            let mut machine_crit_count = vec![0usize; ds.num_machines];
            for &nd in &crit_nodes { machine_crit_count[ds.node_machine[nd]] += 1; }
            let target_m = machine_crit_count.iter().enumerate()
                .max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);
            let candidates: Vec<usize> = crit_nodes.iter().cloned()
                .filter(|&nd| {
                    if ds.node_machine[nd] != target_m { return false; }
                    let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                    let product = pre.job_products[job];
                    pre.product_ops[product][op_idx].machines.len() > 1
                })
                .collect();
            if candidates.is_empty() { ils_no_improve += 1; continue; }
            let ejections = candidates.len().min(2);
            let mut perturbed = 0;
            for i in 0..ejections {
                let node = candidates[i];
                let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                let cur_m = ds.node_machine[node];
                let best_alt = op_info.machines.iter().filter(|&&(m,_)| m != cur_m)
                    .min_by_key(|&&(_,pt)| pt);
                if let Some(&(new_m, new_pt)) = best_alt {
                    let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_m].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[new_m].len();
                    for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[new_m].insert(ins_pos, node);
                    perturbed += 1;
                }
            }
            if perturbed == 0 { ils_no_improve += 1; continue; }
        }

        let Some((_, _)) = eval_disj(&ds, &mut buf) else { ils_no_improve += 1; continue };
        let perturbed_sol = match disj_to_solution(pre, &ds, &buf.start) {
            Ok(s) => s, Err(_) => { ils_no_improve += 1; continue; }
        };
        let after_gr = match greedy_reassign_pass(pre, challenge, &perturbed_sol)? {
            Some((s, mk)) => (s, mk),
            None => {
                if let Some((pmk, _)) = eval_disj(&ds, &mut buf) { (perturbed_sol.clone(), pmk) }
                else { ils_no_improve += 1; continue; }
            }
        };
        let (candidate_sol, candidate_mk) = match critical_block_move_local_search_ex(pre, challenge, &after_gr.0, cb_passes, cb_iters, cb_no_improve) {
            Ok(Some((ls_sol, ls_mk))) => if ls_mk < after_gr.1 { (ls_sol, ls_mk) } else { after_gr },
            _ => after_gr,
        };

        if candidate_mk < ils_best_mk {
            let reward = 1.0 + (ils_best_mk - candidate_mk) as f64 / ils_best_mk as f64;
            op_weights[strategy] = (op_weights[strategy] * 0.8 + reward * 2.0).clamp(1.0, 50.0);
            ils_best_mk = candidate_mk;
            if candidate_mk < best_mk { best_mk = candidate_mk; save_solution(&candidate_sol)?; }
            push_top_solutions(&mut top_solutions, &candidate_sol, candidate_mk, 20);
            ils_no_improve = 0;
        } else {
            op_weights[strategy] = (op_weights[strategy] * 0.95).max(1.0);
            ils_no_improve += 1;
        }
    }

    // Final greedy reassign
    if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &top_solutions[0].0) {
        if m < best_mk { best_mk = m; save_solution(&s)?; }
    }

    // Fuel-based CP-Kick + Tabu loop (total_tabu_budget=310k, no wall clock)
    // Calibrated: 310k / ts_iters ≈ 192 outer iters → cycle ≈ 615s at 5T
    {
        let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 12);
        let ts_iters = (effort.fjsp_medium_iters * 3 / 4).max(60);
        let max_outer_iters = (310_000usize / ts_iters.max(1)).min(250);
        let mut tb_no_improve = 0usize;
        for loop_iter in 0..max_outer_iters {
            if top_solutions.is_empty() { break; }

            // Pick base solution: cycle through pool
            let pool_size = top_solutions.len().min(12);
            let base_idx = loop_iter % pool_size;
            let base_sol = top_solutions[base_idx].0.clone();

            // CP-targeted kick to break out of current basin
            let kick_reassigns = rng.gen_range(2usize..=5);
            let kick_swaps = rng.gen_range(1usize..=4);
            let perturbed = match targeted_cp_kick(pre, challenge, &base_sol, kick_reassigns, kick_swaps, &mut rng) {
                Ok(s) => s,
                Err(_) => base_sol.clone(),
            };

            // Tabu search on perturbed solution
            let (cand_sol, cand_mk) = match tabu_search_fjsp_medium(
                pre, challenge, &perturbed, ts_iters, ts_tenure, save_solution, &mut best_mk,
            )? {
                Some((s, m)) => (s, m),
                None => {
                    // Fallback: CBMLS on perturbed if tabu fails
                    match critical_block_move_local_search_ex(pre, challenge, &perturbed, cb_passes, cb_iters, cb_no_improve) {
                        Ok(Some((ls_sol, ls_mk))) => (ls_sol, ls_mk),
                        _ => (perturbed, u32::MAX),
                    }
                }
            };

            if cand_mk < best_mk {
                best_mk = cand_mk;
                save_solution(&cand_sol)?;
                tb_no_improve = 0;
            } else {
                tb_no_improve += 1;
            }
            if cand_mk < u32::MAX {
                push_top_solutions(&mut top_solutions, &cand_sol, cand_mk, 20);
            }
            if tb_no_improve > 160 { break; }
        }
    }

    Ok(())
}
