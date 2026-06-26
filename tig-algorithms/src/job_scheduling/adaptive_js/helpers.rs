use super::types::*;
use rand::rngs::SmallRng;

#[inline]
pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
    for &(m, pt) in &op.machines {
        if m == machine {
            return Some(pt);
        }
    }
    None
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 {
        return;
    }
    let mut pos = top.len();
    while pos > 0 && top[pos - 1].score < c.score {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    top.insert(pos, c);
    if top.len() > k {
        top.pop();
    }
}

#[inline]
pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
    if k == 0 {
        return;
    }
    let mut pos = top.len();
    while pos > 0 && top[pos - 1].base_score < c.base_score {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    top.insert(pos, c);
    if top.len() > k {
        top.pop();
    }
}

#[inline]
pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
    let mut best = INF;
    let mut second = INF;
    let mut cnt_best = 0usize;
    let mut cnt_best_idle = 0usize;

    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best {
            second = best;
            best = end;
            cnt_best = 1;
            cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 };
        } else if end == best {
            cnt_best += 1;
            if machine_avail[m] <= time {
                cnt_best_idle += 1;
            }
        } else if end < second {
            second = end;
        }
    }
    if cnt_best > 1 {
        second = best;
    }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

#[inline]
pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    use rand::Rng;
    if top.len() <= 1 {
        return top[0];
    }
    let min_s = top.last().unwrap().score;
    let n = top.len().min(8);
    let mut w: [f64; 8] = [0.0; 8];
    let mut sum = 0.0f64;
    for i in 0..n {
        let d = (top[i].score - min_s) + 1e-9;
        let wi = d * d;
        w[i] = wi;
        sum += wi;
    }
    if !(sum > 0.0) {
        return top[rng.gen_range(0..top.len())];
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= w[i];
        if r <= 0.0 {
            return top[i];
        }
    }
    top[n - 1]
}

#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 {
        return;
    }
    let mut pos = top.len();
    while pos > 0 && top[pos - 1].score < c.score {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    top.insert(pos, c);
    if top.len() > k {
        top.pop();
    }
}

#[inline]
pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut best_m = NONE_USIZE;
    let mut best_pt = INF;
    let mut second_m = NONE_USIZE;
    let mut second_pt = INF;

    for &(m, pt) in &op.machines {
        if pt < best_pt || (pt == best_pt && m < best_m) {
            second_m = best_m;
            second_pt = best_pt;
            best_m = m;
            best_pt = pt;
        } else if m != best_m && (pt < second_pt || (pt == second_pt && m < second_m)) {
            second_m = m;
            second_pt = pt;
        }
    }

    [(best_m, best_pt), (second_m, second_pt)]
}

#[inline]
pub fn push_top_solutions(top: &mut Vec<(tig_challenges::job_scheduling::Solution, u32)>, sol: tig_challenges::job_scheduling::Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol, mk));
    if top.len() > cap {
        top.truncate(cap);
    }
}

#[inline]
pub fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() {
            continue;
        }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            let v = comp[k].max(comp[k - 1]).saturating_add(row[k]);
            comp[k] = v;
        }
    }
    *comp.last().unwrap_or(&0)
}

#[inline]
pub fn reentrant_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0);
    let mut mk = 0u32;
    for &j in seq {
        let row = &pt[j];
        let mut prev = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            let p = row[op_idx];
            let st = prev.max(mready[m]);
            let end = st.saturating_add(p);
            mready[m] = end;
            prev = end;
        }
        if prev > mk {
            mk = prev;
        }
    }
    mk
}
