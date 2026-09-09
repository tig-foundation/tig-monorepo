use super::hrefine::{Block, State};

pub struct JetParams {
    pub rounds: usize,
    pub tolerance: usize,
    pub neg_pct: u32,
    pub min_gain: i32,
    pub rebalance_iters: usize,
    pub cap: u32,
}

pub fn jet_refine(st: &mut State, p: &JetParams, budget_ok: &dyn Fn() -> bool) -> i64 {
    let n = st.n;
    let k = st.k;
    let start_cut = st.cut;
    let entry_feasible = st.feasible();
    let mut best_cut = if entry_feasible { st.cut } else { i64::MAX };
    let mut best_part: Vec<Block> = st.part.clone();
    let mut moved_last = vec![false; n];
    let mut moved_now = vec![false; n];
    let mut cnt = vec![0i32; k];
    let mut cands: Vec<(i32, u32, Block)> = Vec::with_capacity(n / 4);
    let c = p.neg_pct as f64 / 100.0;
    let mut no_improve = 0usize;

    for _ in 0..p.rounds {
        if !budget_ok() {
            break;
        }
        cands.clear();
        for v in 0..n {
            if moved_last[v] || !st.is_boundary(v) {
                continue;
            }
            if let Some((g, to, hold)) = st.best_move_unconstrained(v, &mut cnt, p.cap) {
                let thr = (-(c * hold as f64)).floor() as i32;
                if g > thr {
                    cands.push((g, v as u32, to));
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        moved_now.iter_mut().for_each(|f| *f = false);
        let mut applied = 0usize;
        for &(_, v, to) in cands.iter() {
            let vu = v as usize;
            let from = st.part[vu];
            let w = st.wt[vu];
            if from == to || !st.can_leave(from, w) {
                continue;
            }
            if p.cap != u32::MAX && st.block_size[to as usize] + w > p.cap {
                continue;
            }
            if st.move_gain(vu, to) >= p.min_gain {
                st.apply(vu, to);
                moved_now[vu] = true;
                applied += 1;
            }
        }

        let rb = rebalance(st, p.rebalance_iters, &mut cnt, &mut moved_now);

        (moved_last, moved_now) = (moved_now, moved_last);
        if st.feasible() && st.cut < best_cut {
            best_cut = st.cut;
            best_part.copy_from_slice(&st.part);
            no_improve = 0;
        } else {
            no_improve += 1;
            if no_improve >= p.tolerance.max(1) {
                break;
            }
        }
        if applied == 0 && rb == 0 {
            break;
        }
    }

    if best_cut == i64::MAX {
        if st.part != best_part {
            st.part.copy_from_slice(&best_part);
            st.rebuild();
        }
        return 0;
    }
    if !(st.feasible() && st.cut == best_cut) {
        st.part.copy_from_slice(&best_part);
        st.rebuild();
    }
    start_cut - st.cut
}

fn rebalance(st: &mut State, max_iters: usize, cnt: &mut [i32], moved: &mut [bool]) -> usize {
    let mut total = 0usize;
    let mut cands: Vec<(i32, u32, Block)> = Vec::new();
    for _ in 0..max_iters.max(1) {
        let mut over = 0u64;
        for b in 0..st.k {
            if st.block_size[b] > st.max_block_size {
                over |= 1u64 << b;
            }
        }
        if over == 0 {
            break;
        }
        cands.clear();
        for v in 0..st.n {
            if (over >> st.part[v]) & 1 == 0 {
                continue;
            }
            if let Some((g, to)) = st.best_move_any(v, cnt) {
                cands.push((g, v as u32, to));
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut applied = 0usize;
        for &(_, v, to) in cands.iter() {
            let vu = v as usize;
            let from = st.part[vu];
            let w = st.wt[vu];
            if st.block_size[from as usize] <= st.max_block_size || !st.fits(to, w) || !st.can_leave(from, w) {
                continue;
            }
            st.apply(vu, to);
            moved[vu] = true;
            applied += 1;
        }
        total += applied;
        if applied == 0 {
            break;
        }
    }
    total
}
