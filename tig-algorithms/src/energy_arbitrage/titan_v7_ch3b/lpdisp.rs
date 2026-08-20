//! C-H3B — CONCAVE-HULL PWL DISPATCH LP solver (submission-compatible: pure Rust, no deps, no RNG, no I/O).
//!
//! Problem (one step): maximise Σ_b hull_b(u_b) s.t. lo_b ≤ u_b ≤ hi_b and −lim_l ≤ base_l + Σ_b sens[l][b]·u_b ≤ lim_l
//! for all lines l, where hull_b is the upper concave envelope of a PWL sample of titan's exact one-step objective
//! f_b(u) (kinks = SOC-cell crossings, u = 0, clamp points, bounds, + midpoints). It is NOT an exact optimiser of the
//! original (globally non-concave) piecewise-quadratic objective; the caller evaluates the exact objective afterwards.
//!
//! Method: constraint generation over lines + bounded-variable DUAL simplex on the reduced LP.
//!   variables: segment increments y_j ∈ [0, len_j] (u_b = x0_b + Σ_{j∈b} y_j, costs = hull slopes, non-increasing per
//!   battery ⇒ ordered fill is automatically optimal); ranged slacks s_l = Σ_j A_lj y_j ∈ [−lim_l − base'_l, lim_l − base'_l]
//!   for the active lines (base' includes x0). Rows = active lines only (≈ 20–60), columns = segments (≈ 2 000–3 000).
//!   Start: unconstrained optimum (y_j = len_j iff slope_j > 0) = dual feasible; dual simplex iterations: leaving row =
//!   most infeasible basic (smallest index tie-break), pivot row via dense B⁻¹, Harris two-pass + bound-flipping (long-step)
//!   ratio test over the eligible nonbasic columns, pivot with explicit B⁻¹ update, refactorisation every REFACTOR pivots,
//!   Bland rule after STALL non-improving iterations, finite cap. Then all lines are checked; violated lines are added
//!   (deterministic index order, at most ADD_MAX per round) and the LP is re-solved warm (old basis + new slacks basic).
//!   Fallback (cap reached): the current point is clamped to bounds and deflated toward u = 0 until feasible (never titan's
//!   PGA); every fallback is counted.
//! Determinism: fixed orderings (battery, segment, line index), fixed tie-breaks, fixed tolerances, no RNG.

pub const TOL_P: f64 = 1e-9;      // primal feasibility tolerance on basic values (absolute, $/MW scale of y ~ 0.01-50)
pub const TOL_D: f64 = 1e-9;      // dual (reduced cost) tolerance
pub const TOL_PIV: f64 = 1e-11;   // minimum |pivot|
pub const TOL_FLOW: f64 = 1e-9;   // line violation tolerance for constraint generation (absolute MW)
pub const REFACTOR: usize = 40;
pub const STALL: usize = 50;
pub const MAX_PIVOTS: usize = 4000;
pub const MAX_ROUNDS: usize = 40;
pub const ADD_MAX: usize = 40;

#[derive(Clone, Debug)]
pub struct Hull { pub x0: f64, pub len: Vec<f64>, pub slope: Vec<f64> }

/// Upper concave envelope of sorted points (xs ascending, distinct).
pub fn concave_hull(xs: &[f64], ys: &[f64]) -> Hull {
    let mut h: Vec<usize> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        while h.len() >= 2 {
            let (i1, i2) = (h[h.len() - 2], h[h.len() - 1]);
            let cross = (xs[i2] - xs[i1]) * (ys[i] - ys[i1]) - (ys[i2] - ys[i1]) * (xs[i] - xs[i1]);
            if cross >= 0.0 { h.pop(); } else { break; }
        }
        h.push(i);
    }
    let mut len = Vec::with_capacity(h.len()); let mut slope = Vec::with_capacity(h.len());
    for w in h.windows(2) {
        let l = xs[w[1]] - xs[w[0]];
        if l > 0.0 { len.push(l); slope.push((ys[w[1]] - ys[w[0]]) / l); }
    }
    Hull { x0: xs[h[0]], len, slope }
}

/// Build breakpoints for battery b and the hull of f(u). `kinks` must include lo, hi, 0 (if inside), SOC-cell crossings,
/// clamp points; `n_sub` extra points per piece (n_sub=2 => midpoints).
pub fn build_hull<F: Fn(f64) -> f64>(kinks_sorted: &[f64], n_sub: usize, f: F) -> Hull {
    let mut xs: Vec<f64> = Vec::with_capacity(kinks_sorted.len() * n_sub);
    for w in kinks_sorted.windows(2) {
        xs.push(w[0]);
        for k in 1..n_sub { xs.push(w[0] + (w[1] - w[0]) * k as f64 / n_sub as f64); }
    }
    xs.push(*kinks_sorted.last().unwrap());
    xs.dedup();
    let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();
    concave_hull(&xs, &ys)
}

#[derive(Clone, Debug, Default)]
pub struct LpStats { pub pivots: usize, pub bound_flips: usize, pub rounds: usize, pub active_lines: usize, pub refactors: usize, pub degenerate: usize, pub bland_switches: usize, pub fallback: bool, pub max_violation: f64, pub hull_obj: f64, pub status: u8 /*0 optimal,1 fallback_cap,2 fallback_infeasible*/ }

struct Lp<'a> {
    hulls: &'a [Hull], sens: &'a [Vec<f64>], lim: &'a [f64], base: &'a [f64],
    m: usize, n: usize,                    // batteries, structural columns
    col_b: Vec<usize>, col_len: Vec<f64>, col_c: Vec<f64>, // per structural column
    base_eff: Vec<f64>,                   // base + sens·x0
    lines: Vec<usize>,                    // active lines (rows)
    binv: Vec<f64>,                       // r x r row-major
    basis: Vec<usize>,                    // basic variable per row: < n structural, >= n slack (index n + row_of_line? we use n + k where k = position in `lines`)
    x: Vec<f64>,                          // values of all variables (n + lines.len())
    at_upper: Vec<bool>,                  // for nonbasic
    is_basic: Vec<bool>,
    stats: LpStats,
}

impl<'a> Lp<'a> {
    fn r(&self) -> usize { self.lines.len() }
    fn nvar(&self) -> usize { self.n + self.lines.len() }
    fn lb(&self, j: usize) -> f64 { if j < self.n { 0.0 } else { let l = self.lines[j - self.n]; -self.lim[l] - self.base_eff[l] } }
    fn ub(&self, j: usize) -> f64 { if j < self.n { self.col_len[j] } else { let l = self.lines[j - self.n]; self.lim[l] - self.base_eff[l] } }
    fn cost(&self, j: usize) -> f64 { if j < self.n { self.col_c[j] } else { 0.0 } }
    /// column j entry for row k (line lines[k]): structural: sens[line][b]; slack: -1 on its own row
    fn a(&self, k: usize, j: usize) -> f64 { if j < self.n { self.sens[self.lines[k]][self.col_b[j]] } else if j - self.n == k { -1.0 } else { 0.0 } }
    fn refactor(&mut self) -> bool {
        let r = self.r(); if r == 0 { self.binv.clear(); return true; }
        // Gaussian elimination with partial pivoting on B (columns = basis)
        let mut m = vec![0.0; r * r]; let mut inv = vec![0.0; r * r];
        for k in 0..r { for (i, &j) in self.basis.iter().enumerate() { m[k * r + i] = self.a(k, j); } inv[k * r + k] = 1.0; }
        for col in 0..r {
            let mut p = col; let mut best = m[col * r + col].abs();
            for i in col + 1..r { let v = m[i * r + col].abs(); if v > best { best = v; p = i; } }
            if best < 1e-13 { return false; }
            if p != col { for j in 0..r { m.swap(col * r + j, p * r + j); inv.swap(col * r + j, p * r + j); } }
            let piv = m[col * r + col];
            for j in 0..r { m[col * r + j] /= piv; inv[col * r + j] /= piv; }
            for i in 0..r { if i != col { let f = m[i * r + col]; if f != 0.0 { for j in 0..r { m[i * r + j] -= f * m[col * r + j]; inv[i * r + j] -= f * inv[col * r + j]; } } } }
        }
        self.binv = inv; self.stats.refactors += 1; true
    }
    /// recompute basic values from nonbasic: B x_B = -N x_N
    fn recompute_xb(&mut self) {
        let r = self.r(); if r == 0 { return; }
        let mut rhs = vec![0.0; r];
        for j in 0..self.nvar() { if !self.is_basic[j] && self.x[j] != 0.0 { for k in 0..r { rhs[k] -= self.a(k, j) * self.x[j]; } } }
        let mut xb = vec![0.0; r];
        for i in 0..r { let mut s = 0.0; for k in 0..r { s += self.binv[i * r + k] * rhs[k]; } xb[i] = s; }
        for i in 0..r { let j = self.basis[i]; self.x[j] = xb[i]; }
    }
    fn col_binv_a(&self, j: usize) -> Vec<f64> {
        let r = self.r(); let mut w = vec![0.0; r];
        if j < self.n { let b = self.col_b[j]; let a: Vec<f64> = (0..r).map(|k| self.sens[self.lines[k]][b]).collect(); for i in 0..r { let mut s = 0.0; for k in 0..r { s += self.binv[i * r + k] * a[k]; } w[i] = s; } }
        else { let k0 = j - self.n; for i in 0..r { w[i] = -self.binv[i * r + k0]; } }
        w
    }
    /// dual simplex on the current active set; returns true if optimal, false if cap reached
    fn dual_simplex(&mut self, pivot_budget: &mut usize) -> bool {
        let r = self.r(); if r == 0 { return true; }
        let mut since_refactor = 0usize; let mut stall = 0usize; let mut bland = false; let mut last_infeas = f64::INFINITY;
        loop {
            // leaving row
            let mut p = usize::MAX; let mut worst = TOL_P; let mut to_lower = true; let mut total_inf = 0.0;
            for i in 0..r {
                let j = self.basis[i]; let v = self.x[j]; let (lb, ub) = (self.lb(j), self.ub(j));
                let inf_lo = lb - v; let inf_hi = v - ub;
                let inf = if inf_lo > inf_hi { inf_lo } else { inf_hi };
                if inf > TOL_P { total_inf += inf; }
                if bland { if inf > TOL_P && (p == usize::MAX || j < self.basis[p]) { p = i; worst = inf; to_lower = inf_lo >= inf_hi; } }
                else if inf > worst { worst = inf; p = i; to_lower = inf_lo >= inf_hi; }
            }
            if p == usize::MAX { return true; }
            if total_inf >= last_infeas - 1e-12 { stall += 1; if stall > STALL && !bland { bland = true; self.stats.bland_switches += 1; } } else { stall = 0; }
            last_infeas = total_inf;
            if *pivot_budget == 0 { return false; }
            // pivot row rho = e_p^T B^{-1}; alpha_j = rho . a_j ; y = c_B^T B^{-1}
            let rho: Vec<f64> = (0..r).map(|k| self.binv[p * r + k]).collect();
            let mut y = vec![0.0; r];
            for i in 0..r { let cb = self.cost(self.basis[i]); if cb != 0.0 { for k in 0..r { y[k] += cb * self.binv[i * r + k]; } } }
            // per-battery shared quantities
            let mut alpha_b = vec![0.0; self.m]; let mut g_b = vec![0.0; self.m];
            for b in 0..self.m { let mut sa = 0.0; let mut sg = 0.0; for k in 0..r { let s = self.sens[self.lines[k]][b]; sa += rho[k] * s; sg += y[k] * s; } alpha_b[b] = sa; g_b[b] = sg; }
            // eligible nonbasic columns with ratio
            let xbp = self.x[self.basis[p]]; let target = if to_lower { self.lb(self.basis[p]) } else { self.ub(self.basis[p]) };
            let mut delta = (xbp - target).abs(); // amount the leaving var must move
            let mut cands: Vec<(f64, usize, f64, f64)> = Vec::new(); // (ratio, j, alpha, d)
            for j in 0..self.nvar() {
                if self.is_basic[j] { continue; }
                let (alpha, d) = if j < self.n { let b = self.col_b[j]; (alpha_b[b], self.col_c[j] - g_b[b]) } else { let k = j - self.n; (-rho[k], y[k]) };
                if alpha.abs() <= TOL_PIV { continue; }
                // leaving needs to move UP (to_lower) : x_Bp changes by -alpha*dx_j; need -alpha*dx_j>0
                let ok = if to_lower { (alpha < 0.0 && !self.at_upper[j]) || (alpha > 0.0 && self.at_upper[j]) } else { (alpha > 0.0 && !self.at_upper[j]) || (alpha < 0.0 && self.at_upper[j]) };
                if !ok { continue; }
                let dd = d.abs().max(0.0);
                cands.push((dd / alpha.abs(), j, alpha, d));
            }
            if cands.is_empty() { // dual unbounded => primal infeasible on this active set (cannot happen: u=0 feasible) ; treat as fallback
                return false;
            }
            cands.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap().then(a.1.cmp(&c.1)));
            // bound flipping walk with Harris window
            let mut q = usize::MAX; let mut idx = 0usize;
            let mut flips: Vec<usize> = Vec::new();
            while idx < cands.len() {
                let (ratio, j, alpha, _) = cands[idx];
                let span = if j < self.n { self.col_len[j] } else { self.ub(j) - self.lb(j) };
                let move_cap = alpha.abs() * span;
                // Harris window: among candidates with ratio <= ratio_first + TOL_D/|alpha| choose largest |alpha| at the stopping point
                if move_cap < delta - TOL_P && !bland {
                    // flip j entirely; leaving var still infeasible afterwards
                    flips.push(j); delta -= move_cap; idx += 1; continue;
                }
                // stopping group: candidates with ratio within [ratio, ratio + TOL_D] (Harris) -> choose max |alpha| (smallest j tie)
                let mut best = idx; let mut k = idx; let lim_ratio = ratio + TOL_D;
                while k < cands.len() && cands[k].0 <= lim_ratio { if cands[k].2.abs() > cands[best].2.abs() + 1e-15 { best = k; } k += 1; }
                q = cands[best].1; let _ = ratio; break;
            }
            if q == usize::MAX { // all candidates flipped but leaving var still infeasible -> take the last as entering (enter with partial)
                q = *flips.last().unwrap(); flips.pop();
            }
            // apply flips
            for &j in &flips {
                let (lo, hi) = (self.lb(j), self.ub(j)); let newv = if self.at_upper[j] { lo } else { hi }; let dx = newv - self.x[j];
                let w = self.col_binv_a(j); for i in 0..r { let bj = self.basis[i]; self.x[bj] -= w[i] * dx; }
                self.x[j] = newv; self.at_upper[j] = !self.at_upper[j]; self.stats.bound_flips += 1;
            }
            // pivot q in, basis[p] out
            let w = self.col_binv_a(q); let wp = w[p];
            if wp.abs() <= TOL_PIV { // numerically useless pivot: refactor and retry, else give up
                if !self.refactor() { return false; } self.recompute_xb(); since_refactor = 0; *pivot_budget -= 1; continue;
            }
            let xbp_now = self.x[self.basis[p]];
            let dxq = (xbp_now - target) / wp;  // x_Bp_new = x_Bp - wp*dxq = target
            let newq = self.x[q] + dxq;
            // clamp entering value into its bounds (Harris may allow tiny violations)
            let (lq, uq) = (self.lb(q), self.ub(q)); let newq_c = newq.max(lq).min(uq); let dxq_c = newq_c - self.x[q];
            for i in 0..r { let bj = self.basis[i]; self.x[bj] -= w[i] * dxq_c; }
            self.x[q] = newq_c;
            let lv = self.basis[p];
            // update B^{-1}: row p /= wp ; rows i -= w_i * row p
            for k in 0..r { self.binv[p * r + k] /= wp; }
            for i in 0..r { if i != p { let f = w[i]; if f != 0.0 { for k in 0..r { self.binv[i * r + k] -= f * self.binv[p * r + k]; } } } }
            self.basis[p] = q; self.is_basic[lv] = false; self.is_basic[q] = true;
            self.x[lv] = target; self.at_upper[lv] = !to_lower;
            self.stats.pivots += 1; *pivot_budget -= 1; since_refactor += 1;
            if dxq_c.abs() <= 1e-14 { self.stats.degenerate += 1; }
            if since_refactor >= REFACTOR { if !self.refactor() { return false; } self.recompute_xb(); since_refactor = 0; }
        }
    }
    fn current_u(&self) -> Vec<f64> {
        let mut u: Vec<f64> = self.hulls.iter().map(|h| h.x0).collect();
        for j in 0..self.n { u[self.col_b[j]] += self.x[j]; }
        u
    }
}

/// Solve the hull LP for one step. `bounds` = (lo,hi) per battery; hulls start at x0 = lo by construction.
pub fn solve(hulls: &[Hull], sens: &[Vec<f64>], base: &[f64], lim: &[f64], bounds: &[(f64, f64)]) -> (Vec<f64>, LpStats) {
    let m = hulls.len(); let nl = lim.len();
    let mut col_b = Vec::new(); let mut col_len = Vec::new(); let mut col_c = Vec::new();
    for (b, h) in hulls.iter().enumerate() { for (k, &l) in h.len.iter().enumerate() { col_b.push(b); col_len.push(l); col_c.push(h.slope[k]); } }
    let n = col_b.len();
    let mut base_eff = base.to_vec();
    for l in 0..nl { for b in 0..m { base_eff[l] += sens[l][b] * hulls[b].x0; } }
    let mut lp = Lp { hulls, sens, lim, base, m, n, col_b, col_len, col_c, base_eff, lines: Vec::new(), binv: Vec::new(), basis: Vec::new(), x: vec![0.0; n], at_upper: vec![false; n], is_basic: vec![false; n], stats: LpStats::default() };
    // unconstrained optimum
    for j in 0..n { if lp.col_c[j] > TOL_D { lp.x[j] = lp.col_len[j]; lp.at_upper[j] = true; } }
    let mut budget = MAX_PIVOTS;
    let flows = |u: &[f64]| -> Vec<f64> { (0..nl).map(|l| { let mut f = base[l]; for b in 0..m { f += sens[l][b] * u[b]; } f }).collect() };
    let mut optimal = true;
    for round in 0..MAX_ROUNDS {
        lp.stats.rounds = round + 1;
        let ok = lp.dual_simplex(&mut budget);
        if !ok { optimal = false; break; }
        let u = lp.current_u(); let f = flows(&u);
        let mut viol: Vec<(usize, f64)> = (0..nl).filter(|&l| !lp.lines.contains(&l)).map(|l| (l, f[l].abs() - lim[l])).filter(|x| x.1 > TOL_FLOW).collect();
        if viol.is_empty() { break; }
        viol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        viol.truncate(ADD_MAX); viol.sort_by_key(|x| x.0);
        for (l, _) in viol { let k = lp.lines.len(); lp.lines.push(l); let j = n + k; lp.x.push(0.0); lp.at_upper.push(false); lp.is_basic.push(true); lp.basis.push(j); }
        // recompute slack values, refactor
        lp.stats.active_lines = lp.lines.len();
        if !lp.refactor() { optimal = false; break; }
        // slack basic values = A_row . y (x_B = -B^{-1} N x_N handles it)
        lp.recompute_xb();
        if round + 1 == MAX_ROUNDS { optimal = false; }
    }
    lp.stats.active_lines = lp.lines.len();
    let mut u = lp.current_u();
    for b in 0..m { u[b] = u[b].max(bounds[b].0).min(bounds[b].1); }
    let f = flows(&u); let mut mv = f64::NEG_INFINITY; for l in 0..nl { mv = mv.max(f[l].abs() - lim[l]); }
    lp.stats.max_violation = mv;
    if !optimal || mv > 1e-6 * 1.0 {
        // fallback: deflate toward zero until feasible (u=0 feasible by construction of the challenge)
        lp.stats.fallback = true; lp.stats.status = if !optimal { 1 } else { 2 };
        let orig = u.clone(); let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let feas = |s: f64| -> bool { let us: Vec<f64> = orig.iter().map(|x| x * s).collect(); let f = flows(&us); (0..nl).all(|l| f[l].abs() <= lim[l]) };
        if feas(1.0) { lo = 1.0; } else { for _ in 0..40 { let mid = 0.5 * (lo + hi); if feas(mid) { lo = mid; } else { hi = mid; } } }
        u = orig.iter().map(|x| x * lo).collect();
        let f2 = flows(&u); let mut mv2 = f64::NEG_INFINITY; for l in 0..nl { mv2 = mv2.max(f2[l].abs() - lim[l]); } lp.stats.max_violation = mv2;
    }
    let mut hobj = 0.0; for (b, h) in hulls.iter().enumerate() { let mut rem = u[b] - h.x0; for (k, &l) in h.len.iter().enumerate() { let t = rem.max(0.0).min(l); hobj += h.slope[k] * t; rem -= t; } }
    lp.stats.hull_obj = hobj;
    (u, lp.stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid_max<F: Fn(usize, f64) -> f64>(bounds: &[(f64, f64)], sens: &[Vec<f64>], base: &[f64], lim: &[f64], f: F, n: usize) -> f64 {
        // exhaustive over n^m grid (m <= 3)
        let m = bounds.len(); let mut best = f64::NEG_INFINITY;
        let mut idx = vec![0usize; m];
        loop {
            let u: Vec<f64> = (0..m).map(|b| bounds[b].0 + (bounds[b].1 - bounds[b].0) * idx[b] as f64 / (n - 1) as f64).collect();
            let feas = (0..lim.len()).all(|l| { let mut fl = base[l]; for b in 0..m { fl += sens[l][b] * u[b]; } fl.abs() <= lim[l] });
            if feas { let v: f64 = (0..m).map(|b| f(b, u[b])).sum(); if v > best { best = v; } }
            let mut k = 0; loop { if k == m { return best; } idx[k] += 1; if idx[k] < n { break; } idx[k] = 0; k += 1; }
        }
    }
    fn run_case(bounds: Vec<(f64, f64)>, sens: Vec<Vec<f64>>, base: Vec<f64>, lim: Vec<f64>, fs: Vec<Box<dyn Fn(f64) -> f64>>) {
        let m = bounds.len();
        let hulls: Vec<Hull> = (0..m).map(|b| { let (lo, hi) = bounds[b]; let mut k = vec![lo, hi]; if lo < 0.0 && hi > 0.0 { k.push(0.0); } for i in 1..40 { k.push(lo + (hi - lo) * i as f64 / 40.0); } k.sort_by(|a, c| a.partial_cmp(c).unwrap()); k.dedup(); build_hull(&k, 2, |u| fs[b](u)) }).collect();
        let (u, st) = solve(&hulls, &sens, &base, &lim, &bounds);
        let val: f64 = (0..m).map(|b| fs[b](u[b])).sum();
        let ex = grid_max(&bounds, &sens, &base, &lim, |b, x| fs[b](x), 161);
        let feas = (0..lim.len()).all(|l| { let mut fl = base[l]; for b in 0..m { fl += sens[l][b] * u[b]; } fl.abs() <= lim[l] * (1.0 + 1e-6) });
        assert!(feas, "infeasible: {:?} stats {:?}", u, st);
        assert!(!st.fallback, "fallback {:?}", st);
        // hull optimum >= true optimum (hull is concave envelope of samples); exact value at u within grid error of exhaustive
        let lip: f64 = (0..m).map(|b| { let (lo, hi) = bounds[b]; ((fs[b](hi) - fs[b](lo)).abs() / (hi - lo).max(1e-9)).max(50.0) }).sum();
        let gerr = lip * (0..m).map(|b| (bounds[b].1 - bounds[b].0) / 160.0).fold(0.0, f64::max);
        assert!(val >= ex - gerr - 1e-6, "val {} < exhaustive {} - err {} ; u {:?} stats {:?}", val, ex, gerr, u, st);
        // (the hull of samples lies below a concave f between samples, so no hull>=f assertion)
        let _c0: f64 = (0..m).map(|b| fs[b](hulls[b].x0)).sum();
    }
    #[test] fn uncongested() { run_case(vec![(-25.0, 25.0), (-15.0, 15.0)], vec![vec![0.3, -0.2]], vec![5.0], vec![1000.0], vec![Box::new(|u| 20.0 * u - 0.1 * u * u), Box::new(|u| 5.0 * u - 0.2 * u * u)]); }
    #[test] fn single_binding() { run_case(vec![(-25.0, 25.0), (-15.0, 15.0)], vec![vec![0.6, 0.5]], vec![8.0], vec![12.0], vec![Box::new(|u| 20.0 * u - 0.1 * u * u), Box::new(|u| 18.0 * u - 0.2 * u * u)]); }
    #[test] fn same_node_three() { run_case(vec![(-25.0, 25.0), (-25.0, 25.0), (-20.0, 20.0)], vec![vec![0.5, 0.5, 0.5], vec![-0.2, -0.2, -0.2]], vec![3.0, -4.0], vec![15.0, 6.0], vec![Box::new(|u| 22.0 * u - 0.05 * u * u), Box::new(|u| 22.0 * u - 0.05 * u * u - 0.25 * u.abs()), Box::new(|u| 10.0 * u)]); }
    #[test] fn kink_at_zero_and_negative_price() { run_case(vec![(-25.0, 25.0), (-15.0, 15.0)], vec![vec![0.4, -0.3]], vec![2.0], vec![9.0], vec![Box::new(|u| -30.0 * u - 0.25 * u.abs() - 0.01 * u * u), Box::new(|u| -5.0 * u - 0.25 * u.abs())]); }
    #[test] fn thin_slice_parallel() { run_case(vec![(-25.0, 25.0), (-15.0, 15.0)], vec![vec![0.7, 0.69], vec![0.7, -0.69]], vec![9.0, -9.0], vec![10.0, 10.0], vec![Box::new(|u| 25.0 * u - 0.1 * u * u), Box::new(|u| 23.0 * u - 0.1 * u * u)]); }
    #[test] fn degenerate_tie() { run_case(vec![(-25.0, 25.0), (-25.0, 25.0)], vec![vec![0.5, 0.5]], vec![0.0], vec![10.0], vec![Box::new(|u| 10.0 * u), Box::new(|u| 10.0 * u)]); }
    #[test] fn determinism() {
        let bounds = vec![(-25.0, 25.0), (-15.0, 15.0), (-20.0, 20.0)]; let sens = vec![vec![0.5, 0.5, 0.5], vec![-0.2, 0.4, -0.2], vec![0.1, -0.3, 0.6]]; let base = vec![3.0, -4.0, 1.0]; let lim = vec![15.0, 6.0, 8.0];
        let fs: Vec<Box<dyn Fn(f64) -> f64>> = vec![Box::new(|u| 22.0 * u - 0.05 * u * u), Box::new(|u| 22.0 * u - 0.05 * u * u - 0.25 * u.abs()), Box::new(|u| 10.0 * u - 0.3 * u.abs())];
        let hulls: Vec<Hull> = (0..3).map(|b| { let (lo, hi) = bounds[b]; let mut k = vec![lo, hi, 0.0]; for i in 1..40 { k.push(lo + (hi - lo) * i as f64 / 40.0); } k.sort_by(|a, c| a.partial_cmp(c).unwrap()); k.dedup(); build_hull(&k, 2, |u| fs[b](u)) }).collect();
        let (u1, s1) = solve(&hulls, &sens, &base, &lim, &bounds); let (u2, s2) = solve(&hulls, &sens, &base, &lim, &bounds);
        assert_eq!(u1, u2); assert_eq!(s1.pivots, s2.pivots); assert_eq!(s1.hull_obj.to_bits(), s2.hull_obj.to_bits());
    }
}
