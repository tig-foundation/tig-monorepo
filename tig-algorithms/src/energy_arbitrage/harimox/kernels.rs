#[inline(always)]
pub fn dot_seq(a: &[f64], b: &[f64]) -> f64 {
    dot_acc(0.0, a, b)
}
#[inline(always)]
pub fn dot_acc(mut acc: f64, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let m = n - n % 8;
    for (ac, bc) in a[..m].chunks_exact(8).zip(b[..m].chunks_exact(8)) {
        let ac: &[f64; 8] = ac.try_into().unwrap();
        let bc: &[f64; 8] = bc.try_into().unwrap();
        for j in 0..8 { acc += ac[j] * bc[j]; }
    }
    for k in m..n { acc += a[k] * b[k]; }
    acc
}
#[inline(always)]
pub fn axpy_sub(x: &mut [f64], y: &[f64], f: f64) {
    let n = x.len().min(y.len());
    let (x, y) = (&mut x[..n], &y[..n]);
    let m = n - n % 8;
    let (x8, xr) = x.split_at_mut(m);
    for (xc, yc) in x8.chunks_exact_mut(8).zip(y[..m].chunks_exact(8)) {
        let xc: &mut [f64; 8] = xc.try_into().unwrap();
        let yc: &[f64; 8] = yc.try_into().unwrap();
        for j in 0..8 { xc[j] -= f * yc[j]; }
    }
    for (xv, &yv) in xr.iter_mut().zip(&y[m..]) { *xv -= f * yv; }
}
#[inline(always)]
pub fn axpy_add(x: &mut [f64], y: &[f64], w: f64) {
    let n = x.len().min(y.len());
    let (x, y) = (&mut x[..n], &y[..n]);
    let m = n - n % 8;
    let (x8, xr) = x.split_at_mut(m);
    for (xc, yc) in x8.chunks_exact_mut(8).zip(y[..m].chunks_exact(8)) {
        let xc: &mut [f64; 8] = xc.try_into().unwrap();
        let yc: &[f64; 8] = yc.try_into().unwrap();
        for j in 0..8 { xc[j] += w * yc[j]; }
    }
    for (xv, &yv) in xr.iter_mut().zip(&y[m..]) { *xv += w * yv; }
}

#[inline(always)]
pub fn argmin_below(row: &[f64], start: f64) -> Option<usize> {
    let n = row.len();
    let m = n - n % 8;
    let mut best = start;
    let mut ent = usize::MAX;
    let mut j0 = 0;
    while j0 < m {
        let c: &[f64; 8] = row[j0..j0 + 8].try_into().unwrap();
        for q in 0..8 {
            let take = c[q] < best;
            ent = if take { j0 + q } else { ent };
            best = if take { c[q] } else { best };
        }
        j0 += 8;
    }
    for j in m..n {
        let take = row[j] < best;
        ent = if take { j } else { ent };
        best = if take { row[j] } else { best };
    }
    if ent == usize::MAX { None } else { Some(ent) }
}

#[inline(always)]
pub fn dot_skip_small(a: &[f64], b: &[f64], tol: f64) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let m = n - n % 8;
    let mut acc = 0.0f64;
    for (ac, bc) in a[..m].chunks_exact(8).zip(b[..m].chunks_exact(8)) {
        let ac: &[f64; 8] = ac.try_into().unwrap();
        let bc: &[f64; 8] = bc.try_into().unwrap();
        for j in 0..8 {
            let t = ac[j] * bc[j];
            acc = if ac[j].abs() > tol { acc + t } else { acc };
        }
    }
    for k in m..n {
        let t = a[k] * b[k];
        acc = if a[k].abs() > tol { acc + t } else { acc };
    }
    acc
}

#[inline(always)]
pub fn nonzero_blocks(src: &[f64], flags: &mut Vec<bool>) {
    flags.clear();
    let mut it = src.chunks_exact(8);
    for c in &mut it {
        let c: &[f64; 8] = c.try_into().unwrap();
        let mut nz = false;
        for j in 0..8 { nz |= c[j] != 0.0; }
        flags.push(nz);
    }
    let r = it.remainder();
    if !r.is_empty() {
        let mut nz = false;
        for v in r { nz |= *v != 0.0; }
        flags.push(nz);
    }
}

#[inline(always)]
pub fn argmin_below16(row: &[f64], start: f64) -> Option<usize> {
    let n = row.len();
    let mut best = start;
    let mut ent = usize::MAX;
    let mut j0 = 0;
    while j0 + 16 <= n {
        let c: &[f64; 16] = row[j0..j0 + 16].try_into().unwrap();
        for q in 0..16 {
            let take = c[q] < best;
            ent = if take { j0 + q } else { ent };
            best = if take { c[q] } else { best };
        }
        j0 += 16;
    }
    if j0 + 8 <= n {
        let c: &[f64; 8] = row[j0..j0 + 8].try_into().unwrap();
        for q in 0..8 {
            let take = c[q] < best;
            ent = if take { j0 + q } else { ent };
            best = if take { c[q] } else { best };
        }
        j0 += 8;
    }
    for j in j0..n {
        let take = row[j] < best;
        ent = if take { j } else { ent };
        best = if take { row[j] } else { best };
    }
    if ent == usize::MAX { None } else { Some(ent) }
}

#[inline(always)]
pub fn dot_skip_small16(a: &[f64], b: &[f64], tol: f64) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = 0.0f64;
    let mut k = 0;
    while k + 16 <= n {
        let ac: &[f64; 16] = a[k..k + 16].try_into().unwrap();
        let bc: &[f64; 16] = b[k..k + 16].try_into().unwrap();
        for j in 0..16 {
            let t = ac[j] * bc[j];
            acc = if ac[j].abs() > tol { acc + t } else { acc };
        }
        k += 16;
    }
    if k + 8 <= n {
        let ac: &[f64; 8] = a[k..k + 8].try_into().unwrap();
        let bc: &[f64; 8] = b[k..k + 8].try_into().unwrap();
        for j in 0..8 {
            let t = ac[j] * bc[j];
            acc = if ac[j].abs() > tol { acc + t } else { acc };
        }
        k += 8;
    }
    let r = n - k;
    if r > 0 {
        let last = n - 1;
        for q in 0..7usize {
            let i = if k + q < last { k + q } else { last };
            let on = q < r;
            let (av, bv) = unsafe { (*a.get_unchecked(i), *b.get_unchecked(i)) };
            let t = av * bv;
            acc = if on & (av.abs() > tol) { acc + t } else { acc };
        }
    }
    acc
}
