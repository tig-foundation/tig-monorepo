macro_rules! masked_tail {
    ($r:expr, |$q:ident, $i:ident, $on:ident| $body:expr) => {{
        let r: usize = $r;
        if r > 0 {
            let last = r - 1;
            for $q in 0..7usize {
                let $i = if $q < last { $q } else { last };
                let $on = $q < r;
                $body
            }
        }
    }};
}

#[inline(always)]
pub fn dot_acc(mut acc: f64, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut ai = a.chunks_exact(16);
    let mut bi = b.chunks_exact(16);
    for (x, y) in (&mut ai).zip(&mut bi) {
        let x: &[f64; 16] = x.try_into().unwrap();
        let y: &[f64; 16] = y.try_into().unwrap();
        for j in 0..16 {
            acc += x[j] * y[j];
        }
    }
    let (mut x, mut y) = (ai.remainder(), bi.remainder());
    if x.len() >= 8 {
        let x8: &[f64; 8] = x[..8].try_into().unwrap();
        let y8: &[f64; 8] = y[..8].try_into().unwrap();
        for j in 0..8 {
            acc += x8[j] * y8[j];
        }
        x = &x[8..];
        y = &y[8..];
    }
    let y = &y[..x.len()];
    masked_tail!(x.len(), |q, i, on| {
        let t = unsafe { *x.get_unchecked(i) * *y.get_unchecked(i) };
        let s = acc + t;
        acc = if on { s } else { acc };
    });
    acc
}

#[inline(always)]
pub fn dot_seq(a: &[f64], b: &[f64]) -> f64 {
    dot_acc(0.0, a, b)
}

#[inline(always)]
pub fn dot_sub(mut acc: f64, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut ai = a.chunks_exact(16);
    let mut bi = b.chunks_exact(16);
    for (x, y) in (&mut ai).zip(&mut bi) {
        let x: &[f64; 16] = x.try_into().unwrap();
        let y: &[f64; 16] = y.try_into().unwrap();
        for j in 0..16 {
            acc -= x[j] * y[j];
        }
    }
    let (mut x, mut y) = (ai.remainder(), bi.remainder());
    if x.len() >= 8 {
        let x8: &[f64; 8] = x[..8].try_into().unwrap();
        let y8: &[f64; 8] = y[..8].try_into().unwrap();
        for j in 0..8 {
            acc -= x8[j] * y8[j];
        }
        x = &x[8..];
        y = &y[8..];
    }
    let y = &y[..x.len()];
    masked_tail!(x.len(), |q, i, on| {
        let t = unsafe { *x.get_unchecked(i) * *y.get_unchecked(i) };
        let s = acc - t;
        acc = if on { s } else { acc };
    });
    acc
}

#[inline(always)]
pub fn axpy<const SUB: bool>(x: &mut [f64], y: &[f64], f: f64) {
    let n = x.len().min(y.len());
    let (x, y) = (&mut x[..n], &y[..n]);
    let mut xi = x.chunks_exact_mut(16);
    let mut yi = y.chunks_exact(16);
    for (xc, yc) in (&mut xi).zip(&mut yi) {
        let xc: &mut [f64; 16] = xc.try_into().unwrap();
        let yc: &[f64; 16] = yc.try_into().unwrap();
        for j in 0..16 {
            xc[j] = if SUB { xc[j] - f * yc[j] } else { xc[j] + f * yc[j] };
        }
    }
    let mut xr = xi.into_remainder();
    let mut yr = yi.remainder();
    if xr.len() >= 8 {
        let (x8, xrest) = xr.split_at_mut(8);
        let x8: &mut [f64; 8] = x8.try_into().unwrap();
        let y8: &[f64; 8] = yr[..8].try_into().unwrap();
        for j in 0..8 {
            x8[j] = if SUB { x8[j] - f * y8[j] } else { x8[j] + f * y8[j] };
        }
        xr = xrest;
        yr = &yr[8..];
    }
    let yr = &yr[..xr.len()];
    masked_tail!(xr.len(), |q, i, on| unsafe {
        let v = *xr.get_unchecked(i);
        let yv = *yr.get_unchecked(i);
        let nv = if SUB { v - f * yv } else { v + f * yv };
        *xr.get_unchecked_mut(i) = if on { nv } else { v };
    });
}

#[inline(always)]
pub fn scale(x: &mut [f64], s: f64) {
    let mut xi = x.chunks_exact_mut(16);
    for xc in &mut xi {
        let xc: &mut [f64; 16] = xc.try_into().unwrap();
        for j in 0..16 {
            xc[j] *= s;
        }
    }
    let mut xr = xi.into_remainder();
    if xr.len() >= 8 {
        let (x8, xrest) = xr.split_at_mut(8);
        let x8: &mut [f64; 8] = x8.try_into().unwrap();
        for j in 0..8 {
            x8[j] *= s;
        }
        xr = xrest;
    }
    masked_tail!(xr.len(), |q, i, on| unsafe {
        let v = *xr.get_unchecked(i);
        let nv = v * s;
        *xr.get_unchecked_mut(i) = if on { nv } else { v };
    });
}

#[inline(always)]
pub unsafe fn dot4(r: [*const f64; 4], b: &[f64]) -> [f64; 4] {
    let n = b.len();
    let mut acc = [0.0f64; 4];
    let m8 = n - n % 8;
    let mut k = 0;
    while k < m8 {
        let bc: &[f64; 8] = b.get_unchecked(k..k + 8).try_into().unwrap();
        for t in 0..4 {
            let row = r[t].add(k);
            let mut a = acc[t];
            for j in 0..8 {
                a += *row.add(j) * bc[j];
            }
            acc[t] = a;
        }
        k += 8;
    }
    let rem = n - m8;
    if rem > 0 {
        let last = rem - 1;
        for q in 0..7usize {
            let i = m8 + if q < last { q } else { last };
            let on = q < rem;
            let bv = *b.get_unchecked(i);
            for t in 0..4 {
                let v = *r[t].add(i) * bv;
                let s = acc[t] + v;
                acc[t] = if on { s } else { acc[t] };
            }
        }
    }
    acc
}

#[inline(always)]
pub unsafe fn axpy4_sub(r: [*mut f64; 4], y: &[f64], f: [f64; 4]) {
    let n = y.len();
    let m8 = n - n % 8;
    let mut k = 0;
    while k < m8 {
        let yc: &[f64; 8] = y.get_unchecked(k..k + 8).try_into().unwrap();
        for t in 0..4 {
            let row = r[t].add(k);
            for j in 0..8 {
                *row.add(j) -= f[t] * yc[j];
            }
        }
        k += 8;
    }
    let rem = n - m8;
    if rem > 0 {
        let last = rem - 1;
        for q in 0..7usize {
            let i = m8 + if q < last { q } else { last };
            let on = q < rem;
            let yv = *y.get_unchecked(i);
            for t in 0..4 {
                let p = r[t].add(i);
                let v = *p;
                let nv = v - f[t] * yv;
                *p = if on { nv } else { v };
            }
        }
    }
}

#[inline(always)]
pub unsafe fn acc4(out: &mut [f64], r: [*const f64; 4], w: [f64; 4]) {
    let n = out.len();
    let m8 = n - n % 8;
    let mut k = 0;
    while k < m8 {
        let oc: &mut [f64; 8] = out.get_unchecked_mut(k..k + 8).try_into().unwrap();
        for j in 0..8 {
            let mut v = oc[j];
            for t in 0..4 {
                v += w[t] * *r[t].add(k + j);
            }
            oc[j] = v;
        }
        k += 8;
    }
    let rem = n - m8;
    if rem > 0 {
        let last = rem - 1;
        for q in 0..7usize {
            let i = m8 + if q < last { q } else { last };
            let on = q < rem;
            let v0 = *out.get_unchecked(i);
            let mut v = v0;
            for t in 0..4 {
                v += w[t] * *r[t].add(i);
            }
            *out.get_unchecked_mut(i) = if on { v } else { v0 };
        }
    }
}

#[inline(always)]
pub fn acc_rows(out: &mut [f64], mat: &[f64], stride: usize, rows: &[usize], w: &[f64]) {
    let n = out.len();
    let mut p = 0;
    while p + 4 <= rows.len() {
        let ix = [rows[p], rows[p + 1], rows[p + 2], rows[p + 3]];
        let mx = ix[0].max(ix[1]).max(ix[2].max(ix[3]));
        assert!(mx * stride + n <= mat.len());
        let ptr = mat.as_ptr();
        unsafe {
            acc4(
                out,
                [ptr.add(ix[0] * stride), ptr.add(ix[1] * stride), ptr.add(ix[2] * stride), ptr.add(ix[3] * stride)],
                [w[p], w[p + 1], w[p + 2], w[p + 3]],
            );
        }
        p += 4;
    }
    while p < rows.len() {
        let i = rows[p];
        axpy::<false>(out, &mat[i * stride..i * stride + n], w[p]);
        p += 1;
    }
}

#[inline(always)]
pub fn mat_dots(mat: &[f64], stride: usize, b: &[f64], out: &mut [f64]) {
    let rows = out.len();
    let len = b.len();
    if rows == 0 {
        return;
    }
    assert!((rows - 1) * stride + len <= mat.len());
    let ptr = mat.as_ptr();
    let mut i = 0;
    while i + 4 <= rows {
        let a = unsafe {
            dot4([ptr.add(i * stride), ptr.add((i + 1) * stride), ptr.add((i + 2) * stride), ptr.add((i + 3) * stride)], b)
        };
        out[i..i + 4].copy_from_slice(&a);
        i += 4;
    }
    while i < rows {
        out[i] = dot_seq(&mat[i * stride..i * stride + len], b);
        i += 1;
    }
}

#[inline(always)]
pub fn nonzeros(v: &[f64], idx: &mut Vec<usize>, val: &mut Vec<f64>) -> usize {
    let n = v.len();
    idx.clear();
    val.clear();
    idx.resize(n + 8, 0);
    val.resize(n + 8, 0.0);
    let (ip, vp) = (idx.as_mut_ptr(), val.as_mut_ptr());
    let mut kc = 0usize;
    let m8 = n - n % 8;
    let mut i = 0;
    unsafe {
        while i < m8 {
            for q in 0..8 {
                let w = *v.get_unchecked(i + q);
                *ip.add(kc) = i + q;
                *vp.add(kc) = w;
                kc += (w != 0.0) as usize;
            }
            i += 8;
        }
        let rem = n - m8;
        if rem > 0 {
            let last = rem - 1;
            for q in 0..7usize {
                let ii = m8 + if q < last { q } else { last };
                let w = *v.get_unchecked(ii);
                *ip.add(kc) = ii;
                *vp.add(kc) = w;
                kc += ((q < rem) & (w != 0.0)) as usize;
            }
        }
    }
    idx.truncate(kc);
    val.truncate(kc);
    kc
}

#[inline(always)]
pub fn scale_into(out: &mut [f64], src: &[f64], s: f64) {
    let n = out.len().min(src.len());
    let (out, src) = (&mut out[..n], &src[..n]);
    let mut oi = out.chunks_exact_mut(16);
    let mut si = src.chunks_exact(16);
    for (oc, sc) in (&mut oi).zip(&mut si) {
        let oc: &mut [f64; 16] = oc.try_into().unwrap();
        let sc: &[f64; 16] = sc.try_into().unwrap();
        for j in 0..16 {
            oc[j] = s * sc[j];
        }
    }
    for (o, &v) in oi.into_remainder().iter_mut().zip(si.remainder()) {
        *o = s * v;
    }
}

#[inline(always)]
pub fn scatter_add(u: &mut [f64], idx: &[usize], w: &[f64], x: &[f64]) {
    let n = idx.len().min(w.len()).min(x.len());
    if u.is_empty() || n == 0 {
        return;
    }
    let last = u.len() - 1;
    let (idx, w, x) = (&idx[..n], &w[..n], &x[..n]);
    let m8 = n - n % 8;
    let mut j = 0;
    unsafe {
        while j < m8 {
            for q in 0..8 {
                let k = (*idx.get_unchecked(j + q)).min(last);
                *u.get_unchecked_mut(k) += *w.get_unchecked(j + q) * *x.get_unchecked(j + q);
            }
            j += 8;
        }
        while j < n {
            let k = (*idx.get_unchecked(j)).min(last);
            *u.get_unchecked_mut(k) += *w.get_unchecked(j) * *x.get_unchecked(j);
            j += 1;
        }
    }
}

#[inline(always)]
pub fn sum_acc(mut acc: f64, a: &[f64]) -> f64 {
    let mut ai = a.chunks_exact(16);
    for x in &mut ai {
        let x: &[f64; 16] = x.try_into().unwrap();
        for j in 0..16 {
            acc += x[j];
        }
    }
    let mut x = ai.remainder();
    if x.len() >= 8 {
        let x8: &[f64; 8] = x[..8].try_into().unwrap();
        for j in 0..8 {
            acc += x8[j];
        }
        x = &x[8..];
    }
    masked_tail!(x.len(), |q, i, on| {
        let s = acc + unsafe { *x.get_unchecked(i) };
        acc = if on { s } else { acc };
    });
    acc
}

#[inline(always)]
pub fn relax_multi(best: &mut [f64], prices: &[f64], acts: &[(f64, f64, f64)]) {
    let n = best.len().min(prices.len());
    let (best, prices) = (&mut best[..n], &prices[..n]);
    let m8 = n - n % 8;
    let rem = n - m8;
    let last = if rem > 0 { rem - 1 } else { 0 };
    for g in acts.chunks_exact(4) {
        let g: &[(f64, f64, f64); 4] = g.try_into().unwrap();
        let mut k = 0;
        while k < m8 {
            let bc: &mut [f64; 8] = unsafe { best.get_unchecked_mut(k..k + 8) }.try_into().unwrap();
            let pc: &[f64; 8] = unsafe { prices.get_unchecked(k..k + 8) }.try_into().unwrap();
            for a in 0..4 {
                let (rev, cost, cont) = g[a];
                for j in 0..8 {
                    let val = rev * pc[j] - cost + cont;
                    let b = bc[j];
                    bc[j] = if val > b { val } else { b };
                }
            }
            k += 8;
        }
        if rem > 0 && n >= 8 {
            let bc: &mut [f64; 8] = unsafe { best.get_unchecked_mut(n - 8..n) }.try_into().unwrap();
            let pc: &[f64; 8] = unsafe { prices.get_unchecked(n - 8..n) }.try_into().unwrap();
            for a in 0..4 {
                let (rev, cost, cont) = g[a];
                for j in 0..8 {
                    let val = rev * pc[j] - cost + cont;
                    let b = bc[j];
                    bc[j] = if val > b { val } else { b };
                }
            }
        } else if rem > 0 {
            unsafe {
                for q in 0..7usize {
                    let i = m8 + if q < last { q } else { last };
                    let on = q < rem;
                    let p = *prices.get_unchecked(i);
                    let b0 = *best.get_unchecked(i);
                    let mut b = b0;
                    for a in 0..4 {
                        let (rev, cost, cont) = g[a];
                        let val = rev * p - cost + cont;
                        b = if val > b { val } else { b };
                    }
                    *best.get_unchecked_mut(i) = if on { b } else { b0 };
                }
            }
        }
    }
}

#[inline(always)]
pub fn rows_sub(t: &mut [f64], stride: usize, rows: &[usize], fs: &[f64], prow: &[f64]) {
    let len = prow.len();
    let mut p = 0;
    let base = t.as_mut_ptr();
    let tl = t.len();
    while p + 4 <= rows.len() {
        let ix = [rows[p], rows[p + 1], rows[p + 2], rows[p + 3]];
        let mx = ix[0].max(ix[1]).max(ix[2].max(ix[3]));
        assert!(mx * stride + len <= tl);
        unsafe {
            axpy4_sub(
                [base.add(ix[0] * stride), base.add(ix[1] * stride), base.add(ix[2] * stride), base.add(ix[3] * stride)],
                prow,
                [fs[p], fs[p + 1], fs[p + 2], fs[p + 3]],
            );
        }
        p += 4;
    }
    while p < rows.len() {
        let i = rows[p];
        axpy::<true>(&mut t[i * stride..i * stride + len], prow, fs[p]);
        p += 1;
    }
}

#[inline(always)]
pub fn gather<const W: bool>(out: &mut [f64], src: &[f64], idx: &[usize], w: &[f64]) {
    let n = out.len();
    if n == 0 {
        return;
    }
    assert!(!src.is_empty() && idx.len() >= n && (!W || w.len() >= n));
    let last = src.len() - 1;
    let m8 = n - n % 8;
    let mut j = 0;
    unsafe {
        while j < m8 {
            for q in 0..8 {
                let v = *src.get_unchecked((*idx.get_unchecked(j + q)).min(last));
                *out.get_unchecked_mut(j + q) = if W { *w.get_unchecked(j + q) * v } else { v };
            }
            j += 8;
        }
        while j < n {
            let v = *src.get_unchecked((*idx.get_unchecked(j)).min(last));
            *out.get_unchecked_mut(j) = if W { *w.get_unchecked(j) * v } else { v };
            j += 1;
        }
    }
}

#[inline(always)]
pub fn sel(c: bool, a: f64, b: f64) -> f64 {
    let m = (c as u64).wrapping_neg();
    f64::from_bits((a.to_bits() & m) | (b.to_bits() & !m))
}
