//! Fuse four dense contribution passes while retaining the exact i32 value
//! each sequential operation would see. Integer wrapping and row order stay
//! unchanged, including for asymmetric matrices and duplicate input IDs.
pub(super) fn apply(contrib: &mut [i32], q: &[Vec<i32>], ids: &[usize], add: bool) -> i64 {
    let n = contrib.len();
    let mut delta = 0i64;
    for group in ids.chunks(4) {
        for (p, &i) in group.iter().enumerate() {
            let mut c = contrib[i];
            for &prior in &group[..p] {
                c = if add {
                    c.wrapping_add(q[prior][i])
                } else {
                    c.wrapping_sub(q[prior][i])
                };
            }
            delta = if add {
                delta.wrapping_add(c as i64)
            } else {
                delta.wrapping_sub(c as i64)
            };
        }
        if group.len() == 4 {
            let a = q[group[0]].as_ptr();
            let b = q[group[1]].as_ptr();
            let c = q[group[2]].as_ptr();
            let d = q[group[3]].as_ptr();
            let out = contrib.as_mut_ptr();
            // Every row belongs to the same n-by-n challenge matrix.
            unsafe {
                if add {
                    for i in 0..n {
                        *out.add(i) = (*out.add(i))
                            .wrapping_add(*a.add(i))
                            .wrapping_add(*b.add(i))
                            .wrapping_add(*c.add(i))
                            .wrapping_add(*d.add(i));
                    }
                } else {
                    for i in 0..n {
                        *out.add(i) = (*out.add(i))
                            .wrapping_sub(*a.add(i))
                            .wrapping_sub(*b.add(i))
                            .wrapping_sub(*c.add(i))
                            .wrapping_sub(*d.add(i));
                    }
                }
            }
        } else {
            for &id in group {
                let row = &q[id];
                if add {
                    for (c, &v) in contrib.iter_mut().zip(row) {
                        *c = c.wrapping_add(v);
                    }
                } else {
                    for (c, &v) in contrib.iter_mut().zip(row) {
                        *c = c.wrapping_sub(v);
                    }
                }
            }
        }
    }
    delta
}
/// Stable differences of two 0/1 membership arrays. Both output buffers have
/// room for every ID; dummy writes touch only the uncommitted buffer tails.
pub(super) fn differences(
    selected: &[bool],
    target: &[u8],
    removed: &mut Vec<usize>,
    added: &mut Vec<usize>,
) {
    assert_eq!(selected.len(), target.len());
    let n = selected.len();
    removed.clear();
    added.clear();
    removed.reserve(n);
    added.reserve(n);
    let mut nr = 0;
    let mut na = 0;
    unsafe {
        let rp = removed.as_mut_ptr();
        let ap = added.as_mut_ptr();
        let sp = selected.as_ptr();
        let tp = target.as_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                let s = *sp.add(i);
                let t = *tp.add(i) != 0;
                rp.add(nr).write(i);
                ap.add(na).write(i);
                nr += (s && !t) as usize;
                na += (!s && t) as usize;
            }};
        }
        let mut i = 0;
        let full = n / 16 * 16;
        while i < full {
            emit!(i);
            emit!(i + 1);
            emit!(i + 2);
            emit!(i + 3);
            emit!(i + 4);
            emit!(i + 5);
            emit!(i + 6);
            emit!(i + 7);
            emit!(i + 8);
            emit!(i + 9);
            emit!(i + 10);
            emit!(i + 11);
            emit!(i + 12);
            emit!(i + 13);
            emit!(i + 14);
            emit!(i + 15);
            i += 16;
        }
        while i < n {
            emit!(i);
            i += 1;
        }
        removed.set_len(nr);
        added.set_len(na);
    }
}

pub(super) fn bool_bytes(bits: &[bool]) -> &[u8] {
    // Rust bool occupies one byte and has exactly the valid bit patterns 0/1.
    unsafe { std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len()) }
}

/// Evaluate the original sequential removals and additions without materializing
/// every contribution. Each prior move visits one matrix row, and updates all
/// later changed items. Their i32 arithmetic occurs in the original order, even
/// for asymmetric matrices, repeated IDs and wrapped contributions.
pub(super) fn value_delta(
    contrib: &[i32],
    q: &[Vec<i32>],
    removed: &[usize],
    added: &[usize],
) -> i64 {
    let n = contrib.len();
    // These checked gathers validate every target ID used below.
    let mut r: Vec<i32> = removed.iter().map(|&i| contrib[i]).collect();
    let mut a: Vec<i32> = added.iter().map(|&i| contrib[i]).collect();
    for (p, &id) in removed.iter().enumerate() {
        let row = &q[id][..n];
        adjust::<false>(&mut r[p + 1..], &removed[p + 1..], row);
        adjust::<false>(&mut a, added, row);
    }
    for (p, &id) in added.iter().enumerate() {
        adjust::<true>(&mut a[p + 1..], &added[p + 1..], &q[id][..n]);
    }
    let removed_value: i64 = r.iter().map(|&c| c as i64).sum();
    let added_value: i64 = a.iter().map(|&c| c as i64).sum();
    added_value.wrapping_sub(removed_value)
}
fn adjust<const ADD: bool>(values: &mut [i32], ids: &[usize], row: &[i32]) {
    debug_assert_eq!(values.len(), ids.len());
    unsafe {
        let vp = values.as_mut_ptr();
        let ip = ids.as_ptr();
        let qp = row.as_ptr();
        macro_rules! visit {
            ($p:expr) => {{
                let p = $p;
                let old = *vp.add(p);
                let q = *qp.add(*ip.add(p));
                *vp.add(p) = if ADD {
                    old.wrapping_add(q)
                } else {
                    old.wrapping_sub(q)
                };
            }};
        }
        let mut p = 0;
        let full = ids.len() / 8 * 8;
        while p < full {
            visit!(p);
            visit!(p + 1);
            visit!(p + 2);
            visit!(p + 3);
            visit!(p + 4);
            visit!(p + 5);
            visit!(p + 6);
            visit!(p + 7);
            p += 8;
        }
        while p < ids.len() {
            visit!(p);
            p += 1;
        }
    }
}
