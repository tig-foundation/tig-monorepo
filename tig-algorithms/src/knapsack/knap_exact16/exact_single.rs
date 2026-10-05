//! Exact one-for-one search with weight buckets and original window ordinals.
pub(super) fn best<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    if unused.len() > 256 {
        return best_heap(unused, used, weights, contrib, slack, cross);
    }
    // The fast bounds rely on nonnegative TIG interactions and nonoverflowing
    // contributions. Keep the original wrapping delta outside the small-weight
    // range or when contributions approach the i32 limit.
    if unused
        .iter()
        .chain(used)
        .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
    {
        let mut best = 0i32;
        let mut movement = None;
        for &r in used {
            for &a in unused {
                if weights[a] > weights[r] + slack {
                    continue;
                }
                let delta = contrib[a]
                    .wrapping_sub(contrib[r])
                    .wrapping_sub(cross(r, a));
                if delta > best {
                    best = delta;
                    movement = Some((a, r));
                }
            }
        }
        return movement;
    }
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
    }
    for &r in used {
        min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
    }
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
    {
        return None;
    }
    // At most 256 entries can reach any weight class. Uninitialized
    // stack storage avoids the per-class allocation and growth sequence.
    let mut storage = [0u64; 11 * 256];
    let mut counts = [0usize; 11];
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weights[a] as usize;
        let lower = weights[a].saturating_sub(slack).max(1) as usize;
        if contrib[a] <= min_remove[lower] {
            continue;
        }
        let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
        let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
        storage[w * 256 + counts[w]] = entry;
        counts[w] += 1;
    }
    for w in 1..=10 {
        let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut buckets: [&[u64]; 11] = [&[]; 11];
    for w in 0..11 {
        let __lo = w * 256;
        let __n = counts[w];
        buckets[w] = &storage[__lo..__lo + __n];
    }
    let buckets = buckets;
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contrib[r] as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket.iter() {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contrib[r] as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unused[ai];
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}

pub(super) fn best_heap<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    // The fast bounds rely on nonnegative TIG interactions and nonoverflowing
    // contributions. Keep the original wrapping delta outside the small-weight
    // range or when contributions approach the i32 limit.
    if unused
        .iter()
        .chain(used)
        .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
    {
        let mut best = 0i32;
        let mut movement = None;
        for &r in used {
            for &a in unused {
                if weights[a] > weights[r] + slack {
                    continue;
                }
                let delta = contrib[a]
                    .wrapping_sub(contrib[r])
                    .wrapping_sub(cross(r, a));
                if delta > best {
                    best = delta;
                    movement = Some((a, r));
                }
            }
        }
        return movement;
    }
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
    }
    for &r in used {
        min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
    }
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
    {
        return None;
    }
    let mut buckets: [Vec<u64>; 11] = Default::default();
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weights[a] as usize;
        let lower = weights[a].saturating_sub(slack).max(1) as usize;
        if contrib[a] <= min_remove[lower] {
            continue;
        }
        let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
        buckets[w].push((c << 32) | (u32::MAX as usize - ordinal) as u64);
    }
    for bucket in &mut buckets {
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contrib[r] as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contrib[r] as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unused[ai];
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}

/// Fused validation and extrema setup for repeatedly rebuilt windows.
pub(super) fn best_fused<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    if unused.len() > 256 {
        return best_heap(unused, used, weights, contrib, slack, cross);
    }
    // Validate and build extrema in the same pass. If any guard fails,
    // retain the original full-range search through best_heap's fallback.
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        let w = weights[a];
        if !(1..=10).contains(&w) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        let c = contrib[a];
        if (c as i64).abs() >= (1i64 << 28) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        max_add[w as usize] = max_add[w as usize].max(c);
    }
    for &r in used {
        let w = weights[r];
        if !(1..=10).contains(&w) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        let c = contrib[r];
        if (c as i64).abs() >= (1i64 << 28) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        min_remove[w as usize] = min_remove[w as usize].min(c);
    }
    // Every subsequent ID has just been checked in both immutable arrays.
    // Its weight is in 1..=10 and its contribution passes the overflow guard.
    let weight = |i: usize| unsafe { *weights.get_unchecked(i) };
    let contribution = |i: usize| unsafe { *contrib.get_unchecked(i) };
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weight(r) as u64 + slack as u64).min(10) as usize] > contribution(r))
    {
        return None;
    }
    // At most 256 entries can reach any weight class. Uninitialized
    // stack storage avoids the per-class allocation and growth sequence.
    let mut storage = [0u64; 11 * 256];
    let mut counts = [0usize; 11];
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weight(a) as usize;
        let lower = weight(a).saturating_sub(slack).max(1) as usize;
        if contribution(a) <= unsafe { *min_remove.get_unchecked(lower) } {
            continue;
        }
        let c = ((contribution(a) as u32) ^ 0x80000000) as u64;
        let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
        let __i = unsafe { *counts.get_unchecked(w) };
        storage[w * 256 + __i] = entry;
        unsafe {
            *counts.get_unchecked_mut(w) += 1;
        }
    }
    for w in 1..=10 {
        let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut buckets: [&[u64]; 11] = [&[]; 11];
    for w in 0..11 {
        let __lo = w * 256;
        let __n = counts[w];
        buckets[w] = &storage[__lo..__lo + __n];
    }
    let buckets = buckets;
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weight(r) as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contribution(r) as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket.iter() {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contribution(r) as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unsafe { *unused.get_unchecked(ai) };
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}
