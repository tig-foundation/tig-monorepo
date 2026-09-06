//! A usize specialization of the pinned Rust 1.86 ipnsort permutation.
//! Pivot samples, cyclic partition, equal-key partitions and small networks
//! retain their original order. The partition loop is unrolled eight ways.
//! Derived from the Rust standard library's ipnsort and introselect, by Lukas
//! Bergdoll and Orson Peters (MIT OR Apache-2.0; see LICENSE-RUST).
//! Use with the TIG dev:0.0.7 compiler.

#[inline(always)]
fn compare_swap<F: Fn(usize, usize) -> bool>(v: &mut [usize], a: usize, b: usize, less: &F) {
    unsafe {
        let x = *v.get_unchecked(a);
        let y = *v.get_unchecked(b);
        let swap = less(y, x);
        *v.get_unchecked_mut(a) = if swap { y } else { x };
        *v.get_unchecked_mut(b) = if swap { x } else { y };
    }
}

pub(super) fn pack_density(out: &mut [usize], contrib: &[i32], weights: &[u32]) {
    const FACTOR: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    assert_eq!(out.len(), contrib.len());
    assert_eq!(out.len(), weights.len());
    let n = out.len();
    let mut i = 0;
    unsafe {
        let p = out.as_mut_ptr();
        let c = contrib.as_ptr();
        let w = weights.as_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                let factor = FACTOR[(*w.add(i) as usize).clamp(1, 10)];
                *p.add(i) = (((*c.add(i) as i64 * factor) << 16) as usize) | i;
            }};
        }
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
    }
}

pub(super) fn density_order(
    ids: &mut [usize],
    contrib: &[i32],
    weights: &[u32],
    small_weights: bool,
) {
    if usize::BITS == 64 && weights.len() <= u16::MAX as usize && small_weights {
        const F: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        for item in ids.iter_mut() {
            let i = *item;
            *item = (((contrib[i] as i64 * F[weights[i] as usize]) << 16) as usize) | i;
        }
        if compatible() {
            sort(ids, |a, b| ((a as i64) >> 16) > ((b as i64) >> 16));
        } else {
            ids.sort_unstable_by(|&a, &b| ((b as i64) >> 16).cmp(&((a as i64) >> 16)));
        }
        for item in ids {
            *item &= 0xffff;
        }
    } else {
        ids.sort_unstable_by(|&a, &b| {
            ((contrib[b] as i64) * (weights[a] as i64).max(1))
                .cmp(&((contrib[a] as i64) * (weights[b] as i64).max(1)))
        });
    }
}

/// Keep the key inside each sort element, so comparisons require no further
/// indirection. The fallback retains the unbounded original key representation.
pub(super) fn keyed_prefix(
    ids: &mut [usize],
    keys: &[i64],
    weights: &[u32],
    capacity: u32,
    weight_first: bool,
) -> usize {
    if usize::BITS == 64 && weights.len() <= u16::MAX as usize {
        let mut fits = true;
        for item in ids.iter_mut() {
            let i = *item;
            let mut key = keys[i];
            if weight_first {
                fits &= weights[i] <= 1024 && (-(1i64 << 31)..=(1i64 << 31)).contains(&key);
                key = (weights[i] as i64)
                    .wrapping_mul(1i64 << 33)
                    .wrapping_add(key);
            }
            fits &= (-(1i64 << 47)..(1i64 << 47)).contains(&key);
            *item = ((key << 16) as usize) | i;
        }
        if fits {
            let end = capacity_prefix(
                ids,
                |a, b| ((a as i64) >> 16) < ((b as i64) >> 16),
                |key| weights[key & 0xffff],
                capacity,
                0,
            );
            for item in ids {
                *item &= 0xffff;
            }
            return end;
        }
        for item in ids.iter_mut() {
            *item &= 0xffff;
        }
    }
    capacity_prefix(
        ids,
        |a, b| {
            if weight_first {
                weights[a]
                    .cmp(&weights[b])
                    .then(keys[a].cmp(&keys[b]))
                    .is_lt()
            } else {
                keys[a] < keys[b]
            }
        },
        |i| weights[i],
        capacity,
        0,
    )
}

fn insertion<F: Fn(usize, usize) -> bool>(v: &mut [usize], start: usize, less: &F) {
    for i in start..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && less(x, v[j - 1]) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

fn small<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let middle = if n < 18 { n } else { n / 2 };
    for range in [0..middle, middle..n] {
        let part = &mut v[range];
        let sorted = if part.len() >= 13 {
            sort13_optimal(part, less);
            13
        } else if part.len() >= 9 {
            sort9_optimal(part, less);
            9
        } else {
            1
        };
        insertion(part, sorted, less);
    }
    if middle == n {
        return;
    }
    // The original bidirectional merge is stable between its two runs.
    let mut scratch = [0usize; 32];
    let (mut a, mut b) = (0, middle);
    for dest in &mut scratch[..n] {
        if b == n || (a < middle && !less(v[b], v[a])) {
            *dest = v[a];
            a += 1;
        } else {
            *dest = v[b];
            b += 1;
        }
    }
    v.copy_from_slice(&scratch[..n]);
}

#[inline(always)]
fn median<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    a: usize,
    b: usize,
    c: usize,
    less: &F,
) -> usize {
    let x = less(v[a], v[b]);
    let y = less(v[a], v[c]);
    if x == y {
        if less(v[b], v[c]) ^ x {
            c
        } else {
            b
        }
    } else {
        a
    }
}
fn median_rec<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    mut a: usize,
    mut b: usize,
    mut c: usize,
    n: usize,
    less: &F,
) -> usize {
    if n * 8 >= 64 {
        let k = n / 8;
        a = median_rec(v, a, a + 4 * k, a + 7 * k, k, less);
        b = median_rec(v, b, b + 4 * k, b + 7 * k, k, less);
        c = median_rec(v, c, c + 4 * k, c + 7 * k, k, less);
    }
    median(v, a, b, c, less)
}
fn pivot<F: Fn(usize, usize) -> bool>(v: &[usize], less: &F) -> usize {
    let k = v.len() / 8;
    if v.len() < 64 {
        median(v, 0, k * 4, k * 7, less)
    } else {
        median_rec(v, 0, k * 4, k * 7, k, less)
    }
}

fn partition<F: Fn(usize, usize) -> bool, const EQUAL: bool>(
    v: &mut [usize],
    pivot: usize,
    less: &F,
) -> usize {
    v.swap(0, pivot);
    let value = v[0];
    let n = v.len() - 1;
    if n == 0 {
        return 0;
    }
    // After saving element 1, right visits elements 2..len then the saved
    // element. Every cyclic write is the same as the standard partition.
    unsafe {
        let p = v.as_mut_ptr().add(1);
        let saved = *p;
        let mut count = 0usize;
        let mut gap = 0usize;
        let mut right = 1usize;
        macro_rules! step {
            ($x:expr,$to:expr) => {{
                let x = $x;
                let yes = if EQUAL {
                    !less(value, x)
                } else {
                    less(x, value)
                };
                *p.add(gap) = *p.add(count);
                *p.add(count) = x;
                gap = $to;
                count += yes as usize;
            }};
        }
        // Earlier cyclic writes touch only positions <= right. The next
        // four values can therefore be loaded and compared before any write,
        // retaining the exact comparator and movement order of the partition.
        macro_rules! batch4 {
            () => {{
                let x0 = *p.add(right);
                let x1 = *p.add(right + 1);
                let x2 = *p.add(right + 2);
                let x3 = *p.add(right + 3);
                let y0 = if EQUAL {
                    !less(value, x0)
                } else {
                    less(x0, value)
                };
                let y1 = if EQUAL {
                    !less(value, x1)
                } else {
                    less(x1, value)
                };
                let y2 = if EQUAL {
                    !less(value, x2)
                } else {
                    less(x2, value)
                };
                let y3 = if EQUAL {
                    !less(value, x3)
                } else {
                    less(x3, value)
                };
                macro_rules! commit {
                    ($x:expr,$yes:expr) => {{
                        *p.add(gap) = *p.add(count);
                        *p.add(count) = $x;
                        gap = right;
                        count += $yes as usize;
                        right += 1;
                    }};
                }
                commit!(x0, y0);
                commit!(x1, y1);
                commit!(x2, y2);
                commit!(x3, y3);
            }};
        }
        while right + 8 <= n {
            batch4!();
            batch4!();
        }
        while right < n {
            step!(*p.add(right), right);
            right += 1;
        }
        step!(saved, 0);
        let _ = gap;
        v.swap(0, count);
        count
    }
}

fn heap<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    let n = v.len();
    for i in (0..n + n / 2).rev() {
        let mut node = if i >= n {
            i - n
        } else {
            v.swap(0, i);
            0
        };
        let len = i.min(n);
        loop {
            let mut child = node * 2 + 1;
            if child >= len {
                break;
            }
            if child + 1 < len {
                child += less(v[child], v[child + 1]) as usize;
            }
            if !less(v[node], v[child]) {
                break;
            }
            v.swap(node, child);
            node = child;
        }
    }
}

fn quick<F, S>(
    mut v: &mut [usize],
    mut ancestor: Option<usize>,
    mut limit: u32,
    mut offset: usize,
    less: &F,
    sink: &mut S,
) -> bool
where
    F: Fn(usize, usize) -> bool,
    S: FnMut(&[usize], usize) -> bool,
{
    loop {
        if v.len() <= 32 {
            small(v, less);
            return sink(v, offset);
        }
        if limit == 0 {
            heap(v, less);
            return sink(v, offset);
        }
        limit -= 1;
        let pos = pivot(v, less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, less) + 1;
            if !sink(&v[..count], offset) {
                return false;
            }
            offset += count;
            v = &mut v[count..];
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, less);
        let (left, right) = v.split_at_mut(count);
        let (p, right) = right.split_at_mut(1);
        if !quick(left, ancestor, limit, offset, less, sink) {
            return false;
        }
        if !sink(p, offset + count) {
            return false;
        }
        ancestor = Some(p[0]);
        offset += count + 1;
        v = right;
    }
}

pub(super) fn sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: F) {
    visit(v, &less, &mut |_, _| true);
}

fn visit<F, S>(v: &mut [usize], less: &F, sink: &mut S)
where
    F: Fn(usize, usize) -> bool,
    S: FnMut(&[usize], usize) -> bool,
{
    let n = v.len();
    if n <= 20 {
        insertion(v, 1, less);
        sink(v, 0);
        return;
    }
    let descending = less(v[1], v[0]);
    let mut run = 2;
    while run < n
        && if descending {
            less(v[run], v[run - 1])
        } else {
            !less(v[run], v[run - 1])
        }
    {
        run += 1;
    }
    if run == n {
        if descending {
            v.reverse();
        }
        sink(v, 0);
        return;
    }
    quick(v, None, 2 * (n | 1).ilog2(), 0, less, sink);
}

/// Produce the exact sorted prefix needed by a greedy capacity scan plus pad.
/// Positive weights ensure that capacity zero ends the original scan.
pub(super) fn capacity_prefix<F, W>(
    v: &mut [usize],
    less: F,
    weight: W,
    capacity: u32,
    pad: usize,
) -> usize
where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    let n = v.len();
    let mut rem = capacity;
    let mut goal = n;
    if rem == 0 {
        goal = pad.min(n);
    }
    visit(v, &less, &mut |part, offset| {
        for (i, &item) in part.iter().enumerate() {
            let index = offset + i;
            if index >= goal {
                return false;
            }
            if rem > 0 {
                let w = weight(item);
                if w <= rem {
                    rem -= w;
                    if rem == 0 {
                        goal = (index + pad + 1).min(n);
                    }
                }
            }
        }
        offset + part.len() < goal
    });
    goal
}

/// Sort the exact DP core while retaining only membership in the locked prefix.
/// A whole partition may remain unsorted when its total weight leaves enough
/// capacity for `half + 1` further maximum-weight items. Thus every item in it
/// precedes the DP core, regardless of its internal permutation. The pivot and
/// the other partition are unchanged, so subsequent tie permutations stay exact.
pub(super) fn capacity_core<F, W>(
    v: &mut [usize],
    less: F,
    weight: W,
    capacity: u32,
    half: usize,
    max_weight: u32,
) where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    let n = v.len();
    let mut scan = CoreScan {
        rem: capacity as u64,
        goal: if capacity == 0 {
            half.saturating_add(1).min(n)
        } else {
            n
        },
        half,
        reserve: (half as u64 + 1).saturating_mul(max_weight as u64),
        n,
    };
    if n <= 20 {
        insertion(v, 1, &less);
        return;
    }
    let descending = less(v[1], v[0]);
    let mut run = 2;
    while run < n
        && if descending {
            less(v[run], v[run - 1])
        } else {
            !less(v[run], v[run - 1])
        }
    {
        run += 1;
    }
    if run == n {
        if descending {
            v.reverse();
        }
        return;
    }
    core_quick(v, None, 2 * (n | 1).ilog2(), 0, &less, &weight, &mut scan);
}
struct CoreScan {
    rem: u64,
    goal: usize,
    half: usize,
    reserve: u64,
    n: usize,
}
impl CoreScan {
    fn consume<W: Fn(usize) -> u32>(&mut self, part: &[usize], offset: usize, weight: &W) -> bool {
        for (i, &item) in part.iter().enumerate() {
            let index = offset + i;
            if index >= self.goal {
                return false;
            }
            if self.rem > 0 {
                let w = weight(item) as u64;
                if w <= self.rem {
                    self.rem -= w;
                    if self.rem == 0 {
                        self.goal = (index + self.half + 1).min(self.n);
                    }
                }
            }
        }
        offset + part.len() < self.goal
    }
}
fn core_quick<F, W>(
    mut v: &mut [usize],
    mut ancestor: Option<usize>,
    mut limit: u32,
    mut offset: usize,
    less: &F,
    weight: &W,
    scan: &mut CoreScan,
) -> bool
where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    loop {
        if offset >= scan.goal {
            return false;
        }
        if scan.rem >= scan.reserve
            && !v.is_empty()
            && offset + v.len() <= scan.n.saturating_sub(scan.half.saturating_add(1))
        {
            let total: u64 = v.iter().map(|&i| weight(i) as u64).sum();
            if total <= scan.rem - scan.reserve {
                scan.rem -= total;
                return true;
            }
        }
        if v.len() <= 32 {
            small(v, less);
            return scan.consume(v, offset, weight);
        }
        if limit == 0 {
            heap(v, less);
            return scan.consume(v, offset, weight);
        }
        limit -= 1;
        let pos = pivot(v, less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, less) + 1;
            if !scan.consume(&v[..count], offset, weight) {
                return false;
            }
            offset += count;
            v = &mut v[count..];
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, less);
        let (left, right) = v.split_at_mut(count);
        let (p, right) = right.split_at_mut(1);
        if !core_quick(left, ancestor, limit, offset, less, weight, scan) {
            return false;
        }
        if !scan.consume(p, offset + count, weight) {
            return false;
        }
        ancestor = Some(p[0]);
        offset += count + 1;
        v = right;
    }
}

fn sort9_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    compare_swap(v, 0, 3, less);
    compare_swap(v, 1, 7, less);
    compare_swap(v, 2, 5, less);
    compare_swap(v, 4, 8, less);
    compare_swap(v, 0, 7, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 3, 8, less);
    compare_swap(v, 5, 6, less);
    compare_swap(v, 0, 2, less);
    compare_swap(v, 1, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 1, 4, less);
    compare_swap(v, 3, 6, less);
    compare_swap(v, 5, 7, less);
    compare_swap(v, 0, 1, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 3, 5, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 6, 7, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 6, less);
}

fn sort13_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    compare_swap(v, 0, 12, less);
    compare_swap(v, 1, 10, less);
    compare_swap(v, 2, 9, less);
    compare_swap(v, 3, 7, less);
    compare_swap(v, 5, 11, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 1, 6, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 11, less);
    compare_swap(v, 7, 9, less);
    compare_swap(v, 8, 10, less);
    compare_swap(v, 0, 4, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 6, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 11, 12, less);
    compare_swap(v, 4, 6, less);
    compare_swap(v, 5, 9, less);
    compare_swap(v, 8, 11, less);
    compare_swap(v, 10, 12, less);
    compare_swap(v, 0, 5, less);
    compare_swap(v, 3, 8, less);
    compare_swap(v, 4, 7, less);
    compare_swap(v, 6, 11, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 0, 1, less);
    compare_swap(v, 2, 5, less);
    compare_swap(v, 6, 9, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 10, 11, less);
    compare_swap(v, 1, 3, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 5, 6, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 7, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 6, 7, less);
    compare_swap(v, 8, 9, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 6, less);
}

/// Disable this specialization when the host stdlib uses a different tie permutation.
/// Production validation targets exactly nightly-2025-02-10 in TIG dev:0.0.7.
fn compatible_uncached() -> bool {
        for n in [17usize, 31, 89, 257, 1001] {
            for distinct in [3usize, 19, 511] {
                let mut a: Vec<usize> = (0..n).collect();
                let key = |i: usize| i.wrapping_mul(7919).wrapping_add(i / 7) % distinct;
                let mut b = a.clone();
                a.sort_unstable_by(|&x, &y| key(y).cmp(&key(x)));
                sort(&mut b, |x, y| key(x) > key(y));
                if a != b {
                    return false;
                }
                for k in [0, n - 1, n / 3] {
                    let mut a: Vec<usize> = (0..n).collect();
                    let mut b = a.clone();
                    a.select_nth_unstable_by(k, |&x, &y| key(y).cmp(&key(x)));
                    select(&mut b, k, |x, y| key(x) > key(y));
                    if a != b {
                        return false;
                    }
                }
            }
        }
        true
    }

pub(super) fn compatible() -> bool {
    thread_local! { static CHECK: bool = compatible_uncached(); }
    CHECK.with(|c| *c)
}

/// Exact pinned introselect, using the same pivot and cyclic partition order.
pub(super) fn select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: F) {
    assert!(index < v.len());
    if index == 0 || index == v.len() - 1 {
        let p = extreme(v, index != 0, &less);
        v.swap(p, index);
        return;
    }
    let mut limit = 16;
    let mut ancestor = None;
    loop {
        if v.len() <= 16 {
            insertion(v, 1, &less);
            return;
        }
        if limit == 0 {
            median_select(v, index, &less);
            return;
        }
        limit -= 1;
        let pos = pivot(v, &less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, &less) + 1;
            if count > index {
                return;
            }
            v = &mut v[count..];
            index -= count;
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, &less);
        if count == index {
            return;
        }
        if count < index {
            ancestor = Some(v[count]);
            v = &mut v[count + 1..];
            index -= count + 1;
        } else {
            v = &mut v[..count];
        }
    }
}
fn extreme<F: Fn(usize, usize) -> bool>(v: &[usize], max: bool, less: &F) -> usize {
    let mut chosen = 0;
    for i in 1..v.len() {
        if if max {
            less(v[chosen], v[i])
        } else {
            less(v[i], v[chosen])
        } {
            chosen = i;
        }
    }
    chosen
}
fn median_select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: &F) {
    loop {
        if v.len() <= 16 {
            insertion(v, 1, less);
            return;
        }
        if index == 0 || index == v.len() - 1 {
            let p = extreme(v, index != 0, less);
            v.swap(p, index);
            return;
        }
        let frac = if v.len() <= 1024 {
            v.len() / 12
        } else if v.len() <= 128 * 1024 {
            v.len() / 64
        } else {
            v.len() / 1024
        };
        let middle = frac / 2;
        let lo = v.len() / 2 - middle;
        let hi = frac + lo;
        let gap = (v.len() - 9 * frac) / 4;
        let mut a = lo - 4 * frac - gap;
        let mut b = hi + gap;
        for i in lo..hi {
            ninther(
                v,
                [a, i - frac, b, a + 1, i, b + 1, a + 2, i + frac, b + 2],
                less,
            );
            a += 3;
            b += 3;
        }
        median_select(&mut v[lo..lo + frac], middle, less);
        let p = partition::<F, false>(v, lo + middle, less);
        if p == index {
            return;
        }
        if p > index {
            v = &mut v[..p];
        } else {
            v = &mut v[p + 1..];
            index -= p + 1;
        }
    }
}
fn median_index<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    mut a: usize,
    b: usize,
    mut c: usize,
    less: &F,
) -> usize {
    if less(v[c], v[a]) {
        let __t = a; a = c; c = __t;
    }
    if less(v[c], v[b]) {
        return c;
    }
    if less(v[b], v[a]) {
        return a;
    }
    b
}
fn ninther<F: Fn(usize, usize) -> bool>(v: &mut [usize], positions: [usize; 9], less: &F) {
    let [a, mut b, c, mut d, e, mut f, g, mut h, i] = positions;
    b = median_index(v, a, b, c, less);
    h = median_index(v, g, h, i, less);
    if less(v[h], v[b]) {
        let __t = b; b = h; h = __t;
    }
    if less(v[f], v[d]) {
        let __t = d; d = f; f = __t;
    }
    if less(v[e], v[d]) {
    } else if less(v[f], v[e]) {
        d = f;
    } else {
        if less(v[e], v[b]) {
            v.swap(e, b);
        } else if less(v[h], v[e]) {
            v.swap(e, h);
        }
        return;
    }
    if less(v[d], v[b]) {
        d = b;
    } else if less(v[h], v[d]) {
        d = h;
    }
    v.swap(d, e);
}
