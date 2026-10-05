//! Exact rational-density windows for the pinned TIG Rust toolchain.
//!
//! For weights 1..=10, 2520 / weight is integral. Scaling the contribution by
//! this factor preserves every cross-product comparison, including ties.
//! The signed key fits in 48 bits for every i32 contribution; the low 16 bits
//! hold the item ID and are ignored by the comparator.
//!
//! Rust nightly-2025-02-10's select_nth_unstable uses the same partition
//! permutation for these 8-byte elements and the original 24-byte tuples.
//! Both take its <=96-byte cyclic Lomuto path; its different loop unroll
//! counts do not change the order of comparisons or moves. Pivot selection,
//! insertion sort, and the median-of-medians fallback are type-independent.
//! This equivalence is specific to the validated compiler implementations.

const SCALE: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
type Original = (usize, i64, i64);

pub(super) struct Windows<'a> {
    weights: &'a [u32],
    packed: bool,
    unused: Vec<usize>,
    used: Vec<usize>,
    original_unused: Vec<Original>,
    original_used: Vec<Original>,
}

impl<'a> Windows<'a> {
    pub(super) fn new(weights: &'a [u32]) -> Self {
        let n = weights.len();
        let packed = usize::BITS == 64
            && n <= u16::MAX as usize
            && weights.iter().all(|&w| (1..=10).contains(&w));
        Self {
            weights,
            packed,
            unused: Vec::with_capacity(if packed { n } else { 0 }),
            used: Vec::with_capacity(if packed { n } else { 0 }),
            original_unused: Vec::with_capacity(if packed { 0 } else { n }),
            original_used: Vec::with_capacity(if packed { 0 } else { n }),
        }
    }

    pub(super) fn is_packed(&self) -> bool {
        self.packed
    }

    pub(super) fn build(
        &mut self,
        contrib: &[i32],
        selected: &[bool],
        k: usize,
        best_unused: &mut Vec<usize>,
        worst_used: &mut Vec<usize>,
    ) {
        let weights = self.weights;
        assert_eq!(weights.len(), contrib.len());
        assert_eq!(weights.len(), selected.len());
        best_unused.clear();
        worst_used.clear();
        if !self.packed {
            self.original_unused.clear();
            self.original_used.clear();
            for i in 0..weights.len() {
                let score = (i, contrib[i] as i64, (weights[i] as i64).max(1));
                if selected[i] {
                    self.original_used.push(score);
                } else {
                    self.original_unused.push(score);
                }
            }
            let ku = k.min(self.original_unused.len());
            let ks = k.min(self.original_used.len());
            if ku > 0 && ku < self.original_unused.len() {
                self.original_unused
                    .select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
            }
            if ks > 0 && ks < self.original_used.len() {
                self.original_used
                    .select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
            }
            best_unused.extend(self.original_unused[..ku].iter().map(|t| t.0));
            worst_used.extend(self.original_used[..ks].iter().map(|t| t.0));
            return;
        }

        self.unused.clear();
        self.used.clear();
        let mut nu = 0usize;
        let mut ns = 0usize;
        let unused = self.unused.as_mut_ptr();
        let used = self.used.as_mut_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                // All slices have the checked equal length. new() checked every
                // weight, and self retains an immutable borrow of that array.
                let (c, factor, is_selected) = unsafe {
                    (
                        *contrib.get_unchecked(i) as i64,
                        *SCALE.get_unchecked(*weights.get_unchecked(i) as usize),
                        *selected.get_unchecked(i),
                    )
                };
                let entry = ((c * factor) << 16) as usize | i;
                // Before item i, nu + ns == i < n. Both allocations have capacity
                // n. Each dummy write lands on the uncommitted tail, never on a
                // retained item. Committing just the two true lengths reproduces
                // the original stable split, including its input order for ties.
                unsafe {
                    unused.add(nu).write(entry);
                    used.add(ns).write(entry);
                }
                ns += is_selected as usize;
                nu += (!is_selected) as usize;
            }};
        }
        let mut i = 0usize;
        let end = weights.len() / 16 * 16;
        while i < end {
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
        while i < weights.len() {
            emit!(i);
            i += 1;
        }
        unsafe {
            self.unused.set_len(nu);
            self.used.set_len(ns);
        }
        let ku = k.min(self.unused.len());
        let ks = k.min(self.used.len());
        if super::exact_sort::compatible() {
            if ku > 0 && ku < self.unused.len() {
                super::exact_sort::select(&mut self.unused, ku - 1, |a, b| {
                    ((a as i64) >> 16) > ((b as i64) >> 16)
                });
            }
            if ks > 0 && ks < self.used.len() {
                super::exact_sort::select(&mut self.used, ks - 1, |a, b| {
                    ((a as i64) >> 16) < ((b as i64) >> 16)
                });
            }
        } else {
            let ku = k.min(self.unused.len());
            let ks = k.min(self.used.len());
            if ku > 0 && ku < self.unused.len() {
                self.unused.select_nth_unstable_by(ku - 1, |&a, &b| {
                    ((b as i64) >> 16).cmp(&((a as i64) >> 16))
                });
            }
            if ks > 0 && ks < self.used.len() {
                self.used.select_nth_unstable_by(ks - 1, |&a, &b| {
                    ((a as i64) >> 16).cmp(&((b as i64) >> 16))
                });
            }
        }
        best_unused.extend(self.unused[..ku].iter().map(|&entry| entry & 0xffff));
        worst_used.extend(self.used[..ks].iter().map(|&entry| entry & 0xffff));
    }
}
