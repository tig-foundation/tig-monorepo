//! A blocked maximum queue, stable active-item ranks, and exact xorshift jumps.
//! Queue keys encode the original ascending-ID tie order. Random draws are
//! skipped only when the bounded random bonus cannot put an item in the top two.

/// Exact floor(1000*c/w) for positive i32 c and weights 1..=10.
/// With M=ceil(2^48/w), the reciprocal error is <1000*c/2^48
/// <0.008, below the smallest nonzero fractional gap 1/10. Thus
/// truncating the product gives precisely the integer-division result.
#[inline(always)]
pub(super) fn positive_density(c: i32, w: u32) -> i64 {
    const M: [u64; 11] = [
        0,
        1u64 << 48,
        ((1u64 << 48) + 1) / 2,
        ((1u64 << 48) + 2) / 3,
        ((1u64 << 48) + 3) / 4,
        ((1u64 << 48) + 4) / 5,
        ((1u64 << 48) + 5) / 6,
        ((1u64 << 48) + 6) / 7,
        ((1u64 << 48) + 7) / 8,
        ((1u64 << 48) + 8) / 9,
        ((1u64 << 48) + 9) / 10,
    ];
    debug_assert!(c > 0 && (1..=10).contains(&w));
    ((((c as u32 as u64) * 1000) as u128 * M[w as usize] as u128) >> 48) as i64
}

/// The constructor proves c<2^24 for this kernel. With
/// M=ceil(1000*2^30/w), reciprocal error is <1/64, smaller than
/// the minimum nonzero fractional gap 1/10. Also c*M<1000*2^54<2^64.
#[inline(always)]
pub(super) fn positive_density_narrow(c: i32, w: u32) -> i64 {
    const N: u64 = 1000u64 << 30;
    const M: [u64; 11] = [
        0,
        N,
        (N + 1) / 2,
        (N + 2) / 3,
        (N + 3) / 4,
        (N + 4) / 5,
        (N + 5) / 6,
        (N + 6) / 7,
        (N + 7) / 8,
        (N + 8) / 9,
        (N + 9) / 10,
    ];
    debug_assert!(c > 0 && c < (1 << 24) && (1..=10).contains(&w));
    ((c as u32 as u64 * M[w as usize]) >> 30) as i64
}

pub(super) struct Queue {
    small: Vec<u32>,
    wide: Option<Vec<u64>>,
    maxima: Vec<u64>,
    dirty: Vec<bool>,
    bits: Vec<u64>,
    counts: Vec<i32>,
    n: usize,
    len: usize,
}
impl Queue {
    pub(super) fn new(n: usize) -> Self {
        assert!(n <= u16::MAX as usize);
        let blocks = (n + 31) / 32;
        let words = (n + 63) / 64;
        Self {
            small: vec![0; blocks * 32],
            wide: None,
            maxima: vec![0; blocks],
            dirty: vec![false; blocks],
            bits: vec![0; words],
            counts: vec![0; words + 1],
            n,
            len: 0,
        }
    }
    #[inline]
    fn active(&mut self, id: usize, yes: bool) {
        let word = id / 64;
        self.bits[word] ^= 1u64 << (id & 63);
        let mut p = word + 1;
        let delta = if yes { 1 } else { -1 };
        while p < self.counts.len() {
            self.counts[p] += delta;
            p += p & p.wrapping_neg();
        }
    }
    #[inline]
    pub(super) fn rank(&self, id: usize) -> usize {
        let word = id / 64;
        let mask = (1u64 << (id & 63)) - 1;
        let mut rank = (self.bits[word] & mask).count_ones() as usize;
        let mut p = word;
        while p != 0 {
            rank += self.counts[p] as usize;
            p &= p - 1;
        }
        rank
    }
    #[inline]
    pub(super) fn set(&mut self, id: usize, score: i64) {
        debug_assert!(id < self.n && score >= 0 && score < (1i64 << 48));
        let code = score as u64 + 1;
        // Widen exactly once if an unusual challenge exceeds the compact
        // score range. No score is truncated, even during the transition.
        if code > u32::MAX as u64 && self.wide.is_none() {
            self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
        }
        let fresh = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let fresh = *p == 0;
                *p = code;
                fresh
            } else {
                let p = self.small.get_unchecked_mut(id);
                let fresh = *p == 0;
                *p = code as u32;
                fresh
            }
        };
        if fresh {
            self.active(id, true);
            self.len += 1;
        }
        unsafe {
            *self.dirty.get_unchecked_mut(id / 32) = true;
        }
    }
    #[inline]
    pub(super) fn remove(&mut self, id: usize) {
        debug_assert!(id < self.n);
        let existed = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let existed = *p != 0;
                *p = 0;
                existed
            } else {
                let p = self.small.get_unchecked_mut(id);
                let existed = *p != 0;
                *p = 0;
                existed
            }
        };
        if existed {
            self.active(id, false);
            self.len -= 1;
            unsafe {
                *self.dirty.get_unchecked_mut(id / 32) = true;
            }
        }
    }
    #[inline]
    pub(super) fn assign(&mut self, id: usize, score: i64, active: bool) {
        let code = (score as u64 + 1) & 0u64.wrapping_sub(active as u64);
        if code > u32::MAX as u64 && self.wide.is_none() {
            self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
        }
        let existed = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let old = *p != 0;
                *p = code;
                old
            } else {
                let p = self.small.get_unchecked_mut(id);
                let old = *p != 0;
                *p = code as u32;
                old
            }
        };
        if existed != active {
            self.active(id, active);
            if active {
                self.len += 1;
            } else {
                self.len -= 1;
            }
        }
        unsafe {
            *self.dirty.get_unchecked_mut(id / 32) = true;
        }
    }
    /// The constructor proves that every score, including its +1 sentinel,
    /// fits u32 before choosing this kernel. No operation on that queue can
    /// widen it. Keep the full-range assign() path for every other case.
    /// Safety: id is in range, wide is None, and score < u32::MAX.
    #[inline]
    pub(super) unsafe fn assign_small(&mut self, id: usize, score: u32, active: bool) {
        debug_assert!(id < self.n && self.wide.is_none() && score < u32::MAX);
        let code = (score + 1) & 0u32.wrapping_sub(active as u32);
        let p = self.small.get_unchecked_mut(id);
        let existed = *p != 0;
        *p = code;
        if existed != active {
            self.active(id, active);
            if active {
                self.len += 1;
            } else {
                self.len -= 1;
            }
        }
        *self.dirty.get_unchecked_mut(id / 32) = true;
    }
    fn refresh(&mut self) {
        if let Some(wide) = self.wide.as_ref() {
            for block in 0..self.maxima.len() {
                if self.dirty[block] {
                    self.maxima[block] = wide[block * 32..block * 32 + 32]
                        .iter()
                        .copied()
                        .max()
                        .unwrap();
                    self.dirty[block] = false;
                }
            }
        } else {
            for block in 0..self.maxima.len() {
                if self.dirty[block] {
                    self.maxima[block] = self.small[block * 32..block * 32 + 32]
                        .iter()
                        .copied()
                        .max()
                        .unwrap() as u64;
                    self.dirty[block] = false;
                }
            }
        }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn best(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        self.refresh();
        let maximum = self.maxima.iter().copied().max().unwrap();
        let block = self.maxima.iter().position(|&v| v == maximum).unwrap();
        let pos = if let Some(wide) = self.wide.as_ref() {
            wide[block * 32..block * 32 + 32]
                .iter()
                .position(|&v| v == maximum)
                .unwrap()
        } else {
            self.small[block * 32..block * 32 + 32]
                .iter()
                .position(|&v| v as u64 == maximum)
                .unwrap()
        };
        Some(block * 32 + pos)
    }
    pub(super) fn finalists(&mut self, bonus: u64, out: &mut Vec<(usize, i64)>) {
        out.clear();
        if self.len == 0 {
            return;
        }
        self.refresh();
        let mut first = 0u64;
        let mut second = 0u64;
        for &key in &self.maxima {
            if key > first {
                second = first;
                first = key;
            } else {
                second = second.max(key);
            }
        }
        // The second maximum from distinct blocks is a valid lower bound on
        // the true second maximum, including when the two blocks tie.
        let lower = second.saturating_sub(1).saturating_sub(bonus);
        for (block, &maximum) in self.maxima.iter().enumerate() {
            if maximum == 0 || maximum - 1 < lower {
                continue;
            }
            if let Some(wide) = self.wide.as_ref() {
                for (p, &code) in wide[block * 32..block * 32 + 32].iter().enumerate() {
                    if code > 0 && code - 1 >= lower {
                        out.push((block * 32 + p, (code - 1) as i64));
                    }
                }
            } else {
                for (p, &code) in self.small[block * 32..block * 32 + 32].iter().enumerate() {
                    if code > 0 && code as u64 - 1 >= lower {
                        out.push((block * 32 + p, (code - 1) as i64));
                    }
                }
            }
        }
    }
}

pub(super) struct Jumps {
    tables: Vec<[[u64; 256]; 8]>,
}

impl Jumps {
    fn step(mut x: u64) -> u64 {
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        x
    }

    pub(super) fn new() -> Self {
        let mut powers = [0u64; 64];
        for bit in 0..64 {
            powers[bit] = Self::step(1u64 << bit);
        }
        let mut tables = Vec::with_capacity(16);
        for _ in 0..16 {
            let mut table = [[0u64; 256]; 8];
            for byte in 0..8 {
                for value in 1usize..256 {
                    let rest = value & (value - 1);
                    table[byte][value] =
                        table[byte][rest] ^ powers[byte * 8 + value.trailing_zeros() as usize];
                }
            }
            let mut squared = [0u64; 64];
            for bit in 0..64 {
                squared[bit] = Self::apply(&table, powers[bit]);
            }
            tables.push(table);
            powers = squared;
        }
        Self { tables }
    }

    #[inline(always)]
    fn apply(t: &[[u64; 256]; 8], x: u64) -> u64 {
        t[0][(x & 255) as usize]
            ^ t[1][((x >> 8) & 255) as usize]
            ^ t[2][((x >> 16) & 255) as usize]
            ^ t[3][((x >> 24) & 255) as usize]
            ^ t[4][((x >> 32) & 255) as usize]
            ^ t[5][((x >> 40) & 255) as usize]
            ^ t[6][((x >> 48) & 255) as usize]
            ^ t[7][(x >> 56) as usize]
    }

    #[inline]
    pub(super) fn advance(&self, state: &mut u64, mut count: usize) {
        debug_assert!(count <= u16::MAX as usize);
        if count < 8 {
            for _ in 0..count {
                *state = Self::step(*state);
            }
            return;
        }
        while count != 0 {
            let power = count.trailing_zeros() as usize;
            *state = Self::apply(&self.tables[power], *state);
            count &= count - 1;
        }
    }
}
