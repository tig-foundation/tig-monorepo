//! Within a weight class, contribution order equals integer density order.
//! Only one candidate per class needs the original truncating division.
pub(super) struct Density {
    groups: [Vec<usize>; 11],
    positions: Vec<usize>,
}
impl Density {
    pub(super) fn new(weights: &[u32], include: &[bool]) -> Option<Self> {
        if weights.iter().any(|&w| !(1..=10).contains(&w)) {
            return None;
        }
        if weights.len() > u32::MAX as usize {
            return None;
        }
        let mut positions = vec![0usize; weights.len()];
        let mut groups: [Vec<usize>; 11] = Default::default();
        for (i, &w) in weights.iter().enumerate() {
            if include[i] {
                positions[i] = groups[w as usize].len();
                groups[w as usize].push(i);
            }
        }
        Some(Self { groups, positions })
    }
    fn extreme<const MAX: bool>(ids: &[usize], contrib: &[i32]) -> u64 {
        let mut result = if MAX { 0u64 } else { u64::MAX };
        unsafe {
            let ip = ids.as_ptr();
            let cp = contrib.as_ptr();
            macro_rules! visit {
                ($p:expr) => {{
                    let id = *ip.add($p);
                    let c = (*cp.add(id) as u32) ^ 0x80000000;
                    let key = ((c as u64) << 32)
                        | if MAX {
                            (u32::MAX - id as u32) as u64
                        } else {
                            id as u64
                        };
                    result = if MAX {
                        result.max(key)
                    } else {
                        result.min(key)
                    };
                }};
            }
            let mut p = 0usize;
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
        result
    }
    pub(super) fn pop(&mut self, contrib: &[i32], slack: u32, maximum: bool) -> Option<usize> {
        let mut chosen = None;
        let mut best = if maximum { i64::MIN } else { i64::MAX };
        let mut best_id = usize::MAX;
        for w in 1..=10usize.min(slack as usize) {
            let group = &self.groups[w];
            if group.is_empty() {
                continue;
            }
            let key = if maximum {
                Self::extreme::<true>(group, contrib)
            } else {
                Self::extreme::<false>(group, contrib)
            };
            let value = ((key >> 32) as u32 ^ 0x80000000) as i32;
            let id = if maximum {
                u32::MAX - key as u32
            } else {
                key as u32
            } as usize;
            let pos = self.positions[id];
            if maximum && value <= 0 {
                continue;
            }
            let score = value as i64 * 1000 / w as i64;
            if (if maximum { score > best } else { score < best })
                || (score == best && id < best_id)
            {
                best = score;
                best_id = id;
                chosen = Some((w, pos));
            }
        }
        chosen.map(|(w, pos)| {
            let id = self.groups[w].swap_remove(pos);
            if pos < self.groups[w].len() {
                self.positions[self.groups[w][pos]] = pos;
            }
            id
        })
    }

    /// Anchor score is 50*(20*c + 7*max(affinity,0))/weight. Within
    /// one weight class distinct numerators differ by at least 50/10,
    /// so taking its maximum before division retains all integer ties.
    pub(super) fn pop_affinity(
        &mut self,
        contrib: &[i32],
        affinity: &[i32],
        slack: u32,
    ) -> Option<usize> {
        assert!(self.positions.len() <= u16::MAX as usize);
        assert_eq!(contrib.len(), self.positions.len());
        assert_eq!(affinity.len(), self.positions.len());
        let mut chosen = None;
        let mut best = -1i64;
        let mut best_id = usize::MAX;
        for w in 1..=10usize.min(slack as usize) {
            let ids = &self.groups[w];
            // Four independent reductions remove the serial maximum chain.
            // IDs belong to this queue's checked-length arrays; the packed key
            // retains the original smallest-ID rule for all equal scores.
            let mut maxima = [0u64; 4];
            unsafe {
                let ip = ids.as_ptr();
                let cp = contrib.as_ptr();
                let ap = affinity.as_ptr();
                macro_rules! visit {
                    ($p:expr,$lane:expr) => {{
                        let id = *ip.add($p);
                        let c = *cp.add(id);
                        let numerator = c as i64 * 20 + (*ap.add(id)).max(0) as i64 * 7;
                        let key = if c > 0 {
                            ((numerator as u64) << 16) | (0xffff - id) as u64
                        } else {
                            0
                        };
                        maxima[$lane] = maxima[$lane].max(key);
                    }};
                }
                let mut p = 0;
                let full = ids.len() / 4 * 4;
                while p < full {
                    visit!(p, 0);
                    visit!(p + 1, 1);
                    visit!(p + 2, 2);
                    visit!(p + 3, 3);
                    p += 4;
                }
                while p < ids.len() {
                    visit!(p, 0);
                    p += 1;
                }
            }
            let maximum = maxima.into_iter().max().unwrap();
            if maximum == 0 {
                continue;
            }
            let id = 0xffff - (maximum as usize & 0xffff);
            let score = (maximum >> 16) as i64 * 50 / w as i64;
            if score > best || (score == best && id < best_id) {
                best = score;
                best_id = id;
                chosen = Some((w, self.positions[id]));
            }
        }
        chosen.map(|(w, pos)| {
            let id = self.groups[w].swap_remove(pos);
            if pos < self.groups[w].len() {
                self.positions[self.groups[w][pos]] = pos;
            }
            id
        })
    }
}

/// Sum the largest three positive interactions, ignoring `exclude`.
/// Each of four independent streams keeps its top three; their union contains
/// the global top three, including repeated equal values.
/// Safety: every ID must index `row`.
pub(super) unsafe fn top_three_sum(row: &[i32], ids: &[usize], exclude: usize) -> i64 {
    let mut tops = [[0i32; 4]; 3];
    let mut p = 0;
    while p + 4 <= ids.len() {
        for lane in 0..4 {
            let id = *ids.get_unchecked(p + lane);
            let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
            tops[2][lane] = tops[2][lane].max(x.min(tops[1][lane]));
            tops[1][lane] = tops[1][lane].max(x.min(tops[0][lane]));
            tops[0][lane] = tops[0][lane].max(x);
        }
        p += 4;
    }
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    for x in tops.into_iter().flatten() {
        c = c.max(x.min(b));
        b = b.max(x.min(a));
        a = a.max(x);
    }
    for &id in &ids[p..] {
        let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
        c = c.max(x.min(b));
        b = b.max(x.min(a));
        a = a.max(x);
    }
    a as i64 + b as i64 + c as i64
}
