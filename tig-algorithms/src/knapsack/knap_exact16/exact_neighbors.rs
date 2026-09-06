//! Canonical top neighbors using compact exact keys and binary insertion.
//! The original scan visits ascending IDs and inserts after equal scores.
//! Encoding the reversed ID below the score reproduces that complete order.

#[inline]
fn offer(list: &mut Vec<u64>, slots: usize, cutoff: &mut u64, key: u64) {
    if key <= *cutoff {
        return;
    }
    let at = list.partition_point(|&old| old > key);
    list.insert(at, key);
    if list.len() > slots {
        list.pop();
    }
    if list.len() == slots {
        *cutoff = list[slots - 1];
    }
}

pub(super) fn build(
    weights: &[u32],
    matrix: &[Vec<i32>],
    row_ptr: &[u32],
    columns: &[u16],
    k: usize,
    weight_aware: bool,
) -> Option<Vec<Vec<usize>>> {
    let n = weights.len();
    if n > u16::MAX as usize || k == 0 || weights.iter().any(|&w| !(1..=10).contains(&w)) {
        return None;
    }
    // LCM(1..=20). The sum of the two weights is in 2..=20.
    // With interaction <=2,000,000, the largest scaled score is
    // 2,000,000 * LCM/2 < 2^48, leaving 16 exact bits for the ID.
    const LCM: u64 = 232_792_560;
    const FACTOR: [u64; 21] = [
        0,
        LCM,
        LCM / 2,
        LCM / 3,
        LCM / 4,
        LCM / 5,
        LCM / 6,
        LCM / 7,
        LCM / 8,
        LCM / 9,
        LCM / 10,
        LCM / 11,
        LCM / 12,
        LCM / 13,
        LCM / 14,
        LCM / 15,
        LCM / 16,
        LCM / 17,
        LCM / 18,
        LCM / 19,
        LCM / 20,
    ];
    let absolute_slots = if weight_aware { (k + 1) / 2 } else { k };
    let mut friends = Vec::with_capacity(n);
    let mut absolute = Vec::with_capacity(absolute_slots.min(n) + 1);
    let mut density = Vec::with_capacity(k.min(n) + 1);
    for i in 0..n {
        absolute.clear();
        density.clear();
        let mut absolute_cutoff = 0;
        let mut density_cutoff = 0;
        let row = &matrix[i];
        for &column in &columns[row_ptr[i] as usize..row_ptr[i + 1] as usize] {
            let j = column as usize;
            let interaction = row[j];
            if i == j || interaction <= 0 {
                continue;
            }
            if interaction > 2_000_000 {
                return None;
            }
            let tie = (u16::MAX as usize - j) as u64;
            offer(
                &mut absolute,
                absolute_slots,
                &mut absolute_cutoff,
                ((interaction as u64) << 16) | tie,
            );
            if weight_aware {
                let scaled = interaction as u64 * FACTOR[(weights[i] + weights[j]) as usize];
                offer(&mut density, k, &mut density_cutoff, (scaled << 16) | tie);
            }
        }
        let mut neighbors = Vec::with_capacity(k.min(n));
        neighbors.extend(
            absolute
                .iter()
                .map(|&key| u16::MAX as usize - (key as usize & 0xffff)),
        );
        if weight_aware {
            for &key in &density {
                if neighbors.len() >= k {
                    break;
                }
                let j = u16::MAX as usize - (key as usize & 0xffff);
                if !neighbors.contains(&j) {
                    neighbors.push(j);
                }
            }
        }
        friends.push(neighbors);
    }
    Some(friends)
}
