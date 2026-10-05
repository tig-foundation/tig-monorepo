//! Two independent DP rows allow the compiler to vectorize exact integer updates.
//! Returns None when the 32-bit range proof does not apply; the caller then uses
//! its original i64 kernel. Core order and strict tie comparisons are unchanged.
pub fn fill(
    core: &[usize],
    weights: &[u32],
    values: &[i32],
    capacity: usize,
    choices: &mut Vec<u8>,
    scratch: &mut Vec<i32>,
) -> Option<usize> {
    let magnitude: i64 = core.iter().map(|&i| (values[i] as i64).abs()).sum();
    // Every real path lies in [-magnitude, magnitude]. Every unreachable
    // pseudo-path lies below INIT + magnitude < -magnitude. Additions cannot
    // overflow, so changing the sentinel cannot change the winning real path.
    if magnitude >= (1i64 << 29) {
        return None;
    }
    const INIT: i32 = -(1 << 30);
    let width = capacity + 1;
    scratch.resize(2 * width, INIT);
    scratch[..2 * width].fill(INIT);
    choices.resize(core.len() * width, 0);
    choices[..core.len() * width].fill(0);
    scratch[0] = 0;
    let mut row = 0;
    let mut hi = 0usize;
    for (t, &item) in core.iter().enumerate() {
        let weight = weights[item] as usize;
        if weight > capacity {
            continue;
        }
        let value = values[item];
        let next_hi = (hi + weight).min(capacity);
        let (first, second) = scratch[..2 * width].split_at_mut(width);
        let (src, dst) = if row == 0 {
            (first, second)
        } else {
            (second, first)
        };
        dst[..weight].copy_from_slice(&src[..weight]);
        let source = &src[..=next_hi - weight];
        let prior = &src[weight..=next_hi];
        let output = &mut dst[weight..=next_hi];
        let bits = &mut choices[t * width + weight..=t * width + next_hi];
        for (((out, bit), &old), &from) in output.iter_mut().zip(bits).zip(prior).zip(source) {
            let candidate = from + value;
            *bit = (candidate > old) as u8;
            *out = candidate.max(old);
        }
        hi = next_hi;
        row ^= 1;
    }
    let last = &scratch[row * width..(row + 1) * width];
    Some((0..=capacity).max_by_key(|&w| last[w]).unwrap_or(0))
}

/// Certify the current membership as the UNIQUE solution of the linear
/// knapsack that DP would solve. Every selected item must have positive
/// density strictly above every unselected item, and selected weight must
/// fill capacity. A positive separating multiplier then makes every change
/// strictly worse, including changes at equal total weight. Thus DP cannot
/// change membership, regardless of its internal tie order.
///
/// Caller supplies keys in original item order, with exact density in the
/// signed upper 48 bits, and positive weights. No hash equivalence is used.
pub(super) fn strict_density_solution(
    packed: &[usize], selected: &[bool], weights: &[u32], capacity: u32,
) -> bool {
    assert_eq!(packed.len(), selected.len());
    assert_eq!(packed.len(), weights.len());
    let mut lowest_selected = i64::MAX;
    let mut highest_unused = i64::MIN;
    let mut weight = 0u64;
    for ((&key, &chosen), &w) in packed.iter().zip(selected).zip(weights) {
        let density = (key as i64) >> 16;
        lowest_selected = lowest_selected.min(if chosen { density } else { i64::MAX });
        highest_unused = highest_unused.max(if chosen { i64::MIN } else { density });
        weight += w as u64 * chosen as u64;
    }
    weight == capacity as u64 && lowest_selected > 0 && lowest_selected > highest_unused
}
