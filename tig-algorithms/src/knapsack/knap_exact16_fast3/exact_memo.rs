//! A cache address only: callers always compare the complete input state.
pub(super) fn fingerprint(bits: &[bool], parameter: usize) -> u64 {
    let mut hash = 0x517cc1b727220a95u64 ^ parameter as u64;
    let mut i = 0;
    while i + 8 <= bits.len() {
        // bool has size/alignment 1; the complete eight-byte read is in bounds.
        let word = unsafe { (bits.as_ptr().add(i) as *const u64).read_unaligned() };
        hash = (hash ^ word).wrapping_mul(0x9e3779b97f4a7c15);
        i += 8;
    }
    for &bit in &bits[i..] {
        hash = (hash ^ bit as u64).wrapping_mul(0x9e3779b97f4a7c15);
    }
    hash ^= hash >> 32;
    hash
}

pub(super) fn address(bits: &[bool], parameter: usize) -> usize {
    fingerprint(bits, parameter) as usize & 511
}
