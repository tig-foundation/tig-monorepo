//! Reuse only a previously proved local-search fixed point. Full state
//! equality is checked after indexing; hashes cannot cause a false hit.
struct Proof {
    parameter: usize,
    hash: u64,
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
}
pub(super) struct Terminal {
    slots: Vec<Option<Proof>>,
}
impl Terminal {
    pub(super) fn new() -> Self {
        Self { slots: Vec::new() }
    }
    pub(super) fn clear(&mut self) {
        self.slots = Vec::new();
    }
    fn index(hash: u64, parameter: usize) -> usize {
        ((hash ^ (parameter as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_mul(0xbf58476d1ce4e5b9)
            >> 53) as usize
    }
    pub(super) fn contains(
        &self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
    ) -> bool {
        if let Some(Some(p)) = self.slots.get(Self::index(hash, parameter)) {
            p.parameter == parameter
                && p.hash == hash
                && p.value == value
                && p.weight == weight
                && p.bits == bits
                && p.contrib == contrib
        } else {
            false
        }
    }
    pub(super) fn insert(
        &mut self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
    ) {
        if self.slots.is_empty() {
            self.slots.resize_with(2048, || None);
        }
        let p = Proof {
            parameter,
            hash,
            bits: bits.to_vec(),
            contrib: contrib.to_vec(),
            value,
            weight,
        };
        self.slots[Self::index(hash, parameter)] = Some(p);
    }
}
