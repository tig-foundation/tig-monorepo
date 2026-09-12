//! Cache proved deterministic paths to a local-search terminal state. A hit
//! requires full input equality and enough iterations to traverse the original
//! path, including its final certification pass. Terminal snapshots are shared.
use std::rc::Rc;
pub(super) struct Snapshot {
    pub bits: Vec<bool>,
    pub contrib: Vec<i32>,
    pub value: i64,
    pub weight: u32,
    pub hash: u64,
}
impl Snapshot {
    pub(super) fn new(bits: &[bool], contrib: &[i32], value: i64, weight: u32, hash: u64) -> Self {
        Self {
            bits: bits.to_vec(),
            contrib: contrib.to_vec(),
            value,
            weight,
            hash,
        }
    }
}
struct Entry {
    parameter: usize,
    input: Snapshot,
    output: Rc<Snapshot>,
    steps: usize,
}
pub(super) struct Paths {
    slots: Vec<Option<Entry>>,
}
impl Paths {
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
    pub(super) fn lookup(
        &self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
        remaining: usize,
    ) -> Option<(Rc<Snapshot>, usize)> {
        let entry = self.slots.get(Self::index(hash, parameter))?.as_ref()?;
        let s = &entry.input;
        if entry.parameter == parameter
            && entry.steps <= remaining
            && s.hash == hash
            && s.value == value
            && s.weight == weight
            && s.bits == bits
            && s.contrib == contrib
        {
            Some((Rc::clone(&entry.output), entry.steps))
        } else {
            None
        }
    }
    pub(super) fn publish(
        &mut self,
        parameter: usize,
        pending: Vec<(Snapshot, usize)>,
        end: usize,
        output: Rc<Snapshot>,
    ) {
        if self.slots.is_empty() {
            self.slots.resize_with(2048, || None);
        }
        for (input, start) in pending {
            let index = Self::index(input.hash, parameter);
            self.slots[index] = Some(Entry {
                parameter,
                input,
                output: Rc::clone(&output),
                steps: end - start,
            });
        }
    }
}
