//! Exact integer-pair lookup; no enumeration order is exposed to the search.
#[derive(Clone,Copy,Default)]
struct Entry {
    key: (usize,usize),
    expires: usize,
    generation: u32,
}

pub(crate) struct FlatTabu {
    slots: Vec<Entry>,
    len: usize,
    generation: u32,
}

impl FlatTabu {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        // Preserve the donor's RandomState initialization side effect. The
        // private table never exposes iteration and needs no randomized hash.
        let _original_hash_seed = std::collections::hash_map::RandomState::new();
        let size = capacity.checked_mul(2).and_then(|n|n.max(16).checked_next_power_of_two())
            .expect("tabu capacity overflow");
        Self { slots:vec![Entry::default();size],len:0,generation:1 }
    }

    #[inline]
    fn locate(&self,key: (usize,usize)) -> usize {
        let a = key.0.wrapping_mul(0x9e3779b97f4a7c15u64 as usize).rotate_left(17);
        let b = key.1.wrapping_mul(0xbf58476d1ce4e5b9u64 as usize);
        let x = a ^ b;
        let mask = self.slots.len()-1;
        let mut index = (x ^ (x>>29)) & mask;
        loop {
            let entry = &self.slots[index];
            if entry.generation!=self.generation || entry.key==key { return index; }
            index = (index+1)&mask;
        }
    }

    #[inline]
    pub(crate) fn get(&self,key: &(usize,usize)) -> Option<&usize> {
        let entry = &self.slots[self.locate(*key)];
        if entry.generation==self.generation { Some(&entry.expires) } else { None }
    }

    #[inline]
    pub(crate) fn insert(&mut self,key: (usize,usize),expires: usize) -> Option<usize> {
        let mut index = self.locate(key);
        if self.slots[index].generation==self.generation {
            return Some(__tig_safe_replace(&mut self.slots[index].expires,expires));
        }
        if self.len>=self.slots.len()/2 {
            self.grow();
            index = self.locate(key);
        }
        self.slots[index] = Entry {key,expires,generation:self.generation};
        self.len += 1;
        None
    }

    fn grow(&mut self) {
        let size = self.slots.len().checked_mul(2).expect("tabu capacity overflow");
        let old_generation = self.generation;
        let old = __tig_safe_replace(&mut self.slots,vec![Entry::default();size]);
        self.generation = 1;
        self.len = 0;
        for entry in old {
            if entry.generation==old_generation {
                let index = self.locate(entry.key);
                self.slots[index] = Entry {generation:1,..entry};
                self.len += 1;
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation==0 {
            for entry in &mut self.slots { entry.generation=0; }
            self.generation=1;
        }
    }
}

#[inline(always)]
fn __tig_safe_replace<T: Clone>(dest: &mut T, src: T) -> T { let old = dest.clone(); *dest = src; old }
