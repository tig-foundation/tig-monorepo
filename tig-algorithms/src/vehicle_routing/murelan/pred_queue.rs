pub struct IndexDeque {
    pub data: Vec<usize>,
    pub head: usize,
}

impl IndexDeque {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            head: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head >= self.data.len()
    }

    #[inline]
    pub fn push_back(&mut self, v: usize) {
        self.data.push(v);
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            self.data.last().copied()
        }
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.data[self.head])
        }
    }

    #[inline]
    pub fn pop_back(&mut self) {
        if self.is_empty() {
            return;
        }
        self.data.pop();
        if self.head >= self.data.len() {
            self.clear();
        }
    }

    #[inline]
    pub fn pop_front(&mut self) {
        if self.is_empty() {
            return;
        }
        self.head += 1;
        if self.head >= self.data.len() {
            self.clear();
        }
    }
}

/// Doubly linked node indices with sentinel `nil` (= capacity).
/// Removing a node advances matching `feas` and `no_warp` cursors to its successor.
pub struct PredQueue {
    pub prev: Vec<usize>,
    pub next: Vec<usize>,
    pub nil: usize,
    pub size: usize,
    pub feas: usize,
    pub no_warp: usize,
}

impl PredQueue {
    pub fn new(n: usize) -> Self {
        Self {
            prev: vec![n; n + 1],
            next: vec![n; n + 1],
            nil: n,
            size: 0,
            feas: n,
            no_warp: n,
        }
    }

    #[inline(always)]
    fn link(&self, id: usize) -> (usize, usize) {
        debug_assert!(id <= self.nil);
        unsafe {
            (
                *self.prev.get_unchecked(id),
                *self.next.get_unchecked(id),
            )
        }
    }

    #[inline(always)]
    pub fn head(&self) -> usize {
        self.link(self.nil).1
    }

    #[inline(always)]
    pub fn tail(&self) -> usize {
        self.link(self.nil).0
    }

    #[inline(always)]
    pub fn succ(&self, id: usize) -> usize {
        self.link(id).1
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        let h = self.head();
        if h != self.nil {
            Some(h)
        } else {
            None
        }
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        let t = self.tail();
        if t != self.nil {
            Some(t)
        } else {
            None
        }
    }

    #[inline]
    pub fn front2(&self) -> Option<usize> {
        let n = self.succ(self.head());
        if n != self.nil {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    pub fn insert_back(&mut self, id: usize) {
        let nil = self.nil;
        let t = self.tail();
        unsafe {
            *self.next.get_unchecked_mut(t) = id;
            *self.prev.get_unchecked_mut(id) = t;
            *self.next.get_unchecked_mut(id) = nil;
            *self.prev.get_unchecked_mut(nil) = id;
        }
        self.size += 1;
    }

    #[inline]
    pub fn remove_node(&mut self, id: usize) {
        let (p, n) = self.link(id);
        unsafe {
            *self.next.get_unchecked_mut(p) = n;
            *self.prev.get_unchecked_mut(n) = p;
        }
        self.feas = if self.feas == id { n } else { self.feas };
        self.no_warp = if self.no_warp == id { n } else { self.no_warp };
        self.size -= 1;
    }

    #[inline]
    pub fn remove_front(&mut self) {
        let h = self.head();
        if h != self.nil {
            self.remove_node(h);
        }
    }

    #[inline]
    pub fn remove_front2(&mut self) {
        if let Some(h2) = self.front2() {
            self.remove_node(h2);
        }
    }

    #[inline]
    pub fn feas_prev(&self) -> Option<usize> {
        if self.feas == self.nil {
            return None;
        }
        let p = self.link(self.feas).0;
        if p != self.nil {
            Some(p)
        } else {
            None
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        let nil = self.nil;
        unsafe {
            *self.next.get_unchecked_mut(nil) = nil;
            *self.prev.get_unchecked_mut(nil) = nil;
        }
        self.size = 0;
        self.feas = nil;
        self.no_warp = nil;
    }
}
