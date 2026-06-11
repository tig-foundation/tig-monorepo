pub const NIL: usize = usize::MAX;

pub struct IndexDeque {
    pub data: Vec<usize>,
    pub head: usize,
}

impl IndexDeque {
    pub fn with_capacity(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), head: 0 }
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
        if self.is_empty() { None } else { self.data.last().copied() }
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        if self.is_empty() { None } else { Some(self.data[self.head]) }
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

pub struct PredQueue {
    pub prev: Vec<usize>,
    pub next: Vec<usize>,
    pub head: usize,
    pub tail: usize,
    pub size: usize,
    pub feas: usize,
    pub no_warp: usize,
}

impl PredQueue {
    pub fn new(n: usize) -> Self {
        Self {
            prev: vec![NIL; n],
            next: vec![NIL; n],
            head: NIL,
            tail: NIL,
            size: 0,
            feas: NIL,
            no_warp: NIL,
        }
    }

    #[inline]
    pub fn front(&self) -> Option<usize> { if self.head != NIL { Some(self.head) } else { None } }

    #[inline]
    pub fn back(&self) -> Option<usize> { if self.tail != NIL { Some(self.tail) } else { None } }

    #[inline]
    pub fn front2(&self) -> Option<usize> {
        if self.head == NIL {
            return None;
        }
        let n = self.next[self.head];
        if n != NIL { Some(n) } else { None }
    }

    #[inline]
    pub fn insert_back(&mut self, id: usize) {
        debug_assert!(self.prev[id] == NIL && self.next[id] == NIL);
        if self.tail == NIL {
            self.head = id;
            self.tail = id;
        } else {
            let t = self.tail;
            self.next[t] = id;
            self.prev[id] = t;
            self.tail = id;
        }
        self.size += 1;
    }

    #[inline]
    pub fn remove_node(&mut self, id: usize) {
        let p = self.prev[id];
        let n = self.next[id];
        if p != NIL {
            self.next[p] = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.prev[n] = p;
        } else {
            self.tail = p;
        }
        self.prev[id] = NIL;
        self.next[id] = NIL;
        if self.feas == id {
            self.feas = n;
        }
        if self.no_warp == id {
            self.no_warp = n;
        }
        self.size -= 1;
    }

    #[inline]
    pub fn remove_back(&mut self) {
        if self.tail != NIL { self.remove_node(self.tail); }
    }

    #[inline]
    pub fn remove_front(&mut self) {
        if self.head != NIL { self.remove_node(self.head); }
    }

    #[inline]
    pub fn remove_front2(&mut self) {
        if let Some(h2) = self.front2() {
            self.remove_node(h2);
        }
    }

    #[inline]
    pub fn feas_prev(&self) -> Option<usize> {
        if self.feas == NIL {
            None
        } else {
            let p = self.prev[self.feas];
            if p != NIL { Some(p) } else { None }
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.prev.fill(NIL);
        self.next.fill(NIL);
        self.head = NIL;
        self.tail = NIL;
        self.size = 0;
        self.feas = NIL;
        self.no_warp = NIL;
    }
}
