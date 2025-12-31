// Route representation with O(1) positional lookup using SmallVec
use serde::{Serialize, Deserialize};
use smallvec::SmallVec;

/// Route structure optimized for fast positional lookups and in place edits
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Route {
    pub nodes: SmallVec<[usize; 1024]>,
    /// position lookup: pos[node] = index in nodes; usize::MAX if unknown
    pub pos: Vec<usize>,
}

// Allow comparing Route directly with Vec<usize> in tests and other helpers.
impl PartialEq<Vec<usize>> for Route {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.nodes.as_slice() == other.as_slice()
    }
}

impl PartialEq<&[usize]> for Route {
    fn eq(&self, other: &&[usize]) -> bool {
        self.nodes.as_slice() == *other
    }
}

impl Route {
    /// Create a route from an iterator of node indices. `pos` will be sized to
    /// accommodate the largest node index seen.
    pub fn from_nodes<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let nodes_iter = iter.into_iter();
        let mut nodes: SmallVec<[usize; 1024]> = SmallVec::new();
        let mut max_node = 0usize;
        for n in nodes_iter {
            max_node = max_node.max(n);
            nodes.push(n);
        }
        let mut pos = vec![usize::MAX; max_node + 1];
        for (i, &n) in nodes.iter().enumerate() {
            if n >= pos.len() {
                pos.resize(n + 1, usize::MAX);
            }
            pos[n] = i;
        }
        Self { nodes, pos }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert `node` at position `idx`. Updates position lookup in O(1) amortized for pos resize.
    pub fn insert(&mut self, idx: usize, node: usize) {
        let n = self.nodes.len();
        let insert_idx = if idx > n { n } else { idx };
        self.nodes.insert(insert_idx, node);
        if node >= self.pos.len() {
            self.pos.resize(node + 1, usize::MAX);
        }
        // update positions: naive O(n) shift; callers should prefer delta tables when performance-critical
        for i in insert_idx..self.nodes.len() {
            let nd = self.nodes[i];
            if nd >= self.pos.len() {
                self.pos.resize(nd + 1, usize::MAX);
            }
            self.pos[nd] = i;
        }
    }

    /// Remove element at `idx` and return it
    pub fn remove(&mut self, idx: usize) -> usize {
        let node = self.nodes.remove(idx);
        // update positions from idx to end
        for i in idx..self.nodes.len() {
            let nd = self.nodes[i];
            self.pos[nd] = i;
        }
        self.pos[node] = usize::MAX;
        node
    }

    /// Swap positions i and j
    pub fn swap(&mut self, i: usize, j: usize) {
        let n = self.nodes.len();
        if i >= n || j >= n || i == j {
            return;
        }
        self.nodes.swap(i, j);
        let a = self.nodes[i];
        let b = self.nodes[j];
        if a >= self.pos.len() { self.pos.resize(a + 1, usize::MAX); }
        if b >= self.pos.len() { self.pos.resize(b + 1, usize::MAX); }
        self.pos[a] = i;
        self.pos[b] = j;
    }

    /// Reverse segment [i..=j]
    pub fn reverse(&mut self, i: usize, j: usize) {
        let n = self.nodes.len();
        if i >= n || j >= n || i >= j { return; }
        self.nodes[i..=j].reverse();
        for k in i..=j {
            let nd = self.nodes[k];
            if nd >= self.pos.len() { self.pos.resize(nd + 1, usize::MAX); }
            self.pos[nd] = k;
        }
    }

    /// Return a segment starting at i with length len. If len overruns, returns available tail.
    pub fn segment(&self, i: usize, len: usize) -> Vec<usize> {
        if i >= self.nodes.len() { return Vec::new(); }
        let end = (i + len).min(self.nodes.len());
        self.nodes[i..end].iter().copied().collect()
    }
}
