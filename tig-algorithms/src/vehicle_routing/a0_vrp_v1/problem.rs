#[derive(Copy, Clone)]
pub struct NodeData {
    pub start_tw: i32,
    pub end_tw: i32,
    pub service_time: i32,
    pub demand: i32,
}

pub struct Problem {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub is_vrptw: bool,
    pub fixed_distance_offset: i64,
    pub max_capacity: i32,
    pub distance_matrix: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub node_data: Vec<NodeData>,
}

impl Problem {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        debug_assert!(i < self.nb_nodes && j < self.nb_nodes);
        let idx = i * self.nb_nodes + j;
        debug_assert!(idx < self.distance_matrix.len());
        unsafe { *self.distance_matrix.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn nd(&self, i: usize) -> &NodeData {
        debug_assert!(i < self.nb_nodes);
        unsafe { self.node_data.get_unchecked(i) }
    }
}
