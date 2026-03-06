pub struct Instance {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub demands: Vec<i32>,
    pub max_capacity: i32,
    pub distance_matrix: Vec<Vec<i32>>,
    // flattened row-major distance matrix for faster single-index access
    pub distance_flat: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub service_times: Vec<i32>,
    pub start_tw: Vec<i32>,
    pub end_tw: Vec<i32>,
}

impl Instance {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        // use flattened access for faster indexing
        let idx = i * self.nb_nodes + j;
        unsafe { *self.distance_flat.get_unchecked(idx) }
    }
}
