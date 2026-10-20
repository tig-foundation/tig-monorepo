pub struct Problem {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub demands: Vec<i32>,
    pub max_capacity: i32,
    pub distance_matrix: Vec<Vec<i32>>,
    pub node_positions: Vec<(i32, i32)>,
    pub service_times: Vec<i32>,
    pub start_tw: Vec<i32>,
    pub end_tw: Vec<i32>
}

impl Problem {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        debug_assert!(i < self.nb_nodes && j < self.nb_nodes);
        unsafe { *self.distance_matrix.get_unchecked(i).get_unchecked(j) }
    }
}
