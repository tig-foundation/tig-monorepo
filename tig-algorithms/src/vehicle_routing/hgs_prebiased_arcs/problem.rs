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
    pub distance_matrix_transposed: Vec<i32>,
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
    pub fn distance_row(&self, i: usize) -> &[i32] {
        debug_assert!(i < self.nb_nodes);
        let start = i * self.nb_nodes;
        unsafe {
            self.distance_matrix
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    #[inline(always)]
    pub fn distance_column(&self, j: usize) -> &[i32] {
        debug_assert!(j < self.nb_nodes);
        let start = j * self.nb_nodes;
        unsafe {
            self.distance_matrix_transposed
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    pub fn transpose_distances(distance_matrix: &[i32], nb_nodes: usize) -> Vec<i32> {
        let mut transposed = vec![0; distance_matrix.len()];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                transposed[j * nb_nodes + i] = distance_matrix[i * nb_nodes + j];
            }
        }
        transposed
    }

    #[inline(always)]
    pub fn nd(&self, i: usize) -> &NodeData {
        debug_assert!(i < self.nb_nodes);
        unsafe { self.node_data.get_unchecked(i) }
    }
}
