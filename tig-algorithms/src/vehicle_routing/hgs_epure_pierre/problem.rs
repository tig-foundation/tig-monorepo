#[derive(Copy, Clone)]
pub(super) struct NodeData {
    pub(super) start_tw: i32,
    pub(super) end_tw: i32,
    pub(super) service_time: i32,
    pub(super) demand: i32,
}

pub(super) struct Problem {
    pub(super) seed: [u8; 32],
    pub(super) nb_nodes: usize,
    pub(super) nb_vehicles: usize,
    pub(super) lb_vehicles: usize,
    pub(super) fixed_distance_offset: i64,
    pub(super) max_capacity: i32,
    pub(super) distance_matrix: Vec<i32>,
    pub(super) distance_matrix_transposed: Vec<i32>,
    pub(super) node_positions: Vec<(i32, i32)>,
    pub(super) node_data: Vec<NodeData>,
}

impl Problem {
    #[inline(always)]
    pub(super) fn dm(&self, i: usize, j: usize) -> i32 {
        debug_assert!(i < self.nb_nodes && j < self.nb_nodes);
        let idx = i * self.nb_nodes + j;
        debug_assert!(idx < self.distance_matrix.len());
        unsafe { *self.distance_matrix.as_ptr().add(idx) }
    }

    #[inline(always)]
    pub(super) fn distance_row(&self, i: usize) -> &[i32] {
        debug_assert!(i < self.nb_nodes);
        let start = i * self.nb_nodes;
        unsafe {
            self.distance_matrix
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    #[inline(always)]
    pub(super) fn distance_column(&self, j: usize) -> &[i32] {
        debug_assert!(j < self.nb_nodes);
        let start = j * self.nb_nodes;
        unsafe {
            self.distance_matrix_transposed
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    pub(super) fn transpose_distances(distance_matrix: &[i32], nb_nodes: usize) -> Vec<i32> {
        let mut transposed = vec![0; distance_matrix.len()];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                transposed[j * nb_nodes + i] = distance_matrix[i * nb_nodes + j];
            }
        }
        transposed
    }

    #[inline(always)]
    pub(super) fn nd(&self, i: usize) -> &NodeData {
        debug_assert!(i < self.nb_nodes);
        unsafe { self.node_data.get_unchecked(i) }
    }
}
