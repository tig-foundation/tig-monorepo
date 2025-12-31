//! TIG VRPTW Solver Module Registry
//! 
//! This file re-exports all modules from lib for TIG platform compatibility.

pub use crate::vehicle_routing_solver;
pub use crate::adp;
pub use crate::tig_adaptive;
pub use crate::utilities;
pub use crate::constructive;
pub use crate::local_search;
pub use crate::solver;
pub use crate::route;
pub use crate::delta;
pub use crate::instance_gen;
pub use crate::genetic;
pub use crate::problem_loader;
pub use crate::config;
pub use crate::population;
pub use crate::instance;
pub use crate::bundle;
pub use crate::validation;
pub use crate::repair;

// Re-export public API
pub use crate::vehicle_routing_solver::Solver;
pub use crate::utilities::IZS;
pub use crate::delta::DeltaTables;
pub use crate::tig_adaptive::TIGState;
pub use crate::population::Population;

// Compatibility shims
pub mod delta_tables {
    pub use crate::delta::DeltaTables;
}

pub mod izs {
    pub use crate::utilities::IZS;
}

pub mod testing {
    use crate::tig_adaptive::TIGState;

    /// Build a small feasible TIGState for integration tests.
    pub fn small_route_state() -> TIGState {
        let route = vec![0usize, 1, 2, 3];
        let time = 0i32;
        let max_cap = 100i32;
        let tw_start = vec![0i32; route.len()];
        let tw_end = vec![10_000i32; route.len()];
        let service_time = vec![0i32; route.len()];
        let demands = vec![0i32; route.len()];

        // simple symmetric distance matrix with unit cost per hop
        let n = route.len();
        let mut distance_matrix = vec![vec![0i32; n]; n];
        for i in 0..n {
            for j in 0..n {
                distance_matrix[i][j] = ((i as i32 - j as i32).abs()).max(1);
            }
        }

        let mut state = TIGState::new(route, time, max_cap, tw_start, tw_end, service_time, distance_matrix, demands);
        // ensure derived fields are consistent
        state.recompute_all_times();
        state.recompute_load();
        state
    }
}
