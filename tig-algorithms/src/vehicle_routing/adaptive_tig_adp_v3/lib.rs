//! # Adaptive TIG ADP v3 VRPTW Solver
//!
//! Public API entry point: `Solver::solve_challenge_instance`

pub mod vehicle_routing_solver;
pub mod adp;
pub mod tig_adaptive;
pub mod utilities;
pub mod constructive;
pub mod local_search;
pub mod solver;
pub mod route;
pub mod delta;
pub mod instance_gen;
pub mod genetic;
pub mod problem_loader;
pub mod config;
pub mod population;
pub mod instance;
pub mod bundle;
pub mod validation;
pub mod repair;

// Module registry for TIG compatibility
mod mod_;

// Re-export public API
pub use crate::vehicle_routing_solver::Solver;
pub use crate::utilities::IZS;
pub use crate::delta::DeltaTables;
pub use crate::tig_adaptive::TIGState;
pub use crate::population::Population;
pub use crate::mod_::delta_tables;
pub use crate::mod_::izs;
pub use crate::mod_::testing;




