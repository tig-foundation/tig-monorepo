// Vendored compatibility wrapper for c001_a115 / sat_hybrid_v3 by NVX.
#[path = "a115_engine_e.rs"]
mod engine_e;
#[path = "a115_engine_ag.rs"]
mod engine_ag;
pub use engine_ag::solve as solve_challenge;
pub(super) use engine_e::{Prepared,preprocess};
