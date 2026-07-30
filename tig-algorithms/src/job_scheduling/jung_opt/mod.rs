pub mod types;
pub mod knobs;
pub mod preprocess;
mod infra_shared;
pub mod solver;
pub mod flow_shop;
pub mod hybrid_flow_shop;
/// E31 graft: `adaptive_js_v9`'s `hybrid_flow_shop.rs`, byte-identical to the v9 package.
/// Reached only when the `hfs_engine` knob == 1; `hfs_engine` == 0 (the default) keeps the
/// v10-lineage `hybrid_flow_shop` above.  See GRAFT_V9_HYBRID.md.
pub mod hybrid_flow_shop_v9;
pub mod job_shop;
pub mod fjsp_medium;
pub mod fjsp_high;

pub use solver::{solve_challenge, help};
