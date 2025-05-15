pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

pub mod knapsack;
pub use knapsack as c003;
pub mod satisfiability;
pub use satisfiability as c001;
pub mod vehicle_routing;
pub use vehicle_routing as c002;

#[cfg(feature = "cuda")]
pub mod vector_search;
#[cfg(feature = "cuda")]
pub use vector_search as c004;
#[cfg(feature = "cuda")]
pub mod hypergraph;
#[cfg(feature = "cuda")]
pub use hypergraph as c005;
