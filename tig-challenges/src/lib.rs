pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

#[cfg(feature = "c001")]
pub mod satisfiability;
#[cfg(feature = "c001")]
pub use satisfiability as c001;
#[cfg(feature = "c002")]
pub mod vehicle_routing;
#[cfg(feature = "c002")]
pub use vehicle_routing as c002;
#[cfg(feature = "c003")]
pub mod knapsack;
#[cfg(feature = "c003")]
pub use knapsack as c003;
#[cfg(feature = "c004")]
pub mod vector_search;
#[cfg(feature = "c004")]
pub use vector_search as c004;
#[cfg(feature = "c005")]
pub mod hypergraph;
#[cfg(feature = "c005")]
pub use hypergraph as c005;
