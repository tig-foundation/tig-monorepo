pub mod knapsack;
pub use knapsack as c003;
pub mod satisfiability;
pub use satisfiability as c001;
pub mod vehicle_routing;
pub use vehicle_routing as c002;

#[cfg(feature = "cuda")]
mod gpu_algorithms {
    pub mod vector_search;
    pub use vector_search as c004;
}
#[cfg(feature = "cuda")]
pub use gpu_algorithms::*;
