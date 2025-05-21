use ahash::RandomState;
pub fn seeded_hasher(seed: &[u8; 32]) -> RandomState {
    let seed1 = u64::from_be_bytes(seed[0..8].try_into().unwrap());
    let seed2 = u64::from_be_bytes(seed[8..16].try_into().unwrap());
    let seed3 = u64::from_be_bytes(seed[16..24].try_into().unwrap());
    let seed4 = u64::from_be_bytes(seed[24..32].try_into().unwrap());
    RandomState::with_seeds(seed1, seed2, seed3, seed4)
}
pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, RandomState>;
pub(crate) type HashSet<T> = std::collections::HashSet<T, RandomState>;

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
