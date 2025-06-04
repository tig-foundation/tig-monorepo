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
