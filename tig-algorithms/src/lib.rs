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

pub mod balanced_square;
pub use balanced_square as c001;
pub mod min_superstring;
pub use min_superstring as c002;
pub mod travelling_salesman;
pub use travelling_salesman as c003;
