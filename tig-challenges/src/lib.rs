pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

macro_rules! conditional_pub {
    (fn $name:ident $($rest:tt)*) => {
        #[cfg(not(feature = "hide_verification"))]
        pub fn $name $($rest)*

        #[cfg(feature = "hide_verification")]
        fn $name $($rest)*
    };
}

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
#[cfg(feature = "c006")]
pub(crate) mod neuralnet;
#[cfg(feature = "c006")]
pub mod neuralnet_optimizer;
#[cfg(feature = "c006")]
pub use neuralnet_optimizer as c006;
