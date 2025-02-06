mod benchmarker_outbound;
pub use benchmarker_outbound::solve_challenge;
#[cfg(feature = "cuda")]
pub use benchmarker_outbound::{cuda_solve_challenge, KERNEL};