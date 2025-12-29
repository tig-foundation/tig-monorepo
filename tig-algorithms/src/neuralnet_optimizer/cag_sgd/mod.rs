//! CAG-SGD++: Predictive Dual-Phase Consensus Optimizer
//!
//! A state-of-the-art optimization system that combines:
//! - Dual-phase consensus (Fisher + sign)
//! - Predictive Generalization Control (PGC)
//! - Adaptive hyperparameter tuning
//! - Layer-aware scaling
//! - Stochastic initialization with guided exploration

pub mod optimizer;
pub mod gpu;

pub use optimizer::{
    CAGOptimizer,
    CAGState,
    OptimizerConfig,
    OptimizerError,
};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::optimizer::{
        CAGOptimizer,
        CAGState,
        OptimizerConfig,
        OptimizerError,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::CudaDevice;
    use std::sync::Arc;

    #[test]
    fn test_optimizer_creation() {
        let device = CudaDevice::new(0).unwrap();
        let device = Arc::new(device);
        
        let param_sizes = vec![1024, 2048, 512];
        let seed = [42u8; 32];
        
        let config = OptimizerConfig::default();
        let optimizer = CAGOptimizer::new(
            device,
            &param_sizes,
            seed,
            config,
        );
        
        assert!(optimizer.is_ok());
    }
}
