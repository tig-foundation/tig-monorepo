## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** cag_sgd_1
* **Copyright:** 2025 Brent Beane 
* **Identity of Submitter:** Brent Beane
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses 

# CAG-SGD++: Predictive Dual-Phase Consensus Optimizer


**CAG-SGD++** is a  CUDA-accelerated optimizer designed for the TIG NNGD Challenge. It combines predictive generalization control, dual-phase consensus updates, and adaptive hyperparameter tuning to achieve superior convergence performance.

## 🚀 Key Features

### 1. **Predictive Generalization Control (PGC)**
A meta-control layer that anticipates convergence behavior and pre-adjusts parameters based on:
- Loss variance normalization
- Gradient stability tracking
- Plateau risk assessment
- Training progress analysis

### 2. **Dual-Phase Consensus**
- **Fast Phase**: Blends AdamW, Fisher approximation, and sign updates
- **Robust Phase**: Sign-based consensus with exponential filtering for noise-floor precision

### 3. **Adaptive Control Logic**
- Real-time hyperparameter adjustment based on training signals
- Near noise-floor detection and response
- Plateau detection and mitigation
- Trust backoff mechanism for loss spikes

### 4. **Layer-Aware Intelligence**
- Automatic layer role inference (embeddings, attention, batch norm, output)
- Role-specific learning rate scaling
- Role-specific weight decay adjustment

### 5. **Dynamic Lookahead**
- Adaptive lookahead frequency based on training phase
- Slow weight averaging for stability

### 6. **Stochastic Initialization**
- PGS-guided parameter sampling
- Bias toward optimal regions while maintaining diversity

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CAG-SGD++ Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │  Predictive Generalization Score (PGS) Estimator  │    │
│  │  - Loss variance tracking                          │    │
│  │  - Gradient stability analysis                     │    │
│  │  - Plateau risk prediction                         │    │
│  └──────────────────┬──────────────────────────────┘      │
│                     │                                       │
│                     ▼                                       │
│  ┌───────────────────────────────────────────────────┐    │
│  │         Control Logic Engine                      │    │
│  │  - Compute control signals                        │    │
│  │  - Adjust hyperparameters                         │    │
│  │  - Determine phase and blending                   │    │
│  └──────────────────┬──────────────────────────────┘      │
│                     │                                       │
│                     ▼                                       │
│  ┌───────────────────────────────────────────────────┐    │
│  │      Dual-Phase Consensus Core                    │    │
│  │                                                    │    │
│  │  Fast Phase:                                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │    │
│  │  │  AdamW   │ │  Fisher  │ │   Sign   │          │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘          │    │
│  │       └────────────┴─────────────┘                │    │
│  │                    │                               │    │
│  │         Weighted Blend (adaptive)                 │    │
│  │                                                    │    │
│  │  Robust Phase:                                     │    │
│  │  ┌──────────────────────────────────┐             │    │
│  │  │  Sign + Exponential Filtering    │             │    │
│  │  └──────────────────────────────────┘             │    │
│  └──────────────────┬──────────────────────────────┘      │
│                     │                                       │
│                     ▼                                       │
│  ┌───────────────────────────────────────────────────┐    │
│  │       Layer Role Inference & Scaling              │    │
│  │  - Detect layer type (BN, Embed, Attention, etc.) │    │
│  │  - Apply role-specific LR and WD multipliers      │    │
│  └──────────────────┬──────────────────────────────┘      │
│                     │                                       │
│                     ▼                                       │
│  ┌───────────────────────────────────────────────────┐    │
│  │        Dynamic Lookahead (optional)               │    │
│  │  - Slow weight averaging                          │    │
│  │  - Adaptive τ based on training signals           │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- CUDA Toolkit 11.x or later
- Rust 1.70+
- NVIDIA GPU with compute capability 6.0+

### Build

```bash
cargo build --release
```

## 📖 Usage

### Basic Example

```rust
use cag_sgd_plus_plus::prelude::*;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() -> Result<(), OptimizerError> {
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    
    // Define model parameter sizes
    let param_sizes = vec![1024, 2048, 4096, 512];
    
    // Create optimizer with custom seed
    let seed = [42u8; 32];
    let config = OptimizerConfig::default();
    let mut optimizer = CAGOptimizer::new(
        device.clone(),
        &param_sizes,
        seed,
        config,
    )?;
    
    // Training loop
    for step in 0..1000 {
        // ... compute gradients ...
        let gradients: Vec<CudaSlice<f32>> = todo!();
        let val_loss = Some(0.5);
        
        // Perform optimization step
        let updates = optimizer.step(&gradients, val_loss)?;
        
        // Apply updates to parameters
        // ... apply updates ...
        
        println!("Step {}: PGS = {:.4}, Best Loss = {:?}",
            optimizer.step_count(),
            optimizer.pgs_score(),
            optimizer.best_val_loss()
        );
    }
    
    Ok(())
}
```

### Custom Configuration

```rust
let config = OptimizerConfig {
    total_steps: 2000,
    warmup_steps: 128,
    beta1_init: 0.90,
    beta2_init: 0.999,
    eps_init: 1e-8,
    weight_decay_init: 0.02,
    lookahead_alpha_init: 0.6,
    lookahead_tau_init: 0.25,
};

let optimizer = CAGOptimizer::new(device, &param_sizes, seed, config)?;
```

## 🧪 Running Examples

```bash
# Run training example
cargo run --release --example train

# Run benchmarks
cargo bench
```

## 📈 Performance Characteristics

### Expected Behavior

1. **Warmup Phase (Steps 0-64)**:
   - Fast initial descent
   - Normalized updates dominate (60% weight)
   - Adaptive learning rate discovery

2. **Fast Convergence Phase (Steps 64-800)**:
   - Balanced update blending
   - PGS-guided exploration
   - Layer-aware scaling active

3. **Precision Phase (Steps 800-1000)**:
   - Transition to sign-based robust updates
   - High momentum for noise filtering
   - Fine-grained weight decay adjustment

### Target Metrics
- **Final Loss**: < 0.20
- **Convergence Speed**: Top 1% of challenge entrants
- **Stability**: No divergence across 100+ runs
- **Noise Floor**: < 0.042 * 5.0

## 🔬 Technical Deep Dive

### Predictive Generalization Score (PGS)

The PGS is computed using a logistic function:

```
PGS = 1 / (1 + exp(-z))

where z = 0.5 * (1 - loss_variance_norm)
        + 0.3 * gradient_stability
        - 0.4 * plateau_risk
        + 0.2 * step_progress
```

**Interpretation**:
- High PGS (0.8+): Model is stable, increase exploration
- Medium PGS (0.3-0.8): Normal operation
- Low PGS (<0.3): Risk of divergence, increase regularization

### Dual-Phase Consensus

**Fast Phase** (steps 0-800):
```
update = adam_w * AdamW_update
       + norm_w * Fisher_update  
       + sign_w * Sign_update
```

**Robust Phase** (steps 800-1000):
```
ef[t] = β_ef * ef[t-1] + (1 - β_ef) * sign(grad[t])
m[t] = β₁ * m[t-1] + (1 - β₁) * ef[t]
update = -lr * m[t]
```

### Layer Role Inference

```rust
LayerRole::InputEmbed  => LR × 1.8, WD × 0.5
LayerRole::BatchNorm   => LR × 1.8, WD × 0.0
LayerRole::Attention   => LR × 1.2, WD × 1.2
LayerRole::Dense       => LR × 1.0, WD × 1.0
LayerRole::Output      => LR × 0.6, WD × 1.5
```


### Competitive Advantages

1. **Predictive vs. Reactive**:
   - Other optimizers react to training signals
   - CAG-SGD++ predicts and preempts issues

2. **Meta-Adaptive Control**:
   - Not just adaptive hyperparameters
   - Adaptation of the adaptation strategy itself

3. **Layer Intelligence**:
   - Treats different layer types differently
   - No manual tuning required

4. **Dual-Phase Strategy**:
   - Fast descent when safe
   - Robust refinement when needed
   - Seamless transition between phases

5. **Stochastic Diversity with Control**:
   - Explores parameter space intelligently
   - PGS guides exploration to promising regions
   - Maintains stability while maximizing coverage


## 📚 References

This optimizer builds upon ideas from:
- AdamW (Loshchilov & Hutter, 2019)
- Lookahead Optimizer (Zhang et al., 2019)
- SignSGD (Bernstein et al., 2018)
- Fisher Information Matrix approximations
- TIG NNGD Challenge top entrants analysis