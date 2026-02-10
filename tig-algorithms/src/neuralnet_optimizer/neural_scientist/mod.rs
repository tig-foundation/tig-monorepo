use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

thread_local! {
    static HYPERPARAMETERS: std::cell::RefCell<Map<String, Value>> = std::cell::RefCell::new(Map::new());
}

#[derive(Clone)]
struct DualPhaseConsensusState {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    prev_g: Vec<CudaSlice<f32>>,          
    prev_u: Vec<CudaSlice<f32>>,          
    slow_u: Vec<CudaSlice<f32>>,         
    f: Vec<CudaSlice<f32>>,              
    ef: Vec<CudaSlice<f32>>,             
    upd: Vec<CudaSlice<f32>>,            
    cfgs: Vec<LaunchConfig>,
    layer_lrs: Vec<f32>,
    spectral_boost: f32,

    step_count: usize,
    warmup_steps: usize,
    total_steps: usize,

    noise_variance: f32,

    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,

    bn_layer_boost: f32,
    output_layer_damping: f32,

    prev_val_loss: Option<f32>,
    best_val_loss: Option<f32>,
    plateau_count: usize,
    slope_ema: f32,
    lr_boost: f32,
    last_pulse_step: usize,
    last_epoch: usize,
    steps_in_epoch: usize,
    bpe_ema: f32,
    phase_tempo: f32,
}

impl OptimizerStateTrait for DualPhaseConsensusState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let _k_fast = module.load_function("dual_consensus_fisher_kernel")?;
    let _k_robust = module.load_function("sign_ef_consensus_kernel")?;

    let hp = hyperparameters.clone().unwrap_or_default();
    HYPERPARAMETERS.with(|h| {
        *h.borrow_mut() = hp;
    });

    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init_state,
        optimizer_query_at_params,
        optimizer_step,
    )?;

    Ok(())
}

fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let (threads_per_block, blocks_per_sm, total_steps, warmup_steps, noise_variance, spectral_boost, beta1, beta2, eps, weight_decay, bn_layer_boost, output_layer_damping) = 
        HYPERPARAMETERS.with(|h| {
            let hp = h.borrow();
            let threads_per_block = hp.get("threads_per_block")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as u32;
            
            let blocks_per_sm = hp.get("blocks_per_sm")
                .and_then(|v| v.as_u64())
                .unwrap_or(6) as u32;
            
            let total_steps = hp.get("total_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(1000) as usize;
            
            let warmup_steps = hp.get("warmup_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(96) as usize;
            
            let noise_variance = hp.get("noise_variance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.042) as f32;
            
            let spectral_boost = hp.get("spectral_boost")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.05) as f32;
            
            let beta1 = hp.get("beta1")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.92) as f32;
            
            let beta2 = hp.get("beta2")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.997) as f32;
            
            let eps = hp.get("eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-8) as f32;
            
            let weight_decay = hp.get("weight_decay")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0025) as f32;
            
            let bn_layer_boost = hp.get("bn_layer_boost")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.35) as f32;
            
            let output_layer_damping = hp.get("output_layer_damping")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.8) as f32;
            
            (threads_per_block, blocks_per_sm, total_steps, warmup_steps, noise_variance, spectral_boost, beta1, beta2, eps, weight_decay, bn_layer_boost, output_layer_damping)
        });

    let mut m = Vec::new();
    let mut v = Vec::new();
    let mut prev_g = Vec::new();
    let mut prev_u = Vec::new();
    let mut slow_u = Vec::new();
    let mut f = Vec::new();
    let mut ef = Vec::new();
    let mut upd = Vec::new();
    for &n in param_sizes {
        m.push(stream.alloc_zeros::<f32>(n)?);
        v.push(stream.alloc_zeros::<f32>(n)?);
        prev_g.push(unsafe { stream.alloc::<f32>(n)? });
        prev_u.push(unsafe { stream.alloc::<f32>(n)? });
        slow_u.push(stream.alloc_zeros::<f32>(n)?);
        f.push(unsafe { stream.alloc::<f32>(n)? });
        ef.push(stream.alloc_zeros::<f32>(n)?);
        upd.push(unsafe { stream.alloc::<f32>(n)? });
    }
    
    let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(blocks_per_sm).max(1);
    let mut cfgs = Vec::with_capacity(param_sizes.len());
    for &n in param_sizes {
        let calc_blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
        let grid_dim = calc_blocks.min(sm_blocks).max(1);
        cfgs.push(LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        });
    }

    let mut layer_lrs = Vec::with_capacity(param_sizes.len());
    for (i, &param_size) in param_sizes.iter().enumerate() {
        let mut lr = 0.0012f32;
        if i == 0 { lr = 0.0011; }
        if param_size <= 512 { lr = 0.0018; }
        if param_size > 50000 { lr = 0.0009; }
        if i == param_sizes.len() - 1 { lr = 0.0007; }
        layer_lrs.push(lr);
    }

    let state = DualPhaseConsensusState {
        m,
        v,
        prev_g,
        prev_u,
        slow_u,
        f,
        ef,
        upd,
        cfgs,
        layer_lrs,
        spectral_boost,
        step_count: 0,
        warmup_steps,
        total_steps,
        noise_variance,
        beta1,
        beta2,
        eps,
        weight_decay,
        bn_layer_boost,
        output_layer_damping,
        prev_val_loss: None,
        best_val_loss: None,
        plateau_count: 0,
        slope_ema: 0.0,
        lr_boost: 1.0,
        last_pulse_step: 0,
        last_epoch: 0,
        steps_in_epoch: 0,
        bpe_ema: 1.0,
        phase_tempo: 1.0,
    };

    Ok(Box::new(state))
}

fn optimizer_query_at_params(
    _optimizer_state: &dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

#[inline]
fn spectral_phase_lr(s: &DualPhaseConsensusState, base_lr: f32) -> f32 {
    let t = s.step_count as f32;
    let warm = s.warmup_steps as f32;
    let total = s.total_steps as f32;

    if t <= warm {
        return base_lr * (t / warm.max(1.0)) * s.spectral_boost;
    }

    let progress = ((t - warm) / (total - warm).max(1.0)).min(1.0);

    let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    let spec_boost = s.spectral_boost * (1.0 - 0.3 * progress);

    base_lr * cosine_factor * spec_boost
}

#[inline]
fn compute_blends(s: &DualPhaseConsensusState, val_loss: Option<f32>) -> (f32, f32, f32, f32, f32, f32, f32) {    
    let t = s.step_count as f32;
    let warm = s.warmup_steps as f32;
    let total = s.total_steps as f32;
    let progress = (t / total.max(1.0)).min(1.0);

    let (mut blend_adam, mut blend_norm, mut blend_sign, gamma, bb_blend, mut lookahead_alpha, mut lookahead_tau): (f32, f32, f32, f32, f32, f32, f32) = if t <= warm {
        (0.35, 0.65, 0.0, 0.22, 0.6, 0.0, 0.2)
    } else {
        let mut trend = 0.0f32;
        if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
            trend = prev - curr; 
        }

        if trend > 1e-3 {
            (0.60, 0.35, 0.05, 0.28, 0.40, 0.15, 0.15)
        } else if trend.abs() < 1e-4 {
            (0.55, 0.35, 0.10, 0.15, 0.45, 0.30, 0.22)
        } else {
            (0.50, 0.35, 0.15, 0.20, 0.50, 0.20, 0.20)
        }
    };
    
    if t > warm {
        if let Some(curr) = val_loss {
            if curr <= s.noise_variance * 5.0 {                
                blend_sign = (blend_sign + 0.2).min(0.6);
                lookahead_alpha = lookahead_alpha.max(0.45);
                lookahead_tau = (lookahead_tau + 0.05).min(0.35);
                blend_adam *= 0.9;
                blend_norm *= 0.9;
            } else if curr >= s.noise_variance * 6.2 && curr <= s.noise_variance * 8.6 {
                blend_sign = blend_sign.max(0.35);
                blend_adam = (blend_adam * 0.95).max(0.25);
                blend_norm = (blend_norm * 0.95).max(0.15);
                lookahead_alpha = (lookahead_alpha * 0.85).min(0.35);
                lookahead_tau = (lookahead_tau * 0.85).min(0.30);
            }
        }

        if progress > 0.8 {            
            blend_norm = blend_norm.max(0.35);
            blend_sign *= 0.9;
            lookahead_alpha = lookahead_alpha.max(0.5);
            lookahead_tau = (lookahead_tau + 0.05).min(0.4);
        }
    }

    let sum = (blend_adam + blend_norm + blend_sign).max(1e-8);
    (
        blend_adam / sum,
        blend_norm / sum,
        blend_sign / sum,
        gamma,
        bb_blend,
        lookahead_alpha,
        lookahead_tau,
    )
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    _train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let s = optimizer_state.as_any_mut().downcast_mut::<DualPhaseConsensusState>().unwrap();
    s.step_count += 1;
    if s.step_count == 1 {
        s.last_epoch = epoch;
    }
    if s.last_epoch != epoch {
        if s.steps_in_epoch > 0 {
            s.bpe_ema = 0.9 * s.bpe_ema + 0.1 * (s.steps_in_epoch as f32);
        }
        s.steps_in_epoch = 0;
        s.last_epoch = epoch;
    }
    s.steps_in_epoch = s.steps_in_epoch.saturating_add(1);
    let tempo = (1.0 + 0.30 * s.bpe_ema.ln()).clamp(1.0, 2.2);
    s.phase_tempo = tempo;
    let mut global_damp = 1.0f32;

    if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let improvement = prev - curr;
        s.slope_ema = 0.85 * s.slope_ema + 0.15 * improvement;
        if s.step_count > s.warmup_steps {
            if improvement <= 1.0e-4 {
                s.plateau_count += 1;
            } else {
                s.plateau_count = 0;
            }
            if s.plateau_count >= 25 {
                if curr > s.noise_variance * 4.0 {
                    s.lr_boost = (s.lr_boost * 1.12).min(1.60);
                    s.last_pulse_step = s.step_count;
                    s.plateau_count = 0;
                }
            } else if s.lr_boost > 1.0 {
                let decay = if improvement > 5.0e-5 { 0.82 } else { 0.92 };
                s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay;
                if s.step_count.saturating_sub(s.last_pulse_step) > 80 {
                    s.lr_boost *= 0.96;
                }
                if s.lr_boost < 1.02 { s.lr_boost = 1.0; }
            }
        }
    }

    if let Some(loss) = val_loss {
        let dynamic_threshold = s.noise_variance * (1.1 + 0.1 * (s.step_count as f32 / s.total_steps as f32));
        if loss <= dynamic_threshold && s.step_count > s.warmup_steps {
            global_damp *= 0.2;
        }

        if loss <= s.noise_variance * 5.0 {
            let noise_proximity = (loss / (s.noise_variance * 5.0)).min(1.0);
            let noise_damping = 0.8 + 0.2 * noise_proximity;
            global_damp *= noise_damping;
        }
    }

    let t = s.step_count as i32;
    let bias_correction1 = 1.0 - s.beta1.powi(t);
    let bias_correction2 = 1.0 - s.beta2.powi(t);

    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) = compute_blends(s, val_loss);
    let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * 3.0);
    let late_phase = s.step_count > s.total_steps * 3 / 4;
    let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);

    let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain): (bool, f32, f32, f32, f32) = if let Some(loss) = val_loss {
        if s.step_count > s.warmup_steps {
            let z_lo = s.noise_variance * 6.2;
            let z_hi = s.noise_variance * 8.6;
            if loss >= z_lo && loss <= z_hi {
                let pos = ((z_hi - loss) / (z_hi - z_lo + 1e-8)).clamp(0.0, 1.0);
                let pg = 1.02 + 0.06 * pos;
                let gate_lo = 0.70 + 0.02 * pos;
                let gate_hi = 1.50 + 0.05 * pos;
                let forward_gain = if let Some(prev) = s.prev_val_loss {
                    let rel = ((prev - loss).max(0.0)) / (prev.abs() + 1e-6);
                    1.0 + (0.75 * rel).min(0.015)
                } else { 1.0 };
                (true, pg, gate_lo, gate_hi, forward_gain)
            } else { (false, 1.0, 0.66, 1.50, 1.0) }
        } else { (false, 1.0, 0.66, 1.50, 1.0) }
    } else { (false, 1.0, 0.66, 1.50, 1.0) };

    let beta1_eff: f32 = if in_precision_zone { (s.beta1 + 0.02).min(0.995) } else { s.beta1 };
    let beta2_eff: f32 = s.beta2;
    let eps_eff: f32 = if in_precision_zone { s.eps * 0.9 } else { s.eps };
    let mut wd_eff: f32 = if in_precision_zone { s.weight_decay * 1.05 } else { s.weight_decay };
    if s.step_count > s.warmup_steps {
        if near_floor {
            wd_eff *= 1.10;
        } else if s.plateau_count >= 20 {
            wd_eff *= 0.50;
        }
    }
    wd_eff *= (1.0 / s.phase_tempo).clamp(0.6, 1.0);

    let trust_backoff: f32 = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let delta = curr - prev;
        if delta > 2e-4 {
            1.0 / (1.0 + 1.5 * (delta / (prev.abs() + 1e-8)).min(0.02))
        } else {
            1.0
        }
    } else {
        1.0
    };

    let k_fast = module.load_function("dual_consensus_fisher_kernel")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel")?;    

    let mut updates = Vec::with_capacity(gradients.len());

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 {
            updates.push(stream.alloc_zeros::<f32>(0)?);
            continue;
        }

        let base_lr = s.layer_lrs[i];
        let tempo_lr = (1.0 / s.phase_tempo.powf(0.35)).max(0.6);
        let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr;

        let layer_multiplier = if i == gradients.len() - 1 {
            s.output_layer_damping
        } else if n <= 512 {
            s.bn_layer_boost
        } else {
            1.0
        };

        let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff;

        let cfg = s.cfgs[i];

        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream
                    .launch_builder(&k_robust)
                    .arg(g)
                    .arg(&mut s.f[i])
                    .arg(&mut s.ef[i])
                    .arg(&mut s.slow_u[i])
                    .arg(update_buf_ref)
                    .arg(&(n as u32))
                    .arg(&effective_lr)
                    .arg(&eps_eff)
                    .arg(&lookahead_alpha)
                    .arg(&lookahead_tau)
                    .arg(&gate_lo)
                    .arg(&gate_hi)
                    .launch(cfg)?;
            } else {
                stream
                    .launch_builder(&k_fast)
                    .arg(g)
                    .arg(&mut s.m[i])
                    .arg(&mut s.v[i])
                    .arg(&mut s.prev_g[i])
                    .arg(&mut s.prev_u[i])
                    .arg(&mut s.slow_u[i])
                    .arg(&mut s.f[i])
                    .arg(update_buf_ref)
                    .arg(&(n as u32))
                    .arg(&effective_lr)
                    .arg(&beta1_eff)
                    .arg(&beta2_eff)
                    .arg(&eps_eff)
                    .arg(&wd_eff)
                    .arg(&bias_correction1)
                    .arg(&bias_correction2)
                    .arg(&blend_adam)
                    .arg(&blend_norm)
                    .arg(&blend_sign)
                    .arg(&nesterov_gamma)
                    .arg(&bb_blend)
                    .arg(&lookahead_alpha)
                    .arg(&lookahead_tau)
                    .arg(&gate_lo)
                    .arg(&gate_hi)
                    .launch(cfg)?;
            }
        }

        updates.push(s.upd[i].clone());
    }

    if let Some(curr) = val_loss {
        s.best_val_loss = Some(match s.best_val_loss {
            Some(b) => if curr < b { curr } else { b },
            None => curr,
        });
    }
    s.prev_val_loss = val_loss;
    Ok(updates)
}

pub fn help() {
    println!("No help information available.");
}