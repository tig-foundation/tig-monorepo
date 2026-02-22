use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

use super::helpers::{
    OptimizerState,
    spectral_phase_lr, compute_blends, finalize_state,
};

thread_local! {
    static TRACK_CONFIG: std::cell::RefCell<TrackConfig> = std::cell::RefCell::new(TrackConfig::default());
}

struct TrackConfig {
    total_steps: usize,
    warmup_steps: usize,
    noise_variance: f32,
    spectral_boost: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bn_layer_boost: f32,
    output_layer_damping: f32,
    threads_per_block: u32,
    blocks_per_sm: u32,
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            total_steps: 1100,
            warmup_steps: 40,
            noise_variance: 0.040,
            spectral_boost: 1.1,
            beta1: 0.92,
            beta2: 0.997,
            eps: 1e-8,
            weight_decay: 0.0025,
            bn_layer_boost: 1.35,
            output_layer_damping: 0.8,
            threads_per_block: 128,
            blocks_per_sm: 4,
        }
    }
}

fn parse_config(hyperparameters: &Option<Map<String, Value>>) -> TrackConfig {
    let hp = hyperparameters.as_ref();
    TrackConfig {
        total_steps: hp.and_then(|h| h.get("total_steps").and_then(|v| v.as_u64())).unwrap_or(1100) as usize,
        warmup_steps: hp.and_then(|h| h.get("warmup_steps").and_then(|v| v.as_u64())).unwrap_or(40) as usize,
        noise_variance: hp.and_then(|h| h.get("noise_variance").and_then(|v| v.as_f64())).unwrap_or(0.040) as f32,
        spectral_boost: hp.and_then(|h| h.get("spectral_boost").and_then(|v| v.as_f64())).unwrap_or(1.1) as f32,
        beta1: hp.and_then(|h| h.get("beta1").and_then(|v| v.as_f64())).unwrap_or(0.92) as f32,
        beta2: hp.and_then(|h| h.get("beta2").and_then(|v| v.as_f64())).unwrap_or(0.997) as f32,
        eps: hp.and_then(|h| h.get("eps").and_then(|v| v.as_f64())).unwrap_or(1e-8) as f32,
        weight_decay: hp.and_then(|h| h.get("weight_decay").and_then(|v| v.as_f64())).unwrap_or(0.0025) as f32,
        bn_layer_boost: hp.and_then(|h| h.get("bn_layer_boost").and_then(|v| v.as_f64())).unwrap_or(1.35) as f32,
        output_layer_damping: hp.and_then(|h| h.get("output_layer_damping").and_then(|v| v.as_f64())).unwrap_or(0.8) as f32,
        threads_per_block: hp.and_then(|h| h.get("threads_per_block").and_then(|v| v.as_u64())).unwrap_or(128) as u32,
        blocks_per_sm: hp.and_then(|h| h.get("blocks_per_sm").and_then(|v| v.as_u64())).unwrap_or(4) as u32,
    }
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let config = parse_config(hyperparameters);
    TRACK_CONFIG.with(|c| *c.borrow_mut() = config);

    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init,
        optimizer_query,
        optimizer_step,
    )?;
    Ok(())
}

fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    TRACK_CONFIG.with(|c| {
        let cfg = c.borrow();
        
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
            prev_g.push(stream.alloc_zeros::<f32>(n)?);
            prev_u.push(stream.alloc_zeros::<f32>(n)?);
            slow_u.push(stream.alloc_zeros::<f32>(n)?);
            let mut fisher_init = stream.alloc_zeros::<f32>(n)?;
            let init_fisher = vec![1e-4f32; n];
            stream.memcpy_htod(&init_fisher, &mut fisher_init)?;
            f.push(fisher_init);
            ef.push(stream.alloc_zeros::<f32>(n)?);
            upd.push(unsafe { stream.alloc::<f32>(n)? });
        }

        let sm_blocks = (prop.multiProcessorCount as u32)
            .saturating_mul(cfg.blocks_per_sm)
            .max(1);
        let mut cfgs = Vec::with_capacity(param_sizes.len());
        for &n in param_sizes {
            let calc_blocks = (n as u32 + cfg.threads_per_block - 1) / cfg.threads_per_block;
            let grid_dim = calc_blocks.min(sm_blocks).max(1);
            cfgs.push(LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (cfg.threads_per_block, 1, 1),
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

        let state = OptimizerState {
            m, v, prev_g, prev_u, slow_u, f, ef, upd, cfgs, layer_lrs,
            spectral_boost: cfg.spectral_boost,
            step_count: 0,
            warmup_steps: cfg.warmup_steps,
            total_steps: cfg.total_steps,
            noise_variance: cfg.noise_variance,
            val_loss_history: Vec::new(),
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            weight_decay: cfg.weight_decay,
            bn_layer_boost: cfg.bn_layer_boost,
            output_layer_damping: cfg.output_layer_damping,
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

        Ok(Box::new(state) as Box<dyn OptimizerStateTrait>)
    })
}

fn optimizer_query(
    _state: &dyn OptimizerStateTrait,
    _params: &[CudaSlice<f32>],
    _epoch: usize,
    _train: Option<f32>,
    _val: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

fn optimizer_step(
    state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    _train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let s = state.as_any_mut().downcast_mut::<OptimizerState>().unwrap();    
   
    s.step_count += 1;
    if s.step_count == 1 { s.last_epoch = epoch; }
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

    if let Some(loss) = val_loss {
        if s.step_count > s.warmup_steps {
            s.val_loss_history.push(loss);
            if s.val_loss_history.len() > 12 { s.val_loss_history.remove(0); }
            if s.val_loss_history.len() >= 6 {
                let min_loss = s.val_loss_history.iter().copied().fold(f32::INFINITY, f32::min);
                let recent_n = (s.val_loss_history.len().min(10)) as f32;
                let recent_avg = s.val_loss_history.iter().rev().take(10).sum::<f32>() / recent_n.max(1.0);
                let target_nv = (min_loss / 5.0).min(recent_avg / 8.0);
                s.noise_variance = 0.85 * s.noise_variance + 0.15 * target_nv;
                s.noise_variance = s.noise_variance.clamp(0.0, 0.05);
            }
        }
    }

    if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        if s.step_count > s.warmup_steps && s.step_count > 20 {
            let improvement = prev - curr;
            let relative_improvement = improvement / prev.abs().max(1e-8);
            if relative_improvement > 0.008 {
                s.spectral_boost = (s.spectral_boost * 1.010).min(1.40);  
            } else if relative_improvement < -0.003 {
                s.spectral_boost *= 0.985;  
            } else if relative_improvement.abs() < 0.0005 && s.plateau_count > 15 {
                s.spectral_boost = (s.spectral_boost * 1.004).min(1.40);  
            }
            s.spectral_boost = s.spectral_boost.clamp(0.90, 1.40);  
        }
    }

    if s.step_count > s.warmup_steps && s.val_loss_history.len() >= 8 {
        let recent_avg = s.val_loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
        let older_n = (s.val_loss_history.len().saturating_sub(5)).min(5) as f32;
        let older_avg = s.val_loss_history.iter().rev().skip(5).take(5).sum::<f32>() / older_n.max(1.0);
        let trend = older_avg - recent_avg;
        let target_beta1 = if trend > 0.02 { 0.94 } else if trend < -0.02 { 0.88 } else { 0.91 };
        s.beta1 = 0.85 * s.beta1 + 0.15 * target_beta1;
        s.beta1 = s.beta1.clamp(0.87, 0.94);
    }

    let mut global_damp = 1.0f32;
    if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let improvement = prev - curr;
        s.slope_ema = 0.85 * s.slope_ema + 0.15 * improvement;
        if s.step_count > s.warmup_steps {
            let is_stagnant = improvement <= 1.0e-4 && s.slope_ema < 2.0e-4;
            let is_declining = improvement < 0.0 && s.slope_ema < 0.0;
            if is_stagnant || is_declining { s.plateau_count += 1; }
            else if improvement > 5.0e-5 { s.plateau_count = 0; }
            else if s.plateau_count > 0 { s.plateau_count = s.plateau_count.saturating_sub(1); }
            
            if s.plateau_count >= 25 {
                if curr > s.noise_variance * 4.0 {
                    s.lr_boost = (s.lr_boost * 1.08).min(1.45);  
                    s.last_pulse_step = s.step_count;
                    s.plateau_count = 0;
                }
            } else if s.plateau_count >= 15 && curr > s.noise_variance * 8.0 {
                s.lr_boost = (s.lr_boost * 1.10).min(1.50);  
                s.last_pulse_step = s.step_count;
                s.plateau_count = 0;
            } else if s.plateau_count >= 18 && curr > s.noise_variance * 5.0 {
                s.lr_boost = (s.lr_boost * 1.06).min(1.40);  
                s.last_pulse_step = s.step_count;
                s.plateau_count = 0;
            } else if s.lr_boost > 1.0 {
                let relative_improvement = improvement / curr.abs().max(1e-8);
                let decay = if relative_improvement > 0.01 { 0.75 }
                    else if relative_improvement > 0.001 { 0.85 }
                    else if improvement > 0.0 { 0.93 } else { 0.97 };
                s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay;
                let decay2 = if improvement > 5.0e-5 { 0.82 } else { 0.92 };
                s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay2;
                if s.step_count.saturating_sub(s.last_pulse_step) > 80 { s.lr_boost *= 0.96; }
                if s.lr_boost < 1.02 { s.lr_boost = 1.0; }
            }
        }
    }

    if let Some(loss) = val_loss {
        let dynamic_threshold = s.noise_variance * (1.1 + 0.1 * (s.step_count as f32 / s.total_steps as f32));
        if loss <= dynamic_threshold && s.step_count > s.warmup_steps {
            let proximity = (loss / dynamic_threshold).clamp(0.4, 1.0);
            let plateau_mul: f32 = if s.plateau_count > 10 { 0.85 } else { 1.0 };  
            global_damp *= (0.30 + 0.40 * proximity) * 0.9 * plateau_mul;  
        }
        if loss <= s.noise_variance * 5.0 {
            let noise_proximity = (loss / (s.noise_variance * 5.0)).min(1.0);
            let steepness = 1.0 + 0.5 * (1.0 - noise_proximity);
            let noise_damping = 0.70 + 0.30 * noise_proximity.powf(steepness);
            global_damp *= noise_damping;
        }
    }

    let t = s.step_count as i32;
    let bias_correction1 = 1.0 - s.beta1.powi(t);
    let bias_correction2 = 1.0 - s.beta2.powi(t);

    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) = compute_blends(s, val_loss);
    
    let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * 4.0);
    let late_phase = s.step_count > s.total_steps * 3 / 4;
    let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase || s.plateau_count >= 22);

    let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain) = 
        if let Some(loss) = val_loss {
            if s.step_count > s.warmup_steps {
                let z_lo = s.noise_variance * 6.2;
                let z_hi = s.noise_variance * 8.6;
                if loss >= z_lo && loss <= z_hi {
                    let pos = ((z_hi - loss) / (z_hi - z_lo + 1e-8)).clamp(0.0, 1.0);
                    let pg = 1.01 + 0.05 * pos;  
                    let glo = 0.68 + 0.02 * pos;  
                    let ghi = 1.45 + 0.04 * pos;  
                    let fg = if let Some(prev) = s.prev_val_loss {
                        let rel = ((prev - loss).max(0.0)) / (prev.abs() + 1e-6);
                        1.0 + (0.75 * rel).min(0.015)
                    } else { 1.0 };
                    (true, pg, glo, ghi, fg)
                } else { (false, 1.0, 0.68, 1.45, 1.0) }
            } else { (false, 1.0, 0.68, 1.45, 1.0) }
        } else { (false, 1.0, 0.68, 1.45, 1.0) };

    let beta1_eff: f32 = if in_precision_zone { (s.beta1 + 0.02).min(0.995) } else { s.beta1 };
    let beta2_eff: f32 = s.beta2;
    let eps_eff: f32 = if in_precision_zone { s.eps * 0.9 } else { s.eps };
    let mut wd_eff: f32 = if in_precision_zone { s.weight_decay * 1.05 } else { s.weight_decay };
    if s.step_count > s.warmup_steps {
        if near_floor { wd_eff *= 1.10; }
        else if s.plateau_count >= 20 { wd_eff *= 0.50; }
    }
    wd_eff *= (1.0 / s.phase_tempo).clamp(0.6, 1.0);

    let trust_backoff: f32 = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let delta = curr - prev;
        if delta > 2e-4 { 1.0 / (1.0 + 1.5 * (delta / (prev.abs() + 1e-8)).min(0.02)) } else { 1.0 }
    } else { 1.0 };

    let k_fast = module.load_function("dual_consensus_fisher_kernel")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel")?;

    let mut updates = Vec::with_capacity(gradients.len());
    
    let tempo_lr = (1.0 / s.phase_tempo.powf(0.35)).max(0.6);
    let lr_factor = spectral_phase_lr(s, 1.0) * global_damp * s.lr_boost * tempo_lr;

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 { updates.push(stream.alloc_zeros::<f32>(0)?); continue; }

        let p = model_params.get(i).unwrap();  
        let base_lr = s.layer_lrs[i];
        let lr = base_lr * lr_factor;

        let layer_multiplier = if i == gradients.len() - 1 {
            if let Some(loss) = val_loss {
                if s.step_count > s.warmup_steps + 30 {
                    let loss_ratio = (loss / (s.noise_variance * 6.0)).min(1.0);
                    (s.output_layer_damping * (0.7 + 0.3 * loss_ratio)).max(s.output_layer_damping)
                } else { s.output_layer_damping }
            } else { s.output_layer_damping }
        } else if n <= 512 { s.bn_layer_boost } else { 1.0 };

        let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff;
        let cfg = s.cfgs[i];
        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream.launch_builder(&k_robust)
                    .arg(g)
                    .arg(p)  
                    .arg(&mut s.f[i])
                    .arg(&mut s.ef[i])
                    .arg(&mut s.slow_u[i])
                    .arg(update_buf_ref)
                    .arg(&(n as u32))
                    .arg(&effective_lr)
                    .arg(&eps_eff)
                    .arg(&wd_eff)  
                    .arg(&lookahead_alpha)
                    .arg(&lookahead_tau)
                    .arg(&gate_lo)
                    .arg(&gate_hi)
                    .launch(cfg)?;
            } else {
                stream.launch_builder(&k_fast)
                    .arg(g)
                    .arg(p)  
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

    finalize_state(s, val_loss);
    Ok(updates)
}
