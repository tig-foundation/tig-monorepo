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
    spectral_phase_lr, compute_blends, update_state_from_val_loss,
    compute_global_damp, compute_precision_params, finalize_state,
};

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}

fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let threads_per_block: u32 = 128;
    let blocks_per_sm: u32 = 4;
    let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(blocks_per_sm).max(1);

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
        stream.memcpy_htod(&vec![1e-4f32; n], &mut fisher_init)?;
        f.push(fisher_init);
        ef.push(stream.alloc_zeros::<f32>(n)?);
        upd.push(unsafe { stream.alloc::<f32>(n)? });
    }

    let mut cfgs = Vec::with_capacity(param_sizes.len());
    for &n in param_sizes {
        let calc_blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
        cfgs.push(LaunchConfig {
            grid_dim: (calc_blocks.min(sm_blocks).max(1), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        });
    }

    let num_layers = param_sizes.len();
    let mut layer_lrs = Vec::with_capacity(num_layers);
    for (i, &ps) in param_sizes.iter().enumerate() {
        let depth_frac = i as f32 / (num_layers.max(1) as f32);
        let depth_scale = 0.55f32 + 0.45f32 * depth_frac;
        let mut lr = 0.00155f32 * depth_scale;
        if ps <= 512 {
            lr = 0.0022f32;
        }
        if ps > 50000 {
            lr = lr.min(0.00100f32);
        }
        if i == num_layers.saturating_sub(1) {
            lr = 0.00078f32;
        }
        if i == 0 {
            lr = 0.00088f32;
        }
        layer_lrs.push(lr);
    }

    Ok(Box::new(OptimizerState {
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
        spectral_boost: 1.1,
        step_count: 0,
        warmup_steps: 45,
        total_steps: 900,
        noise_variance: 0.040,
        val_loss_history: Vec::new(),
        beta1: 0.92,
        beta2: 0.997,
        eps: 1e-8,
        weight_decay: 0.0025,
        bn_layer_boost: 1.48,
        output_layer_damping: 0.77,
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
    }) as Box<dyn OptimizerStateTrait>)
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

fn compute_trend_factor(history: &[f32], best: f32) -> f32 {    
    let n = history.len();
    if n < 3 {
        return 1.0f32;
    }
    let recent = &history[n.saturating_sub(3)..];
    let all_increasing = recent.windows(2).all(|w| w[1] > w[0]);
    if !all_increasing {
        return 1.0f32;
    }
    let last = recent[recent.len() - 1];
    let r = last / (best + 1e-8f32);
    if r > 1.15f32 {
        0.65f32
    } else if r > 1.06f32 {
        0.80f32
    } else if r > 1.02f32 {
        0.93f32
    } else {
        1.0f32
    }
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
    update_state_from_val_loss(s, epoch, val_loss);
    let mut global_damp = compute_global_damp(s, val_loss);

    s.lr_boost = s.lr_boost.clamp(0.5f32, 1.9f32);

    if s.step_count > s.warmup_steps {
        if let (Some(best), Some(curr)) = (s.best_val_loss, val_loss) {
            let r = curr / (best + 1e-8f32);
            if r > 3.0f32 {
                global_damp *= 0.12f32;
                s.last_pulse_step = s.step_count;
            } else if r > 2.0f32 {
                global_damp *= 0.24f32;
                s.last_pulse_step = s.step_count;
            } else if r > 1.5f32 {
                global_damp *= 0.40f32;
                s.last_pulse_step = s.step_count;
            } else if r > 1.2f32 {
                global_damp *= 0.65f32;
                s.last_pulse_step = s.step_count;
            } else if r > 1.08f32 {
                global_damp *= 0.87f32;
            }
        }
    }

    if s.step_count > s.warmup_steps + 10 {
        if let Some(best) = s.best_val_loss {
            let trend_factor = compute_trend_factor(&s.val_loss_history, best);
            if trend_factor < 1.0f32 {
                global_damp *= trend_factor;
                if trend_factor < 0.85f32 {
                    s.last_pulse_step = s.step_count;
                }
            }
        }
    }

    if s.step_count > s.warmup_steps && s.last_pulse_step > s.warmup_steps {
        let steps_since = s.step_count.saturating_sub(s.last_pulse_step);
        if steps_since > 0 && steps_since < 40 {
            let recovery = 0.22f32 + 0.78f32 * (steps_since as f32 / 40.0f32);
            global_damp *= recovery;
        }
    }

    let t = s.step_count as i32;
    let bias_correction1 = 1.0f32 - s.beta1.powi(t.max(1));
    let bias_correction2 = 1.0f32 - s.beta2.powi(t.max(1));

    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) =
        compute_blends(s, val_loss);

    let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * 3.0f32);
    let late_phase = s.step_count > (s.total_steps * 60 / 100);
    let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);

    let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain) =
        compute_precision_params(s, val_loss);

    let beta1_eff: f32 = if in_precision_zone {
        (s.beta1 + 0.02f32).min(0.995f32)
    } else {
        s.beta1
    };
    let beta2_eff: f32 = s.beta2;
    let eps_eff: f32 = if in_precision_zone { s.eps * 0.85f32 } else { s.eps };

    let mut wd_eff: f32 = if in_precision_zone {
        s.weight_decay * 1.05f32
    } else {
        s.weight_decay
    };
    if s.step_count > s.warmup_steps {
        if near_floor {
            wd_eff *= 1.10f32;
        } else if s.plateau_count >= 20 {
            wd_eff *= 0.50f32;
        }
    }
    wd_eff *= (1.0f32 / s.phase_tempo).clamp(0.6f32, 1.0f32);

    let trust_backoff: f32 = if s.step_count > s.warmup_steps {
        if let (Some(best), Some(curr)) = (s.best_val_loss, val_loss) {
            let r = curr / (best + 1e-8f32);
            if r > 2.5f32 { 0.18f32 }
            else if r > 1.8f32 { 0.32f32 }
            else if r > 1.3f32 { 0.55f32 }
            else if r > 1.1f32 { 0.80f32 }
            else { 1.0f32 }
        } else { 1.0f32 }
    } else { 1.0f32 };

    let trend_trust: f32 = if s.step_count > s.warmup_steps + 10 {
        if let Some(best) = s.best_val_loss {
            let tf = compute_trend_factor(&s.val_loss_history, best);
            0.5f32 + 0.5f32 * tf
        } else {
            1.0f32
        }
    } else {
        1.0f32
    };

    let k_fast = module.load_function("dual_consensus_fisher_kernel_10")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel_10")?;

    let mut updates = Vec::with_capacity(gradients.len());
    let num_layers = gradients.len();

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 {
            updates.push(stream.alloc_zeros::<f32>(0)?);
            continue;
        }

        let base_lr = s.layer_lrs[i];
        let tempo_lr = (1.0f32 / s.phase_tempo.powf(0.35f32)).max(0.6f32);
        let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr;

        let is_bn = n <= 512;
        let is_output = i == num_layers.saturating_sub(1);
        let layer_multiplier = if is_output {
            s.output_layer_damping
        } else if is_bn {
            s.bn_layer_boost
        } else {
            1.0f32
        };

        let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff * trend_trust;

        let rel_update_cap: f32 = if near_floor { 0.11f32 } else { 0.17f32 };
        let rel_update_cap = if is_output {
            (rel_update_cap * 1.45f32).min(0.31f32)
        } else {
            rel_update_cap
        };

        let cfg = s.cfgs[i];
        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream
                    .launch_builder(&k_robust)
                    .arg(g)
                    .arg(&model_params[i])
                    .arg(&mut s.f[i])
                    .arg(&mut s.ef[i])
                    .arg(&mut s.slow_u[i])
                    .arg(update_buf_ref)
                    .arg(&(n as u32))
                    .arg(&effective_lr)
                    .arg(&eps_eff)
                    .arg(&wd_eff)
                    .arg(&rel_update_cap)
                    .arg(&lookahead_alpha)
                    .arg(&lookahead_tau)
                    .arg(&gate_lo)
                    .arg(&gate_hi)
                    .launch(cfg)?;
            } else {
                stream
                    .launch_builder(&k_fast)
                    .arg(g)
                    .arg(&model_params[i])
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