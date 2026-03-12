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

fn optimizer_init(_seed: [u8; 32], param_sizes: &[usize], stream: Arc<CudaStream>, _module: Arc<CudaModule>, prop: &cudaDeviceProp) -> Result<Box<dyn OptimizerStateTrait>> {
    let threads_per_block: u32 = 128;
    let sm_count = prop.multiProcessorCount as u32;
    let sm_blocks = sm_count.saturating_mul(4).max(1);

    let mut m = Vec::new(); let mut v = Vec::new(); let mut prev_g = Vec::new(); let mut prev_u = Vec::new();
    let mut slow_u = Vec::new(); let mut f = Vec::new(); let mut ef = Vec::new(); let mut upd = Vec::new();

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
        let calc_blocks = ((n as u32 + threads_per_block - 1) / threads_per_block).min(sm_blocks).max(1);
        cfgs.push(LaunchConfig { grid_dim: (calc_blocks, 1, 1), block_dim: (threads_per_block, 1, 1), shared_mem_bytes: 0 });
    }

    let n_params = param_sizes.len();
    let mut layer_lrs = Vec::with_capacity(n_params);
    for (i, &ps) in param_sizes.iter().enumerate() {
        let depth_frac = i as f32 / (n_params.saturating_sub(1).max(1)) as f32;
        let depth_scale = 0.50f32 + 0.50f32 * depth_frac;
        let base: f32 = if ps <= 512 {
            0.0018
        } else if ps > 50000 {
            0.0010
        } else {
            0.0012
        };
        let mut lr = base * depth_scale;
        if i == 0 { lr = 0.0005; }
        if i == 1 { lr = base; }
        if i + 1 == n_params { lr = 0.0007; }
        layer_lrs.push(lr);
    }

    Ok(Box::new(OptimizerState {
        m, v, prev_g, prev_u, slow_u, f, ef, upd, cfgs, layer_lrs,
        spectral_boost: 1.1,
        step_count: 0,
        warmup_steps: 40,
        total_steps: 1000,
        noise_variance: 0.040,
        val_loss_history: Vec::new(),
        beta1: 0.92, beta2: 0.997, eps: 1e-8,
        weight_decay: 0.0025,
        bn_layer_boost: 1.47,
        output_layer_damping: 0.80,
        prev_val_loss: None, best_val_loss: None,
        plateau_count: 0, slope_ema: 0.0, lr_boost: 1.0, last_pulse_step: 0,
        last_epoch: 0, steps_in_epoch: 0, bpe_ema: 1.0, phase_tempo: 1.0,
    }) as Box<dyn OptimizerStateTrait>)
}

fn optimizer_query(_state: &dyn OptimizerStateTrait, _params: &[CudaSlice<f32>], _epoch: usize, _train: Option<f32>, _val: Option<f32>, _stream: Arc<CudaStream>, _module: Arc<CudaModule>, _prop: &cudaDeviceProp) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

fn optimizer_step(state: &mut dyn OptimizerStateTrait, model_params: &[CudaSlice<f32>], gradients: &[CudaSlice<f32>], epoch: usize, _train_loss: Option<f32>, val_loss: Option<f32>, stream: Arc<CudaStream>, module: Arc<CudaModule>, _prop: &cudaDeviceProp) -> Result<Vec<CudaSlice<f32>>> {
    let s = state.as_any_mut().downcast_mut::<OptimizerState>().unwrap();
    update_state_from_val_loss(s, epoch, val_loss);
    let global_damp = compute_global_damp(s, val_loss);

    s.lr_boost = s.lr_boost.clamp(0.5, 2.0);

    let divergence_factor: f32 = if let (Some(best), Some(curr)) = (s.best_val_loss, val_loss) {
        let r = curr / (best + 1e-8);
        if r > 1.3 { 0.5 } else if r > 1.15 { 0.75 } else if r > 1.05 { 0.9 } else { 1.0 }
    } else { 1.0 };

    let t = s.step_count as i32;
    let bias_correction1 = 1.0 - s.beta1.powi(t.max(1));
    let bias_correction2 = 1.0 - s.beta2.powi(t.max(1));

    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) = compute_blends(s, val_loss);

    let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * 3.0);
    let late_phase = s.step_count > s.total_steps * 13 / 20;
    let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);

    let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain) = compute_precision_params(s, val_loss);

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

    let k_fast = module.load_function("dual_consensus_fisher_kernel_14")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel_14")?;

    let mut updates = Vec::with_capacity(gradients.len());
    let num_layers = gradients.len();

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 { updates.push(stream.alloc_zeros::<f32>(0)?); continue; }

        let base_lr = s.layer_lrs[i];
        let tempo_lr = (1.0 / s.phase_tempo.powf(0.35)).max(0.6);
        let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr * divergence_factor;

        let is_output = i + 1 == num_layers;
        let layer_multiplier = if is_output {
            if let Some(loss) = val_loss {
                if s.step_count > s.warmup_steps + 30 {
                    (s.output_layer_damping * (0.7 + 0.3 * (loss / (s.noise_variance * 6.0)).min(1.0))).max(s.output_layer_damping)
                } else { s.output_layer_damping }
            } else { s.output_layer_damping }
        } else if n <= 512 {
            s.bn_layer_boost
        } else {
            1.0
        };

        let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff;

        let rel_update_cap: f32 = if near_floor {
            if is_output { 0.22 } else { 0.10 }
        } else {
            if is_output { 0.25 } else { 0.18 }
        };

        let cfg = s.cfgs[i];
        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream.launch_builder(&k_robust)
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
                stream.launch_builder(&k_fast)
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