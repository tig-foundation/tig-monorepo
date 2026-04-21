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
    intra_epoch_factor, loss_curvature_factor,
};

#[derive(Clone)]
struct TrackHP {
    total_steps: usize, warmup_steps: usize,
    beta1: f32, beta2: f32, weight_decay: f32,
    spectral_boost: f32, bn_layer_boost: f32,
}
impl Default for TrackHP {
    fn default() -> Self {
        TrackHP { total_steps: 1300, warmup_steps: 55,
            beta1: 0.92, beta2: 0.997, weight_decay: 0.0025,
            spectral_boost: 1.1, bn_layer_boost: 1.47 }
    }
}
thread_local! {
    static TRACK_HP: std::cell::RefCell<TrackHP> = std::cell::RefCell::new(TrackHP::default());
}
fn parse_f32(hp: Option<&Map<String, Value>>, key: &str, default: f32) -> f32 {
    hp.and_then(|h| h.get(key).and_then(|v| v.as_f64())).map(|v| v as f32).unwrap_or(default)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let hp = hyperparameters.as_ref();
    let d = TrackHP::default();
    let cfg = TrackHP {
        total_steps: hp.and_then(|h| h.get("total_steps").and_then(|v| v.as_u64())).unwrap_or(d.total_steps as u64) as usize,
        warmup_steps: hp.and_then(|h| h.get("warmup_steps").and_then(|v| v.as_u64())).unwrap_or(d.warmup_steps as u64) as usize,
        beta1: parse_f32(hp, "beta1", d.beta1),
        beta2: parse_f32(hp, "beta2", d.beta2),
        weight_decay: parse_f32(hp, "weight_decay", d.weight_decay),
        spectral_boost: parse_f32(hp, "spectral_boost", d.spectral_boost),
        bn_layer_boost: parse_f32(hp, "bn_layer_boost", d.bn_layer_boost),
    };
    TRACK_HP.with(|c| *c.borrow_mut() = cfg);
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

    let mut best_w_bufs = Vec::new();
    for &n in param_sizes {
        best_w_bufs.push(stream.alloc_zeros::<f32>(n)?);
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

    let hp_cfg = TRACK_HP.with(|c| c.borrow().clone());
    Ok(Box::new(OptimizerState {
        m, v, prev_g, prev_u, slow_u, f, ef, upd, cfgs, layer_lrs,
        spectral_boost: hp_cfg.spectral_boost,
        step_count: 0,
        warmup_steps: hp_cfg.warmup_steps,
        total_steps: hp_cfg.total_steps,
        noise_variance: 0.040,
        val_loss_history: Vec::new(),
        beta1: hp_cfg.beta1, beta2: hp_cfg.beta2, eps: 1e-8,
        weight_decay: hp_cfg.weight_decay,
        bn_layer_boost: hp_cfg.bn_layer_boost,
        output_layer_damping: 0.80,
        prev_val_loss: None, best_val_loss: None,
        plateau_count: 0, slope_ema: 0.0, lr_boost: 1.0, last_pulse_step: 0,
        last_epoch: 0, steps_in_epoch: 0, bpe_ema: 1.0, phase_tempo: 1.0, spectral_decay: 0.3, best_w: Some(best_w_bufs), tt_best_loss: f32::INFINITY, tt_cooldown: 0, global_lr_mult: 1.0, tt_prev_epoch: 0,
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

    let mut nitt_save = false;
    let mut nitt_rollback = false;
    if s.tt_prev_epoch != epoch && s.step_count > 1 {
        if s.tt_cooldown > 0 { s.tt_cooldown -= 1; }
        if let Some(vl) = val_loss {
            if vl < s.tt_best_loss {
                s.tt_best_loss = vl;
                nitt_save = true;
            } else if s.step_count > s.warmup_steps + 10
                && s.tt_cooldown == 0
                && s.tt_best_loss < f32::INFINITY
                && vl > s.tt_best_loss * 2.5 + 0.03
            {
                nitt_rollback = true;
                s.tt_cooldown = 15;
                s.global_lr_mult *= 0.5;
                s.plateau_count = 0;
                s.lr_boost = 1.0;
            }
        }
    }
    s.tt_prev_epoch = epoch;

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
    let k_tt_save = module.load_function("tt_save_kernel")?;
    let k_tt_restore = module.load_function("tt_restore_kernel")?;

    let mut updates = Vec::with_capacity(gradients.len());
    let num_layers = gradients.len();

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 { updates.push(stream.alloc_zeros::<f32>(0)?); continue; }

        let p = &model_params[i];
        let cfg_tt = LaunchConfig { grid_dim: (((n as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        if nitt_save {
            if let Some(ref mut bw) = s.best_w {
                unsafe { stream.launch_builder(&k_tt_save).arg(p).arg(&mut bw[i]).arg(&(n as u32)).launch(cfg_tt)?; }
            }
        }

        if nitt_rollback {
            if let Some(ref bw) = s.best_w {
                let update_buf = &mut s.upd[i];
                unsafe {
                    stream.launch_builder(&k_tt_restore)
                        .arg(p).arg(&bw[i]).arg(update_buf)
                        .arg(&mut s.m[i]).arg(&mut s.v[i]).arg(&mut s.prev_g[i])
                        .arg(&mut s.prev_u[i]).arg(&mut s.slow_u[i]).arg(&mut s.f[i]).arg(&mut s.ef[i])
                        .arg(&(n as u32)).launch(cfg_tt)?;
                }
                updates.push(s.upd[i].clone());
                continue;
            }
        }

        let base_lr = s.layer_lrs[i];
        let tempo_lr = (1.0 / s.phase_tempo.powf(0.35)).max(0.6);
        let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr * divergence_factor * s.global_lr_mult;

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