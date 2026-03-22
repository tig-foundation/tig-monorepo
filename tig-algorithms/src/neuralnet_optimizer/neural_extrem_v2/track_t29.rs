use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg}, 
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

use super::helpers::{
    OptimizerState,
    spectral_phase_lr, compute_blends, update_state_from_val_loss,
    compute_global_damp, compute_precision_params, finalize_state,
    intra_epoch_factor, loss_curvature_factor,
};

const TOTAL_STEPS: usize = 1024;
const WARMUP_STEPS: usize = 32;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
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
    let mut m = Vec::with_capacity(param_sizes.len());
    let mut v = Vec::with_capacity(param_sizes.len());
    let mut prev_g = Vec::with_capacity(param_sizes.len());
    let mut prev_u = Vec::with_capacity(param_sizes.len());
    let mut slow_u = Vec::with_capacity(param_sizes.len());
    let mut f = Vec::with_capacity(param_sizes.len());
    let mut ef = Vec::with_capacity(param_sizes.len());
    let mut upd = Vec::with_capacity(param_sizes.len());

    for &n in param_sizes {
        m.push(stream.alloc_zeros::<f32>(n)?);
        v.push(stream.alloc_zeros::<f32>(n)?);
        prev_g.push(stream.alloc_zeros::<f32>(n)?);
        prev_u.push(stream.alloc_zeros::<f32>(n)?);
        slow_u.push(stream.alloc_zeros::<f32>(n)?);
        
        f.push(stream.alloc_zeros::<f32>(n)?);

        ef.push(stream.alloc_zeros::<f32>(n)?);
        upd.push(unsafe { stream.alloc::<f32>(n)? });
    }

    let threads_per_block: u32 = 256;
    let blocks_per_sm: u32 = 3;
    let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(blocks_per_sm).max(1);

    let mut cfgs = Vec::with_capacity(param_sizes.len());
    for &n in param_sizes {
        let calc_blocks = ((n as u32) + threads_per_block - 1) / threads_per_block;
        let grid_dim = calc_blocks.min(sm_blocks).max(1);
        cfgs.push(LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        });
    }

    let last = param_sizes.len();
    let mut layer_lrs = Vec::with_capacity(last);
    for (i, &n) in param_sizes.iter().enumerate() {
        let mut lr = if n > 50_000 {
            0.00145
        } else if n > 10_000 {
            0.00165
        } else {
            0.00195
        };
        if n <= 512 {
            lr = 0.0030;
        }
        if i == 0 && n > 10_000 {
            lr *= 0.93;
        }
        if i + 2 >= last {
            lr = 0.00105;
        }
        layer_lrs.push(lr);
    }

    let state = OptimizerState {
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

        spectral_boost: 1.12,
        step_count: 0,
        warmup_steps: WARMUP_STEPS,
        total_steps: TOTAL_STEPS,
        noise_variance: 0.035,
        val_loss_history: Vec::new(),

        beta1: 0.90,
        beta2: 0.995,
        eps: 1.0e-8,
        weight_decay: 0.0018,
        bn_layer_boost: 1.50,
        output_layer_damping: 0.86,

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
        spectral_decay: 0.3,
    };

    Ok(Box::new(state) as Box<dyn OptimizerStateTrait>)
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

    let prev_loss = s.prev_val_loss;
    update_state_from_val_loss(s, epoch, val_loss);

    let mut global_damp = compute_global_damp(s, val_loss);
    let (in_zone, precision_gain, mut gate_lo, mut gate_hi, forward_gain) =
        compute_precision_params(s, val_loss);
    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) =
        compute_blends(s, val_loss);

    if s.step_count <= s.warmup_steps {
        s.lr_boost = 1.0;
        s.plateau_count = 0;
    }
    s.lr_boost = s.lr_boost.clamp(0.70, 1.55);

    if s.step_count < s.warmup_steps + 32 {
        gate_lo *= 0.88;
        gate_hi *= 1.10;
    }
    if in_zone {
        gate_lo = gate_lo.max(0.72);
        gate_hi = gate_hi.min(1.38);
        global_damp *= 0.82;
        s.lr_boost = s.lr_boost.min(1.18);
    }

    let trust_backoff = if let (Some(p), Some(c)) = (prev_loss, val_loss) {
        let rel_up = (c - p) / (p.abs() + 1e-6);
        if rel_up > 0.012 {
            0.78
        } else if rel_up > 0.006 {
            0.88
        } else {
            1.0
        }
    } else {
        1.0
    };
    global_damp *= trust_backoff;

    let near_floor = val_loss.map_or(false, |l| l <= s.noise_variance * 3.6);
    let late_phase = s.step_count > (s.total_steps * 19 / 20);
    let use_robust = s.step_count > s.warmup_steps + 120
        && (near_floor || late_phase || (in_zone && s.plateau_count >= 12));
    
    let t = (s.step_count + 1) as i32;
    let bias_correction1 = 1.0 - s.beta1.powi(t);
    let bias_correction2 = 1.0 - s.beta2.powi(t);

    let k_fast = module.load_function("dual_consensus_fisher_kernel")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel")?;

    let mut lr_factor = spectral_phase_lr(s, 1.0) * global_damp * s.lr_boost
        * intra_epoch_factor(s);
    if s.step_count < s.warmup_steps + 56 {
        lr_factor *= 1.05;
    }
    if in_zone {
        lr_factor *= 0.80;
    }
    if use_robust {
        lr_factor *= 0.75;
    }
    lr_factor = lr_factor.clamp(0.0, 3.0);

    let eps_eff = if in_zone { s.eps * 0.85 } else { s.eps };

    let last = gradients.len();
    let mut updates = Vec::with_capacity(last);

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 {
            updates.push(stream.alloc_zeros::<f32>(0)?);
            continue;
        }

        let p = &model_params[i];
        let cfg = s.cfgs[i];

        let mut layer_mul = 1.0f32;

        if n <= 512 {
            let ramp = (s.step_count as f32 / 180.0).clamp(0.0, 1.0);
            layer_mul *= 1.0 + (s.bn_layer_boost - 1.0) * ramp;
        }
        if i + 2 >= last {
            layer_mul *= s.output_layer_damping;
            if in_zone {
                layer_mul *= 0.94;
            }
        }

        let mut effective_lr =
            s.layer_lrs[i] * lr_factor * layer_mul * precision_gain * forward_gain;
        if use_robust {
            effective_lr *= 0.72;
        }
        effective_lr = effective_lr.clamp(0.0, 0.0130);

        let mut wd_eff = s.weight_decay;
        if n <= 512 {
            wd_eff *= 0.15;
        } else if i + 2 >= last {
            wd_eff *= 0.55;
        }
        if near_floor {
            wd_eff *= 1.05;
        }

        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream
                    .launch_builder(&k_robust)
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
                stream
                    .launch_builder(&k_fast)
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
                    .arg(&s.beta1)
                    .arg(&s.beta2)
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