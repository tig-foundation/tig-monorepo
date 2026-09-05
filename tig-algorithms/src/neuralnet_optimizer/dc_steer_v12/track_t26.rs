use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

use super::helpers::{
    FusedMeta, OptimizerState,
    spectral_phase_lr, compute_blends, update_state_from_val_loss,
    compute_global_damp, compute_precision_params, finalize_state,
    intra_epoch_factor, loss_curvature_factor,
    dc,
};

thread_local! {
    static TRACK_CONFIG: std::cell::RefCell<TrackConfig> = std::cell::RefCell::new(TrackConfig::default());
    /// DC steering state (see `helpers::dc`). Inert unless `dc_enable != 0`.
    static DC: std::cell::RefCell<dc::DcState> = std::cell::RefCell::new(dc::DcState::default());
}

/// DC steering is ON by default on this track (see the port measurement in
/// RESULTS.md); `{"dc_enable":0}` restores the pre-port champion bit-for-bit.
const DC_ENABLE_DEFAULT_T26: u64 = 1;
const DC_AXPY_T26: &str = "dc_axpy_t26";

struct TrackConfig {
    total_steps: usize, anneal_steps: usize, warmup_steps: usize, noise_variance: f32, spectral_boost: f32,
    beta1: f32, beta2: f32, eps: f32, weight_decay: f32, bn_layer_boost: f32,
    output_layer_damping: f32, threads_per_block: u32, blocks_per_sm: u32,
    init_scale: f32,
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            total_steps: 1000, anneal_steps: 0, warmup_steps: 40, noise_variance: 0.036, spectral_boost: 1.18,
            beta1: 0.92, beta2: 0.997, eps: 1e-8, weight_decay: 0.0020,
            bn_layer_boost: 1.48, output_layer_damping: 0.77, threads_per_block: 128, blocks_per_sm: 6,
            init_scale: 1.0,
        }
    }
}

fn parse_config(hyperparameters: &Option<Map<String, Value>>) -> TrackConfig {
    let hp = hyperparameters.as_ref();
    TrackConfig {
        total_steps: hp.and_then(|h| h.get("total_steps").and_then(|v| v.as_u64())).unwrap_or(1000) as usize,
        anneal_steps: hp.and_then(|h| h.get("anneal_steps").and_then(|v| v.as_u64())).unwrap_or(0) as usize,
        warmup_steps: hp.and_then(|h| h.get("warmup_steps").and_then(|v| v.as_u64())).unwrap_or(40) as usize,
        noise_variance: hp.and_then(|h| h.get("noise_variance").and_then(|v| v.as_f64())).unwrap_or(0.036) as f32,
        spectral_boost: hp.and_then(|h| h.get("spectral_boost").and_then(|v| v.as_f64())).unwrap_or(1.18) as f32,
        beta1: hp.and_then(|h| h.get("beta1").and_then(|v| v.as_f64())).unwrap_or(0.92) as f32,
        beta2: hp.and_then(|h| h.get("beta2").and_then(|v| v.as_f64())).unwrap_or(0.997) as f32,
        eps: hp.and_then(|h| h.get("eps").and_then(|v| v.as_f64())).unwrap_or(1e-8) as f32,
        weight_decay: hp.and_then(|h| h.get("weight_decay").and_then(|v| v.as_f64())).unwrap_or(0.0020) as f32,
        bn_layer_boost: hp.and_then(|h| h.get("bn_layer_boost").and_then(|v| v.as_f64())).unwrap_or(1.48) as f32,
        output_layer_damping: hp.and_then(|h| h.get("output_layer_damping").and_then(|v| v.as_f64())).unwrap_or(0.77) as f32,
        threads_per_block: hp.and_then(|h| h.get("threads_per_block").and_then(|v| v.as_u64())).unwrap_or(128) as u32,
        blocks_per_sm: hp.and_then(|h| h.get("blocks_per_sm").and_then(|v| v.as_u64())).unwrap_or(6) as u32,
        init_scale: hp.and_then(|h| h.get("init_scale").and_then(|v| v.as_f64())).unwrap_or(1.0) as f32,
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

    // ── DC steering setup (no-op unless dc_enable != 0) ──────────────────────
    let dcfg = dc::config_from_hp(hyperparameters, DC_ENABLE_DEFAULT_T26);
    DC.with(|c| *c.borrow_mut() = dc::DcState::default());
    if dcfg.enable != 0 {
        let m = dc::target_mean(challenge, &stream)?;
        let nb = dc::batches_per_epoch(challenge);
        DC.with(|c| c.borrow_mut().arm(&dcfg, m, nb));
    }

    training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}

/// Same harness-layout fact as track_t29: `apply_optimizer_updates` (nn/mlp.rs:331) dereferences
/// `updates[i]` only for trainable linear w/b and trainable BN w/b, skipping frozen layers and
/// every BN running_mean/running_var. Those entries must exist for index alignment but are never
/// read, so computing them is pure waste.
fn dead_update_mask_t26(count: usize) -> Vec<bool> {
    const FROZEN: usize = 2;
    let hidden_layers = count.saturating_sub(2) / 6;
    let linear_layers = hidden_layers + 1;
    let mut dead = vec![false; count];
    let mut idx = 0usize;
    for l in 0..linear_layers {
        let trainable = l + FROZEN < linear_layers;
        dead[idx] = !trainable; dead[idx + 1] = !trainable; idx += 2;
    }
    for b in 0..hidden_layers {
        let trainable = b + FROZEN < linear_layers;
        dead[idx] = !trainable; dead[idx + 1] = !trainable;
        dead[idx + 2] = true; dead[idx + 3] = true; idx += 4;
    }
    dead
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
        let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(cfg.blocks_per_sm).max(1);

        let mut m = Vec::new(); let mut v = Vec::new(); let mut prev_g = Vec::new(); let mut prev_u = Vec::new();
        let mut slow_u = Vec::new(); let mut f = Vec::new(); let mut ef = Vec::new();

        for &n in param_sizes {
            m.push(stream.alloc_zeros::<f32>(n)?); v.push(stream.alloc_zeros::<f32>(n)?);
            prev_g.push(stream.alloc_zeros::<f32>(n)?); prev_u.push(stream.alloc_zeros::<f32>(n)?);
            slow_u.push(stream.alloc_zeros::<f32>(n)?);
            let mut fisher_init = stream.alloc_zeros::<f32>(n)?;
            stream.memcpy_htod(&vec![1e-4f32; n], &mut fisher_init)?;
            f.push(fisher_init); ef.push(stream.alloc_zeros::<f32>(n)?);
        }

        let mut cfgs = Vec::with_capacity(param_sizes.len());
        for &n in param_sizes {
            let calc_blocks = (n as u32 + cfg.threads_per_block - 1) / cfg.threads_per_block;
            cfgs.push(LaunchConfig { grid_dim: (calc_blocks.min(sm_blocks).max(1), 1, 1), block_dim: (cfg.threads_per_block, 1, 1), shared_mem_bytes: 0 });
        }

        let num_layers = param_sizes.len();
        let mut layer_lrs = Vec::with_capacity(num_layers);
        for (i, &ps) in param_sizes.iter().enumerate() {
            let depth_frac = i as f32 / (num_layers.max(1) as f32);
            let depth_scale = 0.55f32 + 0.45f32 * depth_frac;
            let mut lr = 0.00155f32 * depth_scale;
            if ps <= 512 { lr = 0.0022f32; }
            if ps > 50000 { lr = lr.min(0.00100f32); }
            if i == num_layers.saturating_sub(1) { lr = 0.00078f32; }
            if i == 0 { lr = 0.00088f32; }
            layer_lrs.push(lr);
        }

        // ── P6: static grouped-launch metadata ───────────────────────────────
        // Fixed for the whole run: which tensors get a launch, how many blocks each
        // gets (identical to `cfgs[i].grid_dim.0`, so each element sees exactly the
        // (tid, stride) the per-tensor launch gave it), and each length.
        let n_returned = num_layers.saturating_sub(6);
        let dead = dead_update_mask_t26(num_layers);
        let live: Vec<usize> = (0..n_returned)
            .filter(|&i| !dead[i] && param_sizes[i] > 0)
            .collect();
        let t_count = live.len();

        let mut h_meta: Vec<i32> = Vec::with_capacity(2 * t_count + 1);
        let mut acc: i32 = 0;
        h_meta.push(0);
        for &i in &live {
            acc += cfgs[i].grid_dim.0 as i32;   // == min(ceil(n/tpb), sm_blocks).max(1)
            h_meta.push(acc);
        }
        for &i in &live {
            h_meta.push(param_sizes[i] as i32);
        }

        let p6 = if t_count > 0 {
            Some(FusedMeta {
                live,
                ptr_dev: stream.alloc_zeros::<u64>(10 * t_count)?,
                scal_dev: stream.alloc_zeros::<f32>(2 * t_count)?,
                meta_dev: stream.memcpy_stod(&h_meta)?,   // uploaded ONCE
                fused_cfg: LaunchConfig {
                    grid_dim: (acc.max(1) as u32, 1, 1),
                    block_dim: (cfg.threads_per_block, 1, 1),
                    shared_mem_bytes: 0,
                },
            })
        } else {
            None
        };

        Ok(Box::new(OptimizerState {
            m, v, prev_g, prev_u, slow_u, f, ef, upd: Vec::new(), cfgs, layer_lrs,
            spectral_boost: cfg.spectral_boost, step_count: 0, warmup_steps: cfg.warmup_steps,
            total_steps: cfg.total_steps, anneal_steps: cfg.anneal_steps, noise_variance: cfg.noise_variance,
            val_loss_history: Vec::new(), beta1: cfg.beta1, beta2: cfg.beta2, eps: cfg.eps,
            weight_decay: cfg.weight_decay, bn_layer_boost: cfg.bn_layer_boost,
            output_layer_damping: cfg.output_layer_damping,
            prev_val_loss: None, best_val_loss: None, plateau_count: 0, slope_ema: 0.0,
            lr_boost: 1.0, last_pulse_step: 0, last_epoch: 0, steps_in_epoch: 0,
            bpe_ema: 1.0, phase_tempo: 1.0, spectral_decay: 0.4,
            nv_hi_hi_mult: 8.6, nv_hi_lo_mult: 6.2,
            p6,
        }) as Box<dyn OptimizerStateTrait>)
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

    // P5 + P6: exactly one `load_function` and exactly one launch per step.
    // `use_robust` depends only on `s.step_count` / loss state (see above), so it is
    // uniform across tensors and is dispatched here on the host.
    let (k_fast, k_robust) = if use_robust {
        (None, Some(module.load_function("sign_ef_consensus_fused_t26")?))
    } else {
        (Some(module.load_function("dual_consensus_fisher_fused_t26")?), None)
    };

    // Prepare init_scale additive at step 0 (hidden weights only).
    // NOTE: `update_state_from_val_loss` above already did `step_count += 1`, so on the
    // first call `step_count == 1`; this branch is structurally unreachable and the
    // kernel is never launched (fire rate 0/N steps). Kept verbatim for parity.
    let step0_scale: Option<(_, f32)> = if s.step_count == 0 {
        let init_scale = TRACK_CONFIG.with(|c| c.borrow().init_scale);
        let sm1 = init_scale - 1.0f32;
        if sm1.abs() > 1e-6 {
            Some((module.load_function("scale_additive_kernel_t26")?, sm1))
        } else { None }
    } else { None };

    let num_layers = gradients.len();

    let dead_t26 = dead_update_mask_t26(num_layers);
    let n_returned_t26 = num_layers.saturating_sub(6);

    let mut updates = Vec::with_capacity(n_returned_t26);

    // P6: T = number of tensors that actually get computed (36 for n_hidden=10).
    let t_count = s.p6.as_ref().map(|p| p.live.len()).unwrap_or(0);
    // Role-major, matching `all_ptrs[role * T + t]` in the kernel.
    let mut h_ptrs: Vec<u64> = vec![0u64; 10 * t_count];
    let mut h_scal: Vec<f32> = vec![0.0f32; 2 * t_count];
    let mut k = 0usize;   // fused slot; must advance in lockstep with `live`

    for (i, g) in gradients.iter().take(n_returned_t26).enumerate() {
        let n = g.len();
        if n == 0 || dead_t26[i] {
            updates.push(unsafe { stream.alloc::<f32>(0)? });
            continue;
        }

        let base_lr = s.layer_lrs[i];
        let tempo_lr = (1.0f32 / s.phase_tempo.powf(0.35f32)).max(0.6f32);
        let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr
;

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

        // P3: fresh, un-memset output buffer, moved into the returned Vec. The fused
        // kernel writes THIS buffer (we take its device pointer here and never
        // clone/realloc it), so `updates[i]` is exactly what the kernel filled. Every
        // element in 0..n is covered by the grid-stride loop, so nothing is left stale.
        let update_buf = unsafe { stream.alloc::<f32>(n)? };

        // P6: no launch here -- just record the pointers and the two per-tensor scalars.
        // Device pointers are re-read every step on purpose: `OptimizerState::box_clone`
        // deep-copies every `CudaSlice`, so a cached pointer could aim at a stale buffer.
        h_ptrs[0 * t_count + k] = { let (q, _g) = g.device_ptr(&stream); q };
        h_ptrs[1 * t_count + k] = { let (q, _g) = model_params[i].device_ptr(&stream); q };
        h_ptrs[2 * t_count + k] = { let (q, _g) = s.m[i].device_ptr(&stream); q };
        h_ptrs[3 * t_count + k] = { let (q, _g) = s.v[i].device_ptr(&stream); q };
        h_ptrs[4 * t_count + k] = { let (q, _g) = s.prev_g[i].device_ptr(&stream); q };
        h_ptrs[5 * t_count + k] = { let (q, _g) = s.prev_u[i].device_ptr(&stream); q };
        h_ptrs[6 * t_count + k] = { let (q, _g) = s.slow_u[i].device_ptr(&stream); q };
        h_ptrs[7 * t_count + k] = { let (q, _g) = s.f[i].device_ptr(&stream); q };
        h_ptrs[8 * t_count + k] = { let (q, _g) = s.ef[i].device_ptr(&stream); q };
        h_ptrs[9 * t_count + k] = { let (q, _g) = update_buf.device_ptr(&stream); q };

        h_scal[0 * t_count + k] = effective_lr;
        h_scal[1 * t_count + k] = rel_update_cap;

        k += 1;
        updates.push(update_buf);
    }

    if k != t_count {
        // The live set is derived from `dead_update_mask_t26(param_sizes.len())` at init
        // and must reproduce exactly here; a mismatch means the harness changed layout.
        return Err(anyhow!(
            "P6/t26: live-tensor count changed ({} at step, {} at init)", k, t_count
        ));
    }

    if t_count > 0 {
        let p6 = s.p6.as_mut().unwrap();
        // Small pageable H2D is staged through the driver's pinned buffer and does NOT
        // drain the pipeline (unlike the fuel check's blocking D2H).
        stream.memcpy_htod(&h_ptrs, &mut p6.ptr_dev)?;
        stream.memcpy_htod(&h_scal, &mut p6.scal_dev)?;

        let num_tensors = t_count as u32;
        let fcfg = p6.fused_cfg;

        unsafe {
            if use_robust {
                stream
                    .launch_builder(k_robust.as_ref().unwrap())
                    .arg(&p6.ptr_dev)
                    .arg(&p6.scal_dev)
                    .arg(&p6.meta_dev)
                    .arg(&num_tensors)
                    .arg(&eps_eff)
                    .arg(&wd_eff)
                    .arg(&lookahead_alpha)
                    .arg(&lookahead_tau)
                    .arg(&gate_lo)
                    .arg(&gate_hi)
                    .launch(fcfg)?;
            } else {
                stream
                    .launch_builder(k_fast.as_ref().unwrap())
                    .arg(&p6.ptr_dev)
                    .arg(&p6.scal_dev)
                    .arg(&p6.meta_dev)
                    .arg(&num_tensors)
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
                    .launch(fcfg)?;
            }
        }
    }

    // At step 0: add (init_scale-1)*W_0 to the update (hidden weights only). Runs AFTER
    // the main kernel, exactly as before. Unreachable in practice (see `step0_scale`).
    if let Some((ref k_scale_add, sm1)) = step0_scale {
        for (i, g) in gradients.iter().take(n_returned_t26).enumerate() {
            let n = g.len();
            if n == 0 || dead_t26[i] { continue; }
            let is_bn = n <= 512;
            let is_output = i == num_layers.saturating_sub(1);
            if !is_bn && !is_output {
                let blk = 128u32;
                let grid = ((n as u32 + blk - 1) / blk).max(1);
                let cfg_k = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (blk, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    stream.launch_builder(k_scale_add)
                        .arg(&model_params[i])
                        .arg(&mut updates[i])
                        .arg(&(n as u32))
                        .arg(&sm1)
                        .launch(cfg_k)?;
                }
            }
        }
    }

    // ── DC pulse / restore on the last trainable BN bias ─────────────────────
    // Runs after the fused kernel has written that update; same stream, so the
    // add is ordered behind it. Two extra launches per epoch at most, and none
    // at all while `dc_enable == 0`.
    {
        let sie = s.steps_in_epoch;
        let res: Result<()> = DC.with(|cell| {
            let mut d = cell.borrow_mut();
            if !d.enable {
                return Ok(());
            }
            dc::step(
                &mut *d, sie, epoch, val_loss, model_params, &mut updates,
                &stream, &module, &mut s.upd, DC_AXPY_T26,
            )
        });
        res?;
    }

    finalize_state(s, val_loss);
    Ok(updates)
}