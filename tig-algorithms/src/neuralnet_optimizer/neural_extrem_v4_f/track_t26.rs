// T26 (n_hidden=10) — V4_F : SOTA adaptatif + innovations greffées
//
// Base : code SOTA neural_advanced (adaptive beta1 / noise_variance / spectral_boost / blends)
// Innovation A : Fisher-ASAM — dénominateur ASAM basé sur F_diag (immunité beta1 adaptatif)
// Innovation B : Trust-Gated WD — WD post-gate modulé par trust (dans kernel)
// Innovation C : Jacobian Trust Damping — s_l depth-adaptive passé par couche au kernel
// Innovation 2 : Pont de Fisher (dans kernel)
// Innovation 4 : SWA Intra-Epoch — lookahead_alpha boost aux bornes d'epoch
// Innovation 5 : WD Post-Gate (dans kernel sign)
use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

// ── State (calqué sur SOTA DualPhaseConsensusState + champs pour innovations) ──

#[derive(Clone)]
struct T26State {
    m:        Vec<CudaSlice<f32>>,
    v:        Vec<CudaSlice<f32>>,
    prev_g:   Vec<CudaSlice<f32>>,
    prev_u:   Vec<CudaSlice<f32>>,
    slow_u:   Vec<CudaSlice<f32>>,
    f:        Vec<CudaSlice<f32>>,  // Fisher diag (aussi utilisé pour Innovation A ASAM)
    ef:       Vec<CudaSlice<f32>>,
    upd:      Vec<CudaSlice<f32>>,
    cfgs:     Vec<LaunchConfig>,
    layer_lrs: Vec<f32>,
    layer_s:   Vec<f32>,  // Innovation C : s_l par couche [0.2, 0.8]
    num_layers: usize,

    spectral_boost: f32,
    step_count:  usize,
    warmup_steps: usize,
    total_steps:  usize,

    noise_variance:  f32,
    val_loss_history: Vec<f32>,

    beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    bn_layer_boost: f32, output_layer_damping: f32,

    prev_val_loss: Option<f32>,
    best_val_loss: Option<f32>,
    plateau_count: usize,
    slope_ema:     f32,
    lr_boost:      f32,
    last_pulse_step: usize,
    last_epoch:    usize,
    steps_in_epoch: usize,
    bpe_ema:       f32,
    phase_tempo:   f32,

    // HP pour ASAM (Innovation A)
    sam_rho: f32,
}

impl OptimizerStateTrait for T26State {
    fn as_any(&self)     -> &dyn std::any::Any     { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self)  -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

// ── HP ──

#[derive(Clone)]
struct TrackHP {
    total_steps: usize, warmup_steps: usize,
    beta1: f32, beta2: f32, weight_decay: f32,
    spectral_boost: f32, bn_layer_boost: f32,
    noise_variance: f32,
    threads_per_block: u32, blocks_per_sm: u32,
    sam_rho: f32,
}
impl Default for TrackHP {
    fn default() -> Self {
        TrackHP {
            total_steps: 2000, warmup_steps: 40,
            beta1: 0.92, beta2: 0.997, weight_decay: 0.0025,
            spectral_boost: 1.1, bn_layer_boost: 1.35,  // SOTA default
            noise_variance: 0.040,
            threads_per_block: 256, blocks_per_sm: 8,
            sam_rho: 0.20,  // ASAM max rho — Fisher-ASAM permet ce niveau sans divergence
        }
    }
}

thread_local! {
    static TRACK_HP: std::cell::RefCell<TrackHP> =
        std::cell::RefCell::new(TrackHP::default());
}

fn parse_f32(hp: Option<&Map<String, Value>>, key: &str, d: f32) -> f32 {
    hp.and_then(|h| h.get(key).and_then(|v| v.as_f64())).map(|v| v as f32).unwrap_or(d)
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
        total_steps:  hp.and_then(|h| h.get("total_steps").and_then(|v| v.as_u64())).unwrap_or(d.total_steps as u64) as usize,
        warmup_steps: hp.and_then(|h| h.get("warmup_steps").and_then(|v| v.as_u64())).unwrap_or(d.warmup_steps as u64) as usize,
        threads_per_block: hp.and_then(|h| h.get("threads_per_block").and_then(|v| v.as_u64())).unwrap_or(d.threads_per_block as u64) as u32,
        blocks_per_sm: hp.and_then(|h| h.get("blocks_per_sm").and_then(|v| v.as_u64())).unwrap_or(d.blocks_per_sm as u64) as u32,
        beta1:          parse_f32(hp, "beta1",          d.beta1),
        beta2:          parse_f32(hp, "beta2",          d.beta2),
        weight_decay:   parse_f32(hp, "weight_decay",   d.weight_decay),
        spectral_boost: parse_f32(hp, "spectral_boost", d.spectral_boost),
        bn_layer_boost: parse_f32(hp, "bn_layer_boost", d.bn_layer_boost),
        noise_variance: parse_f32(hp, "noise_variance", d.noise_variance),
        sam_rho:        parse_f32(hp, "sam_rho",        d.sam_rho),
    };
    TRACK_HP.with(|c| *c.borrow_mut() = cfg);
    training_loop(challenge, save_solution, module, stream, prop,
        optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}

// ── LR schedule identique SOTA ──
#[inline]
fn spectral_phase_lr(s: &T26State, base_lr: f32) -> f32 {
    let t    = s.step_count as f32;
    let warm = s.warmup_steps as f32;
    let total = s.total_steps as f32;
    if t <= warm {
        return base_lr * (t / warm.max(1.0)) * s.spectral_boost;
    }
    let progress = ((t - warm) / (total - warm).max(1.0)).min(1.0);
    let cosine   = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    base_lr * cosine * s.spectral_boost * (1.0 - 0.3 * progress)
}

// ── compute_blends identique SOTA (trend-adaptive) ──
#[inline]
fn compute_blends(s: &T26State, val_loss: Option<f32>)
    -> (f32, f32, f32, f32, f32, f32, f32)
{
    let t = s.step_count as f32;
    let warm  = s.warmup_steps as f32;
    let total = s.total_steps as f32;
    let progress = (t / total.max(1.0)).min(1.0);

    let (mut ba, mut bn, mut bs, gamma, bb_blend, mut la_alpha, mut la_tau): (f32,f32,f32,f32,f32,f32,f32) =
        if t <= warm {
            (0.35, 0.65, 0.0, 0.22, 0.6, 0.0, 0.2)
        } else {
            let trend = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
                prev - curr
            } else { 0.0 };
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
                bs = (bs + 0.2).min(0.6);
                la_alpha = la_alpha.max(0.45);
                la_tau   = (la_tau + 0.05).min(0.35);
                ba *= 0.9;
                bn *= 0.9;
            } else if curr >= s.noise_variance * 6.2 && curr <= s.noise_variance * 8.6 {
                bs = bs.max(0.35);
                ba = (ba * 0.95).max(0.25);
                bn = (bn * 0.95).max(0.15);
                la_alpha = (la_alpha * 0.85).min(0.35);
                la_tau   = (la_tau   * 0.85).min(0.30);
            }
        }
        if progress > 0.8 {
            bn = bn.max(0.35);
            bs *= 0.9;
            la_alpha = la_alpha.max(0.5);
            la_tau   = (la_tau + 0.05).min(0.4);
        }
    }

    let sum = (ba + bn + bs).max(1e-8);
    (ba/sum, bn/sum, bs/sum, gamma, bb_blend, la_alpha, la_tau)
}

// ── Init ──
fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let hp_cfg = TRACK_HP.with(|c| c.borrow().clone());
    let tpb = hp_cfg.threads_per_block;
    let sm_blocks = (prop.multiProcessorCount as u32)
        .saturating_mul(hp_cfg.blocks_per_sm).max(1);

    let mut m      = Vec::new();
    let mut v      = Vec::new();
    let mut prev_g = Vec::new();
    let mut prev_u = Vec::new();
    let mut slow_u = Vec::new();
    let mut f      = Vec::new();
    let mut ef     = Vec::new();
    let mut upd    = Vec::new();

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
        let calc = (n as u32 + tpb - 1) / tpb;
        cfgs.push(LaunchConfig {
            grid_dim:  (calc.min(sm_blocks).max(1), 1, 1),
            block_dim: (tpb, 1, 1),
            shared_mem_bytes: 0,
        });
    }

    // Layer LRs — identiques SOTA
    let num_layers = param_sizes.len();
    let mut layer_lrs = Vec::with_capacity(num_layers);
    for (i, &n) in param_sizes.iter().enumerate() {
        let mut lr = 0.0012f32;
        if i == 0               { lr = 0.0011; }
        if n <= 512              { lr = 0.0018; }
        if n > 50_000            { lr = 0.0009; }
        if i == num_layers - 1  { lr = 0.0007; }
        layer_lrs.push(lr);
    }

    // Innovation C : coefficients Jacobian Trust Damping par couche
    // s_l = 0.8 - 0.6 * (l/(L-1))  → entrée forte, sortie douce
    // BN layers (n<=512) → s = 0.5 (SOTA default)
    let mut layer_s = Vec::with_capacity(num_layers);
    for (i, &n) in param_sizes.iter().enumerate() {
        let s = if n <= 512 || num_layers <= 1 {
            0.5f32  // BN/bias : comportement SOTA neutre
        } else {
            0.8f32 - 0.6f32 * (i as f32 / (num_layers - 1) as f32)
        };
        layer_s.push(s);
    }

    Ok(Box::new(T26State {
        m, v, prev_g, prev_u, slow_u, f, ef, upd,
        cfgs, layer_lrs, layer_s, num_layers,
        spectral_boost: hp_cfg.spectral_boost,
        step_count: 0, warmup_steps: hp_cfg.warmup_steps, total_steps: hp_cfg.total_steps,
        noise_variance: hp_cfg.noise_variance,
        val_loss_history: Vec::new(),
        beta1: hp_cfg.beta1, beta2: hp_cfg.beta2, eps: 1e-8,
        weight_decay: hp_cfg.weight_decay,
        bn_layer_boost: hp_cfg.bn_layer_boost, output_layer_damping: 0.8,
        prev_val_loss: None, best_val_loss: None,
        plateau_count: 0, slope_ema: 0.0, lr_boost: 1.0,
        last_pulse_step: 0, last_epoch: 0, steps_in_epoch: 0,
        bpe_ema: 1.0, phase_tempo: 1.0,
        sam_rho: hp_cfg.sam_rho,
    }) as Box<dyn OptimizerStateTrait>)
}

// ── Query : Innovation A — Fisher-ASAM Topologique ──
fn optimizer_query(
    state: &dyn OptimizerStateTrait,
    params: &[CudaSlice<f32>],
    _epoch: usize,
    _train: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    let s = state.as_any().downcast_ref::<T26State>().unwrap();
    if s.step_count < s.warmup_steps + 15 { return Ok(None); }

    // Innovation A : perturbation active près de la convergence
    let mut rho_max = 0.0f32;
    if let Some(vl) = val_loss {
        let threshold = s.noise_variance * 6.0;
        if vl < threshold {
            let proximity = (vl / threshold).clamp(0.2, 1.0);
            rho_max = s.sam_rho * proximity;
        }
    }
    if rho_max <= 1e-4 { return Ok(None); }

    let k_asam = module.load_function("asam_topo_kernel_26")?;
    let num_layers = params.len();
    let t   = (s.step_count + 1) as i32;
    let bc1 = 1.0f32 - s.beta1.powi(t);

    let mut modified = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let n = params[i].len();
        if n == 0 { modified.push(unsafe { stream.alloc::<f32>(0)? }); continue; }

        // Profondeur quadratique — BN layers pas perturbés
        let depth = if n <= 512 || num_layers <= 1 {
            0.0f32
        } else {
            (i as f32 / (num_layers - 1) as f32).powi(2)
        };
        let layer_rho = rho_max * depth;

        let mut p_out = unsafe { stream.alloc::<f32>(n)? };
        unsafe {
            stream.launch_builder(&k_asam)
                .arg(&params[i])
                .arg(&s.m[i])
                .arg(&s.f[i])   // ← F_diag au lieu de v (Innovation A)
                .arg(&mut p_out)
                .arg(&(n as u32))
                .arg(&layer_rho)
                .arg(&s.eps)
                .arg(&bc1)      // seulement bc1 (F_diag pas bias-corrigé)
                .launch(s.cfgs[i])?;
        }
        modified.push(p_out);
    }
    Ok(Some(modified))
}

// ── Step : SOTA adaptatif + Innovation 4 (SWA) ──
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
    let s = state.as_any_mut().downcast_mut::<T26State>().unwrap();

    // ── Step counter + bpe_ema (identique SOTA) ──
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

    // ── Adaptive noise_variance (SOTA) ──
    if let Some(loss) = val_loss {
        if s.step_count > s.warmup_steps {
            s.val_loss_history.push(loss);
            if s.val_loss_history.len() > 12 { s.val_loss_history.remove(0); }
            if s.val_loss_history.len() >= 6 {
                let min_loss = s.val_loss_history.iter().copied().fold(f32::INFINITY, f32::min);
                let recent_avg = {
                    let n = s.val_loss_history.len().min(10);
                    s.val_loss_history.iter().rev().take(n).sum::<f32>() / n as f32
                };
                let target_nv = (min_loss / 5.0).min(recent_avg / 8.0);
                s.noise_variance = (0.85 * s.noise_variance + 0.15 * target_nv).clamp(0.0, 0.05);
            }
        }
    }

    // ── Adaptive spectral_boost (SOTA) ──
    if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        if s.step_count > s.warmup_steps && s.step_count > 20 {
            let improvement = prev - curr;
            let rel = improvement / prev.abs().max(1e-8);
            if rel > 0.008 {
                s.spectral_boost = (s.spectral_boost * 1.015).min(1.5);
            } else if rel < -0.003 {
                s.spectral_boost *= 0.97;
            } else if improvement.abs() < 5e-4 && s.plateau_count > 15 {
                s.spectral_boost = (s.spectral_boost * 1.008).min(1.5);
            }
            s.spectral_boost = s.spectral_boost.clamp(0.85, 1.5);
        }
    }

    // ── Adaptive beta1 (SOTA) ──
    if s.step_count > s.warmup_steps && s.val_loss_history.len() >= 8 {
        let recent_avg = {
            let n = s.val_loss_history.len().min(5);
            s.val_loss_history.iter().rev().take(n).sum::<f32>() / n as f32
        };
        let older_avg = {
            let items: Vec<f32> = s.val_loss_history.iter().rev().skip(5).take(5).copied().collect();
            if items.is_empty() { recent_avg } else { items.iter().sum::<f32>() / items.len() as f32 }
        };
        let trend = older_avg - recent_avg;
        let target_beta1 = if trend > 0.02 { 0.94 } else if trend < -0.02 { 0.88 } else { 0.91 };
        s.beta1 = (0.85 * s.beta1 + 0.15 * target_beta1).clamp(0.87, 0.94);
    }

    // ── global_damp : 2 composantes SOTA ──
    let mut global_damp = 1.0f32;
    if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let improvement = prev - curr;
        s.slope_ema = 0.85 * s.slope_ema + 0.15 * improvement;

        if s.step_count > s.warmup_steps {
            let is_stagnant  = improvement <= 1e-4 && s.slope_ema < 2e-4;
            let is_declining = improvement < 0.0 && s.slope_ema < 0.0;
            if is_stagnant || is_declining { s.plateau_count += 1; }
            else if improvement > 5e-5     { s.plateau_count = 0; }
            else if s.plateau_count > 0    { s.plateau_count = s.plateau_count.saturating_sub(1); }

            // Pulses lr_boost (SOTA)
            if s.plateau_count >= 25 {
                if curr > s.noise_variance * 4.0 {
                    s.lr_boost = (s.lr_boost * 1.12).min(1.60);
                    s.last_pulse_step = s.step_count;
                    s.plateau_count = 0;
                }
            } else if s.plateau_count >= 15 && curr > s.noise_variance * 8.0 {
                s.lr_boost = (s.lr_boost * 1.15).min(1.70);
                s.last_pulse_step = s.step_count;
                s.plateau_count = 0;
            } else if s.plateau_count >= 18 && curr > s.noise_variance * 5.0 {
                s.lr_boost = (s.lr_boost * 1.10).min(1.45);
                s.last_pulse_step = s.step_count;
                s.plateau_count = 0;
            } else if s.lr_boost > 1.0 {
                let rel = improvement / curr.abs().max(1e-8);
                let decay = if rel > 0.01 { 0.75 } else if rel > 0.001 { 0.85 }
                            else if improvement > 0.0 { 0.93 } else { 0.97 };
                s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay;
                let decay2 = if improvement > 5e-5 { 0.82 } else { 0.92 };
                s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay2;
                if s.step_count.saturating_sub(s.last_pulse_step) > 80 { s.lr_boost *= 0.96; }
                if s.lr_boost < 1.02 { s.lr_boost = 1.0; }
            }
        }
    }

    if let Some(loss) = val_loss {
        // Composante 1 : dynamic threshold (SOTA)
        let dynamic_threshold = s.noise_variance * (1.1 + 0.1 * (s.step_count as f32 / s.total_steps as f32));
        if loss <= dynamic_threshold && s.step_count > s.warmup_steps {
            let proximity = (loss / dynamic_threshold).clamp(0.4, 1.0);
            let plateau_factor = if s.plateau_count > 10 { 1.2f32 } else { 1.0 };
            global_damp *= (0.25 + 0.35 * proximity) * plateau_factor.min(0.9);
        }
        // Composante 2 : plancher anti-bruit (SOTA)
        if loss <= s.noise_variance * 5.0 {
            let np = (loss / (s.noise_variance * 5.0)).min(1.0);
            let steepness = 1.0 + 0.5 * (1.0 - np);
            global_damp *= 0.70 + 0.30 * np.powf(steepness);
        }
    }

    // ── trust_backoff identique SOTA ──
    let trust_backoff: f32 = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
        let delta = curr - prev;
        if delta > 2e-4 { 1.0 / (1.0 + 1.5 * (delta / (prev.abs() + 1e-8)).min(0.02)) }
        else { 1.0 }
    } else { 1.0 };

    // ── use_robust SOTA (75%) ──
    let near_floor = val_loss.map_or(false, |l| l <= s.noise_variance * 3.0);
    let late_phase = s.step_count > s.total_steps * 3 / 4;
    let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);

    // ── Bias corrections ──
    let t = s.step_count as i32;
    let bc1 = 1.0f32 - s.beta1.powi(t);
    let bc2 = 1.0f32 - s.beta2.powi(t);

    // ── Blends SOTA ──
    let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, mut lookahead_alpha, lookahead_tau) =
        compute_blends(s, val_loss);

    // ── Innovation 4 : SWA Intra-Epoch ──
    // Aux bornes d'epoch, booste lookahead_alpha → SWA effect
    // Fournit un signal val_loss plus propre aux contrôleurs adaptatifs du SOTA
    if val_loss.is_some() && s.step_count > s.warmup_steps {
        lookahead_alpha = (lookahead_alpha * 2.0).min(0.85);
    }

    // ── Precision zone SOTA ──
    let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain): (bool, f32, f32, f32, f32) =
        if let Some(loss) = val_loss {
            if s.step_count > s.warmup_steps {
                let z_lo = s.noise_variance * 6.2;
                let z_hi = s.noise_variance * 8.6;
                if loss >= z_lo && loss <= z_hi {
                    let pos = ((z_hi - loss) / (z_hi - z_lo + 1e-8)).clamp(0.0, 1.0);
                    let pg = 1.02 + 0.06 * pos;
                    let g_lo = 0.70 + 0.02 * pos;
                    let g_hi = 1.50 + 0.05 * pos;
                    let fg = if let Some(prev) = s.prev_val_loss {
                        let rel = ((prev - loss).max(0.0)) / (prev.abs() + 1e-6);
                        1.0 + (0.75 * rel).min(0.015)
                    } else { 1.0 };
                    (true, pg, g_lo, g_hi, fg)
                } else { (false, 1.0, 0.66, 1.50, 1.0) }
            } else { (false, 1.0, 0.66, 1.50, 1.0) }
        } else { (false, 1.0, 0.66, 1.50, 1.0) };

    let beta1_eff = if in_precision_zone { (s.beta1 + 0.02).min(0.995) } else { s.beta1 };
    let beta2_eff = s.beta2;
    let eps_eff   = if in_precision_zone { s.eps * 0.9 } else { s.eps };
    let mut wd_eff = if in_precision_zone { s.weight_decay * 1.05 } else { s.weight_decay };
    if s.step_count > s.warmup_steps {
        if near_floor          { wd_eff *= 1.10; }
        if s.plateau_count >= 20 { wd_eff *= 0.50; }
    }
    wd_eff *= (1.0 / s.phase_tempo).clamp(0.6, 1.0);  // tempo SOTA

    let k_fast   = module.load_function("dual_consensus_fisher_kernel_10")?;
    let k_robust = module.load_function("sign_ef_consensus_kernel_10")?;

    let tempo_lr = (1.0 / s.phase_tempo.powf(0.35f32)).max(0.6);

    let mut updates = Vec::with_capacity(gradients.len());

    for (i, g) in gradients.iter().enumerate() {
        let n = g.len();
        if n == 0 { updates.push(stream.alloc_zeros::<f32>(0)?); continue; }

        let p   = &model_params[i];
        let cfg = s.cfgs[i];

        // Layer multiplier (SOTA adaptive output damping)
        let layer_multiplier = if i == gradients.len() - 1 {
            if let Some(loss) = val_loss {
                if s.step_count > s.warmup_steps + 30 {
                    let loss_ratio = (loss / (s.noise_variance * 6.0)).min(1.0);
                    let ad = s.output_layer_damping * (0.7 + 0.3 * loss_ratio);
                    ad.max(s.output_layer_damping)
                } else { s.output_layer_damping }
            } else { s.output_layer_damping }
        } else if n <= 512 {
            s.bn_layer_boost
        } else {
            1.0
        };

        let effective_lr = spectral_phase_lr(s, s.layer_lrs[i])
            * global_damp * s.lr_boost * tempo_lr
            * layer_multiplier * precision_gain * forward_gain * trust_backoff;

        let wd_layer = if n <= 512 { wd_eff * 0.15 }
                       else if i == gradients.len() - 1 { wd_eff * 0.55 }
                       else { wd_eff };

        // Innovation C : s_l par couche pour Jacobian Trust Damping
        let s_l = s.layer_s[i];

        let update_buf_ref = &mut s.upd[i];

        unsafe {
            if use_robust {
                stream.launch_builder(&k_robust)
                    .arg(g).arg(p)
                    .arg(&mut s.f[i]).arg(&mut s.ef[i])
                    .arg(&mut s.slow_u[i]).arg(update_buf_ref)
                    .arg(&(n as u32)).arg(&effective_lr).arg(&eps_eff)
                    .arg(&wd_layer)
                    .arg(&lookahead_alpha).arg(&lookahead_tau)
                    .arg(&gate_lo).arg(&gate_hi)
                    .launch(cfg)?;
            } else {
                stream.launch_builder(&k_fast)
                    .arg(g).arg(p)
                    .arg(&mut s.m[i]).arg(&mut s.v[i])
                    .arg(&mut s.prev_g[i]).arg(&mut s.prev_u[i])
                    .arg(&mut s.slow_u[i]).arg(&mut s.f[i])
                    .arg(update_buf_ref)
                    .arg(&(n as u32)).arg(&effective_lr)
                    .arg(&beta1_eff).arg(&beta2_eff).arg(&eps_eff).arg(&wd_layer)
                    .arg(&bc1).arg(&bc2)
                    .arg(&blend_adam).arg(&blend_norm).arg(&blend_sign)
                    .arg(&nesterov_gamma).arg(&bb_blend)
                    .arg(&lookahead_alpha).arg(&lookahead_tau)
                    .arg(&gate_lo).arg(&gate_hi)
                    .arg(&s_l)   // Innovation C
                    .launch(cfg)?;
            }
        }

        updates.push(s.upd[i].clone());
    }

    // ── best_val_loss update ──
    if let Some(curr) = val_loss {
        s.best_val_loss = Some(match s.best_val_loss {
            Some(b) => if curr < b { curr } else { b },
            None => curr,
        });
    }
    s.prev_val_loss = val_loss;

    Ok(updates)
}
