
// Cautious AdanW role-scaled + cosine LR  AUGMENTÉ d'un détecteur de
// convergence grossier côté-algo qui FREEZE l'optimiseur au genou → le harness
// early-stop (patience=50) fire plus tôt → coupe les epochs de fin gaspillées.

// hp={} = reference behaviour (tous les conv_* / cap_epochs / anneal_to_cap OFF par défaut).
// Knobs HP: t_max_epochs, plateau_patience, plateau_decay (port) +
//   conv_delta, conv_rel, conv_patience, cap_epochs (i11 convergence-cap) +

//   médians au freeze; sinon frozen mid-anneal @cap400 → médiane plafonnée 696219).
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

thread_local! {
    static TRACK_CONFIG: std::cell::RefCell<TrackConfig> = std::cell::RefCell::new(TrackConfig::default());
}


const LR_MAX: f32 = 3.4e-3;
const LR_MIN: f32 = 2e-5;
const WARMUP_EPOCHS: usize = 8;
const EPS: f32 = 1e-8;
const HIDDEN_WEIGHT_DECAY: f32 = 1.6e-3;
const OUTPUT_WEIGHT_DECAY: f32 = 1e-5;
const MIN_LR_SCALE: f32 = 0.35;
const VAL_IMPROVEMENT_EPS: f32 = 1e-7;
const GROUP_HIDDEN_WEIGHT: i32 = 0;
const GROUP_HIDDEN_BIAS: i32 = 1;
const GROUP_OUTPUT_WEIGHT: i32 = 2;
const GROUP_OUTPUT_BIAS: i32 = 3;
const GROUP_BN_WEIGHT: i32 = 4;
const GROUP_BN_BIAS: i32 = 5;
const GROUP_RUNNING_STAT: i32 = 6;

#[derive(Clone, Copy)]
struct TrackConfig {
    t_max_epochs: usize,
    plateau_patience: usize,
    plateau_decay: f32,
    
    // by default → hp={} reproduces the reference behaviour. See parse_config docs.
    conv_delta: f32,
    conv_rel: f32,
    conv_patience: usize,
    cap_epochs: usize,
    
    // cap_epochs is set, the cosine anneals to LR_MIN by `cap_epochs` instead of
    // `t_max_epochs` → mid-difficulty nonces finish annealing BEFORE the freeze
    // (they are otherwise frozen mid-anneal at ~40% LR_MAX @cap400). OFF by default
    // → hp={} byte-exact port.
    anneal_to_cap: bool,
    
    // in 2-3 layer nets, early training favours asymmetric rates (the FIRST linear
    // layer absorbs the label signal, the output layer transmits it); balanced rates
    // become optimal LATER (9326e5cf). We boost the first linear layer's LR (param
    // idx 0/1) by `lr_asym_scale` at epoch 0, linearly decaying the *excess* to 1.0
    // (symmetric) by `lr_asym_epochs`. lr_asym_scale=1.0 → multiplier ≡ 1.0 → the
    // group-static role scaling is untouched → BYTE-EXACT port (sentinel).
    lr_asym_scale: f32,
    lr_asym_epochs: usize,
}

impl Default for TrackConfig {
    fn default() -> Self {
        
        // PLATEAU_PATIENCE=12, PLATEAU_DECAY=0.82) → hp={} reproduit le port.
        // conv_* / cap_epochs OFF → aucun freeze, comportement port inchangé.
        TrackConfig {
            t_max_epochs: 700,
            plateau_patience: 12,
            plateau_decay: 0.82,
            conv_delta: 0.0,
            conv_rel: 0.0,
            conv_patience: 5,
            cap_epochs: usize::MAX,
            anneal_to_cap: false,
            // scale=1.0 → asymmetry OFF (multiplier ≡ 1.0) → byte-exact port.
            lr_asym_scale: 1.0,
            lr_asym_epochs: 120,
        }
    }
}

// i11 — DETERMINISTIC CONVERGENCE-CAP (nouvelle brique training_termination).
//

// early-stopping mechanism via optimizer_query/step ». Le harness framework
// early-stop à `patience=50` epochs SANS amélioration > `min_loss_delta=1e-7`
// (tig-challenges/neuralnet_optimizer/mod.rs L573-582). Ce seuil 1e-7 est si
// serré que le port continue à grappiller des micro-gains bien APRÈS le genou

//
// Ce cap est un détecteur de convergence GROSSIER (delta absolu OU relatif,
// découplé du 1e-7 du harness) : une fois l'amélioration val_loss stagnante
// pendant `conv_patience` epochs consécutifs (ou epoch >= cap_epochs, backstop
// déterministe), on FREEZE l'optimiseur (updates == 0). val_loss cesse alors de
// s'améliorer → le compteur `patience` du harness monte → break au genou au lieu
// de la fin naturelle. La meilleure solution est déjà sauvée (save_solution ne
// fire que sur amélioration stricte). DÉTERMINISTE : dépend de `epoch`/val_loss
// (fournis par le harness), JAMAIS de wall-clock. Trade Q↔temps calibré au
// screening (headroom +2.3% Q au-dessus du gate 696219).
fn parse_config(hyperparameters: &Option<Map<String, Value>>) -> TrackConfig {
    let hp = hyperparameters.as_ref();
    let d = TrackConfig::default();
    TrackConfig {
        t_max_epochs: hp.and_then(|h| h.get("t_max_epochs").and_then(|v| v.as_u64())).unwrap_or(700) as usize,
        plateau_patience: hp.and_then(|h| h.get("plateau_patience").and_then(|v| v.as_u64())).unwrap_or(12) as usize,
        plateau_decay: hp.and_then(|h| h.get("plateau_decay").and_then(|v| v.as_f64())).unwrap_or(0.82) as f32,
        // Freeze quand l'amélioration val_loss epoch-sur-epoch < conv_delta (absolu)
        // OU < conv_rel (relatif) pendant conv_patience epochs. 0.0 = détecteur OFF.
        conv_delta: hp.and_then(|h| h.get("conv_delta").and_then(|v| v.as_f64())).unwrap_or(0.0) as f32,
        conv_rel: hp.and_then(|h| h.get("conv_rel").and_then(|v| v.as_f64())).unwrap_or(0.0) as f32,
        conv_patience: hp.and_then(|h| h.get("conv_patience").and_then(|v| v.as_u64())).unwrap_or(5) as usize,
        // Backstop déterministe: freeze inconditionnel à cet epoch (MAX = off).
        cap_epochs: hp.and_then(|h| h.get("cap_epochs").and_then(|v| v.as_u64())).map(|v| v as usize).unwrap_or(d.cap_epochs),
        
        anneal_to_cap: hp.and_then(|h| h.get("anneal_to_cap").and_then(|v| v.as_bool())).unwrap_or(false),
        // i25 (RUNG2): early-phase asymmetric first-layer LR boost. 1.0 = OFF (byte-exact).
        lr_asym_scale: hp.and_then(|h| h.get("lr_asym_scale").and_then(|v| v.as_f64())).unwrap_or(d.lr_asym_scale as f64) as f32,
        lr_asym_epochs: hp.and_then(|h| h.get("lr_asym_epochs").and_then(|v| v.as_u64())).map(|v| v as usize).unwrap_or(d.lr_asym_epochs),
    }
}


#[derive(Clone)]
struct T30State {
    m: Vec<CudaSlice<f32>>, // first moment (EMA of g)
    v: Vec<CudaSlice<f32>>, // EMA of gradient differences (Adan)
    s: Vec<CudaSlice<f32>>, // Adan second moment n: EMA of (g + 0.92*g_diff)^2
    prev_grad: Vec<CudaSlice<f32>>,
    param_groups: Vec<i32>,
    step: usize,
    best_val_loss: f32,
    plateau_epochs: usize,
    lr_scale: f32,
    last_sched_epoch: usize,
    cfg: TrackConfig,
    // i11 convergence-cap runtime state.
    frozen: bool,
    conv_stale: usize,
    conv_prev_val: f32,
}

impl OptimizerStateTrait for T30State {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

// Cosine-annealed LR schedule with a short linear warmup (verbatim, horizon HP).
fn schedule_lr(epoch: usize, t_max_epochs: usize) -> f32 {
    if epoch < WARMUP_EPOCHS {
        LR_MAX * ((epoch as f32) + 1.0) / (WARMUP_EPOCHS as f32).max(1.0)
    } else {
        let denom = (t_max_epochs.saturating_sub(WARMUP_EPOCHS)).max(1) as f32;
        let p = (((epoch - WARMUP_EPOCHS) as f32) / denom).min(1.0);
        LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

fn infer_param_groups(param_count: usize) -> Vec<i32> {
    let hidden_layers = param_count.saturating_sub(2) / 6;
    let linear_layers = hidden_layers + 1;
    let mut groups = Vec::with_capacity(param_count);

    for layer_idx in 0..linear_layers {
        if layer_idx + 1 == linear_layers {
            groups.push(GROUP_OUTPUT_WEIGHT);
            groups.push(GROUP_OUTPUT_BIAS);
        } else {
            groups.push(GROUP_HIDDEN_WEIGHT);
            groups.push(GROUP_HIDDEN_BIAS);
        }
    }

    for _ in 0..hidden_layers {
        groups.push(GROUP_BN_WEIGHT);
        groups.push(GROUP_BN_BIAS);
        groups.push(GROUP_RUNNING_STAT);
        groups.push(GROUP_RUNNING_STAT);
    }

    groups.resize(param_count, GROUP_HIDDEN_WEIGHT);
    groups
}

fn group_hyperparams(group: i32, base_lr: f32) -> (f32, f32) {
    match group {
        GROUP_HIDDEN_WEIGHT => (base_lr, HIDDEN_WEIGHT_DECAY),
        GROUP_HIDDEN_BIAS => (base_lr * 1.25, 0.0),
        GROUP_OUTPUT_WEIGHT => (base_lr * 1.75, OUTPUT_WEIGHT_DECAY),
        GROUP_OUTPUT_BIAS => (base_lr * 2.0, 0.0),
        GROUP_BN_WEIGHT => (base_lr * 0.55, 0.0),
        GROUP_BN_BIAS => (base_lr * 0.8, 0.0),
        GROUP_RUNNING_STAT => (0.0, 0.0),
        _ => (base_lr, HIDDEN_WEIGHT_DECAY),
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
    training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}

fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let cfg = TRACK_CONFIG.with(|c| *c.borrow());
    let mut m = Vec::with_capacity(param_sizes.len());
    let mut v = Vec::with_capacity(param_sizes.len());
    let mut s = Vec::with_capacity(param_sizes.len());
    let mut prev_grad = Vec::with_capacity(param_sizes.len());
    for &sz in param_sizes {
        m.push(stream.alloc_zeros::<f32>(sz)?);
        v.push(stream.alloc_zeros::<f32>(sz)?);
        s.push(stream.alloc_zeros::<f32>(sz)?);
        prev_grad.push(stream.alloc_zeros::<f32>(sz)?);
    }
    Ok(Box::new(T30State {
        m,
        v,
        s,
        prev_grad,
        param_groups: infer_param_groups(param_sizes.len()),
        step: 0,
        best_val_loss: f32::INFINITY,
        plateau_epochs: 0,
        lr_scale: 1.0,
        last_sched_epoch: usize::MAX,
        cfg,
        frozen: false,
        conv_stale: 0,
        conv_prev_val: f32::INFINITY,
    }))
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
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    _train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<T30State>()
        .ok_or_else(|| anyhow!("Invalid optimizer state"))?;

    state.step += 1;
    if state.last_sched_epoch != epoch {
        state.last_sched_epoch = epoch;
        if let Some(v) = val_loss.filter(|v| v.is_finite()) {
            if v + VAL_IMPROVEMENT_EPS < state.best_val_loss {
                state.best_val_loss = v;
                state.plateau_epochs = 0;
                state.lr_scale = (state.lr_scale * 1.03).min(1.0);
            } else {
                state.plateau_epochs += 1;
                if state.plateau_epochs >= state.cfg.plateau_patience {
                    state.lr_scale = (state.lr_scale * state.cfg.plateau_decay).max(MIN_LR_SCALE);
                    state.plateau_epochs = 0;
                }
            }

            // i11 convergence-cap: coarse deterministic knee detector. Once the
            // epoch-over-epoch val_loss improvement stalls below conv_delta
            // (absolute) OR conv_rel (relative) for conv_patience epochs, freeze
            // the optimizer so the harness patience break fires at the knee.
            if !state.frozen {
                let prev = state.conv_prev_val;
                let abs_impr = prev - v;
                let rel_impr = if prev.is_finite() && prev.abs() > 1e-12 {
                    abs_impr / prev.abs()
                } else {
                    f32::INFINITY
                };
                let stalled = (state.cfg.conv_delta > 0.0 && abs_impr < state.cfg.conv_delta)
                    || (state.cfg.conv_rel > 0.0 && rel_impr < state.cfg.conv_rel);
                if stalled {
                    state.conv_stale += 1;
                } else {
                    state.conv_stale = 0;
                }
                state.conv_prev_val = v;
                if state.cfg.conv_patience > 0 && state.conv_stale >= state.cfg.conv_patience {
                    state.frozen = true;
                }
            }
        }
        // Deterministic backstop (epoch budget), independent of val_loss.
        if epoch >= state.cfg.cap_epochs {
            state.frozen = true;
        }
    }

    // Frozen: emit zero updates (no param change) → val_loss stays flat → the
    // harness accumulates `patience` no-improvement epochs and early-stops at the
    // knee. Best-so-far solution already captured by save_solution.
    if state.frozen {
        let mut updates = Vec::with_capacity(gradients.len());
        for g in gradients {
            updates.push(stream.alloc_zeros::<f32>(g.len())?);
        }
        return Ok(updates);
    }
    // On the very first step there is no previous gradient yet; the kernel
    // forces g_diff = 0 instead of treating prev_grad's zeros as a real value.
    let first_step: i32 = if state.step == 1 { 1 } else { 0 };
    
    // point so mid-nonces fully converge by cap_epochs (byte-exact port when OFF).
    let anneal_horizon = if state.cfg.anneal_to_cap && state.cfg.cap_epochs != usize::MAX {
        state.cfg.cap_epochs
    } else {
        state.cfg.t_max_epochs
    };
    let scheduled_lr = schedule_lr(epoch, anneal_horizon);
    let lr = LR_MIN + (scheduled_lr - LR_MIN) * state.lr_scale;
    let eps = EPS;

    // i25 (RUNG2): early-phase asymmetric first-layer LR multiplier. The excess
    // (lr_asym_scale-1) decays linearly to 0 over [0, lr_asym_epochs] → the boost
    // is present only while the first layer is absorbing the label signal, then the
    // rates return to symmetric (group-static). `epoch` is harness-provided (no
    // wall-clock). scale=1.0 → asym_mult ≡ 1.0 → byte-exact port.
    let asym_mult: f32 = if state.cfg.lr_asym_scale != 1.0 && state.cfg.lr_asym_epochs > 0 {
        let phase = (1.0 - (epoch as f32) / (state.cfg.lr_asym_epochs as f32)).max(0.0);
        1.0 + (state.cfg.lr_asym_scale - 1.0) * phase
    } else {
        1.0
    };

    let update_kernel = module.load_function("adan_cautious_update_kernel_t30")?;
    let mut updates = Vec::with_capacity(gradients.len());

    for i in 0..gradients.len() {
        let n = gradients[i].len();
        let n_i = n as i32;
        let mut delta = stream.alloc_zeros::<f32>(n)?;
        let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            updates.push(delta);
            continue;
        }
        let (mut lr_i, wd_i) = group_hyperparams(group, lr);
        // First linear layer = param indices 0 (weight) / 1 (bias): boost early.
        if (i == 0 || i == 1) && asym_mult != 1.0 {
            lr_i *= asym_mult;
        }
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(&update_kernel)
                .arg(&gradients[i])
                .arg(&model_params[i])
                .arg(&mut state.m[i])
                .arg(&mut state.v[i])
                .arg(&mut state.s[i])
                .arg(&mut state.prev_grad[i])
                .arg(&lr_i)
                .arg(&eps)
                .arg(&wd_i)
                .arg(&first_step)
                .arg(&mut delta)
                .arg(&n_i)
                .launch(cfg)?;
        }
        updates.push(delta);
    }

    Ok(updates)
}
