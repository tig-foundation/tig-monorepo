mod helpers {
    use cudarc::driver::{CudaSlice, LaunchConfig};
    use tig_challenges::neuralnet_optimizer::*;

    #[derive(Clone)]
    pub struct OptimizerState {
        pub m: Vec<CudaSlice<f32>>,
        pub v: Vec<CudaSlice<f32>>,
        pub prev_g: Vec<CudaSlice<f32>>,
        pub prev_u: Vec<CudaSlice<f32>>,
        pub slow_u: Vec<CudaSlice<f32>>,
        pub f: Vec<CudaSlice<f32>>,
        pub ef: Vec<CudaSlice<f32>>,
        pub upd: Vec<CudaSlice<f32>>,
        pub cfgs: Vec<LaunchConfig>,
        pub layer_lrs: Vec<f32>,
        pub spectral_boost: f32,

        pub step_count: usize,
        pub warmup_steps: usize,
        pub total_steps: usize,

        pub noise_variance: f32,
        pub val_loss_history: Vec<f32>,

        pub beta1: f32,
        pub beta2: f32,
        pub eps: f32,
        pub weight_decay: f32,

        pub bn_layer_boost: f32,
        pub output_layer_damping: f32,

        pub prev_val_loss: Option<f32>,
        pub best_val_loss: Option<f32>,
        pub plateau_count: usize,
        pub slope_ema: f32,
        pub lr_boost: f32,
        pub last_pulse_step: usize,
        pub last_epoch: usize,
        pub steps_in_epoch: usize,
        pub bpe_ema: f32,
        pub phase_tempo: f32,
        pub spectral_decay: f32,

        pub best_w: Option<Vec<CudaSlice<f32>>>,
        pub tt_best_loss: f32,
        pub tt_cooldown: usize,
        pub global_lr_mult: f32,
        pub tt_prev_epoch: usize,
    }

    impl OptimizerStateTrait for OptimizerState {
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
    }

    #[inline]
    pub fn spectral_phase_lr(s: &OptimizerState, base_lr: f32) -> f32 {
        let t = s.step_count as f32;
        let warm = s.warmup_steps as f32;
        let total = s.total_steps as f32;

        if t <= warm {
            return base_lr * (t / warm.max(1.0)) * s.spectral_boost;
        }

        let t_active = t - warm;
        let active_steps = (total - warm).max(1.0);

        let num_cycles = 3.0f32;
        let cycle_len = active_steps / num_cycles;

        let cycle_idx = (t_active / cycle_len).floor().min(num_cycles - 1.0);
        let progress_in_cycle = (t_active - cycle_idx * cycle_len) / cycle_len;

        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress_in_cycle).cos());

        let cycle_decay = 0.80f32.powf(cycle_idx);
        let spec_boost = s.spectral_boost * (1.0 - s.spectral_decay * (t_active / active_steps));

        base_lr * cosine_factor * spec_boost * cycle_decay
    }

    #[inline]
    pub fn intra_epoch_factor(s: &OptimizerState) -> f32 {
        let bpe = s.bpe_ema.max(1.0);
        let pos = (s.steps_in_epoch as f32) / bpe;
        0.96 + 0.08 * (std::f32::consts::PI * pos).cos()
    }

    #[inline]
    pub fn loss_curvature_factor(s: &OptimizerState) -> f32 {
        let h = &s.val_loss_history;
        if h.len() < 3 { return 1.0; }
        let n = h.len();
        let d1 = h[n - 2] - h[n - 1];
        let d2 = h[n - 3] - h[n - 2];
        let accel = d1 - d2;
        if accel > 0.003 { 1.04 }
        else if accel < -0.003 { 0.96 }
        else { 1.0 }
    }

    #[inline]
    pub fn compute_blends(s: &OptimizerState, val_loss: Option<f32>) -> (f32, f32, f32, f32, f32, f32, f32) {
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

    pub fn update_state_from_val_loss(s: &mut OptimizerState, epoch: usize, val_loss: Option<f32>) {
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

        if let Some(loss) = val_loss {
            if s.step_count > s.warmup_steps {
                s.val_loss_history.push(loss);
                if s.val_loss_history.len() > 12 {
                    s.val_loss_history.remove(0);
                }

                if s.val_loss_history.len() >= 6 {
                    let min_loss = s.val_loss_history.iter().copied().fold(f32::INFINITY, f32::min);
                    let recent_avg = s.val_loss_history.iter().rev().take(10).sum::<f32>() / 10.0;
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
                    s.spectral_boost = (s.spectral_boost * 1.015).min(1.5);
                } else if relative_improvement < -0.003 {
                    s.spectral_boost *= 0.97;
                } else if relative_improvement.abs() < 0.0005 && s.plateau_count > 15 {
                    s.spectral_boost = (s.spectral_boost * 1.008).min(1.5);
                }

                s.spectral_boost = s.spectral_boost.clamp(0.85, 1.5);
            }
        }

        if s.step_count > s.warmup_steps && s.val_loss_history.len() >= 8 {
            let recent_avg = s.val_loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let older_avg = s.val_loss_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;
            let trend = older_avg - recent_avg;

            let target_beta1 = if trend > 0.02 {
                0.94
            } else if trend < -0.02 {
                0.88
            } else {
                0.91
            };

            s.beta1 = 0.85 * s.beta1 + 0.15 * target_beta1;
            s.beta1 = s.beta1.clamp(0.87, 0.94);
        }
    }

    pub fn compute_global_damp(s: &mut OptimizerState, val_loss: Option<f32>) -> f32 {
        let mut global_damp = 1.0f32;

        if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
            let improvement = prev - curr;
            s.slope_ema = 0.85 * s.slope_ema + 0.15 * improvement;
            if s.step_count > s.warmup_steps {
                let is_stagnant = improvement <= 1.0e-4 && s.slope_ema < 2.0e-4;
                let is_declining = improvement < 0.0 && s.slope_ema < 0.0;

                if is_stagnant || is_declining {
                    s.plateau_count += 1;
                } else if improvement > 5.0e-5 {
                    s.plateau_count = 0;
                } else if s.plateau_count > 0 {
                    s.plateau_count = s.plateau_count.saturating_sub(1);
                }
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
                    let relative_improvement = improvement / curr.abs().max(1e-8);
                    let decay = if relative_improvement > 0.01 {
                        0.75
                    } else if relative_improvement > 0.001 {
                        0.85
                    } else if improvement > 0.0 {
                        0.93
                    } else {
                        0.97
                    };
                    s.lr_boost = 1.0 + (s.lr_boost - 1.0) * decay;
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
                let proximity = (loss / dynamic_threshold).clamp(0.4, 1.0);
                let plateau_factor: f32 = if s.plateau_count > 10 { 1.2 } else { 1.0 };
                global_damp *= (0.25 + 0.35 * proximity) * plateau_factor.min(0.9);
            }

            if loss <= s.noise_variance * 5.0 {
                let noise_proximity = (loss / (s.noise_variance * 5.0)).min(1.0);
                let steepness = 1.0 + 0.5 * (1.0 - noise_proximity);
                let noise_damping = 0.70 + 0.30 * noise_proximity.powf(steepness);
                global_damp *= noise_damping;
            }
        }

        global_damp
    }

    pub fn compute_precision_params(s: &OptimizerState, val_loss: Option<f32>) -> (bool, f32, f32, f32, f32) {
        if let Some(loss) = val_loss {
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
                    return (true, pg, gate_lo, gate_hi, forward_gain);
                }
            }
        }
        (false, 1.0, 0.66, 1.50, 1.0)
    }

    pub fn finalize_state(s: &mut OptimizerState, val_loss: Option<f32>) {
        if let Some(curr) = val_loss {
            s.best_val_loss = Some(match s.best_val_loss {
                Some(b) => if curr < b { curr } else { b },
                None => curr,
            });
        }
        s.prev_val_loss = val_loss;
    }
}
pub mod track_t29 {
    use anyhow::Result;
    use cudarc::{
        driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
        runtime::sys::cudaDeviceProp,
    };
    use serde_json::{Map, Value};
    use std::sync::Arc;
    use tig_challenges::neuralnet_optimizer::*;

    const TOTAL_STEPS: usize = 400;
    const WARMUP_STEPS: usize = 40;
    const RESTART_STEP: usize = 200;

    #[derive(Clone)]
    struct T29HP {
        total_steps: usize,
        warmup_steps: usize,
        restart_step: usize,
        beta1: f32,
        weight_decay: f32,
        base_lr: f32,
        sam_rho: f32,
    }

    impl Default for T29HP {
        fn default() -> Self {
            T29HP {
                total_steps: TOTAL_STEPS,
                warmup_steps: WARMUP_STEPS,
                restart_step: RESTART_STEP,
                beta1: 0.90,
                weight_decay: 0.01,
                base_lr: 0.040,
                sam_rho: 0.004,
            }
        }
    }

    thread_local! {
        static T29_HP: std::cell::RefCell<T29HP> = std::cell::RefCell::new(T29HP::default());
    }

    #[derive(Clone)]
    struct T29State {
        m: Vec<CudaSlice<f32>>,
        upd: Vec<CudaSlice<f32>>,
        cfgs: Vec<LaunchConfig>,
        layer_lrs: Vec<f32>,
        total_steps: usize,
        warmup_steps: usize,
        restart_step: usize,
        step_count: usize,
        beta1: f32,
        weight_decay: f32,
        sam_rho: f32,
    }

    impl OptimizerStateTrait for T29State {
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
    }

    fn parse_usize(hp: Option<&Map<String, Value>>, key: &str, default: usize) -> usize {
        hp.and_then(|h| h.get(key).and_then(|v| v.as_u64())).map(|v| v as usize).unwrap_or(default)
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
        T29_HP.with(|c| {
            let mut cfg = c.borrow_mut();
            cfg.total_steps = parse_usize(hp, "total_steps", TOTAL_STEPS);
            cfg.warmup_steps = parse_usize(hp, "warmup_steps", WARMUP_STEPS);
            cfg.restart_step = parse_usize(hp, "restart_step", RESTART_STEP);
            cfg.beta1 = parse_f32(hp, "beta1", 0.90);
            cfg.weight_decay = parse_f32(hp, "weight_decay", 0.01);
            cfg.base_lr = parse_f32(hp, "base_lr", 0.040);
            cfg.sam_rho = parse_f32(hp, "sam_rho", 0.004);
        });

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
        let hp = T29_HP.with(|c| c.borrow().clone());

        let mut m = Vec::with_capacity(param_sizes.len());
        let mut upd = Vec::with_capacity(param_sizes.len());

        for &n in param_sizes {
            m.push(stream.alloc_zeros::<f32>(n)?);
            upd.push(unsafe { stream.alloc::<f32>(n)? });
        }

        let threads_per_block: u32 = 256;
        let blocks_per_sm: u32 = 4;
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
        let head_lr = 0.008_f32;
        let mut layer_lrs = Vec::with_capacity(last);
        for (i, &n) in param_sizes.iter().enumerate() {
            let lr = if i + 2 >= last {
                head_lr
            } else if n <= 512 {
                hp.base_lr * 1.5
            } else if n > 50_000 {
                hp.base_lr
            } else {
                hp.base_lr * 1.1
            };
            layer_lrs.push(lr);
        }

        Ok(Box::new(T29State {
            m,
            upd,
            cfgs,
            layer_lrs,
            total_steps: hp.total_steps,
            warmup_steps: hp.warmup_steps,
            restart_step: hp.restart_step,
            step_count: 0,
            beta1: hp.beta1,
            weight_decay: hp.weight_decay,
            sam_rho: hp.sam_rho,
        }))
    }

    fn optimizer_query(
        state: &dyn OptimizerStateTrait,
        params: &[CudaSlice<f32>],
        _epoch: usize,
        _train: Option<f32>,
        _val_loss: Option<f32>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        _prop: &cudaDeviceProp,
    ) -> Result<Option<Vec<CudaSlice<f32>>>> {
        let s = state.as_any().downcast_ref::<T29State>().unwrap();

        if s.step_count < s.warmup_steps || s.step_count >= s.total_steps {
            return Ok(None);
        }

        let sam_kernel = module.load_function("momentum_sam_kernel")?;
        let mut perturbed = Vec::with_capacity(params.len());

        for (i, p) in params.iter().enumerate() {
            let n = p.len();
            if n == 0 {
                perturbed.push(stream.alloc_zeros::<f32>(0)?);
                continue;
            }
            let mut out = unsafe { stream.alloc::<f32>(n)? };
            let cfg = s.cfgs[i];
            unsafe {
                stream
                    .launch_builder(&sam_kernel)
                    .arg(p)
                    .arg(&s.m[i])
                    .arg(&mut out)
                    .arg(&(n as u32))
                    .arg(&s.sam_rho)
                    .launch(cfg)?;
            }
            perturbed.push(out);
        }

        Ok(Some(perturbed))
    }

    #[inline]
    fn compute_sgdr_lr(step: usize, warmup: usize, restart: usize, total: usize) -> f32 {
        if step <= warmup {
            return 0.1 + 0.9 * (step as f32 / warmup.max(1) as f32);
        }

        let (cycle_start, cycle_end) = if step <= restart {
            (warmup, restart)
        } else {
            (restart, total)
        };
        let t = (step - cycle_start) as f32 / (cycle_end - cycle_start).max(1) as f32;
        0.5 * (1.0 + (std::f32::consts::PI * t.min(1.0)).cos())
    }

    fn optimizer_step(
        state: &mut dyn OptimizerStateTrait,
        model_params: &[CudaSlice<f32>],
        gradients: &[CudaSlice<f32>],
        _epoch: usize,
        _train_loss: Option<f32>,
        _val_loss: Option<f32>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        _prop: &cudaDeviceProp,
    ) -> Result<Vec<CudaSlice<f32>>> {
        let s = state.as_any_mut().downcast_mut::<T29State>().unwrap();
        s.step_count += 1;

        let lr_factor = compute_sgdr_lr(s.step_count, s.warmup_steps, s.restart_step, s.total_steps);
        let kernel = module.load_function("lion_kernel")?;
        let mut updates = Vec::with_capacity(gradients.len());

        for (i, g) in gradients.iter().enumerate() {
            let n = g.len();
            if n == 0 {
                updates.push(stream.alloc_zeros::<f32>(0)?);
                continue;
            }
            let p = &model_params[i];
            let effective_lr = s.layer_lrs[i] * lr_factor;
            let cfg = s.cfgs[i];

            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(g)
                    .arg(p)
                    .arg(&mut s.m[i])
                    .arg(&mut s.upd[i])
                    .arg(&(n as u32))
                    .arg(&effective_lr)
                    .arg(&s.beta1)
                    .arg(&s.weight_decay)
                    .launch(cfg)?;
            }
            updates.push(s.upd[i].clone());
        }
        Ok(updates)
    }
}
