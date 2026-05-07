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

        let progress = ((t - warm) / (total - warm).max(1.0)).min(1.0);
        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        let spec_boost = s.spectral_boost * (1.0 - s.spectral_decay * progress);

        base_lr * cosine_factor * spec_boost
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
pub mod track_t26 {
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
            total_steps: 1000,
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
            spectral_decay: 0.4,
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
}
