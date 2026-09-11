use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

#[allow(dead_code, clippy::all)]
mod inner {

fn dead_update_mask_t30(count: usize) -> Vec<bool> {
    const FROZEN: usize = 2;
    let hidden_layers = count.saturating_sub(2) / 6;
    let linear_layers = hidden_layers + 1;
    let mut dead = vec![false; count];
    let mut idx = 0usize;
    for l in 0..linear_layers {
        let t = l + FROZEN < linear_layers;
        dead[idx] = !t; dead[idx + 1] = !t; idx += 2;
    }
    for b in 0..hidden_layers {
        let t = b + FROZEN < linear_layers;
        dead[idx] = !t; dead[idx + 1] = !t;
        dead[idx + 2] = true; dead[idx + 3] = true; idx += 4;
    }
    dead
}

    mod helpers {
        use cudarc::driver::{CudaSlice, LaunchConfig};
        use tig_challenges::neuralnet_optimizer::*;

        #[derive(Clone)]
        pub struct FusedMeta {
            pub live: Vec<usize>,
            pub affine_bias_source: Vec<Option<usize>>,
            pub ptr_dev: CudaSlice<u64>,  
            pub scal_dev: CudaSlice<f32>,  
            pub meta_dev: CudaSlice<i32>,  
            pub fused_cfg: LaunchConfig,
        }

        pub const T30_NORM_FUSE: bool = true;

        #[derive(Clone)]
        pub struct NormFuse1 {
            pub idx: Vec<usize>,
            pub lens_dev: CudaSlice<i32>,
            pub slots_dev: CudaSlice<i32>,
            pub cfg: LaunchConfig,
        }

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

            pub consensus_sign_bias: f32,
            pub near_floor_mult: f32,
            pub late_phase_frac: f32,
            pub plateau_norm_bias: f32,
            pub nv_lo_mult: f32,
            pub transition_sign: f32,
            pub p6: Option<FusedMeta>,
            pub nrm1: Option<NormFuse1>,
            pub trust_partials: Option<CudaSlice<f32>>,
            pub gradient_update_history: Vec<[f32; 4]>,
            pub dc_scratch: Vec<CudaSlice<f32>>,
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
            let spec_boost = s.spectral_boost * (1.0 - 0.3 * progress);

            base_lr * cosine_factor * spec_boost
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
                    let pnb = s.plateau_norm_bias;
                    let ba = (0.55_f32 - pnb).max(0.0);
                    let bn = (0.35_f32 + pnb).max(0.0);
                    (ba, bn, 0.10, 0.15, 0.45, 0.30, 0.22)
                } else {
                    let ts = s.transition_sign;
                    let ta = 0.50_f32 + (0.15_f32 - ts);
                    (ta, 0.35, ts, 0.20, 0.50, 0.20, 0.20)
                }
            };

            if t > warm {
                if let Some(curr) = val_loss {
                    if curr <= s.noise_variance * s.nv_lo_mult {
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
                blend_sign = (blend_sign + s.consensus_sign_bias).max(0.0);
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

                if loss <= s.noise_variance * s.nv_lo_mult {
                    let noise_proximity = (loss / (s.noise_variance * s.nv_lo_mult)).min(1.0);
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
    pub mod track_7 {
        use anyhow::Result;
        use cudarc::{
            driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
            runtime::sys::cudaDeviceProp,
        };
        use std::sync::Arc;
        use tig_challenges::neuralnet_optimizer::*;
        use serde_json::{Map, Value};

        use super::helpers::{
            FusedMeta, OptimizerState, NormFuse1, T30_NORM_FUSE,
            spectral_phase_lr, compute_blends, update_state_from_val_loss,
            compute_global_damp, compute_precision_params, finalize_state,
        };
        use super::super::super::helpers::dc;

        const DC_ENABLE_DEFAULT_T30: u64 = 1;
        const DC_AXPY_T30: &str = "dc_axpy_t30";

        thread_local! {
            static DC: std::cell::RefCell<dc::DcState> = std::cell::RefCell::new(dc::DcState::default());
            static CONSENSUS_SIGN_BIAS_T7: std::cell::Cell<f32> = std::cell::Cell::new(0.0f32);
            static NEAR_FLOOR_MULT_T7: std::cell::Cell<f32> = std::cell::Cell::new(3.0f32);
            static LATE_PHASE_FRAC_T7: std::cell::Cell<f32> = std::cell::Cell::new(0.75f32);
            static PLATEAU_NORM_BIAS_T7: std::cell::Cell<f32> = std::cell::Cell::new(0.0f32);
            static NV_LO_MULT_T7: std::cell::Cell<f32> = std::cell::Cell::new(5.0f32);
            static TRANSITION_SIGN_T7: std::cell::Cell<f32> = std::cell::Cell::new(0.13f32);
            static TR_HI_T7: std::cell::Cell<f32> = std::cell::Cell::new(2.0f32);
            static BNS_MIX_T7: std::cell::Cell<f32> = std::cell::Cell::new(1.0f32);
        }

        pub fn solve(
            challenge: &Challenge,
            save_solution: &dyn Fn(&Solution) -> Result<()>,
            hyperparameters: &Option<Map<String, Value>>,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<()> {
            let csb: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("consensus_sign_bias").and_then(|v| v.as_f64()))
                .unwrap_or(-0.05) as f32;
            CONSENSUS_SIGN_BIAS_T7.with(|c| c.set(csb));
            let tr_hi_hp: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("tr_hi").and_then(|v| v.as_f64()))
                .unwrap_or(2.0) as f32;
            TR_HI_T7.with(|c| c.set(tr_hi_hp));
            let bns_mix_hp: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("bns_mix").and_then(|v| v.as_f64()))
                .unwrap_or(1.0) as f32;
            BNS_MIX_T7.with(|c| c.set(bns_mix_hp));
            let nfm: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("near_floor_mult").and_then(|v| v.as_f64()))
                .unwrap_or(2.2) as f32;
            NEAR_FLOOR_MULT_T7.with(|c| c.set(nfm));
            let lpf: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("late_phase_frac").and_then(|v| v.as_f64()))
                .unwrap_or(0.75) as f32;
            LATE_PHASE_FRAC_T7.with(|c| c.set(lpf));
            let pnb: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("plateau_norm_bias").and_then(|v| v.as_f64()))
                .unwrap_or(0.05) as f32;
            PLATEAU_NORM_BIAS_T7.with(|c| c.set(pnb));
            let nv_lo_mult: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("nv_lo_mult").and_then(|v| v.as_f64()))
                .unwrap_or(5.0) as f32;
            NV_LO_MULT_T7.with(|c| c.set(nv_lo_mult));
            let transition_sign: f32 = hyperparameters.as_ref()
                .and_then(|h| h.get("transition_sign").and_then(|v| v.as_f64()))
                .unwrap_or(0.13) as f32;
            TRANSITION_SIGN_T7.with(|c| c.set(transition_sign));

            let dcfg = dc::config_from_hp(hyperparameters, DC_ENABLE_DEFAULT_T30);
            DC.with(|c| *c.borrow_mut() = dc::DcState::default());
            if dcfg.enable != 0 {
                let m = dc::target_mean(challenge, &stream)?;
                let nb = dc::batches_per_epoch(challenge);
                DC.with(|c| c.borrow_mut().arm(&dcfg, m, nb));
            }

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
            let consensus_sign_bias = CONSENSUS_SIGN_BIAS_T7.with(|c| c.get());
            let near_floor_mult = NEAR_FLOOR_MULT_T7.with(|c| c.get());
            let late_phase_frac = LATE_PHASE_FRAC_T7.with(|c| c.get());
            let plateau_norm_bias = PLATEAU_NORM_BIAS_T7.with(|c| c.get());
            let nv_lo_mult = NV_LO_MULT_T7.with(|c| c.get());
            let transition_sign = TRANSITION_SIGN_T7.with(|c| c.get());
            let mut m = Vec::new();
            let mut v = Vec::new();
            let mut prev_g = Vec::new();
            let mut prev_u = Vec::new();
            let mut slow_u = Vec::new();
            let mut f = Vec::new();
            let mut ef = Vec::new();
            let mut upd = Vec::new();
            let max_n = param_sizes.iter().copied().max().unwrap_or(0);
            let host_fisher = vec![1e-4f32; max_n];
            for &n in param_sizes {
                m.push(stream.alloc_zeros::<f32>(n)?);
                v.push(stream.alloc_zeros::<f32>(n)?);
                prev_g.push(stream.alloc_zeros::<f32>(n)?);
                prev_u.push(stream.alloc_zeros::<f32>(n)?);
                slow_u.push(stream.alloc_zeros::<f32>(n)?);
                let mut fisher_init = stream.alloc_zeros::<f32>(n)?;
                if n > 0 { stream.memcpy_htod(&host_fisher[..n], &mut fisher_init)?; }
                f.push(fisher_init);
                ef.push(stream.alloc_zeros::<f32>(n)?);
                upd.push(unsafe { stream.alloc::<f32>(n)? });
            }
            let threads_per_block: u32 = 256;
            let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(3).max(1);
            let mut cfgs = Vec::with_capacity(param_sizes.len());
            for &n in param_sizes {
                let calc_blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
                let grid_dim = calc_blocks.min(sm_blocks).max(1);
                cfgs.push(LaunchConfig { grid_dim: (grid_dim, 1, 1), block_dim: (threads_per_block, 1, 1), shared_mem_bytes: 0 });
            }
            let num_layers = param_sizes.len();
            let mut layer_lrs = Vec::with_capacity(num_layers);
            for (i, &param_size) in param_sizes.iter().enumerate() {
                let lr = if i == num_layers - 1 {
                    0.00065f32
                } else if i == 0 {
                    0.00110f32
                } else if param_size <= 256 {
                    0.00240f32
                } else if param_size <= 512 {
                    0.00200f32
                } else if param_size > 50000 {
                    0.00095f32
                } else {
                    0.00130f32
                };
                layer_lrs.push(lr);
            }

            let n_returned = param_sizes.len().saturating_sub(6);
            let dead = super::dead_update_mask_t30(param_sizes.len());
            let live: Vec<usize> = (0..n_returned)
                .filter(|&i| param_sizes[i] > 0 && !dead[i])
                .collect();
            let t_count = live.len();

            let mut h_meta: Vec<i32> = Vec::with_capacity(2 * t_count + 1);
            let mut acc: i32 = 0;
            h_meta.push(0);
            for &i in &live {
                acc += cfgs[i].grid_dim.0 as i32;   
                h_meta.push(acc);
            }
            for &i in &live {
                h_meta.push(param_sizes[i] as i32);
            }

            let mut affine_bias_source: Vec<Option<usize>> =
                vec![None; t_count];
            let linear_layers = param_sizes.len().saturating_sub(2) / 6 + 1;
            for layer in 0..linear_layers {
                let weight_index = 2 * layer;
                let bias_index = weight_index + 1;
                if let (Some(weight_slot), Some(bias_slot)) = (
                    live.iter().position(|&i| i == weight_index),
                    live.iter().position(|&i| i == bias_index),
                ) {
                    affine_bias_source[bias_slot] = Some(weight_slot);
                }
            }

            let p6 = if t_count > 0 {
                Some(FusedMeta {
                    live,
                    affine_bias_source,
                    ptr_dev: stream.alloc_zeros::<u64>(10 * t_count)?,
                    scal_dev: stream.alloc_zeros::<f32>(3 * t_count)?,
                    meta_dev: stream.memcpy_stod(&h_meta)?,   
                    fused_cfg: LaunchConfig {
                        grid_dim: (acc.max(1) as u32, 1, 1),
                        block_dim: (threads_per_block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                })
            } else {
                None
            };

            let nrm1 = if T30_NORM_FUSE {
                let dead_n = super::dead_update_mask_t30(param_sizes.len());
                let idx: Vec<usize> = (0..param_sizes.len())
                    .filter(|&i| param_sizes[i] > 0 && !dead_n[i] && cfgs[i].grid_dim.0 == 1)
                    .collect();
                if idx.is_empty() { None } else {
                    let lens: Vec<i32> = idx.iter().map(|&i| param_sizes[i] as i32).collect();
                    let slots: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
                    Some(NormFuse1 {
                        cfg: LaunchConfig {
                            grid_dim: (idx.len() as u32, 1, 1),
                            block_dim: (threads_per_block, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        lens_dev: stream.memcpy_stod(&lens)?,
                        slots_dev: stream.memcpy_stod(&slots)?,
                        idx,
                    })
                }
            } else { None };

            let trust_partials = if t_count > 0 {
                Some(stream.alloc_zeros::<f32>(8 * acc as usize)?)
            } else {
                None
            };
            let gradient_update_history = vec![[0.0f32; 4]; t_count];

            let state = OptimizerState {
                m, v, prev_g, prev_u, slow_u, f, ef, upd, cfgs, layer_lrs,
                spectral_boost: 1.1, step_count: 0, warmup_steps: 40, total_steps: 1100,
                noise_variance: 0.040, val_loss_history: Vec::new(), beta1: 0.92, beta2: 0.997,
                eps: 1e-8, weight_decay: 0.0022, bn_layer_boost: 1.45, output_layer_damping: 0.80,
                prev_val_loss: None, best_val_loss: None, plateau_count: 0, slope_ema: 0.0, lr_boost: 1.0,
                last_pulse_step: 0, last_epoch: 0, steps_in_epoch: 0, bpe_ema: 1.0, phase_tempo: 1.0,
                consensus_sign_bias,
                near_floor_mult,
                late_phase_frac,
                plateau_norm_bias,
                nv_lo_mult,
                transition_sign,
                p6,
                nrm1,
                trust_partials,
                gradient_update_history,
                dc_scratch: Vec::new(),
            };
            Ok(Box::new(state) as Box<dyn OptimizerStateTrait>)
        }

        fn optimizer_query(_state: &dyn OptimizerStateTrait, _params: &[CudaSlice<f32>], _epoch: usize, _train: Option<f32>, _val: Option<f32>, _stream: Arc<CudaStream>, _module: Arc<CudaModule>, _prop: &cudaDeviceProp) -> Result<Option<Vec<CudaSlice<f32>>>> { Ok(None) }

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
            let global_damp = compute_global_damp(s, val_loss);
            let t = (s.step_count + 1) as i32;
            let bias_correction1 = 1.0 - s.beta1.powi(t);
            let bias_correction2 = 1.0 - s.beta2.powi(t);
            let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) = compute_blends(s, val_loss);
            let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * s.near_floor_mult);
            let late_phase = (s.step_count as f32) > (s.total_steps as f32) * s.late_phase_frac;
            let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);
            let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain) = compute_precision_params(s, val_loss);
            let beta1_eff: f32 = if in_precision_zone { (s.beta1 + 0.02).min(0.995) } else { s.beta1 };
            let beta2_eff: f32 = s.beta2;
            let eps_eff: f32 = if in_precision_zone { s.eps * 0.9 } else { s.eps };
            let mut wd_eff: f32 = if in_precision_zone { s.weight_decay * 1.05 } else { s.weight_decay };
            if s.step_count > s.warmup_steps {
                if near_floor { wd_eff *= 1.10; } else if s.plateau_count >= 20 { wd_eff *= 0.50; }
            }
            wd_eff *= (1.0f32 / s.phase_tempo).clamp(0.6, 1.0);
            let trust_backoff: f32 = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
                let rel = (curr - prev) / (prev.abs() + 1e-8);
                if rel > 0.020 { 0.70 }
                else if rel > 0.012 { 0.80 }
                else if rel > 0.005 { 0.90 }
                else { 1.0 }
            } else { 1.0 };
            let trend_factor: f32 = if s.val_loss_history.len() >= 3 {
                let h = &s.val_loss_history;
                let n = h.len();
                let recent_slope = h[n-1] - h[n-3];
                if recent_slope > 0.015 { 0.88 } else if recent_slope > 0.008 { 0.94 } else { 1.0 }
            } else { 1.0 };
            let k_fast = module.load_function("dual_consensus_fisher_fused_7")?;
            let k_robust = module.load_function("sign_ef_consensus_fused_7")?;
            let k_rayleigh = module.load_function("tensorwise_rayleigh_scale_fused_7")?;
            let k_svrg_fused = module.load_function("svrg_variance_reduce_fused_7")?;
            let k_norm = module.load_function("grad_norm_reduction_kernel_7")?;
            let k_norm_fused = if T30_NORM_FUSE {
                Some(module.load_function("grad_norm_fused_1blk_7")?)
            } else { None };
            let mut updates = Vec::with_capacity(gradients.len());
            let num_layers = gradients.len();
            let do_adaptive_lr = s.step_count > s.warmup_steps;
            let norms_dev = stream.alloc_zeros::<f32>(num_layers)?;
            let (norms_base, _norms_guard) = norms_dev.device_ptr(&stream);
            let dead_t30 = super::dead_update_mask_t30(num_layers);
            let fused_norm: Vec<bool> = if let (Some(kf), Some(nf)) = (k_norm_fused.as_ref(), s.nrm1.as_ref()) {
                let mut mask = vec![false; num_layers];
                let mut hp: Vec<u64> = Vec::with_capacity(nf.idx.len());
                let mut ok = true;
                for &i in &nf.idx {
                    if i >= num_layers || gradients[i].len() == 0 || dead_t30[i] { ok = false; break; }
                    let (q, _g) = gradients[i].device_ptr(&stream);
                    hp.push(q);
                }
                if ok && !hp.is_empty() {
                    let ptrs = stream.memcpy_stod(&hp)?;
                    unsafe {
                        stream.launch_builder(kf)
                            .arg(&ptrs)
                            .arg(&nf.lens_dev)
                            .arg(&nf.slots_dev)
                            .arg(&(norms_base))
                            .arg(&(nf.idx.len() as u32))
                            .launch(nf.cfg)?;
                    }
                    for &i in &nf.idx { mask[i] = true; }
                }
                mask
            } else { vec![false; num_layers] };

            for i in 0..num_layers {
                let n = gradients[i].len();
                if n == 0 || dead_t30[i] { continue; }
                if fused_norm[i] { continue; }
                let cfg = s.cfgs[i];
                let g = &gradients[i];
                unsafe {
                    stream.launch_builder(&k_norm)
                        .arg(g)
                        .arg(&(norms_base + 4u64 * i as u64))
                        .arg(&(n as u32))
                        .launch(cfg)?;
                }
            }
            let do_svrg = !use_robust && s.step_count >= s.warmup_steps;
            let mut vr_grad_bufs: Vec<Option<CudaSlice<f32>>> = (0..num_layers).map(|_| None).collect();
            let mut ef_mean_bufs: Vec<Option<CudaSlice<f32>>> = (0..num_layers).map(|_| None).collect();

            if do_svrg {
                for i in 0..num_layers {
                    let n = gradients[i].len();
                    if n == 0 || dead_t30[i] { continue; }
                    let vr_buf = unsafe { stream.alloc::<f32>(n)? };
                    vr_grad_bufs[i] = Some(vr_buf);
                    let mean_buf = stream.alloc_zeros::<f32>(1)?;
                    ef_mean_bufs[i] = Some(mean_buf);
                }
                let svrg_blend: f32 = ((s.step_count - s.warmup_steps) as f32 / 60.0f32).min(0.35f32);
                if let Some(p6) = s.p6.as_ref() {
                    let t_count = p6.live.len();
                    if t_count > 0 {
                        let mut h3: Vec<u64> = vec![0u64; 3 * t_count];
                        let mut ok = true;
                        for (k, &i) in p6.live.iter().enumerate() {
                            match vr_grad_bufs[i].as_ref() {
                                Some(vr) => {
                                    h3[0 * t_count + k] = { let (q, _g) = gradients[i].device_ptr(&stream); q };
                                    h3[1 * t_count + k] = { let (q, _g) = s.ef[i].device_ptr(&stream); q };
                                    h3[2 * t_count + k] = { let (q, _g) = vr.device_ptr(&stream); q };
                                }
                                None => { ok = false; break; }
                            }
                        }
                        if ok {
                            let ptr3 = stream.memcpy_stod(&h3)?;
                            unsafe {
                                stream.launch_builder(&k_svrg_fused)
                                    .arg(&ptr3)
                                    .arg(&p6.meta_dev)
                                    .arg(&(t_count as u32))
                                    .arg(&svrg_blend)
                                    .launch(p6.fused_cfg)?;
                            }
                        }
                    }
                }
            }

            let dead_t30 = super::dead_update_mask_t30(gradients.len());
            let n_ret_t30 = gradients.len().saturating_sub(6);

            let t_count = s.p6.as_ref().map(|p| p.live.len()).unwrap_or(0);
            let mut h_ptrs: Vec<u64> = vec![0u64; 10 * t_count];
            let mut h_scal: Vec<f32> = vec![0.0f32; 3 * t_count];
            let mut k = 0usize;  

            for (i, g) in gradients.iter().take(n_ret_t30).enumerate() {
                let n = g.len();
                if n == 0 || dead_t30[i] { continue; }
                let p = model_params.get(i).unwrap();
                let base_lr = s.layer_lrs[i];
                let tempo_lr = (1.0f32 / s.phase_tempo.powf(0.35)).max(0.6);
                let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr;
                let layer_multiplier = if i == num_layers - 1 {
                    if let Some(loss) = val_loss {
                        if s.step_count > s.warmup_steps + 30 {
                            let loss_ratio = (loss / (s.noise_variance * 6.0)).min(1.0);
                            (s.output_layer_damping * (0.7 + 0.3 * loss_ratio)).max(s.output_layer_damping)
                        } else { s.output_layer_damping }
                    } else { s.output_layer_damping }
                } else if n <= 512 { s.bn_layer_boost } else { 1.0 };
                let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff * trend_factor;
                let g_input: &CudaSlice<f32> = if do_svrg {
                    vr_grad_bufs[i].as_ref().unwrap_or(g)
                } else {
                    g
                };
                let g_arg: &CudaSlice<f32> = if use_robust { g } else { g_input };

                h_ptrs[0 * t_count + k] = { let (q, _g) = g_arg.device_ptr(&stream); q };
                h_ptrs[1 * t_count + k] = { let (q, _g) = p.device_ptr(&stream); q };
                h_ptrs[2 * t_count + k] = { let (q, _g) = s.m[i].device_ptr(&stream); q };
                h_ptrs[3 * t_count + k] = { let (q, _g) = s.v[i].device_ptr(&stream); q };
                h_ptrs[4 * t_count + k] = { let (q, _g) = s.prev_g[i].device_ptr(&stream); q };
                h_ptrs[5 * t_count + k] = { let (q, _g) = s.prev_u[i].device_ptr(&stream); q };
                h_ptrs[6 * t_count + k] = { let (q, _g) = s.slow_u[i].device_ptr(&stream); q };
                h_ptrs[7 * t_count + k] = { let (q, _g) = s.f[i].device_ptr(&stream); q };
                h_ptrs[8 * t_count + k] = { let (q, _g) = s.ef[i].device_ptr(&stream); q };
                h_ptrs[9 * t_count + k] = { let (q, _g) = s.upd[i].device_ptr(&stream); q };

                h_scal[0 * t_count + k] = effective_lr;
                h_scal[1 * t_count + k] = wd_eff;

                let [a, b, c, d] = s.gradient_update_history[k];
                let positive_cycle =
                    a > 0.0 && b < 0.0 && c > 0.0 && d < 0.0;
                let negative_cycle =
                    a < 0.0 && b > 0.0 && c < 0.0 && d > 0.0;
                h_scal[2 * t_count + k] =
                    i32::from(positive_cycle || negative_cycle) as f32;

                k += 1;
            }

            if k != t_count {
                return Err(anyhow::anyhow!(
                    "P6/t30: live-tensor count changed ({} at step, {} at init)", k, t_count
                ));
            }

            if t_count > 0 {
                {
                    let p6 = s.p6.as_mut().unwrap();
                    stream.memcpy_htod(&h_ptrs, &mut p6.ptr_dev)?;
                    stream.memcpy_htod(&h_scal, &mut p6.scal_dev)?;

                    let num_tensors = t_count as u32;
                    let fcfg = p6.fused_cfg;
                    let tr_hi_k: f32 = TR_HI_T7.with(|c| c.get());
                    let bns_mix_k: f32 = BNS_MIX_T7.with(|c| c.get());

                    unsafe {
                        if use_robust {
                            stream.launch_builder(&k_robust)
                                .arg(&p6.ptr_dev).arg(&p6.scal_dev).arg(&p6.meta_dev)
                                .arg(&num_tensors)
                                .arg(&eps_eff)
                                .arg(&lookahead_alpha).arg(&lookahead_tau)
                                .arg(&gate_lo).arg(&gate_hi)
                                .launch(fcfg)?;
                        } else {
                            let partials = s.trust_partials.as_mut().unwrap();
                            stream.launch_builder(&k_fast)
                                .arg(&p6.ptr_dev).arg(&p6.scal_dev).arg(&p6.meta_dev)
                                .arg(&num_tensors)
                                .arg(&beta1_eff).arg(&beta2_eff).arg(&eps_eff)
                                .arg(&bias_correction1).arg(&bias_correction2)
                                .arg(&blend_adam).arg(&blend_norm).arg(&blend_sign)
                                .arg(&nesterov_gamma).arg(&bb_blend)
                                .arg(&lookahead_alpha).arg(&lookahead_tau)
                                .arg(&gate_lo).arg(&gate_hi).arg(&tr_hi_k).arg(&bns_mix_k)
                                .arg(partials)
                                .launch(fcfg)?;
                        }
                    }
                }

                if !use_robust {
                    let block_partials: Vec<f32> =
                        stream.memcpy_dtov(s.trust_partials.as_ref().unwrap())?;
                    let mut differential_scales = Vec::with_capacity(t_count);
                    let mut common_components = Vec::with_capacity(t_count);
                    let mut radial_components = Vec::with_capacity(t_count);
                    let mut emitted_centroids = Vec::with_capacity(t_count);
                    let mut block = 0usize;
                    for (slot, &i) in s.p6.as_ref().unwrap().live.iter().enumerate() {
                        let mut parameter_sq = 0.0f32;
                        let mut update_sq = 0.0f32;
                        let mut parameter_update_dot = 0.0f32;
                        let mut gradient_update_dot = 0.0f32;
                        let mut gradient_sq = 0.0f32;
                        let mut update_sum = 0.0f32;
                        let mut gradient_sum = 0.0f32;
                        let mut parameter_sum = 0.0f32;
                        for _ in 0..s.cfgs[i].grid_dim.0 as usize {
                            parameter_sq += block_partials[8 * block];
                            update_sq += block_partials[8 * block + 1];
                            parameter_update_dot += block_partials[8 * block + 2];
                            gradient_update_dot += block_partials[8 * block + 3];
                            gradient_sq += block_partials[8 * block + 4];
                            update_sum += block_partials[8 * block + 5];
                            gradient_sum += block_partials[8 * block + 6];
                            parameter_sum += block_partials[8 * block + 7];
                            block += 1;
                        }

                        let count = gradients[i].len() as f32;
                        let common_update = update_sum / count;
                        let common_supported = update_sum * gradient_sum <= 0.0;
                        let retained_common = if common_supported { common_update } else { 0.0 };
                        let differential_sq =
                            (update_sq - update_sum * update_sum / count).max(0.0);
                        let differential_parameter_dot =
                            parameter_update_dot - parameter_sum * common_update;
                        let directional_cosine = if update_sq > 0.0 && gradient_sq > 0.0 {
                            gradient_update_dot / (update_sq.sqrt() * gradient_sq.sqrt())
                        } else {
                            0.0
                        };
                        let [_, b, c, d] = s.gradient_update_history[slot];
                        s.gradient_update_history[slot] =
                            [b, c, d, gradient_update_dot];
                        let tangent_only = directional_cosine > 0.0 && parameter_sq > 0.0;
                        let radial_component = if tangent_only {
                            differential_parameter_dot / parameter_sq
                        } else {
                            0.0
                        };
                        let geometry_update_sq = if tangent_only {
                            (differential_sq
                                - differential_parameter_dot * differential_parameter_dot / parameter_sq)
                                .max(0.0)
                        } else {
                            differential_sq
                        };
                        let scale = if parameter_sq > 0.0 && geometry_update_sq > 0.0 {
                            h_scal[slot] * parameter_sq.sqrt() / geometry_update_sq.sqrt()
                        } else {
                            1.0
                        };
                        emitted_centroids.push(
                            scale * (
                                common_update - retained_common
                                - radial_component * parameter_sum / count
                            ) + retained_common
                                - h_scal[slot] * h_scal[t_count + slot]
                                    * parameter_sum / count
                        );
                        differential_scales.push(scale);
                        common_components.push(retained_common);
                        radial_components.push(radial_component);
                    }
                    let affine_transport: Vec<f32> = s.p6.as_ref().unwrap()
                        .affine_bias_source
                        .iter()
                        .map(|source| source.map_or_else(f32::default, |slot| emitted_centroids[slot]))
                        .collect();

                    let mut rayleigh_scales = differential_scales;
                    rayleigh_scales.extend(common_components);
                    rayleigh_scales.extend(radial_components);
                    rayleigh_scales.extend(affine_transport);

                    let d_rayleigh_scales = stream.memcpy_stod(&rayleigh_scales)?;
                    let p6 = s.p6.as_ref().unwrap();
                    let num_tensors = t_count as u32;
                    unsafe {
                        stream.launch_builder(&k_rayleigh)
                            .arg(&p6.ptr_dev).arg(&p6.scal_dev).arg(&p6.meta_dev)
                            .arg(&d_rayleigh_scales).arg(&num_tensors)
                            .launch(p6.fused_cfg)?;
                    }
                }
            }

            for (i, g) in gradients.iter().take(n_ret_t30).enumerate() {
                let n = g.len();
                if n == 0 || dead_t30[i] { updates.push(unsafe { stream.alloc::<f32>(0)? }); continue; }
                updates.push(s.upd[i].clone());
            }

            {
                let sie = s.steps_in_epoch;
                let res: Result<()> = DC.with(|cell| {
                    let mut d = cell.borrow_mut();
                    if !d.enable {
                        return Ok(());
                    }
                    dc::step(
                        &mut *d, sie, epoch, val_loss, model_params, &mut updates,
                        &stream, &module, &mut s.dc_scratch, DC_AXPY_T30,
                    )
                });
                res?;
            }

            if do_adaptive_lr {
                let mut curr_norms = vec![0.0f32; num_layers];
                let mut host_norms = vec![0.0f32; num_layers];
                stream.memcpy_dtoh(&norms_dev, &mut host_norms)?;
                curr_norms.copy_from_slice(&host_norms);
                let curr_norm_sum: f32 = curr_norms.iter().sum::<f32>();
                let prev_norm_sum = s.bpe_ema;
                if prev_norm_sum > 1e-8 {
                    for i in 0..num_layers {
                        if gradients[i].len() == 0 { continue; }
                        let curr = curr_norms[i];
                        let layer_share = if curr_norm_sum > 1e-8 { curr / (curr_norm_sum + 1e-8) } else { 1.0 / num_layers as f32 };
                        let prev_est = prev_norm_sum * layer_share;
                        let ratio = curr / (prev_est + 1e-8);
                        let initial_lr = if i == num_layers - 1 { 0.00065f32 }
                            else if i == 0 { 0.00110f32 }
                            else if gradients[i].len() <= 256 { 0.00240f32 }
                            else if gradients[i].len() <= 512 { 0.00200f32 }
                            else if gradients[i].len() > 50000 { 0.00095f32 }
                            else { 0.00130f32 };
                        let scale = if ratio > 1.5 { 0.9f32 } else if ratio < 0.5 { 1.1f32 } else { 1.0f32 };
                        let new_lr = (s.layer_lrs[i] * scale).clamp(initial_lr * 0.3, initial_lr * 3.0);
                        s.layer_lrs[i] = new_lr;
                    }
                }
                s.bpe_ema = curr_norm_sum;
            }

            finalize_state(s, val_loss);
            Ok(updates)
        }
    }
    pub mod track_14 {
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
                consensus_sign_bias: 0.0,
                near_floor_mult: 3.0,
                late_phase_frac: 0.75,
                plateau_norm_bias: 0.0,
                nv_lo_mult: 5.0,
                transition_sign: 0.15,
                p6: None,
                nrm1: None,
                gradient_update_history: Vec::new(),
                dc_scratch: Vec::new(),
                trust_partials: None,
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

            let dead_t30 = super::dead_update_mask_t30(gradients.len());
            let n_ret_t30 = gradients.len().saturating_sub(6);
            for (i, g) in gradients.iter().take(n_ret_t30).enumerate() {
                let n = g.len();
                if n == 0 || dead_t30[i] { updates.push(unsafe { stream.alloc::<f32>(0)? }); continue; }

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
    }
    pub mod track_18 {
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
                    total_steps: 1200,
                    warmup_steps: 40,
                    noise_variance: 0.040,
                    spectral_boost: 1.1,
                    beta1: 0.92,
                    beta2: 0.997,
                    eps: 1e-8,
                    weight_decay: 0.0025,
                    bn_layer_boost: 1.52,
                    output_layer_damping: 0.8,
                    threads_per_block: 128,
                    blocks_per_sm: 4,
                }
            }
        }

        fn parse_config(hyperparameters: &Option<Map<String, Value>>) -> TrackConfig {
            let hp = hyperparameters.as_ref();
            TrackConfig {
                total_steps: hp.and_then(|h| h.get("total_steps").and_then(|v| v.as_u64())).unwrap_or(1200) as usize,
                warmup_steps: hp.and_then(|h| h.get("warmup_steps").and_then(|v| v.as_u64())).unwrap_or(40) as usize,
                noise_variance: hp.and_then(|h| h.get("noise_variance").and_then(|v| v.as_f64())).unwrap_or(0.040) as f32,
                spectral_boost: hp.and_then(|h| h.get("spectral_boost").and_then(|v| v.as_f64())).unwrap_or(1.1) as f32,
                beta1: hp.and_then(|h| h.get("beta1").and_then(|v| v.as_f64())).unwrap_or(0.92) as f32,
                beta2: hp.and_then(|h| h.get("beta2").and_then(|v| v.as_f64())).unwrap_or(0.997) as f32,
                eps: hp.and_then(|h| h.get("eps").and_then(|v| v.as_f64())).unwrap_or(1e-8) as f32,
                weight_decay: hp.and_then(|h| h.get("weight_decay").and_then(|v| v.as_f64())).unwrap_or(0.0025) as f32,
                bn_layer_boost: hp.and_then(|h| h.get("bn_layer_boost").and_then(|v| v.as_f64())).unwrap_or(1.52) as f32,
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
            training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
            Ok(())
        }

        fn optimizer_init(_seed: [u8; 32], param_sizes: &[usize], stream: Arc<CudaStream>, _module: Arc<CudaModule>, prop: &cudaDeviceProp) -> Result<Box<dyn OptimizerStateTrait>> {
            TRACK_CONFIG.with(|c| {
                let cfg = c.borrow();
                let mut m = Vec::new(); let mut v = Vec::new(); let mut prev_g = Vec::new(); let mut prev_u = Vec::new();
                let mut slow_u = Vec::new(); let mut f = Vec::new(); let mut ef = Vec::new(); let mut upd = Vec::new();
                for &n in param_sizes {
                    m.push(stream.alloc_zeros::<f32>(n)?); v.push(stream.alloc_zeros::<f32>(n)?);
                    prev_g.push(stream.alloc_zeros::<f32>(n)?); prev_u.push(stream.alloc_zeros::<f32>(n)?);
                    slow_u.push(stream.alloc_zeros::<f32>(n)?);
                    let mut fisher_init = stream.alloc_zeros::<f32>(n)?;
                    stream.memcpy_htod(&vec![1e-4f32; n], &mut fisher_init)?;
                    f.push(fisher_init); ef.push(stream.alloc_zeros::<f32>(n)?);
                    upd.push(unsafe { stream.alloc::<f32>(n)? });
                }
                let sm_blocks = (prop.multiProcessorCount as u32).saturating_mul(cfg.blocks_per_sm).max(1);
                let mut cfgs = Vec::with_capacity(param_sizes.len());
                for &n in param_sizes {
                    let calc_blocks = (n as u32 + cfg.threads_per_block - 1) / cfg.threads_per_block;
                    cfgs.push(LaunchConfig { grid_dim: (calc_blocks.min(sm_blocks).max(1), 1, 1), block_dim: (cfg.threads_per_block, 1, 1), shared_mem_bytes: 0 });
                }
                let mut layer_lrs = Vec::with_capacity(param_sizes.len());
                for (i, &ps) in param_sizes.iter().enumerate() {
                    let mut lr = 0.0012f32;
                    if i == 0 { lr = 0.0011; }
                    if ps <= 512 { lr = 0.0018; }
                    if ps > 50000 { lr = 0.0009; }
                    if i == param_sizes.len() - 1 { lr = 0.0007; }
                    layer_lrs.push(lr);
                }
                Ok(Box::new(OptimizerState { m, v, prev_g, prev_u, slow_u, f, ef, upd, cfgs, layer_lrs, spectral_boost: cfg.spectral_boost, step_count: 0, warmup_steps: cfg.warmup_steps, total_steps: cfg.total_steps, noise_variance: cfg.noise_variance, val_loss_history: Vec::new(), beta1: cfg.beta1, beta2: cfg.beta2, eps: cfg.eps, weight_decay: cfg.weight_decay, bn_layer_boost: cfg.bn_layer_boost, output_layer_damping: cfg.output_layer_damping, prev_val_loss: None, best_val_loss: None, plateau_count: 0, slope_ema: 0.0, lr_boost: 1.0, last_pulse_step: 0, last_epoch: 0, steps_in_epoch: 0, bpe_ema: 1.0, phase_tempo: 1.0, consensus_sign_bias: 0.0, near_floor_mult: 3.0, late_phase_frac: 0.75, plateau_norm_bias: 0.0, nv_lo_mult: 5.0, transition_sign: 0.15, p6: None, nrm1: None, gradient_update_history: Vec::new(), dc_scratch: Vec::new(), trust_partials: None}) as Box<dyn OptimizerStateTrait>)
            })
        }

        fn optimizer_query(_state: &dyn OptimizerStateTrait, _params: &[CudaSlice<f32>], _epoch: usize, _train: Option<f32>, _val: Option<f32>, _stream: Arc<CudaStream>, _module: Arc<CudaModule>, _prop: &cudaDeviceProp) -> Result<Option<Vec<CudaSlice<f32>>>> { Ok(None) }

        fn optimizer_step(state: &mut dyn OptimizerStateTrait, model_params: &[CudaSlice<f32>], gradients: &[CudaSlice<f32>], epoch: usize, _train_loss: Option<f32>, val_loss: Option<f32>, stream: Arc<CudaStream>, module: Arc<CudaModule>, _prop: &cudaDeviceProp) -> Result<Vec<CudaSlice<f32>>> {
            let s = state.as_any_mut().downcast_mut::<OptimizerState>().unwrap();
            update_state_from_val_loss(s, epoch, val_loss);
            let global_damp = compute_global_damp(s, val_loss);
            let t = s.step_count as i32;
            let bias_correction1 = 1.0f32 - s.beta1.powi(t.max(1));
            let bias_correction2 = 1.0f32 - s.beta2.powi(t.max(1));
            let (blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau) = compute_blends(s, val_loss);
            let near_floor = val_loss.map_or(false, |loss| loss <= s.noise_variance * 3.0);
            let late_phase = s.step_count > s.total_steps * 3 / 4;
            let use_robust = s.step_count > s.warmup_steps && (near_floor || late_phase);
            let (in_precision_zone, precision_gain, gate_lo, gate_hi, forward_gain) = compute_precision_params(s, val_loss);
            let beta1_eff: f32 = if in_precision_zone { (s.beta1 + 0.02).min(0.995) } else { s.beta1 };
            let beta2_eff: f32 = s.beta2;
            let eps_eff: f32 = if in_precision_zone { s.eps * 0.9 } else { s.eps };
            let mut wd_eff: f32 = if in_precision_zone { s.weight_decay * 1.05 } else { s.weight_decay };
            if s.step_count > s.warmup_steps {
                if near_floor { wd_eff *= 1.12; } else if s.plateau_count >= 20 { wd_eff *= 0.50; }
            }
            wd_eff *= (1.0 / s.phase_tempo).clamp(0.6, 1.0);
            let trust_backoff: f32 = if let (Some(prev), Some(curr)) = (s.prev_val_loss, val_loss) {
                let delta = curr - prev;
                if delta > 2e-4 { 1.0 / (1.0 + 1.5 * (delta / (prev.abs() + 1e-8)).min(0.02)) } else { 1.0 }
            } else { 1.0 };
            let k_fast = module.load_function("dual_consensus_fisher_kernel_18")?;
            let k_robust = module.load_function("sign_ef_consensus_kernel_18")?;
            let mut updates = Vec::with_capacity(gradients.len());
            for (i, g) in gradients.iter().enumerate() {
                let n = g.len(); if n == 0 { updates.push(stream.alloc_zeros::<f32>(0)?); continue; }
                let base_lr = s.layer_lrs[i];
                let tempo_lr = (1.0 / s.phase_tempo.powf(0.35)).max(0.6);
                let lr = spectral_phase_lr(s, base_lr) * global_damp * s.lr_boost * tempo_lr;
                let layer_multiplier = if i == gradients.len() - 1 {
                    if let Some(loss) = val_loss {
                        if s.step_count > s.warmup_steps + 30 {
                            (s.output_layer_damping * (0.7 + 0.3 * (loss / (s.noise_variance * 6.0)).min(1.0))).max(s.output_layer_damping)
                        } else { s.output_layer_damping }
                    } else { s.output_layer_damping }
                } else if n <= 512 { s.bn_layer_boost } else { 1.0 };
                let effective_lr = lr * layer_multiplier * precision_gain * forward_gain * trust_backoff;
                let rel_update_cap: f32 = { let base: f32 = if near_floor { 0.12 } else { 0.18 }; if i == gradients.len() - 1 { (base * 1.4f32).min(0.32f32) } else { base } };
                let cfg = s.cfgs[i];
                let update_buf_ref = &mut s.upd[i];
                unsafe {
                    if use_robust {
                        stream.launch_builder(&k_robust)
                            .arg(g).arg(&model_params[i]).arg(&mut s.f[i]).arg(&mut s.ef[i])
                            .arg(&mut s.slow_u[i]).arg(update_buf_ref).arg(&(n as u32))
                            .arg(&effective_lr).arg(&eps_eff).arg(&wd_eff).arg(&rel_update_cap)
                            .arg(&lookahead_alpha).arg(&lookahead_tau).arg(&gate_lo).arg(&gate_hi)
                            .launch(cfg)?;
                    } else {
                        stream.launch_builder(&k_fast)
                            .arg(g).arg(&model_params[i]).arg(&mut s.m[i]).arg(&mut s.v[i])
                            .arg(&mut s.prev_g[i]).arg(&mut s.prev_u[i]).arg(&mut s.slow_u[i])
                            .arg(&mut s.f[i]).arg(update_buf_ref).arg(&(n as u32))
                            .arg(&effective_lr).arg(&beta1_eff).arg(&beta2_eff).arg(&eps_eff)
                            .arg(&wd_eff).arg(&bias_correction1).arg(&bias_correction2)
                            .arg(&blend_adam).arg(&blend_norm).arg(&blend_sign).arg(&nesterov_gamma)
                            .arg(&bb_blend).arg(&lookahead_alpha).arg(&lookahead_tau)
                            .arg(&gate_lo).arg(&gate_hi)
                            .launch(cfg)?;
                    }
                }
                updates.push(s.upd[i].clone());
            }
            finalize_state(s, val_loss); Ok(updates)
        }
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
    inner::track_7::solve(challenge, save_solution, hyperparameters, module, stream, prop)
}

