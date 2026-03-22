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

/// Intra-epoch LR modulation: first batches explore, last batches consolidate
#[inline]
pub fn intra_epoch_factor(s: &OptimizerState) -> f32 {
    let bpe = s.bpe_ema.max(1.0);
    let pos = (s.steps_in_epoch as f32) / bpe;
    0.96 + 0.08 * (std::f32::consts::PI * pos).cos()
}

/// Loss curvature: if improvement is decelerating, push harder
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
