use serde_json::{Map, Value};

#[derive(Clone)]
pub(super) struct Cfg {
    pub(super) pstop_z: Option<f32>,
    pub(super) pstop_w: usize,
    pub(super) pstop_s0: usize,
    pub(super) lr_max: f32,
    pub(super) lr_min: f32,
    pub(super) warmup_epochs: Option<usize>,
    pub(super) t_max: usize,
    pub(super) wd: f32,
    pub(super) eps: f32,
    pub(super) beta1: f32,
    pub(super) beta2: f32,
    pub(super) b3: f32,
    pub(super) cautious: f32,
    pub(super) plateau_patience: usize,
    pub(super) plateau_decay: f32,
    pub(super) plateau_grow: f32,
    pub(super) plateau_floor: f32,
    pub(super) head_mult: f32,
    pub(super) kink: f32,
    pub(super) depth_lr: f32,
    pub(super) fuel_reserve: usize,
    pub(super) dc_start: usize,
    pub(super) dc_refresh: usize,
    pub(super) dc_max_abs: f32,
    pub(super) dc_gain: Option<f32>,
    pub(super) ab_blend: f32,
    pub(super) wd_gate: i32,
    pub(super) save_eager: usize,
    pub(super) bn_w: f32,
    pub(super) bn_b: f32,
    pub(super) cool_stale: Option<usize>,
    pub(super) cool_decay: Option<f32>,
    pub(super) cool_floor: f32,
    pub(super) stop_age: Option<usize>,
    pub(super) stop_stale: Option<usize>,
    pub(super) freeze_mask: Option<u64>,
    pub(super) cos_fix: Option<bool>,
    pub(super) orth: Option<bool>,
    pub(super) orth_quintic: usize,
    pub(super) orth_cubic: usize,
    pub(super) orth_scale: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            pstop_z: None,
            pstop_w: 15,
            pstop_s0: 30,
            lr_max: 1.0e-3,
            lr_min: 2e-5,
            warmup_epochs: None,
            t_max: 400,
            wd: 0.02,
            eps: 1e-8,
            beta1: 0.98,
            beta2: 0.92,
            b3: 0.99,
            cautious: 0.25,
            plateau_patience: 12,
            plateau_decay: 0.82,
            plateau_floor: 0.15,
            plateau_grow: 1.03,
            head_mult: 1.0,
            kink: 1.5,
            depth_lr: 1.0,
            fuel_reserve: 55,
            dc_start: 20,
            dc_refresh: 8,
            dc_max_abs: 2.0,
            dc_gain: None,
            ab_blend: 1.0,
            wd_gate: 0,
            save_eager: 1,
            bn_w: 0.55,
            bn_b: 0.80,
            cool_stale: None,
            cool_decay: None,
            cool_floor: 0.0,
            stop_age: None,
            stop_stale: None,
            freeze_mask: None,
            cos_fix: None,
            orth: None,
            orth_quintic: 12,
            orth_cubic: 6,
            orth_scale: 1.0,
        }
    }
}

pub(super) fn parse_cfg(hp: &Option<Map<String, Value>>) -> Cfg {
    let mut c = Cfg::default();
    let Some(m) = hp else { return c };
    c.pstop_z = m.get("pstop_z").and_then(|v| v.as_f64())
        .map(|v| v as f32).filter(|v| v.is_finite() && *v >= 0.0);
    c.pstop_w = m.get("pstop_w").and_then(|v| v.as_u64())
        .filter(|v| (2..=64).contains(v)).map(|v| v as usize).unwrap_or(c.pstop_w);
    c.pstop_s0 = m.get("pstop_s0").and_then(|v| v.as_u64())
        .map(|v| v as usize).unwrap_or(c.pstop_s0);
    let f = |k: &str, d: f32| m.get(k).and_then(|v| v.as_f64())
        .map(|v| v as f32).filter(|v| v.is_finite()).unwrap_or(d);
    let u = |k: &str, d: usize| m.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(d);
    let positive = |k: &str, d: f32| { let v = f(k, d); if v > 0.0 { v } else { d } };
    let nonnegative = |k: &str, d: f32| { let v = f(k, d); if v >= 0.0 { v } else { d } };
    let beta = |k: &str, d: f32| { let v = f(k, d); if (0.0..1.0).contains(&v) { v } else { d } };
    c.lr_max = positive("lr_max", c.lr_max);
    c.lr_min = nonnegative("lr_min", c.lr_min);
    c.warmup_epochs = m.get("warmup_epochs").and_then(|v| v.as_u64()).map(|v| v as usize);
    c.t_max = m.get("t_max").and_then(|v| v.as_u64()).filter(|v| *v > 0)
        .map(|v| v as usize).unwrap_or(c.t_max);
    c.wd = nonnegative("wd", c.wd);
    c.eps = positive("eps", c.eps);
    c.beta1 = beta("beta1", c.beta1);
    c.beta2 = beta("beta2", c.beta2);
    c.b3 = beta("b3", c.b3);
    c.cautious = f("cautious", c.cautious);
    c.plateau_patience = u("plateau_patience", c.plateau_patience);
    c.plateau_decay = f("plateau_decay", c.plateau_decay);
    c.plateau_grow = f("plateau_grow", c.plateau_grow);
    c.plateau_floor = f("plateau_floor", c.plateau_floor);
    c.head_mult = f("head_mult", c.head_mult);
    c.kink = f("kink", c.kink);
    c.depth_lr = f("depth_lr", c.depth_lr);
    c.fuel_reserve = u("fuel_reserve", c.fuel_reserve);
    c.dc_start = u("dc_start", c.dc_start);
    c.dc_refresh = m.get("dc_refresh").and_then(|v| v.as_u64()).filter(|v| *v > 0)
        .map(|v| v as usize).unwrap_or(c.dc_refresh);
    c.dc_max_abs = positive("dc_max_abs", c.dc_max_abs);
    c.dc_gain = m.get("dc_gain").and_then(|v| v.as_f64())
        .map(|v| v as f32).filter(|v| v.is_finite());
    c.ab_blend = f("ab_blend", c.ab_blend);
    c.wd_gate = m.get("wd_gate").and_then(|v| v.as_u64()).filter(|v| *v <= 1)
        .map(|v| v as i32).unwrap_or(c.wd_gate);
    c.save_eager = u("save_eager", c.save_eager);
    c.bn_w = f("bn_w", c.bn_w);
    c.bn_b = f("bn_b", c.bn_b);
    c.cool_stale = m.get("cool_stale").and_then(|v| v.as_u64()).map(|v| v as usize);
    c.cool_decay = m.get("cool_decay").and_then(|v| v.as_f64())
        .map(|v| v as f32).filter(|v| v.is_finite());
    c.cool_floor = f("cool_floor", c.cool_floor);
    c.stop_age = m.get("stop_age").and_then(|v| v.as_u64()).map(|v| v as usize);
    c.stop_stale = m.get("stop_stale").and_then(|v| v.as_u64()).map(|v| v as usize);
    c.freeze_mask = m.get("freeze_mask").and_then(|v| v.as_u64());
    c.cos_fix = m.get("cos_fix").and_then(|v| v.as_u64()).filter(|v| *v <= 1).map(|v| v == 1);
    c.orth = m.get("orth").and_then(|v| v.as_u64()).filter(|v| *v <= 1).map(|v| v == 1);
    c.orth_quintic = m.get("orth_quintic").and_then(|v| v.as_u64()).filter(|v| *v <= 32)
        .map(|v| v as usize).unwrap_or(c.orth_quintic);
    c.orth_cubic = m.get("orth_cubic").and_then(|v| v.as_u64()).filter(|v| *v <= 32)
        .map(|v| v as usize).unwrap_or(c.orth_cubic);
    let scale = f("orth_scale", c.orth_scale);
    c.orth_scale = if scale > 0.0 { scale } else { c.orth_scale };
    c
}

pub(super) struct DepthDefaults {
    pub(super) warmup_epochs: usize,
    pub(super) dc_gain: f32,
    pub(super) pstop_z: f32,
    pub(super) orth: bool,
    pub(super) freeze_mask: u64,
    pub(super) cos_fix: bool,
    pub(super) cool_stale: usize,
    pub(super) cool_decay: f32,
    pub(super) stop_age: usize,
    pub(super) stop_stale: usize,
}

pub(super) fn depth_defaults(n_hidden: usize, warmup_explicit: Option<usize>) -> DepthDefaults {
    let base = DepthDefaults {
        warmup_epochs: if n_hidden == 4 { 4 } else { 8 },
        dc_gain: match n_hidden {
            4 | 14 => 1.0,
            18 => 1.1,
            _ => 1.2,
        },
        pstop_z: 0.0,
        orth: false,
        freeze_mask: 0, cos_fix: false, cool_stale: 15, cool_decay: 0.7,
        stop_age: 0, stop_stale: 0,
    };
    match n_hidden {
        4 => base,
        7 => DepthDefaults { pstop_z: 8.0, freeze_mask: 4, cos_fix: true, ..base },
        10 | 14 => DepthDefaults { pstop_z: 8.0, freeze_mask: 4, stop_age: 25, stop_stale: 45, ..base },
        18 if warmup_explicit.map_or(false, |w| w >= 20) => DepthDefaults {
            pstop_z: 8.0,
            orth: true, freeze_mask: 4, cool_stale: 35, cool_decay: 1.0,
            stop_stale: 35, stop_age: 0, ..base
        },
        _ => base,
    }
}

pub(super) fn help() {
    println!("parallax_proxima: neural network optimizer for the neuralnet_optimizer challenge.");
    println!("Deterministic: no thread, no clock, no filesystem access; the only pseudo-random draws");
    println!("are seeded by the challenge seed.");
    println!("Defaults are the intended operating point: {{}} selects them. Every key below overrides its");
    println!("default, and a value outside its domain (checked after conversion to the internal type)");
    println!("falls back to that default.");
    println!("gp_mode (0 or 1, default 1): selects the default method (1) or the alternative optimizer (0).");
    println!("Keys used when gp_mode is 1:");
    println!("  integers: gp_steps 1..=20000, gp_n 64..=8192, gp_na even in 2..=4096, gp_m 8..=256, gp_warm >= 0");
    println!("  reals > 0: gp_lr, gp_eps, gp_ell, gp_var, gp_noise");
    println!("  reals >= 0: gp_lr_min, gp_kink, gp_blr, gp_bnlr");
    println!("  reals in [0, 1): gp_b1, gp_b2");
    println!("  reals in [1, f32::MAX]: gp_alo, gp_ahi (raised to gp_alo when below it); real >= 1.05: gp_lb");
    println!("  flags (0 or 1): gp_val, gp_save0");
    println!("Keys used by the alternative optimizer (gp_mode 0, also the fallback of gp_mode 1):");
    println!("  integers >= 0: warmup_epochs, plateau_patience, fuel_reserve, dc_start, save_eager,");
    println!("    cool_stale, stop_age, stop_stale, freeze_mask, pstop_s0");
    println!("  integers > 0: t_max, dc_refresh");
    println!("  integers in ranges: pstop_w 2..=64, orth_quintic 0..=32, orth_cubic 0..=32");
    println!("  reals > 0: lr_max, eps, dc_max_abs, orth_scale");
    println!("  reals >= 0: lr_min, wd, pstop_z");
    println!("  reals in [0, 1): beta1, beta2, b3");
    println!("  finite reals: cautious, plateau_decay, plateau_grow, plateau_floor, head_mult, kink,");
    println!("    depth_lr, dc_gain, ab_blend, bn_w, bn_b, cool_decay, cool_floor");
    println!("  flags (0 or 1): wd_gate, cos_fix, orth");
}
