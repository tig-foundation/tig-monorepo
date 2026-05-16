use super::Hyperparameters;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HwProfile {
    Zen4,
    Zen5,
    Zen5c,
    GenericAvx512,
    Generic,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileParams {
    pub(crate) base_prob: f64,
    pub(crate) max_prob: f64,
    pub(crate) make_mult: i32,
    pub(crate) break_mult: i32,
    pub(crate) use_make_score: bool,
    pub(crate) use_clause_weights: bool,
    pub(crate) weight_update_interval: usize,
    pub(crate) max_clause_weight: u16,
    pub(crate) clause_pick_samples: usize,
    pub(crate) check_interval: usize,
    pub(crate) stagnation_limit: usize,
    pub(crate) perturbation_flips: usize,
    pub(crate) phase_restart_prob: f64,
    pub(crate) phase_noise_divisor: usize,
    pub(crate) age_shift: u32,
    pub(crate) age_cap: i32,
}

pub(crate) fn select_hw_profile(hp: &Hyperparameters) -> HwProfile {
    if let Some(s) = hp.hw_profile.as_deref() {
        match s {
            "zen4" => return HwProfile::Zen4,
            "zen5" => return HwProfile::Zen5,
            "zen5c" => return HwProfile::Zen5c,
            "generic_avx512" => return HwProfile::GenericAvx512,
            "generic" => return HwProfile::Generic,
            "auto" | "" => {}
            _ => {}
        }
    }

    let brand = cpu_brand_string().to_ascii_lowercase();
    if brand.contains("9654") {
        HwProfile::Zen4
    } else if brand.contains("9755") || brand.contains("100-000001535") {
        HwProfile::Zen5
    } else if brand.contains("9845") {
        HwProfile::Zen5c
    } else if has_avx512_core() {
        HwProfile::GenericAvx512
    } else {
        HwProfile::Generic
    }
}

pub(crate) fn profile_params(
    profile: HwProfile,
    nv: usize,
    nc: usize,
    hp: &Hyperparameters,
) -> ProfileParams {
    let large = nv >= 30_000 || nc >= 100_000;
    let density = nc as f64 / nv.max(1) as f64;
    let hard_small = !large && density >= 4.24;
    let mut p = match profile {
        HwProfile::Zen4 => ProfileParams {
            base_prob: if large { 0.010 } else { 0.020 },
            max_prob: 0.080,
            make_mult: 8,
            break_mult: 4,
            use_make_score: true,
            use_clause_weights: true,
            weight_update_interval: if large { 8192 } else { 4096 },
            max_clause_weight: 96,
            clause_pick_samples: 1,
            check_interval: if large { 50_000 } else { 10_000 },
            stagnation_limit: 8,
            perturbation_flips: if large { 256 } else { 64 },
            phase_restart_prob: 0.25,
            phase_noise_divisor: 100,
            age_shift: 12,
            age_cap: 64,
        },
        HwProfile::Zen5 => ProfileParams {
            base_prob: if large { 0.012 } else { 0.025 },
            max_prob: 0.100,
            make_mult: 10,
            break_mult: 4,
            use_make_score: true,
            use_clause_weights: true,
            weight_update_interval: if large { 8192 } else { 2048 },
            max_clause_weight: 128,
            clause_pick_samples: if hard_small { 2 } else { 1 },
            check_interval: if large { 50_000 } else { 10_000 },
            stagnation_limit: 8,
            perturbation_flips: if large { 384 } else { 96 },
            phase_restart_prob: 0.35,
            phase_noise_divisor: 80,
            age_shift: 12,
            age_cap: 64,
        },
        HwProfile::Zen5c => ProfileParams {
            base_prob: if large { 0.008 } else { 0.018 },
            max_prob: 0.080,
            make_mult: 8,
            break_mult: 5,
            use_make_score: true,
            use_clause_weights: true,
            weight_update_interval: if large { 16384 } else { 4096 },
            max_clause_weight: 96,
            clause_pick_samples: 1,
            check_interval: if large { 40_000 } else { 10_000 },
            stagnation_limit: 6,
            perturbation_flips: if large { 192 } else { 48 },
            phase_restart_prob: 0.20,
            phase_noise_divisor: 120,
            age_shift: 12,
            age_cap: 64,
        },
        HwProfile::GenericAvx512 => ProfileParams {
            base_prob: if large { 0.010 } else { 0.020 },
            max_prob: 0.080,
            make_mult: 8,
            break_mult: 4,
            use_make_score: true,
            use_clause_weights: true,
            weight_update_interval: if large { 8192 } else { 4096 },
            max_clause_weight: 96,
            clause_pick_samples: 1,
            check_interval: if large { 50_000 } else { 10_000 },
            stagnation_limit: 8,
            perturbation_flips: if large { 256 } else { 64 },
            phase_restart_prob: 0.25,
            phase_noise_divisor: 100,
            age_shift: 12,
            age_cap: 64,
        },
        HwProfile::Generic => ProfileParams {
            base_prob: 0.015,
            max_prob: 0.080,
            make_mult: 8,
            break_mult: 4,
            use_make_score: true,
            use_clause_weights: true,
            weight_update_interval: if large { 8192 } else { 4096 },
            max_clause_weight: 96,
            clause_pick_samples: 1,
            check_interval: if large { 50_000 } else { 10_000 },
            stagnation_limit: 8,
            perturbation_flips: if large { 256 } else { 64 },
            phase_restart_prob: 0.25,
            phase_noise_divisor: 100,
            age_shift: 12,
            age_cap: 64,
        },
    };

    if let Some(x) = hp.base_prob {
        p.base_prob = x;
    }
    if let Some(x) = hp.max_prob {
        p.max_prob = x;
    }
    if let Some(x) = hp.make_mult {
        p.make_mult = x.max(0);
    }
    if let Some(x) = hp.break_mult {
        p.break_mult = x.max(0);
    }
    if hp.disable_make_score.unwrap_or(false) {
        p.use_make_score = false;
    }
    if hp.disable_clause_weights.unwrap_or(false) {
        p.use_clause_weights = false;
    }
    if let Some(x) = hp.weight_update_interval {
        p.weight_update_interval = x;
    }
    if let Some(x) = hp.max_clause_weight {
        p.max_clause_weight = x.max(1);
    }
    if let Some(x) = hp.check_interval {
        p.check_interval = x.max(1);
    }
    if hard_small {
        p.clause_pick_samples = 2;
        p.stagnation_limit = 1_000_000_000;
        p.perturbation_flips = 0;
    }
    if let Some(x) = hp.clause_pick_samples {
        p.clause_pick_samples = x.clamp(1, 16);
    }
    if let Some(x) = hp.stagnation_limit {
        p.stagnation_limit = x.max(1);
    }
    if let Some(x) = hp.perturbation_flips {
        p.perturbation_flips = x;
    }
    if let Some(x) = hp.phase_restart_prob {
        p.phase_restart_prob = x.clamp(0.0, 1.0);
    }
    if let Some(x) = hp.phase_noise_divisor {
        p.phase_noise_divisor = x.max(1);
    }
    if let Some(x) = hp.age_shift {
        p.age_shift = x.min(31);
    }
    if let Some(x) = hp.age_cap {
        p.age_cap = x.max(0);
    }
    p
}

#[cfg(target_arch = "x86_64")]
fn has_avx512_core() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
fn has_avx512_core() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn cpu_brand_string() -> String {
    use std::arch::x86_64::__cpuid;
    unsafe {
        let max_ext = __cpuid(0x8000_0000).eax;
        if max_ext < 0x8000_0004 {
            return String::new();
        }
        let mut bytes = Vec::with_capacity(48);
        for leaf in 0x8000_0002..=0x8000_0004 {
            let r = __cpuid(leaf);
            for x in [r.eax, r.ebx, r.ecx, r.edx] {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        String::from_utf8_lossy(&bytes)
            .trim_matches(char::from(0))
            .trim()
            .to_string()
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand_string() -> String {
    String::new()
}
