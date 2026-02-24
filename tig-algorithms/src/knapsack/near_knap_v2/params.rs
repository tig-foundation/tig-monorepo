use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const N_IT_CONSTRUCT: usize = 2;
pub const CORE_HALF: usize = 25;

pub const DIFF_LIM: usize = 9;

pub const MICRO_K: usize = 16;
pub const MICRO_RM_K: usize = 8;
pub const MICRO_ADD_K: usize = 8;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Params {
    pub effort: usize,
    pub stall_limit: usize,
    pub perturbation_strength: Option<usize>,
    pub perturbation_rounds: Option<usize>,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            effort: 1,
            stall_limit: 6,
            perturbation_strength: None,
            perturbation_rounds: None,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("effort").and_then(|v| v.as_u64()) {
                p.effort = (v as usize).clamp(1, 6);
            }
            if let Some(v) = m.get("stall_limit").and_then(|v| v.as_u64()) {
                p.stall_limit = (v as usize).clamp(1, 20);
            }
            if let Some(v) = m.get("perturbation_strength").and_then(|v| v.as_u64()) {
                p.perturbation_strength = Some((v as usize).clamp(1, 20));
            }
            if let Some(v) = m.get("perturbation_rounds").and_then(|v| v.as_u64()) {
                p.perturbation_rounds = Some((v as usize).clamp(1, 100));
            }
        }
        p
    }

    pub fn n_perturbation_rounds(&self, n: usize) -> usize {
        if let Some(v) = self.perturbation_rounds {
            return v;
        }
        if n >= 2500 { 15 } else { 15 + (self.effort - 1) * 7 }
    }

    pub fn perturbation_strength_base(&self, n: usize) -> usize {
        if let Some(v) = self.perturbation_strength {
            return v;
        }
        if n >= 2500 { 3 } else if self.effort >= 3 { 4 } else { 3 }
    }

    pub fn vnd_max_iterations(&self, n: usize) -> usize {
        if n >= 4500 { 180 } else if n >= 3000 { 260 } else if n >= 1000 { 300 } else { 80 }
    }

    pub fn n_starts(&self, n: usize, hard: bool, team_est: usize) -> usize {
        let mut base = if n <= 600 {
            if hard { 4 } else { 3 }
        } else if n <= 1500 {
            if hard { 3 } else { 2 }
        } else if n >= 2500 {
            if hard { 4 } else { 3 }
        } else {
            2
        };
        if n <= 1500 && team_est >= 200 {
            base = (base + 1).min(4);
        }
        let bonus = if n >= 2500 {
            0
        } else if self.effort >= 5 {
            2
        } else if self.effort >= 3 {
            1
        } else {
            0
        };
        base + bonus
    }

    pub fn stall_limit_effective(&self) -> usize {
        self.stall_limit
    }
}

#[derive(Clone, Copy)]
pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 { s = 1; }
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}
