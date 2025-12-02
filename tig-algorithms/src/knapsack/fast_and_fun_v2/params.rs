use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Params {
    /// Max |Δw| for 1–1 swap neighborhoods
    pub diff_lim: usize,

    /// Window size (best-unused & worst-used)
    pub core_half: usize,

    /// Max DP+LS iterations in the ILS loop
    pub n_maxils: usize,

    /// Stage-2 construction iterations
    pub n_it_construct: usize,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            diff_lim: 4,
            core_half: 30,
            n_maxils: 3,
            n_it_construct: 2,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("diff_lim").and_then(|v| v.as_u64()) { p.diff_lim = v as usize; }
            if let Some(v) = m.get("core_half").and_then(|v| v.as_u64()) { p.core_half = v as usize; }
            if let Some(v) = m.get("n_maxils").and_then(|v| v.as_u64()) { p.n_maxils = v as usize; }
            if let Some(v) = m.get("n_it_construct").and_then(|v| v.as_u64()) { p.n_it_construct = v as usize; }
        }
        p
    }
}
