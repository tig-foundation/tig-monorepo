use serde::{Serialize, Deserialize};
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverStatus {
    Ok,
    Error,
    Panic,
    Repaired,
}

impl fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverStatus::Ok => write!(f, "ok"),
            SolverStatus::Error => write!(f, "error"),
            SolverStatus::Panic => write!(f, "panic"),
            SolverStatus::Repaired => write!(f, "repaired"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonceResult {
    pub nonce_idx: usize,
    pub seed: u64,
    pub solver_status: SolverStatus,
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_errors: Option<Vec<String>>,
    pub score: f64,
    pub total_cost: i64,
    pub fuel_assigned: u64,
    pub fuel_consumed: u64,
    pub elapsed_ms: u128,
    pub solution_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleStats {
    pub mean_score: f64,
    pub median_score: f64,
    pub valid_count: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BundleStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleResult {
    pub track_id: String,
    pub bundle_idx: usize,
    pub nonce_results: Vec<NonceResult>,
    pub stats: BundleStats,
    pub bundle_status: BundleStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_reasons: Option<Vec<String>>,
}

impl BundleResult {
    pub fn compute_stats(nonces: &[NonceResult]) -> BundleStats {
        let total = nonces.len();
        let mut scores: Vec<f64> = nonces.iter().map(|n| n.score).collect();
        scores.sort_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mean = if scores.is_empty() { 0.0 } else { scores.iter().sum::<f64>() / (scores.len() as f64) };
        let median = if scores.is_empty() { 0.0 } else if scores.len() % 2 == 1 {
            scores[scores.len()/2]
        } else {
            let hi = scores.len()/2;
            (scores[hi-1] + scores[hi]) / 2.0
        };
        let valid_count = nonces.iter().filter(|n| n.valid).count();
        BundleStats { mean_score: mean, median_score: median, valid_count, total }
    }
}
