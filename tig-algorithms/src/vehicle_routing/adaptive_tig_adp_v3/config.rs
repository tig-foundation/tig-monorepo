// Phase 2: Unified solver configuration system
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use anyhow::{Context, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    // Learning parameters
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub flexibility_weight: f64,
    pub penalty_weight: f64,
    pub learning_decay: bool,

    // Rollout parameters
    pub rollout_depth: usize,
    pub rollout_fallback: f64,

    // Search parameters
    pub local_search_time_ms: u64,
    pub neighborhood_size: usize,
    pub izs_threshold: i64,
    pub beam_width: usize,

    // Runtime parameters
    pub max_iterations: usize,
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            // Learning defaults
            learning_rate: 0.1,
            discount_factor: 0.9,
            flexibility_weight: 5.0,
            penalty_weight: 100.0,
            learning_decay: true,

            // Rollout defaults
            rollout_depth: 10,
            rollout_fallback: 0.0,

            // Search defaults
            local_search_time_ms: 20,
            neighborhood_size: 8,
            izs_threshold: 1000,
            beam_width: 5,

            // Runtime defaults
            max_iterations: 100,
            verbose: false,
        }
    }
}

impl SolverConfig {
    /// Fast configuration preset
    pub fn fast() -> Self {
        Self {
            local_search_time_ms: 10,
            neighborhood_size: 5,
            rollout_depth: 5,
            max_iterations: 50,
            ..Default::default()
        }
    }

    /// Quality configuration preset
    pub fn quality() -> Self {
        Self {
            local_search_time_ms: 50,
            neighborhood_size: 12,
            rollout_depth: 20,
            max_iterations: 500,
            beam_width: 10,
            ..Default::default()
        }
    }

    /// Experimental configuration preset
    pub fn experimental() -> Self {
        Self {
            learning_rate: 0.05,
            discount_factor: 0.95,
            learning_decay: true,
            rollout_depth: 15,
            max_iterations: 1000,
            verbose: true,
            ..Default::default()
        }
    }
}

/// Sigma II runner configuration (separate from solver `SolverConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaConfig {
    #[serde(default = "SigmaConfig::default_num_bundles")]
    pub num_bundles: usize,

    #[serde(default)]
    pub selected_track_ids: Vec<String>,

    #[serde(default)]
    pub hyperparameters: BTreeMap<String, serde_json::Value>,

    #[serde(default = "SigmaConfig::default_runtime")]
    pub runtime_config: RuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "RuntimeConfig::default_max_fuel")]
    pub max_fuel: u64,
}

impl SigmaConfig {
    fn default_num_bundles() -> usize { 1 }

    fn default_runtime() -> RuntimeConfig { RuntimeConfig { max_fuel: 1_000_000 } }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path).with_context(|| format!("Failed to read config: {}", path))?;
        let cfg: SigmaConfig = serde_json::from_str(&content).with_context(|| format!("Invalid JSON config: {}", path))?;
        Ok(cfg)
    }

    /// Validate that selected tracks exist in provided track mapping.
    pub fn validate_tracks(&self, tracks: &BTreeMap<String, TrackConfig>) -> Result<()> {
        for t in &self.selected_track_ids {
            anyhow::ensure!(tracks.contains_key(t), "Selected track id not found: {}", t);
        }
        Ok(())
    }
}

impl Default for SigmaConfig {
    fn default() -> Self {
        Self { num_bundles: 1, selected_track_ids: Vec::new(), hyperparameters: BTreeMap::new(), runtime_config: RuntimeConfig::default() }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self { RuntimeConfig { max_fuel: RuntimeConfig::default_max_fuel() } }
}

impl RuntimeConfig {
    fn default_max_fuel() -> u64 { 1_000_000 }
}

/// Simple track config loaded from `tracks.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackConfig {
    pub id: String,
    pub num_customers: usize,
    #[serde(default)]
    pub description: Option<String>,
}

impl TrackConfig {
    pub fn load_tracks(path: &str) -> Result<BTreeMap<String, TrackConfig>> {
        let content = fs::read_to_string(path).with_context(|| format!("Failed to read tracks file: {}", path))?;
        let v: Vec<TrackConfig> = serde_json::from_str(&content).with_context(|| format!("Failed to parse tracks.json: {}", path))?;
        let mut m: BTreeMap<String, TrackConfig> = BTreeMap::new();
        for t in v { m.insert(t.id.clone(), t); }
        Ok(m)
    }
}
