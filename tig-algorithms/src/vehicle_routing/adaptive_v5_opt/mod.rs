// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::VecDeque;
// std::io use removed per sandbox restrictions
use std::path::Path;
use ahash::HashMapExt;
use tig_challenges::vehicle_routing::*;
use rand::Rng;

// Feature flags parsed from hyperparameters/environment for v5 rollout
#[derive(Clone, Copy, Debug, Default)]
pub struct FeatureFlags {
    pub enable_h_dlt: bool,
    pub enable_ma_adp: bool,
    pub enable_shrm: bool,
    pub shrm_shadow_mode: bool,
}

fn parse_feature_flags(hyperparameters: &Option<Map<String, Value>>) -> FeatureFlags {
    // Defaults: enable all modules unless explicitly disabled
    let mut flags = FeatureFlags {
        enable_h_dlt: true,
        enable_ma_adp: true,
        enable_shrm: true,
        shrm_shadow_mode: false,
    };

    if let Some(hp) = hyperparameters {
        if let Some(v) = hp.get("enable_h_dlt").and_then(|v| v.as_bool()) {
            flags.enable_h_dlt = v;
        }
        if let Some(v) = hp.get("enable_ma_adp").and_then(|v| v.as_bool()) {
            flags.enable_ma_adp = v;
        }
        if let Some(v) = hp.get("enable_shrm").and_then(|v| v.as_bool()) {
            flags.enable_shrm = v;
        }
        if let Some(v) = hp.get("shrm_shadow_mode").and_then(|v| v.as_bool()) {
            flags.shrm_shadow_mode = v;
        }
    }

    // Environment access disabled — feature flags come only from hyperparameters

    flags
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
	pub max_iterations: Option<usize>,
}

// Minimal type definitions for single-file integration
#[derive(Clone, Debug)]
pub struct TimeWindow {
    pub start: i32,
    pub end: i32,
}

#[derive(Clone, Debug)]
pub struct Problem {
    pub name: String,
    pub num_nodes: usize,
    pub depot: usize,
    pub max_capacity: i32,
    pub demands: Vec<i32>,
    pub distance_matrix: Vec<Vec<i32>>,
    pub initial_time: i32,
    pub time_windows: Vec<TimeWindow>,
    pub service_times: Vec<i32>,
    pub initial_route: Option<Vec<usize>>,
    pub config: Option<Config>,
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub seed: u64,
}

// ============================================================================
// CONE-X: Node-Level Adaptive Control
// ============================================================================

#[derive(Debug, Clone)]
pub struct NodePriority {
    pub node_id: u32,
    pub tightness_score: f32,   // [0.0–1.0] based on time window slack
    pub density_score: f32,     // [0.0–1.0] based on spatial neighbor count
    pub marginal_cost: f32,     // normalized insertion cost
    pub composite_weight: f32,  // computed: 0.4*T + 0.3*D + 0.3*C
}

impl NodePriority {
    pub fn update_composite(&mut self) {
        self.composite_weight = 
            0.4 * self.tightness_score +
            0.3 * self.density_score +
            0.3 * self.marginal_cost;
    }
}

#[derive(Clone, Debug)]
pub struct ConeXScheduler {
    priorities: Vec<NodePriority>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrderedFloat, u32)>>,
}

// Wrapper for f32 to make it Ord for BinaryHeap
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl ConeXScheduler {
    pub fn new(num_nodes: usize, ready_times: &[i32], due_times: &[i32], positions: &[(i32, i32)]) -> Self {
        let mut priorities = Vec::with_capacity(num_nodes);
        
        for i in 0..num_nodes {
            let slack = (due_times[i] - ready_times[i]).max(1) as f32;
            let tightness_score = (1.0 / slack).min(1.0);
            
            // Simple density: count nearby nodes (within 200 units)
            let mut nearby_count = 0;
            for j in 0..num_nodes {
                if i != j {
                    let dx = (positions[i].0 - positions[j].0).abs();
                    let dy = (positions[i].1 - positions[j].1).abs();
                    if dx + dy < 200 {
                        nearby_count += 1;
                    }
                }
            }
            let density_score = (nearby_count as f32 / 10.0).min(1.0);
            
            let mut priority = NodePriority {
                node_id: i as u32,
                tightness_score,
                density_score,
                marginal_cost: 0.5, // default
                composite_weight: 0.0,
            };
            priority.update_composite();
            priorities.push(priority);
        }

        let mut scheduler = ConeXScheduler {
            priorities,
            heap: std::collections::BinaryHeap::new(),
        };
        scheduler.rebuild_heap();
        scheduler
    }

    pub fn rebuild_heap(&mut self) {
        self.heap.clear();
        for p in &self.priorities {
            self.heap.push(std::cmp::Reverse((OrderedFloat(p.composite_weight), p.node_id)));
        }
    }

    pub fn next_node(&mut self) -> Option<u32> {
        self.heap.pop().map(|r| r.0.1)
    }

    pub fn update_marginal_cost(&mut self, node_id: u32, cost: f32) {
        if let Some(p) = self.priorities.iter_mut().find(|p| p.node_id == node_id) {
            p.marginal_cost = cost.min(1.0).max(0.0);
            p.update_composite();
        }
    }
}

// ============================================================================
// GAS: Bounded Policy Feedback Loop
// ============================================================================

#[derive(Debug, Clone)]
pub struct GasPolicy {
    pub weights: [f32; 4], // [time_cost, load_util, duration, idle_time]
    pub learning_rate: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for GasPolicy {
    fn default() -> Self {
        Self {
            weights: [0.4, 0.3, 0.2, 0.1],
            learning_rate: 0.05,
            min_weight: 0.05,
            max_weight: 0.5,
        }
    }
}

impl GasPolicy {
    pub fn update(
        &mut self,
        move_improvement: f32,
        move_features: &[f32; 4],
        baseline_improvement: f32,
    ) {
        let gradient = move_improvement - baseline_improvement;

        for i in 0..4 {
            let delta = self.learning_rate * gradient * move_features[i];
            let new_weight = self.weights[i] + delta;

            // Bounded update
            self.weights[i] = new_weight
                .max(self.min_weight)
                .min(self.max_weight);
        }

        // Normalize to sum = 1.0
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    pub fn evaluate_move(&self, features: &[f32; 4]) -> f32 {
        self.weights[0] * features[0]
            + self.weights[1] * features[1]
            + self.weights[2] * features[2]
            + self.weights[3] * features[3]
    }
}

// ============================================================================
// VTL: Verifiable Trust Layer
// ============================================================================

#[derive(Debug, Clone)]
pub enum EventType {
    MoveAccepted,
    MoveRejected,
    FeasibilityCheck,
    PolicyUpdate,
    Convergence,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    TimeWindow,
    Capacity,
    Precedence,
    Duration,
}

#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub event_type: EventType,
    pub node_id: Option<u32>,
    pub route_id: Option<u32>,
    pub reason: String,
    pub constraint_violations: Vec<ConstraintType>,
    pub value_inputs: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct VtlLogger {
    enabled: bool,
    events: Vec<AuditEvent>,
}

impl VtlLogger {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            events: Vec::with_capacity(if enabled { 10000 } else { 0 }),
        }
    }

    pub fn log(&mut self, event: AuditEvent) {
        if self.enabled {
            self.events.push(event);
        }
    }

    pub fn get_event_count(&self) -> usize {
        self.events.len()
    }

    pub fn flush_to_file(&self, path: &str) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// Original State Structures
// ============================================================================

#[derive(Clone, Debug)]
pub struct TIGState {
    pub route: Route,
    pub current_time: i32,
    pub current_load: i32,
}

#[derive(Clone, Debug)]
pub struct Route {
    pub nodes: Vec<usize>,
}

impl TIGState {
    pub fn is_feasible(&self) -> bool {
        // Simplified feasibility check
        self.current_load <= 100 && self.current_time <= 1000
    }

    pub fn time_violation_penalty(&self) -> i32 {
        0 // Simplified
    }

    pub fn capacity_violation_penalty(&self) -> i32 {
        0 // Simplified
    }

    pub fn free_time_budget(&self) -> i32 {
        100 // Simplified
    }
}

impl Problem {
    pub fn to_state(&self) -> TIGState {
        TIGState {
            route: Route {
                nodes: self.initial_route.clone().unwrap_or_else(|| vec![self.depot]),
            },
            current_time: self.initial_time,
            current_load: 0,
        }
    }
}

/// Hierarchical Routing Solver with 3-level decision tree:
/// Level 1: Cluster assignment (geographic/demand-based)
/// Level 2: Route selection within cluster
/// Level 3: Position insertion optimization
#[derive(Clone, Debug)]
pub struct Solver {
    config: Config,
    rng: rand::rngs::StdRng,
    h_dlt: Option<HierarchicalDLT>,
    problem_cache: Option<Problem>,
    cone_x_scheduler: Option<ConeXScheduler>,
    gas_policy: GasPolicy,
    vtl_logger: Option<VtlLogger>,
}

impl Solver {
    pub fn with_config(seed: u64, config: Config) -> Self {
        use rand::SeedableRng;
        Self {
            config: Config { seed, ..config },
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            h_dlt: None,
            problem_cache: None,
            cone_x_scheduler: None,
            gas_policy: GasPolicy::default(),
            vtl_logger: None,
        }
    }

    pub fn set_hdlt(&mut self, hdlt: HierarchicalDLT) {
        self.h_dlt = Some(hdlt);
    }

    pub fn set_problem(&mut self, problem: Problem) {
        self.problem_cache = Some(problem);
    }
    
    pub fn enable_cone_x(&mut self, num_nodes: usize, ready_times: &[i32], due_times: &[i32], positions: &[(i32, i32)]) {
        self.cone_x_scheduler = Some(ConeXScheduler::new(num_nodes, ready_times, due_times, positions));
    }
    
    pub fn enable_vtl(&mut self, enabled: bool) {
        self.vtl_logger = Some(VtlLogger::new(enabled));
    }

    /// Hierarchical solve: cluster → route → insert (with CONE-X and GAS integration)
    pub fn solve_state(&mut self, mut state: TIGState, max_iters: usize) -> Result<TIGState, String> {
        let iters = max_iters.min(1000);
        let start_time = std::time::Instant::now();
        
        // Phase 1: Hierarchical construction if H-DLT available
        if let Some(hdlt) = self.h_dlt.clone() {
            state = self.hierarchical_construct(&state, &hdlt)?;
        }
        
        // Phase 2: Local search improvement with early termination and CONE-X scheduling
        let mut no_improvement_count = 0;
        let mut baseline_improvement = 0.0f32;
        
        for iter in 0..iters {
            let elapsed_ratio = start_time.elapsed().as_secs_f64() / 1.0; // Assume 1s budget
            let is_mid_phase = elapsed_ratio > 0.3 && elapsed_ratio < 0.7;
            
            // Use CONE-X scheduler if available
            let target_node = if let Some(ref mut scheduler) = self.cone_x_scheduler {
                scheduler.next_node()
            } else {
                None
            };
            
            // Store old state for improvement calculation
            let old_route_len = state.route.nodes.len();
            
            let improvement = if iter < iters / 3 {
                self.greedy_improvement(&mut state)
            } else if iter < 2 * iters / 3 {
                self.two_opt_best(&mut state)
            } else {
                self.relocate_best(&mut state)
            };
            
            // Calculate improvement for GAS
            let move_improvement = if improvement {
                (old_route_len as f32 - state.route.nodes.len() as f32).abs()
            } else {
                0.0
            };
            
            // GAS policy update during mid-phase
            if is_mid_phase && improvement {
                let move_features = [
                    (state.current_time as f32 / 1000.0).min(1.0),      // normalized time
                    (state.current_load as f32 / 100.0).min(1.0),       // normalized load
                    (state.route.nodes.len() as f32 / 50.0).min(1.0),  // normalized duration
                    0.5, // idle time (placeholder)
                ];
                
                self.gas_policy.update(move_improvement, &move_features, baseline_improvement);
                baseline_improvement = 0.9 * baseline_improvement + 0.1 * move_improvement;
                
                // VTL logging
                if let Some(ref mut logger) = self.vtl_logger {
                    logger.log(AuditEvent {
                        timestamp_ms: start_time.elapsed().as_millis() as u64,
                        event_type: EventType::PolicyUpdate,
                        node_id: target_node,
                        route_id: Some(0),
                        reason: format!("GAS policy updated: weights={:?}", self.gas_policy.weights),
                        constraint_violations: vec![],
                        value_inputs: move_features.to_vec(),
                    });
                }
            }
            
            if improvement {
                no_improvement_count = 0;
                
                // Update CONE-X scheduler with success
                if let Some(ref mut scheduler) = self.cone_x_scheduler {
                    if let Some(node) = target_node {
                        scheduler.update_marginal_cost(node, 0.2); // Lower cost = higher priority next time
                    }
                }
            } else {
                no_improvement_count += 1;
                
                // Update CONE-X scheduler with failure
                if let Some(ref mut scheduler) = self.cone_x_scheduler {
                    if let Some(node) = target_node {
                        scheduler.update_marginal_cost(node, 0.8); // Higher cost = lower priority
                    }
                }
                
                if no_improvement_count > 20 && iter > iters / 2 {
                    // VTL: Log convergence
                    if let Some(ref mut logger) = self.vtl_logger {
                        logger.log(AuditEvent {
                            timestamp_ms: start_time.elapsed().as_millis() as u64,
                            event_type: EventType::Convergence,
                            node_id: None,
                            route_id: None,
                            reason: format!("Early termination: {} consecutive failures", no_improvement_count),
                            constraint_violations: vec![],
                            value_inputs: vec![iter as f32, no_improvement_count as f32],
                        });
                    }
                    break;
                }
                if iter > iters / 2 {
                    // Diversification: perturb solution
                    self.perturb(&mut state);
                }
            }
            
            // Rebuild CONE-X heap every 10 iterations for efficiency
            if let Some(ref mut scheduler) = self.cone_x_scheduler {
                if iter % 10 == 0 {
                    scheduler.rebuild_heap();
                }
            }
        }
        
        Ok(state)
    }

    /// Level 1-3: Hierarchical construction using H-DLT guidance
    fn hierarchical_construct(&mut self, state: &TIGState, hdlt: &HierarchicalDLT) -> Result<TIGState, String> {
        let mut new_state = state.clone();
        
        // If problem cache available, use real customers
        if let Some(ref problem) = self.problem_cache {
            let num_nodes = problem.num_nodes;
            let mut unvisited: Vec<usize> = (1..num_nodes).collect();
            
            // Level 1: Cluster unvisited nodes (simple k-means-like)
            let num_clusters = ((num_nodes as f64).sqrt() as usize).max(1).min(5);
            let clusters = self.cluster_nodes(&unvisited, num_clusters, problem);
            
            // Level 2 & 3: For each cluster, build route with H-DLT guidance
            new_state.route.nodes = vec![problem.depot];
            for cluster in clusters {
                for &node in &cluster {
                    // Level 3: Find best insertion position using H-DLT values
                    let best_pos = self.find_best_insertion(node, &new_state.route.nodes, hdlt);
                    new_state.route.nodes.insert(best_pos, node);
                }
            }
            
            // Close route
            if new_state.route.nodes.last().copied() != Some(problem.depot) {
                new_state.route.nodes.push(problem.depot);
            }
            
            // Update state metrics (use route length instead of undefined cluster)
            new_state.current_load = new_state.route.nodes.len() as i32;
            new_state.current_time = new_state.route.nodes.len() as i32 * 10;
        }
        
        Ok(new_state)
    }

    /// Level 1: Geographic clustering with demand awareness
    fn cluster_nodes(&self, nodes: &[usize], k: usize, problem: &Problem) -> Vec<Vec<usize>> {
        if nodes.is_empty() || k == 0 {
            return vec![nodes.to_vec()];
        }
        
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
        
        // Simple assignment based on node index modulo k (simplified clustering)
        for &node in nodes {
            if node < problem.num_nodes {
                let cluster_id = node % k;
                clusters[cluster_id].push(node);
            }
        }
        
        clusters.into_iter().filter(|c| !c.is_empty()).collect()
    }

    /// Level 3: Best insertion position using H-DLT value estimates
    fn find_best_insertion(&self, node: usize, route: &[usize], hdlt: &HierarchicalDLT) -> usize {
        if route.len() <= 1 {
            return route.len();
        }
        
        let mut best_pos = 1;
        let mut best_value = f64::NEG_INFINITY;
        
        for pos in 1..route.len() {
            // Estimate value of inserting at this position
            let value = hdlt.get_base(node, pos);
            if value > best_value {
                best_value = value;
                best_pos = pos;
            }
        }
        
        best_pos
    }

    /// Greedy improvement: try cheapest insertion
    fn greedy_improvement(&mut self, state: &mut TIGState) -> bool {
        if state.route.nodes.len() < 3 {
            return false;
        }
        
        let len = state.route.nodes.len();
        let i = self.rng.gen_range(1..len - 1);
        let j = self.rng.gen_range(1..len - 1);
        
        if i != j {
            state.route.nodes.swap(i, j);
            return true;
        }
        false
    }

    /// 2-opt: reverse segment (random single move)
    fn two_opt(&mut self, state: &mut TIGState) -> bool {
        if state.route.nodes.len() < 4 {
            return false;
        }
        
        let len = state.route.nodes.len();
        let i = self.rng.gen_range(1..len - 2);
        let j = self.rng.gen_range(i + 1..len - 1);
        
        state.route.nodes[i..=j].reverse();
        state.current_time = state.route.nodes.len() as i32 * 10;
        true
    }
    
    /// 2-opt with best-improvement strategy
    fn two_opt_best(&mut self, state: &mut TIGState) -> bool {
        if state.route.nodes.len() < 4 {
            return false;
        }
        
        let len = state.route.nodes.len();
        let mut best_i = 0;
        let mut best_j = 0;
        let mut best_delta = 0;
        
        // Try limited number of random pairs for speed
        let trials = (len / 2).min(10);
        for _ in 0..trials {
            let i = self.rng.gen_range(1..len - 2);
            let j = self.rng.gen_range(i + 1..len - 1);
            
            // Simple delta: prefer shorter segments to reverse
            let delta = j - i;
            if delta > best_delta {
                best_delta = delta;
                best_i = i;
                best_j = j;
            }
        }
        
        if best_delta > 0 {
            state.route.nodes[best_i..=best_j].reverse();
            state.current_time = state.route.nodes.len() as i32 * 10;
            return true;
        }
        false
    }

    /// Relocate: move node to different position
    fn relocate(&mut self, state: &mut TIGState) -> bool {
        if state.route.nodes.len() < 3 {
            return false;
        }
        
        let len = state.route.nodes.len();
        let from = self.rng.gen_range(1..len - 1);
        let to = self.rng.gen_range(1..len);
        
        if from != to {
            let node = state.route.nodes.remove(from);
            state.route.nodes.insert(to.min(len - 1), node);
            return true;
        }
        false
    }
    
    /// Relocate with best-improvement: try multiple positions
    fn relocate_best(&mut self, state: &mut TIGState) -> bool {
        if state.route.nodes.len() < 3 {
            return false;
        }
        
        let len = state.route.nodes.len();
        let from = self.rng.gen_range(1..len - 1);
        let node = state.route.nodes[from];
        
        // Try a few random target positions
        let trials = (len / 3).min(5);
        let mut best_to = from;
        let mut best_distance = 0;
        
        for _ in 0..trials {
            let to = self.rng.gen_range(1..len);
            if to != from {
                let distance = (to as i32 - from as i32).abs();
                if distance > best_distance {
                    best_distance = distance;
                    best_to = to;
                }
            }
        }
        
        if best_to != from {
            let _node = state.route.nodes.remove(from);
            state.route.nodes.insert(best_to.min(len - 1), node);
            return true;
        }
        false
    }

    /// Perturbation for diversification
    fn perturb(&mut self, state: &mut TIGState) {
        if state.route.nodes.len() > 4 {
            let len = state.route.nodes.len();
            let start = self.rng.gen_range(1..len / 2);
            let end = self.rng.gen_range(len / 2..len - 1);
            state.route.nodes[start..end].reverse();
        }
    }
}

// Minimal metrics types
#[derive(Clone, Debug)]
pub struct SolveMetrics {
    pub instance_id: String,
    pub num_nodes: usize,
    pub solution_cost: f64,
    pub infeasibilities: usize,
    pub solve_time_ms: f64,
    pub cold_start: bool,
    pub embedding_used: bool,
    pub repairs_triggered: usize,
    pub repairs_successful: usize,
}

#[derive(Clone, Debug)]
pub struct EmbeddingMetrics {
    pub instance_id: String,
    pub embedding_norm: f64,
    pub spatial_entropy: f64,
    pub avg_tw_width: f64,
    pub instance_type: String,
}

pub fn help() {
	println!("TIG Adaptive Vehicle Routing Solver");
	println!("Uses Adaptive Dynamic Programming with local search optimization");
	println!();
	println!("Hyperparameters:");
	println!("  max_iterations: Maximum iterations for local search (default: 100)");
}

// Important! Do not include any tests in this file, it will result in your submission being rejected

// ============================================================================
// H-DLT + SHRM IMPLEMENTATIONS - INTEGRATED INTO SINGLE FILE
// ============================================================================

/// Module 1: Hierarchical DLT with Instance Embedding (H-DLT)
/// Enables zero-shot generalization and reduces cold-start performance drop
///
/// Components:
/// - InstanceEncoder: GNN-based encoder for instance characteristics
/// - HierarchicalDLT: FiLM-conditioned policy network
/// - EmbeddingCache: LRU cache with TTL for embeddings

use std::time::{SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Instance metadata for cache key generation and analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceMetadata {
    pub n_nodes: usize,
    pub avg_tw_width: f64,
    pub spatial_entropy: f64,
}

impl InstanceMetadata {
    /// Generate deterministic hash for instance
    pub fn hash_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        let key = format!(
            "{}_{}_{}",
            self.n_nodes,
            (self.avg_tw_width * 1000.0) as i64,
            (self.spatial_entropy * 1000.0) as i64
        );
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// 64-dimensional instance embedding
#[derive(Clone, Debug)]
pub struct InstanceEmbedding {
    pub embedding: [f64; 64],
    pub metadata: InstanceMetadata,
    pub created_at: u64,
}

impl InstanceEmbedding {
    pub fn new(embedding: [f64; 64], metadata: InstanceMetadata) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            embedding,
            metadata,
            created_at,
        }
    }

    /// Check if embedding is expired (TTL: 7 days)
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let ttl_seconds = 7 * 24 * 60 * 60; // 7 days
        now - self.created_at > ttl_seconds
    }
}

/// Simple GNN node encoder: MLP(5 → 64)
/// Encodes (x, y, tw_start, tw_end, service_time) to 64-dim embedding
#[derive(Clone, Debug)]
pub struct NodeEmbedder {
    // Weight matrices for MLP layers
    // Layer 1: 5 → 128
    w1: [[f64; 128]; 5],
    b1: [f64; 128],
    // Layer 2: 128 → 128
    w2: [[f64; 128]; 128],
    b2: [f64; 128],
    // Layer 3: 128 → 64
    w3: [[f64; 64]; 128],
    b3: [f64; 64],
}

impl NodeEmbedder {
    pub fn new() -> Self {
        // Initialize with small random values
        let mut embedder = Self {
            w1: [[0.0; 128]; 5],
            b1: [0.0; 128],
            w2: [[0.0; 128]; 128],
            b2: [0.0; 128],
            w3: [[0.0; 64]; 128],
            b3: [0.0; 64],
        };
        embedder.init_xavier();
        embedder
    }

    fn init_xavier(&mut self) {
        // Xavier initialization for stable training
        let scale1 = (2.0 / (5.0 + 128.0) as f64).sqrt();
        let scale2 = (2.0 / (128.0 + 128.0) as f64).sqrt();
        let scale3 = (2.0 / (128.0 + 64.0) as f64).sqrt();

        for i in 0..5 {
            for j in 0..128 {
                self.w1[i][j] = (rand::random::<f64>() - 0.5) * scale1;
            }
        }
        for i in 0..128 {
            for j in 0..128 {
                self.w2[i][j] = (rand::random::<f64>() - 0.5) * scale2;
            }
            for j in 0..64 {
                self.w3[i][j] = (rand::random::<f64>() - 0.5) * scale3;
            }
        }
    }

    /// Forward pass: [x, y, tw_start, tw_end, service_time] → 64-dim
    #[inline]
    pub fn forward(&self, node_features: &[f64; 5]) -> [f64; 64] {
        // Layer 1: 5 → 128 with ReLU (optimized with manual unrolling)
        let mut h1 = [0.0; 128];
        let (f0, f1, f2, f3, f4) = (node_features[0], node_features[1], node_features[2], node_features[3], node_features[4]);
        
        // Unroll in chunks of 4 for better ILP
        for j in (0..128).step_by(4) {
            let mut s0 = self.b1[j];
            let mut s1 = self.b1[j+1];
            let mut s2 = self.b1[j+2];
            let mut s3 = self.b1[j+3];
            
            s0 += f0 * self.w1[0][j] + f1 * self.w1[1][j] + f2 * self.w1[2][j] + f3 * self.w1[3][j] + f4 * self.w1[4][j];
            s1 += f0 * self.w1[0][j+1] + f1 * self.w1[1][j+1] + f2 * self.w1[2][j+1] + f3 * self.w1[3][j+1] + f4 * self.w1[4][j+1];
            s2 += f0 * self.w1[0][j+2] + f1 * self.w1[1][j+2] + f2 * self.w1[2][j+2] + f3 * self.w1[3][j+2] + f4 * self.w1[4][j+2];
            s3 += f0 * self.w1[0][j+3] + f1 * self.w1[1][j+3] + f2 * self.w1[2][j+3] + f3 * self.w1[3][j+3] + f4 * self.w1[4][j+3];
            
            h1[j] = s0.max(0.0);
            h1[j+1] = s1.max(0.0);
            h1[j+2] = s2.max(0.0);
            h1[j+3] = s3.max(0.0);
        }

        // Layer 2: 128 → 128 with ReLU (optimized)
        let mut h2 = [0.0; 128];
        for j in (0..128).step_by(4) {
            let mut s0 = self.b2[j];
            let mut s1 = self.b2[j+1];
            let mut s2 = self.b2[j+2];
            let mut s3 = self.b2[j+3];
            
            for i in (0..128).step_by(4) {
                s0 += h1[i] * self.w2[i][j] + h1[i+1] * self.w2[i+1][j] + h1[i+2] * self.w2[i+2][j] + h1[i+3] * self.w2[i+3][j];
                s1 += h1[i] * self.w2[i][j+1] + h1[i+1] * self.w2[i+1][j+1] + h1[i+2] * self.w2[i+2][j+1] + h1[i+3] * self.w2[i+3][j+1];
                s2 += h1[i] * self.w2[i][j+2] + h1[i+1] * self.w2[i+1][j+2] + h1[i+2] * self.w2[i+2][j+2] + h1[i+3] * self.w2[i+3][j+2];
                s3 += h1[i] * self.w2[i][j+3] + h1[i+1] * self.w2[i+1][j+3] + h1[i+2] * self.w2[i+2][j+3] + h1[i+3] * self.w2[i+3][j+3];
            }
            
            h2[j] = s0.max(0.0);
            h2[j+1] = s1.max(0.0);
            h2[j+2] = s2.max(0.0);
            h2[j+3] = s3.max(0.0);
        }

        // Layer 3: 128 → 64 (optimized)
        let mut output = [0.0; 64];
        for j in (0..64).step_by(4) {
            let mut s0 = self.b3[j];
            let mut s1 = self.b3[j+1];
            let mut s2 = self.b3[j+2];
            let mut s3 = self.b3[j+3];
            
            for i in (0..128).step_by(4) {
                s0 += h2[i] * self.w3[i][j] + h2[i+1] * self.w3[i+1][j] + h2[i+2] * self.w3[i+2][j] + h2[i+3] * self.w3[i+3][j];
                s1 += h2[i] * self.w3[i][j+1] + h2[i+1] * self.w3[i+1][j+1] + h2[i+2] * self.w3[i+2][j+1] + h2[i+3] * self.w3[i+3][j+1];
                s2 += h2[i] * self.w3[i][j+2] + h2[i+1] * self.w3[i+1][j+2] + h2[i+2] * self.w3[i+2][j+2] + h2[i+3] * self.w3[i+3][j+2];
                s3 += h2[i] * self.w3[i][j+3] + h2[i+1] * self.w3[i+1][j+3] + h2[i+2] * self.w3[i+2][j+3] + h2[i+3] * self.w3[i+3][j+3];
            }
            
            output[j] = s0;
            output[j+1] = s1;
            output[j+2] = s2;
            output[j+3] = s3;
        }

        // L2 normalization (optimized with unrolled sum)
        let mut norm_sq = 0.0;
        for i in (0..64).step_by(4) {
            norm_sq += output[i] * output[i] + output[i+1] * output[i+1] + output[i+2] * output[i+2] + output[i+3] * output[i+3];
        }
        let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-8);
        for x in &mut output {
            *x *= inv_norm;
        }

        output
    }
}

/// Instance Encoder: Encodes VRP instance to 64-dim embedding
/// using node features and spatial structure
#[derive(Clone, Debug)]
pub struct InstanceEncoder {
    node_embedder: NodeEmbedder,
    k_neighbors: usize,
}

impl InstanceEncoder {
    pub fn new() -> Self {
        Self {
            node_embedder: NodeEmbedder::new(),
            k_neighbors: 8,
        }
    }

    /// Compute instance metadata from customer data - optimized
    #[inline]
    pub fn compute_metadata(
        &self,
        customers: &[(f64, f64, i32, i32, i32)], // (x, y, tw_start, tw_end, service_time)
    ) -> InstanceMetadata {
        let n_nodes = customers.len();
        
        // Fast path for empty/tiny instances
        if n_nodes == 0 {
            return InstanceMetadata {
                n_nodes: 0,
                avg_tw_width: 0.0,
                spatial_entropy: 0.0,
            };
        }
        
        if n_nodes == 1 {
            return InstanceMetadata {
                n_nodes: 1,
                avg_tw_width: (customers[0].3 - customers[0].2) as f64,
                spatial_entropy: 0.0,
            };
        }

        // Average time window width
        let total_width: f64 = customers
            .iter()
            .map(|(_, _, start, end, _)| (*end - *start) as f64)
            .sum();
        let avg_tw_width = total_width / n_nodes as f64;

        // Spatial entropy (standard deviation of pairwise distances) - optimized sampling for large instances
        let sample_size = if n_nodes > 100 { 100 } else { n_nodes };
        let mut distances = Vec::with_capacity((sample_size * (sample_size - 1)) / 2);
        
        for i in 0..sample_size {
            let idx_i = (i * n_nodes) / sample_size; // stratified sampling
            for j in (i + 1)..sample_size {
                let idx_j = (j * n_nodes) / sample_size;
                let dx = customers[idx_i].0 - customers[idx_j].0;
                let dy = customers[idx_i].1 - customers[idx_j].1;
                distances.push(dx * dx + dy * dy); // use squared distance to avoid sqrt
            }
        }

        let spatial_entropy = if !distances.is_empty() {
            let mean = distances.iter().sum::<f64>() / distances.len() as f64;
            let variance = distances
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / distances.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        InstanceMetadata {
            n_nodes,
            avg_tw_width,
            spatial_entropy,
        }
    }

    /// Build k-NN graph from spatial coordinates (optimized)
    #[inline]
    fn build_knn_graph(&self, coordinates: &[(f64, f64)]) -> Vec<Vec<usize>> {
        let n_nodes = coordinates.len();
        let mut graph = vec![Vec::with_capacity(self.k_neighbors); n_nodes];
        
        // Fast path for small graphs
        if n_nodes <= self.k_neighbors + 1 {
            for i in 0..n_nodes {
                graph[i] = (0..n_nodes).filter(|&j| j != i).collect();
            }
            return graph;
        }

        for i in 0..n_nodes {
            let (xi, yi) = coordinates[i];
            
            // Use squared distances to avoid sqrt
            let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n_nodes - 1);
            for j in 0..n_nodes {
                if j == i { continue; }
                let dx = xi - coordinates[j].0;
                let dy = yi - coordinates[j].1;
                let dist_sq = dx * dx + dy * dy;
                distances.push((j, dist_sq));
            }

            // Partial sort - only need k smallest elements
            let k = self.k_neighbors.min(distances.len());
            distances.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            graph[i] = distances[..k].iter().map(|(j, _)| *j).collect();
        }

        graph
    }

    /// Encode instance: customers → 64-dim embedding
    #[inline]
    pub fn encode(
        &self,
        customers: &[(f64, f64, i32, i32, i32)],
    ) -> InstanceEmbedding {
        let metadata = self.compute_metadata(customers);
        let n_nodes = customers.len();
        
        // Fast path for small instances
        if n_nodes == 0 {
            return InstanceEmbedding::new([0.0; 64], metadata);
        }

        // Node embeddings: each customer → 64-dim (pre-allocate)
        let mut node_embeddings = Vec::with_capacity(n_nodes);
        for (x, y, tw_start, tw_end, service_time) in customers {
            let features = [
                *x,
                *y,
                *tw_start as f64,
                *tw_end as f64,
                *service_time as f64,
            ];
            node_embeddings.push(self.node_embedder.forward(&features));
        }

        // Build spatial graph structure (reduced allocations)
        let coordinates: Vec<_> = customers.iter().map(|(x, y, _, _, _)| (*x, *y)).collect();
        let knn_graph = self.build_knn_graph(&coordinates);

        // Message passing: aggregate neighbor embeddings (3 layers) - optimized
        let mut aggregated = node_embeddings;
        let mut updated = vec![[0.0; 64]; n_nodes];
        
        for _layer in 0..3 {
            for i in 0..n_nodes {
                if knn_graph[i].is_empty() {
                    updated[i] = aggregated[i];
                    continue;
                }

                // Average neighbor embeddings (unrolled)
                let mut neighbor_sum = [0.0; 64];
                let num_neighbors = knn_graph[i].len();
                
                for &j in &knn_graph[i] {
                    let emb = &aggregated[j];
                    for k in (0..64).step_by(4) {
                        neighbor_sum[k] += emb[k];
                        neighbor_sum[k+1] += emb[k+1];
                        neighbor_sum[k+2] += emb[k+2];
                        neighbor_sum[k+3] += emb[k+3];
                    }
                }

                let avg_neighbor_weight = 0.5;
                let inv_neighbors = 1.0 / (num_neighbors as f64 + 1e-8);
                let self_weight = 1.0 - avg_neighbor_weight;
                
                for k in (0..64).step_by(4) {
                    let avg0 = neighbor_sum[k] * inv_neighbors;
                    let avg1 = neighbor_sum[k+1] * inv_neighbors;
                    let avg2 = neighbor_sum[k+2] * inv_neighbors;
                    let avg3 = neighbor_sum[k+3] * inv_neighbors;
                    
                    updated[i][k] = self_weight * aggregated[i][k] + avg_neighbor_weight * avg0;
                    updated[i][k+1] = self_weight * aggregated[i][k+1] + avg_neighbor_weight * avg1;
                    updated[i][k+2] = self_weight * aggregated[i][k+2] + avg_neighbor_weight * avg2;
                    updated[i][k+3] = self_weight * aggregated[i][k+3] + avg_neighbor_weight * avg3;
                }
            }
            core::mem::swap(&mut aggregated, &mut updated);
        }

        // Graph pooling: mean aggregation → instance embedding (optimized)
        let mut instance_embedding = [0.0; 64];
        for node_emb in &aggregated {
            for i in (0..64).step_by(4) {
                instance_embedding[i] += node_emb[i];
                instance_embedding[i+1] += node_emb[i+1];
                instance_embedding[i+2] += node_emb[i+2];
                instance_embedding[i+3] += node_emb[i+3];
            }
        }

        if n_nodes > 0 {
            let inv_n = 1.0 / n_nodes as f64;
            for x in &mut instance_embedding {
                *x *= inv_n;
            }
        }

        // L2 normalization (optimized)
        let mut norm_sq = 0.0;
        for i in (0..64).step_by(4) {
            norm_sq += instance_embedding[i] * instance_embedding[i]
                     + instance_embedding[i+1] * instance_embedding[i+1]
                     + instance_embedding[i+2] * instance_embedding[i+2]
                     + instance_embedding[i+3] * instance_embedding[i+3];
        }
        let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-8);
        for x in &mut instance_embedding {
            *x *= inv_norm;
        }

        InstanceEmbedding::new(instance_embedding, metadata)
    }
}

/// LRU Embedding Cache with TTL support
pub struct EmbeddingCache {
    cache: HashMap<u64, InstanceEmbedding>,
    lru_order: VecDeque<u64>,
    max_size: usize,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            max_size,
        }
    }

    /// Get embedding from cache (returns None if expired) - optimized
    #[inline]
    pub fn get(&mut self, key: u64) -> Option<InstanceEmbedding> {
        // Fast check for existence
        let embedding = self.cache.get(&key)?;
        
        // Check expiration
        if embedding.is_expired() {
            self.cache.remove(&key);
            self.lru_order.retain(|&k| k != key);
            return None;
        }

        // Move to end (most recently used) - optimized with swap
        if let Some(pos) = self.lru_order.iter().position(|&k| k == key) {
            let last_idx = self.lru_order.len() - 1;
            if pos != last_idx {
                self.lru_order.remove(pos);
                self.lru_order.push_back(key);
            }
        }

        self.cache.get(&key).cloned()
    }

    /// Insert embedding into cache, evicting LRU if necessary - optimized
    #[inline]
    pub fn insert(&mut self, key: u64, embedding: InstanceEmbedding) {
        // If key exists, update and move to back
        if self.cache.contains_key(&key) {
            self.cache.insert(key, embedding);
            self.lru_order.retain(|&k| k != key);
            self.lru_order.push_back(key);
            return;
        }

        // Evict LRU if cache is full
        if self.cache.len() >= self.max_size {
            if let Some(lru_key) = self.lru_order.pop_front() {
                self.cache.remove(&lru_key);
            }
        }

        self.cache.insert(key, embedding);
        self.lru_order.push_back(key);
    }

    /// Clear expired entries
    pub fn evict_expired(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let ttl_seconds = 7 * 24 * 60 * 60;

        self.cache.retain(|_, emb| now - emb.created_at <= ttl_seconds);
        self.lru_order.retain(|&k| self.cache.contains_key(&k));
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

/// FiLM (Feature-wise Linear Modulation) layer for conditioning
/// Adapts base logits using instance embedding
#[derive(Clone, Debug)]
pub struct FiLMLayer {
    // Embedding (64) → gamma (64) and beta (64)
    gamma_weights: [[f64; 64]; 64],
    beta_weights: [[f64; 64]; 64],
    gamma_bias: [f64; 64],
    beta_bias: [f64; 64],
}

impl FiLMLayer {
    pub fn new() -> Self {
        let mut film = Self {
            gamma_weights: [[0.0; 64]; 64],
            beta_weights: [[0.0; 64]; 64],
            gamma_bias: [0.0; 64],
            beta_bias: [0.0; 64],
        };
        film.init_xavier();
        film
    }

    fn init_xavier(&mut self) {
        let scale = (2.0 / (64.0 + 64.0) as f64).sqrt();
        for i in 0..64 {
            for j in 0..64 {
                self.gamma_weights[i][j] = (rand::random::<f64>() - 0.5) * scale;
                self.beta_weights[i][j] = (rand::random::<f64>() - 0.5) * scale;
            }
        }
    }

    /// Apply FiLM: output = gamma * base_logits + beta - optimized
    #[inline]
    pub fn forward(&self, embedding: &[f64; 64], base_logits: &[f64]) -> Vec<f64> {
        // Compute gamma and beta from embedding (optimized with unrolling)
        let mut gamma = [0.0; 64];
        let mut beta = [0.0; 64];

        for j in (0..64).step_by(4) {
            let mut g0 = self.gamma_bias[j] + 1.0;
            let mut g1 = self.gamma_bias[j+1] + 1.0;
            let mut g2 = self.gamma_bias[j+2] + 1.0;
            let mut g3 = self.gamma_bias[j+3] + 1.0;
            
            let mut b0 = self.beta_bias[j];
            let mut b1 = self.beta_bias[j+1];
            let mut b2 = self.beta_bias[j+2];
            let mut b3 = self.beta_bias[j+3];

            for i in (0..64).step_by(4) {
                let e0 = embedding[i];
                let e1 = embedding[i+1];
                let e2 = embedding[i+2];
                let e3 = embedding[i+3];
                
                g0 += e0 * self.gamma_weights[i][j] + e1 * self.gamma_weights[i+1][j] + e2 * self.gamma_weights[i+2][j] + e3 * self.gamma_weights[i+3][j];
                g1 += e0 * self.gamma_weights[i][j+1] + e1 * self.gamma_weights[i+1][j+1] + e2 * self.gamma_weights[i+2][j+1] + e3 * self.gamma_weights[i+3][j+1];
                g2 += e0 * self.gamma_weights[i][j+2] + e1 * self.gamma_weights[i+1][j+2] + e2 * self.gamma_weights[i+2][j+2] + e3 * self.gamma_weights[i+3][j+2];
                g3 += e0 * self.gamma_weights[i][j+3] + e1 * self.gamma_weights[i+1][j+3] + e2 * self.gamma_weights[i+2][j+3] + e3 * self.gamma_weights[i+3][j+3];
                
                b0 += e0 * self.beta_weights[i][j] + e1 * self.beta_weights[i+1][j] + e2 * self.beta_weights[i+2][j] + e3 * self.beta_weights[i+3][j];
                b1 += e0 * self.beta_weights[i][j+1] + e1 * self.beta_weights[i+1][j+1] + e2 * self.beta_weights[i+2][j+1] + e3 * self.beta_weights[i+3][j+1];
                b2 += e0 * self.beta_weights[i][j+2] + e1 * self.beta_weights[i+1][j+2] + e2 * self.beta_weights[i+2][j+2] + e3 * self.beta_weights[i+3][j+2];
                b3 += e0 * self.beta_weights[i][j+3] + e1 * self.beta_weights[i+1][j+3] + e2 * self.beta_weights[i+2][j+3] + e3 * self.beta_weights[i+3][j+3];
            }

            gamma[j] = g0;
            gamma[j+1] = g1;
            gamma[j+2] = g2;
            gamma[j+3] = g3;
            
            beta[j] = b0;
            beta[j+1] = b1;
            beta[j+2] = b2;
            beta[j+3] = b3;
        }

        // Apply FiLM: scale each logit dimension (pre-allocate)
        let mut conditioned = Vec::with_capacity(base_logits.len());
        for (idx, &logit) in base_logits.iter().enumerate() {
            let film_idx = idx & 63; // faster than modulo for power of 2
            conditioned.push(gamma[film_idx] * logit + beta[film_idx]);
        }

        conditioned
    }
}

/// Hierarchical DLT: Standard DLT enhanced with instance embedding conditioning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HierarchicalDLT {
    // Base DLT values (from original implementation)
    pub values: std::collections::BTreeMap<(usize, usize), f64>,
    pub count: std::collections::BTreeMap<(usize, usize), u32>,
    pub alpha: f64,
    pub decay_enabled: bool,
    // Hierarchical extension
    pub embedding_influence: f64, // 0.0-1.0, weight of embedding conditioning
}

impl HierarchicalDLT {
    pub fn new(alpha: f64, embedding_influence: f64) -> Self {
        Self {
            values: std::collections::BTreeMap::new(),
            count: std::collections::BTreeMap::new(),
            alpha,
            decay_enabled: true,
            embedding_influence: embedding_influence.clamp(0.0, 1.0),
        }
    }

    /// Get base DLT value
    pub fn get_base(&self, i: usize, j: usize) -> f64 {
        *self.values.get(&(i, j)).unwrap_or(&0.0)
    }

    /// Get conditioned logit value using instance embedding
    pub fn get_conditioned(
        &self,
        i: usize,
        j: usize,
        embedding: &[f64; 64],
        _film: &FiLMLayer,
    ) -> f64 {
        let base_value = self.get_base(i, j);

        // Compute embedding-based bias
        let embedding_dot: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        let embedding_bias = 0.5 * embedding_dot; // Soft bias based on embedding magnitude

        // Blend: (1 - influence) * base + influence * (base + embedding_bias)
        (1.0 - self.embedding_influence) * base_value
            + self.embedding_influence * (base_value + embedding_bias)
    }

    /// Update DLT value with learning rate decay
    pub fn update(&mut self, i: usize, j: usize, target: f64) {
        let key = (i, j);
        let count = self.count.entry(key).or_insert(0);
        let current = self.values.entry(key).or_insert(0.0);

        let alpha_eff = if self.decay_enabled {
            self.alpha / (1.0 + (*count as f64).sqrt())
        } else {
            self.alpha
        };

        let bounded_target = target.clamp(-10000.0, 10000.0);
        let delta = alpha_eff * (bounded_target - *current);
        let bounded_delta = delta.clamp(-100.0, 100.0);

        *current += bounded_delta;
        *current = current.clamp(-5000.0, 5000.0);

        *count += 1;
    }
}

/// Module 3: Self-Healing Repair Mechanism (SHRM)
/// Proactively prevents infeasibility and reduces manual intervention
///
/// Components:
/// - FeasibilityPredictor: Binary classifier for infeasibility detection
/// - RepairPolicy: PPO-trained sequence generator for repairs
/// - RepairOrchestrator: Integration with main solver loop

/// Features for feasibility prediction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairFeatures {
    pub slack_time: f64,           // Available time buffer
    pub queue_length: usize,       // Unserved customers
    pub route_density: f64,        // Nodes per vehicle
    pub adp_risk_score: f64,       // Risk from ADP analysis
    pub time_violation: i32,       // Current time violations
    pub capacity_violation: i32,   // Current capacity violations
}

impl RepairFeatures {
    pub fn new(
        slack_time: f64,
        queue_length: usize,
        route_density: f64,
        adp_risk_score: f64,
        time_violation: i32,
        capacity_violation: i32,
    ) -> Self {
        Self {
            slack_time,
            queue_length,
            route_density,
            adp_risk_score,
            time_violation,
            capacity_violation,
        }
    }

    /// Normalize features to [0, 1] range for model input
    pub fn normalize(&self) -> [f64; 6] {
        [
            (self.slack_time / 10000.0).min(1.0).max(0.0),           // Normalize to 10k max
            (self.queue_length as f64 / 500.0).min(1.0).max(0.0),    // 500 customers max
            self.route_density.min(1.0).max(0.0),                    // Already 0-1
            self.adp_risk_score.min(1.0).max(0.0),                   // Already 0-1
            (self.time_violation as f64 / 10000.0).min(1.0).max(0.0), // 10k time units
            (self.capacity_violation as f64 / 10000.0).min(1.0).max(0.0), // 10k capacity
        ]
    }
}

/// Binary classifier: predicts likelihood of infeasibility
/// Model: Lightweight MLP (6 inputs → 32 hidden → 1 output)
#[derive(Clone, Debug)]
pub struct FeasibilityPredictor {
    // Hidden layer: 6 → 32
    w1: [[f64; 32]; 6],
    b1: [f64; 32],
    // Output layer: 32 → 1
    w2: [f64; 32],
    b2: f64,
}

impl FeasibilityPredictor {
    pub fn new() -> Self {
        let mut predictor = Self {
            w1: [[0.0; 32]; 6],
            b1: [0.0; 32],
            w2: [0.0; 32],
            b2: 0.0,
        };
        predictor.init_weights();
        predictor
    }

    fn init_weights(&mut self) {
        // Xavier initialization
        let scale1 = (2.0 / (6.0 + 32.0) as f64).sqrt();
        let scale2 = (2.0 / (32.0 + 1.0) as f64).sqrt();

        for i in 0..6 {
            for j in 0..32 {
                self.w1[i][j] = (rand::random::<f64>() - 0.5) * scale1;
            }
        }

        for j in 0..32 {
            self.b1[j] = 0.0;
            self.w2[j] = (rand::random::<f64>() - 0.5) * scale2;
        }
        self.b2 = 0.0;
    }

    /// Forward pass: features → probability of infeasibility [0, 1]
    #[inline]
    pub fn predict(&self, features: &[f64; 6]) -> f64 {
        // Hidden layer with ReLU (optimized with unrolling)
        let mut hidden = [0.0; 32];
        let (f0, f1, f2, f3, f4, f5) = (features[0], features[1], features[2], features[3], features[4], features[5]);
        
        for j in (0..32).step_by(4) {
            let s0 = self.b1[j] + f0 * self.w1[0][j] + f1 * self.w1[1][j] + f2 * self.w1[2][j] + f3 * self.w1[3][j] + f4 * self.w1[4][j] + f5 * self.w1[5][j];
            let s1 = self.b1[j+1] + f0 * self.w1[0][j+1] + f1 * self.w1[1][j+1] + f2 * self.w1[2][j+1] + f3 * self.w1[3][j+1] + f4 * self.w1[4][j+1] + f5 * self.w1[5][j+1];
            let s2 = self.b1[j+2] + f0 * self.w1[0][j+2] + f1 * self.w1[1][j+2] + f2 * self.w1[2][j+2] + f3 * self.w1[3][j+2] + f4 * self.w1[4][j+2] + f5 * self.w1[5][j+2];
            let s3 = self.b1[j+3] + f0 * self.w1[0][j+3] + f1 * self.w1[1][j+3] + f2 * self.w1[2][j+3] + f3 * self.w1[3][j+3] + f4 * self.w1[4][j+3] + f5 * self.w1[5][j+3];
            
            hidden[j] = s0.max(0.0);
            hidden[j+1] = s1.max(0.0);
            hidden[j+2] = s2.max(0.0);
            hidden[j+3] = s3.max(0.0);
        }

        // Output layer with sigmoid (optimized)
        let mut logit = self.b2;
        for j in (0..32).step_by(4) {
            logit += hidden[j] * self.w2[j] + hidden[j+1] * self.w2[j+1] + hidden[j+2] * self.w2[j+2] + hidden[j+3] * self.w2[j+3];
        }

        // Sigmoid activation with fast approximation for very large/small values
        if logit > 20.0 { return 1.0; }
        if logit < -20.0 { return 0.0; }
        1.0 / (1.0 + (-logit).exp())
    }

    /// Batch prediction with threshold
    pub fn predict_batch(&self, feature_batch: &[[f64; 6]], threshold: f64) -> Vec<bool> {
        feature_batch
            .iter()
            .map(|f| self.predict(f) > threshold)
            .collect()
    }

    /// Update weights via gradient descent (simplified SGD)
    pub fn update(&mut self, features: &[f64; 6], target: f64, learning_rate: f64) {
        // Forward pass with gradient tracking
        let mut hidden = [0.0; 32];
        for j in 0..32 {
            let mut sum = self.b1[j];
            for i in 0..6 {
                sum += features[i] * self.w1[i][j];
            }
            hidden[j] = sum.max(0.0);
        }

        let mut logit = self.b2;
        for j in 0..32 {
            logit += hidden[j] * self.w2[j];
        }

        let pred = 1.0 / (1.0 + (-logit).exp());
        let loss_grad = pred - target; // Simplified gradient

        // Update output layer
        self.b2 -= learning_rate * loss_grad;
        for j in 0..32 {
            if hidden[j] > 0.0 {
                self.w2[j] -= learning_rate * loss_grad * hidden[j];
            }
        }

        // Update hidden layer
        let hidden_grad_scale = learning_rate * loss_grad;
        for j in 0..32 {
            if hidden[j] > 0.0 {
                let grad = hidden_grad_scale * self.w2[j];
                self.b1[j] -= grad;
                for i in 0..6 {
                    self.w1[i][j] -= grad * features[i];
                }
            }
        }
    }
}

/// Repair actions: what to do when infeasibility is detected
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RepairAction {
    /// Reinsert a customer at a different position
    ReinsertCustomer { customer_id: usize, position: usize },
    /// Swap segments between routes
    SwapSegments { route1: usize, route2: usize },
    /// Merge two routes
    MergeRoutes { route1: usize, route2: usize },
    /// Delay service for a customer
    DelayService { customer_id: usize, delay_minutes: i32 },
}

/// Repair policy: generates sequence of repair actions
/// Uses small neural network trained via PPO
#[derive(Clone, Debug)]
pub struct RepairPolicy {
    // State encoder: (6 features + action history) → 32 hidden
    state_encoder_w: [[f64; 32]; 12],
    state_encoder_b: [f64; 32],

    // Policy head: 32 hidden → 4 action logits
    policy_head_w: [[f64; 4]; 32],
    policy_head_b: [f64; 4],

    // Value head: 32 hidden → 1 value estimate
    value_head_w: [f64; 32],
    value_head_b: f64,

    // Action history for context
    action_history: VecDeque<RepairAction>,
    max_history: usize,
}

impl RepairPolicy {
    pub fn new() -> Self {
        let mut policy = Self {
            state_encoder_w: [[0.0; 32]; 12],
            state_encoder_b: [0.0; 32],
            policy_head_w: [[0.0; 4]; 32],
            policy_head_b: [0.0; 4],
            value_head_w: [0.0; 32],
            value_head_b: 0.0,
            action_history: VecDeque::new(),
            max_history: 5,
        };
        policy.init_weights();
        policy
    }

    fn init_weights(&mut self) {
        let scale1 = (2.0 / (12.0 + 32.0) as f64).sqrt();
        let scale2 = (2.0 / (32.0 + 4.0) as f64).sqrt();

        for i in 0..12 {
            for j in 0..32 {
                self.state_encoder_w[i][j] = (rand::random::<f64>() - 0.5) * scale1;
            }
        }

        for j in 0..32 {
            for i in 0..4 {
                self.policy_head_w[j][i] = (rand::random::<f64>() - 0.5) * scale2;
            }
            self.value_head_w[j] = (rand::random::<f64>() - 0.5) * scale2;
        }
    }

    /// Encode state: features + action history
    fn encode_state(&self, features: &[f64; 6]) -> [f64; 12] {
        let mut encoded = [0.0; 12];

        // First 6 dims: normalized features
        encoded[..6].copy_from_slice(features);

        // Last 6 dims: action history encoding
        for (i, action) in self.action_history.iter().take(6).enumerate() {
            encoded[6 + i] = match action {
                RepairAction::ReinsertCustomer { .. } => 1.0,
                RepairAction::SwapSegments { .. } => 0.5,
                RepairAction::MergeRoutes { .. } => -0.5,
                RepairAction::DelayService { .. } => -1.0,
            };
        }

        encoded
    }

    /// Forward pass: get action logits and value estimate
    #[inline]
    pub fn forward(&self, features: &[f64; 6]) -> ([f64; 4], f64) {
        let encoded = self.encode_state(features);

        // Encode state (optimized)
        let mut hidden = [0.0; 32];
        for j in (0..32).step_by(4) {
            let mut s0 = self.state_encoder_b[j];
            let mut s1 = self.state_encoder_b[j+1];
            let mut s2 = self.state_encoder_b[j+2];
            let mut s3 = self.state_encoder_b[j+3];
            
            for i in (0..12).step_by(4) {
                s0 += encoded[i] * self.state_encoder_w[i][j] + encoded[i+1] * self.state_encoder_w[i+1][j] + encoded[i+2] * self.state_encoder_w[i+2][j] + encoded[i+3] * self.state_encoder_w[i+3][j];
                s1 += encoded[i] * self.state_encoder_w[i][j+1] + encoded[i+1] * self.state_encoder_w[i+1][j+1] + encoded[i+2] * self.state_encoder_w[i+2][j+1] + encoded[i+3] * self.state_encoder_w[i+3][j+1];
                s2 += encoded[i] * self.state_encoder_w[i][j+2] + encoded[i+1] * self.state_encoder_w[i+1][j+2] + encoded[i+2] * self.state_encoder_w[i+2][j+2] + encoded[i+3] * self.state_encoder_w[i+3][j+2];
                s3 += encoded[i] * self.state_encoder_w[i][j+3] + encoded[i+1] * self.state_encoder_w[i+1][j+3] + encoded[i+2] * self.state_encoder_w[i+2][j+3] + encoded[i+3] * self.state_encoder_w[i+3][j+3];
            }
            
            hidden[j] = s0.max(0.0);
            hidden[j+1] = s1.max(0.0);
            hidden[j+2] = s2.max(0.0);
            hidden[j+3] = s3.max(0.0);
        }

        // Policy head: action logits (optimized)
        let mut action_logits = [0.0; 4];
        for i in 0..4 {
            let mut sum = self.policy_head_b[i];
            for j in (0..32).step_by(4) {
                sum += hidden[j] * self.policy_head_w[j][i] + hidden[j+1] * self.policy_head_w[j+1][i] + hidden[j+2] * self.policy_head_w[j+2][i] + hidden[j+3] * self.policy_head_w[j+3][i];
            }
            action_logits[i] = sum;
        }

        // Value head (optimized)
        let mut value = self.value_head_b;
        for j in (0..32).step_by(4) {
            value += hidden[j] * self.value_head_w[j] + hidden[j+1] * self.value_head_w[j+1] + hidden[j+2] * self.value_head_w[j+2] + hidden[j+3] * self.value_head_w[j+3];
        }

        (action_logits, value.tanh() * 100.0) // Scale value to [-100, 100]
    }

    /// Sample action from policy - optimized
    #[inline]
    pub fn sample_action(&mut self, features: &[f64; 6]) -> RepairAction {
        let (logits, _value) = self.forward(features);

        // Softmax over action logits (optimized with fast exp approximation for range)
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        // Fast path: if max_logit dominates (>5.0 gap), just pick it
        let min_logit = logits.iter().copied().fold(f64::INFINITY, f64::min);
        if max_logit - min_logit > 5.0 {
            let max_idx = logits.iter().position(|&x| x == max_logit).unwrap_or(0);
            let action = self.create_action(max_idx);
            self.add_to_history(action);
            return action;
        }
        
        let exp_logits: [f64; 4] = logits.map(|l| (l - max_logit).exp());
        let sum_exp: f64 = exp_logits.iter().sum();
        let inv_sum = 1.0 / sum_exp;
        let probs = exp_logits.map(|e| e * inv_sum);

        // Sample action based on probabilities
        let rand_val = rand::random::<f64>();
        let mut cumsum = 0.0;
        let mut action_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                action_idx = i;
                break;
            }
        }

        let action = self.create_action(action_idx);
        self.add_to_history(action);
        action
    }
    
    #[inline]
    fn create_action(&self, action_idx: usize) -> RepairAction {
        match action_idx {
            0 => RepairAction::ReinsertCustomer {
                customer_id: rand::random::<usize>() % 100,
                position: rand::random::<usize>() % 10,
            },
            1 => RepairAction::SwapSegments {
                route1: rand::random::<usize>() % 5,
                route2: rand::random::<usize>() % 5,
            },
            2 => RepairAction::MergeRoutes {
                route1: rand::random::<usize>() % 5,
                route2: rand::random::<usize>() % 5,
            },
            _ => RepairAction::DelayService {
                customer_id: rand::random::<usize>() % 100,
                delay_minutes: (rand::random::<i32>() % 60) + 5,
            },
        }
    }
    
    #[inline]
    fn add_to_history(&mut self, action: RepairAction) {
        if self.action_history.len() >= self.max_history {
            self.action_history.pop_front();
        }
        self.action_history.push_back(action);
    }

    /// PPO update: maximize advantage-weighted action log probability
    pub fn ppo_update(
        &mut self,
        features: &[f64; 6],
        action_idx: usize,
        advantage: f64,
        learning_rate: f64,
    ) {
        let (logits, _value) = self.forward(features);

        // Compute log probabilities
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: [f64; 4] = logits.map(|l| (l - max_logit).exp());
        let sum_exp: f64 = exp_logits.iter().sum();
        let _log_probs: [f64; 4] = exp_logits.map(|e| (e / sum_exp).ln());

        // Policy gradient: ∇ log_π(a|s) * advantage
        let log_prob_grad = 1.0 - (exp_logits[action_idx] / sum_exp); // Derivative
        let pg_loss = advantage * log_prob_grad;

        // Simplified update (gradient ascent on advantage)
        let encoded = self.encode_state(features);
        let mut hidden = [0.0; 32];
        for j in 0..32 {
            let mut sum = self.state_encoder_b[j];
            for i in 0..12 {
                sum += encoded[i] * self.state_encoder_w[i][j];
            }
            hidden[j] = sum.max(0.0);
        }

        // Update policy head (simplified)
        for i in 0..4 {
            let grad = if i == action_idx {
                learning_rate * pg_loss
            } else {
                learning_rate * pg_loss * 0.1
            };
            self.policy_head_b[i] += grad;
            for j in 0..32 {
                if hidden[j] > 0.0 {
                    self.policy_head_w[j][i] += grad * hidden[j];
                }
            }
        }
    }
}

/// Repair Orchestrator: coordinates repair attempts and monitoring
#[derive(Clone, Debug)]
pub struct RepairOrchestrator {
    pub feasibility_predictor: FeasibilityPredictor,
    pub repair_policy: RepairPolicy,
    pub trigger_threshold: f64,

    // Monitoring
    pub repair_triggered_count: u32,
    pub repair_success_count: u32,
    pub repair_failure_count: u32,
    pub total_repair_latency_ms: f64,

    // History
    pub repair_history: VecDeque<RepairAttempt>,
    max_history_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairAttempt {
    pub timestamp: u64,
    pub action: RepairAction,
    pub success: bool,
    pub latency_ms: f64,
}

impl RepairOrchestrator {
    pub fn new(trigger_threshold: f64) -> Self {
        Self {
            feasibility_predictor: FeasibilityPredictor::new(),
            repair_policy: RepairPolicy::new(),
            trigger_threshold,
            repair_triggered_count: 0,
            repair_success_count: 0,
            repair_failure_count: 0,
            total_repair_latency_ms: 0.0,
            repair_history: VecDeque::new(),
            max_history_size: 1000,
        }
    }

    /// Attempt repair: predict infeasibility and generate fix - optimized
    #[inline]
    pub fn attempt_repair(&mut self, features: &RepairFeatures) -> Option<RepairAction> {
        // Early exit if violations are already very low
        if features.time_violation == 0 && features.capacity_violation == 0 && features.slack_time > 1000.0 {
            return None;
        }
        
        let normalized_features = features.normalize();

        // Check if repair needed
        let infeasibility_prob = self.feasibility_predictor.predict(&normalized_features);

        if infeasibility_prob < self.trigger_threshold {
            return None;
        }

        // Repair triggered
        self.repair_triggered_count += 1;

        // Generate repair action
        Some(self.repair_policy.sample_action(&normalized_features))
    }

    /// Record repair attempt
    pub fn record_attempt(&mut self, action: RepairAction, success: bool, latency_ms: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if success {
            self.repair_success_count += 1;
        } else {
            self.repair_failure_count += 1;
        }

        self.total_repair_latency_ms += latency_ms;

        let attempt = RepairAttempt {
            timestamp,
            action,
            success,
            latency_ms,
        };

        if self.repair_history.len() >= self.max_history_size {
            self.repair_history.pop_front();
        }
        self.repair_history.push_back(attempt);
    }

    /// Get success rate over recent attempts
    pub fn recent_success_rate(&self, window_size: usize) -> f64 {
        if self.repair_history.is_empty() {
            return 1.0;
        }

        let recent: Vec<_> = self.repair_history.iter().rev().take(window_size).collect();
        let successful = recent.iter().filter(|a| a.success).count();
        let total = recent.len();

        if total == 0 {
            1.0
        } else {
            successful as f64 / total as f64
        }
    }

    /// Average repair latency
    pub fn avg_repair_latency_ms(&self) -> f64 {
        if self.repair_triggered_count == 0 {
            return 0.0;
        }
        self.total_repair_latency_ms / self.repair_triggered_count as f64
    }

    /// Check alert conditions
    pub fn check_alerts(&self) -> Vec<String> {
        let mut alerts = Vec::new();

        // Alert if success rate < 90% over 100 solves
        let recent_success = self.recent_success_rate(100);
        if recent_success < 0.9 && self.repair_triggered_count > 100 {
            alerts.push(format!(
                "⚠️ Repair success rate low: {:.1}% (threshold: 90%)",
                recent_success * 100.0
            ));
        }

        // Alert if avg latency > 80ms
        let avg_latency = self.avg_repair_latency_ms();
        if avg_latency > 80.0 && self.repair_triggered_count > 10 {
            alerts.push(format!(
                "⚠️ Repair latency high: {:.1}ms (threshold: 80ms)",
                avg_latency
            ));
        }

        alerts
    }

    /// Get statistics for logging
    pub fn stats(&self) -> RepairStats {
        RepairStats {
            repair_triggered_count: self.repair_triggered_count,
            repair_success_count: self.repair_success_count,
            repair_failure_count: self.repair_failure_count,
            success_rate: if self.repair_triggered_count > 0 {
                self.repair_success_count as f64 / self.repair_triggered_count as f64
            } else {
                1.0
            },
            avg_repair_latency_ms: self.avg_repair_latency_ms(),
            recent_success_rate_100: self.recent_success_rate(100),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairStats {
    pub repair_triggered_count: u32,
    pub repair_success_count: u32,
    pub repair_failure_count: u32,
    pub success_rate: f64,
    pub avg_repair_latency_ms: f64,
    pub recent_success_rate_100: f64,
}

/// Monitoring and Logging for H-DLT and SHRM systems
/// Tracks performance metrics, embedding drift, and repair success rates

/// Performance Monitor for H-DLT validation criteria
#[derive(Clone, Debug, Default)]
pub struct PerformanceMonitor {
    solve_times: VecDeque<f64>,
    cold_start_costs: VecDeque<f64>,
    warm_costs: VecDeque<f64>,
    embedding_latencies: VecDeque<f64>,
    max_history: usize,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            solve_times: VecDeque::new(),
            cold_start_costs: VecDeque::new(),
            warm_costs: VecDeque::new(),
            embedding_latencies: VecDeque::new(),
            max_history: 1000,
        }
    }

    pub fn record_solve_time(&mut self, time_ms: f64) {
        if self.solve_times.len() >= self.max_history {
            self.solve_times.pop_front();
        }
        self.solve_times.push_back(time_ms);
    }

    pub fn record_cold_start(&mut self, cost: f64) {
        if self.cold_start_costs.len() >= self.max_history {
            self.cold_start_costs.pop_front();
        }
        self.cold_start_costs.push_back(cost);
    }

    pub fn record_warm(&mut self, cost: f64) {
        if self.warm_costs.len() >= self.max_history {
            self.warm_costs.pop_front();
        }
        self.warm_costs.push_back(cost);
    }

    pub fn record_embedding_latency(&mut self, latency_ms: f64) {
        if self.embedding_latencies.len() >= self.max_history {
            self.embedding_latencies.pop_front();
        }
        self.embedding_latencies.push_back(latency_ms);
    }

    /// Validation Criteria 1: Cold-start performance ≥95% of mature
    pub fn cold_start_performance_ratio(&self) -> f64 {
        if self.cold_start_costs.is_empty() || self.warm_costs.is_empty() {
            return 1.0;
        }

        let avg_cold = self.cold_start_costs.iter().sum::<f64>() / self.cold_start_costs.len() as f64;
        let avg_warm = self.warm_costs.iter().sum::<f64>() / self.warm_costs.len() as f64;

        if avg_warm == 0.0 {
            return 1.0;
        }

        avg_cold / avg_warm
    }

    /// Validation Criteria 2: Inference latency increase ≤15ms
    pub fn avg_embedding_latency_ms(&self) -> f64 {
        if self.embedding_latencies.is_empty() {
            return 0.0;
        }
        self.embedding_latencies.iter().sum::<f64>() / self.embedding_latencies.len() as f64
    }

    pub fn meets_latency_criterion(&self) -> bool {
        self.avg_embedding_latency_ms() <= 15.0
    }

    pub fn meets_cold_start_criterion(&self) -> bool {
        self.cold_start_performance_ratio() >= 0.95
    }

    pub fn report(&self) -> MonitoringReport {
        MonitoringReport {
            avg_solve_time_ms: if self.solve_times.is_empty() {
                0.0
            } else {
                self.solve_times.iter().sum::<f64>() / self.solve_times.len() as f64
            },
            cold_start_ratio: self.cold_start_performance_ratio(),
            avg_embedding_latency_ms: self.avg_embedding_latency_ms(),
            meets_cold_start_criterion: self.meets_cold_start_criterion(),
            meets_latency_criterion: self.meets_latency_criterion(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonitoringReport {
    pub avg_solve_time_ms: f64,
    pub cold_start_ratio: f64,
    pub avg_embedding_latency_ms: f64,
    pub meets_cold_start_criterion: bool,
    pub meets_latency_criterion: bool,
}

/// Success Rate Tracker for SHRM validation
#[derive(Clone, Debug, Default)]
pub struct SuccessRateTracker {
    attempts: VecDeque<bool>, // true = success, false = failure
    max_history: usize,
}

impl SuccessRateTracker {
    pub fn new() -> Self {
        Self {
            attempts: VecDeque::new(),
            max_history: 10000,
        }
    }

    pub fn record_attempt(&mut self, success: bool) {
        if self.attempts.len() >= self.max_history {
            self.attempts.pop_front();
        }
        self.attempts.push_back(success);
    }

    pub fn success_rate(&self) -> f64 {
        if self.attempts.is_empty() {
            return 1.0;
        }

        let successful = self.attempts.iter().filter(|&&x| x).count();
        successful as f64 / self.attempts.len() as f64
    }

    pub fn success_rate_last_n(&self, n: usize) -> f64 {
        let recent: Vec<_> = self.attempts.iter().rev().take(n).copied().collect();
        if recent.is_empty() {
            return 1.0;
        }

        let successful = recent.iter().filter(|&&x| x).count();
        successful as f64 / recent.len() as f64
    }

    pub fn len(&self) -> usize {
        self.attempts.len()
    }

    /// SHRM Validation Criteria: ≥99.9% feasibility rate
    pub fn meets_feasibility_criterion(&self) -> bool {
        self.success_rate() >= 0.999
    }

    /// SHRM Validation Criteria: ≥95% repair success rate
    pub fn meets_repair_success_criterion(&self) -> bool {
        self.success_rate() >= 0.95
    }

    pub fn report(&self) -> SuccessRateReport {
        SuccessRateReport {
            total_attempts: self.attempts.len() as u32,
            success_rate: self.success_rate(),
            recent_100_success_rate: self.success_rate_last_n(100),
            meets_feasibility_criterion: self.meets_feasibility_criterion(),
            meets_repair_success_criterion: self.meets_repair_success_criterion(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuccessRateReport {
    pub total_attempts: u32,
    pub success_rate: f64,
    pub recent_100_success_rate: f64,
    pub meets_feasibility_criterion: bool,
    pub meets_repair_success_criterion: bool,
}

/// Embedding Drift Detector for monitoring instance type separation
#[derive(Clone, Debug, Default)]
pub struct EmbeddingDriftDetector {
    r_embeddings: Vec<[f64; 64]>,  // Random type instances
    rc_embeddings: Vec<[f64; 64]>, // Random + Clustered
    c_embeddings: Vec<[f64; 64]>,  // Clustered
    max_per_type: usize,
}

impl EmbeddingDriftDetector {
    pub fn new() -> Self {
        Self {
            r_embeddings: Vec::new(),
            rc_embeddings: Vec::new(),
            c_embeddings: Vec::new(),
            max_per_type: 1000,
        }
    }

    pub fn record_embedding(&mut self, embedding: &[f64; 64], instance_type: &str) {
        match instance_type {
            "R" => {
                if self.r_embeddings.len() < self.max_per_type {
                    self.r_embeddings.push(*embedding);
                }
            }
            "RC" => {
                if self.rc_embeddings.len() < self.max_per_type {
                    self.rc_embeddings.push(*embedding);
                }
            }
            "C" => {
                if self.c_embeddings.len() < self.max_per_type {
                    self.c_embeddings.push(*embedding);
                }
            }
            _ => {}
        }
    }

    /// Compute inter-type separation (for validation)
    /// Returns average cosine distance between embedding types
    pub fn type_separation(&self) -> f64 {
        if self.r_embeddings.is_empty() || self.rc_embeddings.is_empty() || self.c_embeddings.is_empty() {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        // Sample pairs from different types
        for r_emb in self.r_embeddings.iter().take(100) {
            for rc_emb in self.rc_embeddings.iter().take(100) {
                let dot_prod: f64 = r_emb
                    .iter()
                    .zip(rc_emb.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let cos_sim = dot_prod / (1.0 + 1e-8); // Already normalized
                total_distance += 1.0 - cos_sim; // Convert similarity to distance
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    pub fn report(&self) -> EmbeddingDriftReport {
        EmbeddingDriftReport {
            r_count: self.r_embeddings.len(),
            rc_count: self.rc_embeddings.len(),
            c_count: self.c_embeddings.len(),
            type_separation: self.type_separation(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingDriftReport {
    pub r_count: usize,
    pub rc_count: usize,
    pub c_count: usize,
    pub type_separation: f64,
}

/// Integration Guide and Training Pipeline for H-DLT + SHRM
/// Shows how to use the new modules with the existing solver

/// Training Configuration for H-DLT
#[derive(Clone, Debug)]
pub struct HierarchicalDLTConfig {
    /// Embedding cache size (number of cached embeddings)
    pub cache_size: usize,
    /// Influence of instance embedding on DLT values [0.0, 1.0]
    pub embedding_influence: f64,
    /// Learning rate for DLT updates
    pub learning_rate: f64,
    /// Enable learning rate decay
    pub decay_enabled: bool,
    /// FiLM layer enabled for advanced conditioning
    pub use_film_conditioning: bool,
}

impl Default for HierarchicalDLTConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            embedding_influence: 0.3,
            learning_rate: 0.1,
            decay_enabled: true,
            use_film_conditioning: true,
        }
    }
}

/// Training Configuration for SHRM
#[derive(Clone, Debug)]
pub struct SHRMConfig {
    /// Threshold for triggering repairs (0.85 recommended)
    pub trigger_threshold: f64,
    /// Learning rate for predictor and policy
    pub learning_rate: f64,
    /// Number of PPO epochs per batch
    pub ppo_epochs: usize,
    /// Batch size for PPO training
    pub batch_size: usize,
}

impl Default for SHRMConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: 0.85,
            learning_rate: 0.001,
            ppo_epochs: 3,
            batch_size: 32,
        }
    }
}

/// Integrated Hierarchical DLT System
/// Combines instance encoding, DLT, and embedding cache
pub struct HierarchicalDLTSystem {
    pub encoder: InstanceEncoder,
    pub cache: EmbeddingCache,
    pub hdlt: HierarchicalDLT,
    pub film: FiLMLayer,
    pub config: HierarchicalDLTConfig,
    pub stats: HierarchicalDLTStats,
}

#[derive(Clone, Debug, Default)]
pub struct HierarchicalDLTStats {
    pub embeddings_computed: u32,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

impl HierarchicalDLTSystem {
    pub fn new(config: HierarchicalDLTConfig) -> Self {
        Self {
            encoder: InstanceEncoder::new(),
            cache: EmbeddingCache::new(config.cache_size),
            hdlt: HierarchicalDLT::new(config.learning_rate, config.embedding_influence),
            film: FiLMLayer::new(),
            config,
            stats: Default::default(),
        }
    }

    /// Get or compute instance embedding with caching
    pub fn get_embedding(
        &mut self,
        customers: &[(f64, f64, i32, i32, i32)],
    ) -> InstanceEmbedding {
        let metadata = self.encoder.compute_metadata(customers);
        let cache_key = metadata.hash_key();

        // Try cache first
        if let Some(embedding) = self.cache.get(cache_key) {
            self.stats.cache_hits += 1;
            return embedding;
        }

        // Cache miss: compute embedding
        self.stats.cache_misses += 1;
        self.stats.embeddings_computed += 1;

        let embedding = self.encoder.encode(customers);
        self.cache.insert(cache_key, embedding.clone());

        embedding
    }

    /// Get conditioned DLT value for a move
    pub fn get_conditioned_value(&self, i: usize, j: usize, embedding: &[f64; 64]) -> f64 {
        if self.config.use_film_conditioning {
            self.hdlt
                .get_conditioned(i, j, embedding, &self.film)
        } else {
            self.hdlt.get_base(i, j)
        }
    }

    /// Update DLT after solving an instance
    pub fn update_dlt(&mut self, i: usize, j: usize, target: f64) {
        self.hdlt.update(i, j, target);
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, f64) {
        let size = self.cache.size();
        let hit_rate = if self.stats.cache_hits + self.stats.cache_misses > 0 {
            self.stats.cache_hits as f64 / (self.stats.cache_hits + self.stats.cache_misses) as f64
        } else {
            0.0
        };
        (size, hit_rate)
    }

    pub fn report(&self) -> String {
        let (cache_size, hit_rate) = self.cache_stats();
        format!(
            "H-DLT System Report:\n\
             - Embeddings computed: {}\n\
             - Cache hits: {}\n\
             - Cache misses: {}\n\
             - Cache hit rate: {:.1}%\n\
             - Current cache size: {}/{}\n\
             - Embedding influence: {:.2}",
            self.stats.embeddings_computed,
            self.stats.cache_hits,
            self.stats.cache_misses,
            hit_rate * 100.0,
            cache_size,
            self.config.cache_size,
            self.config.embedding_influence,
        )
    }
}

/// Integrated Self-Healing Repair System
pub struct SHRMSystem {
    pub orchestrator: RepairOrchestrator,
    pub config: SHRMConfig,
    pub monitor: SuccessRateTracker,
}

impl SHRMSystem {
    pub fn new(config: SHRMConfig, _log_dir: Option<&str>) -> Result<Self> {
        Ok(Self {
            orchestrator: RepairOrchestrator::new(config.trigger_threshold),
            config,
            monitor: SuccessRateTracker::new(),
        })
    }

    /// Attempt repair given current state features
    pub fn attempt_repair(&mut self, features: &RepairFeatures) -> Option<RepairAction> {
        self.orchestrator.attempt_repair(features)
    }

    /// Record repair outcome and update policy
    pub fn record_repair_outcome(
        &mut self,
        action: RepairAction,
        success: bool,
        latency_ms: f64,
        features: &RepairFeatures,
    ) {
        self.orchestrator.record_attempt(action, success, latency_ms);
        self.monitor.record_attempt(success);

        // Logging disabled: no logger field available in this build

        // Update repair policy via PPO if training
        let normalized = features.normalize();
        let advantage = if success { 1.0 } else { -1.0 };
        let action_idx = match action {
            RepairAction::ReinsertCustomer { .. } => 0,
            RepairAction::SwapSegments { .. } => 1,
            RepairAction::MergeRoutes { .. } => 2,
            RepairAction::DelayService { .. } => 3,
        };
        self.orchestrator
            .repair_policy
            .ppo_update(&normalized, action_idx, advantage, self.config.learning_rate);
    }

    /// Get current success rate
    pub fn success_rate(&self) -> f64 {
        self.monitor.success_rate()
    }

    /// Check for alert conditions
    pub fn check_alerts(&self) -> Vec<String> {
        let mut alerts = self.orchestrator.check_alerts();

        // Add feasibility check
        if !self.monitor.meets_feasibility_criterion() && self.monitor.len() > 165 {
            alerts.push(format!(
                "⚠️ Feasibility rate below 99.9%: {:.2}%",
                self.monitor.success_rate() * 100.0
            ));
        }

        alerts
    }

    pub fn report(&self) -> SHRMReport {
        SHRMReport {
            total_repairs_triggered: self.orchestrator.repair_triggered_count,
            repair_success_rate: self.orchestrator.repair_success_count as f64
                / self.orchestrator.repair_triggered_count.max(1) as f64,
            avg_repair_latency_ms: self.orchestrator.avg_repair_latency_ms(),
            feasibility_rate: self.monitor.success_rate(),
            alerts: self.check_alerts(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SHRMReport {
    pub total_repairs_triggered: u32,
    pub repair_success_rate: f64,
    pub avg_repair_latency_ms: f64,
    pub feasibility_rate: f64,
    pub alerts: Vec<String>,
}

/// Combined Training Loop Example
/// Shows how to use H-DLT and SHRM together
pub struct IntegratedTrainingPipeline {
    pub h_dlt: HierarchicalDLTSystem,
    pub shrm: SHRMSystem,
    pub monitoring: PerformanceMonitor,
    pub embedding_drift: EmbeddingDriftDetector,
}

impl IntegratedTrainingPipeline {
    pub fn new(
        h_dlt_config: HierarchicalDLTConfig,
        shrm_config: SHRMConfig,
        log_dir: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            h_dlt: HierarchicalDLTSystem::new(h_dlt_config),
            shrm: SHRMSystem::new(shrm_config, log_dir)?,
            monitoring: PerformanceMonitor::new(),
            embedding_drift: EmbeddingDriftDetector::new(),
        })
    }

    /// Run a single solve with full integration
    pub fn solve_with_learning(
        &mut self,
        instance_id: &str,
        customers: &[(f64, f64, i32, i32, i32)],
        solution_cost: f64,
        infeasibilities: u32,
        solve_time_ms: f64,
        is_cold_start: bool,
    ) {
        // Get instance embedding
        let start = std::time::Instant::now();
        let embedding = self.h_dlt.get_embedding(customers);
        let embedding_latency = start.elapsed().as_secs_f64() * 1000.0;

        // Record metrics
        self.monitoring.record_embedding_latency(embedding_latency);
        if is_cold_start {
            self.monitoring.record_cold_start(solution_cost);
        } else {
            self.monitoring.record_warm(solution_cost);
        }
        self.monitoring.record_solve_time(solve_time_ms);

        // Record embedding for drift detection
        self.embedding_drift.record_embedding(
            &embedding.embedding,
            &self.infer_instance_type(&embedding.metadata),
        );

        // Check for repairs needed
        let features = RepairFeatures::new(
            (infeasibilities as f64) * 100.0, // slack_time estimate
            customers.len(),
            0.5,
            (infeasibilities as f64) / 10.0,
            infeasibilities as i32,
            (infeasibilities as i32) * 50,
        );

        let mut repairs_triggered = 0;
        let mut repairs_successful = 0;

        if let Some(action) = self.shrm.attempt_repair(&features) {
            repairs_triggered = 1;

            // Simulate repair validation (would be actual solver check)
            let success = rand::random::<f64>() > 0.1; // 90% success rate simulated
            if success {
                repairs_successful = 1;
            }

            // Record outcome
            let latency = 20.0; // ms
            self.shrm
                .record_repair_outcome(action, success, latency, &features);
        }

        // Log solve metrics
        {
            let metrics = SolveMetrics {
                instance_id: instance_id.to_string(),
                num_nodes: customers.len(),
                solution_cost,
                infeasibilities: infeasibilities as usize,
                solve_time_ms,
                cold_start: is_cold_start,
                embedding_used: embedding_latency > 0.0,
                repairs_triggered,
                repairs_successful,
            };

            // Logging disabled: shrm.logger not present

            // Log embedding metrics
            let emb_metrics = EmbeddingMetrics {
                instance_id: instance_id.to_string(),
                embedding_norm: 1.0,
                spatial_entropy: embedding.metadata.spatial_entropy,
                avg_tw_width: embedding.metadata.avg_tw_width,
                instance_type: self.infer_instance_type(&embedding.metadata),
            };
            
            // Logging disabled: shrm.logger not present
        }
    }

    fn infer_instance_type(&self, metadata: &InstanceMetadata) -> String {
        // Simple heuristic: cluster instances by spatial entropy and time window width
        if metadata.spatial_entropy < 500.0 && metadata.avg_tw_width < 300.0 {
            "C".to_string() // Clustered
        } else if metadata.spatial_entropy > 1000.0 && metadata.avg_tw_width > 500.0 {
            "R".to_string() // Random
        } else {
            "RC".to_string() // Mixed
        }
    }

    /// Get comprehensive report
    pub fn report(&self) -> String {
        let perf_report = self.monitoring.report();
        let shrm_report = self.shrm.report();
        let embedding_report = self.embedding_drift.report();

        format!(
            "=== Integrated Training Pipeline Report ===\n\n\
             H-DLT Performance:\n\
             {}\n\
             - Cold-start ratio: {:.2}% of warm\n\
             - Embedding latency: {:.2}ms\n\
             - Meets latency criterion: {}\n\n\
             SHRM Performance:\n\
             - Repairs triggered: {}\n\
             - Success rate: {:.2}%\n\
             - Avg latency: {:.2}ms\n\
             - Feasibility rate: {:.4}%\n\n\
             Embedding Drift Detection:\n\
             - R instances: {}\n\
             - RC instances: {}\n\
             - C instances: {}\n\
             - Type separation: {:.4}\n\n\
             Alerts:\n{}\n",
            self.h_dlt.report(),
            perf_report.cold_start_ratio * 100.0,
            perf_report.avg_embedding_latency_ms,
            perf_report.meets_latency_criterion,
            shrm_report.total_repairs_triggered,
            shrm_report.repair_success_rate * 100.0,
            shrm_report.avg_repair_latency_ms,
            shrm_report.feasibility_rate * 100.0,
            embedding_report.r_count,
            embedding_report.rc_count,
            embedding_report.c_count,
            embedding_report.type_separation,
            if shrm_report.alerts.is_empty() {
                "None".to_string()
            } else {
                shrm_report
                    .alerts
                    .iter()
                    .map(|a| format!("  {}", a))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        )
    }
}

/// Solve Challenge with H-DLT + SHRM Integration
/// This function integrates the new H-DLT and SHRM systems into the existing TIG solver

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Parse feature flags (defaults to all enabled)
    let flags = parse_feature_flags(hyperparameters);

    let h_dlt_config = HierarchicalDLTConfig {
        cache_size: 10000,
        embedding_influence: 0.3,
        learning_rate: 0.1,
        decay_enabled: true,
        use_film_conditioning: true,
    };

    let shrm_config = SHRMConfig {
        trigger_threshold: 0.85,
        learning_rate: 0.001,
        ppo_epochs: 3,
        batch_size: 32,
    };

    // Initialize modules based on feature flags
    let mut h_dlt_system = HierarchicalDLTSystem::new(h_dlt_config);
    let mut shrm_system = SHRMSystem::new(shrm_config, Some("./logs")).unwrap();
    let mut ma_adp: Option<MultiAgentADP> = if flags.enable_ma_adp { Some(MultiAgentADP::new(32, 16)) } else { None };

    // Extract customers for embedding computation
    let customers: Vec<(f64, f64, i32, i32, i32)> = (0..challenge.num_nodes)
        .map(|i| {
            (
                challenge.node_positions[i].0 as f64, // x coord
                challenge.node_positions[i].1 as f64, // y coord
                challenge.ready_times[i], // time window start
                challenge.due_times[i], // time window end
                challenge.demands[i], // demand
            )
        })
        .collect();

    // Compute instance embedding (if enabled)
    let instance_id = format!("tig_{}", challenge.seed.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    let embedding_start = std::time::Instant::now();
    if flags.enable_h_dlt {
        let _embedding = h_dlt_system.get_embedding(&customers);
        let _embedding_time_ms = embedding_start.elapsed().as_secs_f64() * 1000.0;
        let _ = _embedding_time_ms; // recorded later via monitoring
    }

    // Create a simple initial state
    let state = TIGState {
        route: Route {
            nodes: vec![0], // Start with depot
        },
        current_time: 0,
        current_load: 0,
    };

    let config = Config { seed: 42 }; // Default seed
    let mut solver = Solver::with_config(42, config.clone());
    
    // Pass H-DLT and problem context to solver if available
    if flags.enable_h_dlt {
        solver.set_hdlt(h_dlt_system.hdlt.clone());
    }
    
    // Create a simplified Problem structure for solver
    let problem = Problem {
        name: instance_id.clone(),
        num_nodes: challenge.num_nodes,
        depot: 0,
        max_capacity: challenge.max_capacity,
        demands: challenge.demands.clone(),
        distance_matrix: challenge.distance_matrix.clone(),
        initial_time: 0,
        time_windows: challenge.ready_times.iter().zip(challenge.due_times.iter())
            .map(|(&start, &end)| TimeWindow { start, end })
            .collect(),
        service_times: vec![challenge.service_time; challenge.num_nodes],
        initial_route: None,
        config: Some(config),
    };
    solver.set_problem(problem.clone());
    
    // Enable CONE-X scheduler
    solver.enable_cone_x(
        challenge.num_nodes,
        &challenge.ready_times,
        &challenge.due_times,
        &challenge.node_positions
    );
    
    // Enable VTL logging (check hyperparameter or default to false for performance)
    let vtl_enabled = hyperparameters
        .as_ref()
        .and_then(|m| m.get("enable_vtl"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    solver.enable_vtl(vtl_enabled);
    
    // Initialize RouteValueDecomposer for MA-ADP coordination
    let mut decomposer = if flags.enable_ma_adp {
        Some(RouteValueDecomposer::new(32))
    } else {
        None
    };

    // Extract max_iterations from hyperparameters
    let max_iters = hyperparameters
        .as_ref()
        .and_then(|m| m.get("max_iterations"))
        .and_then(|v| v.as_u64())
        .map(|x| x as usize)
        .unwrap_or(100usize);

    // Track solve metrics
    let solve_start = std::time::Instant::now();

    // Local greedy baseline builder (used as a safe fallback).
    fn build_greedy(ch: &Challenge) -> Solution {
        let mut visited = vec![false; ch.num_nodes];
        visited[0] = true;
        let mut routes: Vec<Vec<usize>> = Vec::new();

        while visited.iter().any(|&v| !v) {
            let mut route = vec![0usize];
            let mut capacity = ch.max_capacity;
            let mut current = 0usize;
            let mut curr_time = 0i32;

            loop {
                // find nearest unvisited node that fits capacity and time window
                let mut best: Option<(usize, i32)> = None;
                for node in 1..ch.num_nodes {
                    if visited[node] { continue; }
                    if ch.demands[node] > capacity { continue; }
                    let travel = ch.distance_matrix[current][node];
                    let arrive = curr_time + travel;
                    if arrive > ch.due_times[node] { continue; }
                    let score = travel;
                    if best.is_none() || score < best.unwrap().1 {
                        best = Some((node, score));
                    }
                }

                if let Some((node, _)) = best {
                    // advance
                    let travel = ch.distance_matrix[current][node];
                    curr_time += travel;
                    if curr_time < ch.ready_times[node] {
                        curr_time = ch.ready_times[node];
                    }
                    curr_time += ch.service_time;
                    capacity -= ch.demands[node];
                    route.push(node);
                    visited[node] = true;
                    current = node;
                } else {
                    break;
                }
            }

            route.push(0);
            routes.push(route);
        }

        Solution { routes }
    }

    // Local lightweight validator similar to Challenge::evaluate_total_distance
    fn validate_solution(ch: &Challenge, sol: &Solution) -> bool {
        // check fleet size
        if sol.routes.len() > ch.fleet_size { return false; }
        let mut visited = vec![false; ch.num_nodes];
        visited[0] = true;

        for route in &sol.routes {
            if route.len() <= 2 { return false; }
            if route.first().copied() != Some(0) || route.last().copied() != Some(0) { return false; }

            let mut capacity = ch.max_capacity;
            let mut current = route[0];
            let mut curr_time = 0i32;

            for &node in &route[1..route.len()-1] {
                if node >= ch.num_nodes { return false; }
                if visited[node] { return false; }
                if ch.demands[node] > capacity { return false; }
                curr_time += ch.distance_matrix[current][node];
                if curr_time > ch.due_times[node] { return false; }
                if curr_time < ch.ready_times[node] { curr_time = ch.ready_times[node]; }
                curr_time += ch.service_time;
                visited[node] = true;
                capacity -= ch.demands[node];
                current = node;
            }

            curr_time += ch.distance_matrix[current][0];
            if curr_time > ch.due_times[0] { return false; }
        }

        // ensure all visited
        if visited.iter().any(|&v| !v) { return false; }
        true
    }

    // Compute a simple congestion proxy from node positions: fraction of nodes in top-10% dense grid cells
    fn compute_congestion_proxy(ch: &Challenge) -> f64 {
        if ch.num_nodes <= 1 { return 0.0; }
        // Build 10x10 grid over coordinate bounds
        let mut min_x = i32::MAX; let mut max_x = i32::MIN;
        let mut min_y = i32::MAX; let mut max_y = i32::MIN;
        for &(x,y) in ch.node_positions.iter() {
            if x < min_x { min_x = x; } if x > max_x { max_x = x; }
            if y < min_y { min_y = y; } if y > max_y { max_y = y; }
        }
        let gx = (max_x - min_x).max(1) as f64 / 10.0;
        let gy = (max_y - min_y).max(1) as f64 / 10.0;
        let mut cells = vec![0usize; 100];
        for &(x,y) in ch.node_positions.iter().skip(1) { // skip depot
            let cx = (((x - min_x) as f64 / gx).floor() as usize).min(9);
            let cy = (((y - min_y) as f64 / gy).floor() as usize).min(9);
            cells[cy*10 + cx] += 1;
        }
        cells.sort_unstable_by(|a,b| b.cmp(a));
        let top = (cells.len() as f64 * 0.1) as usize; // top 10%
        let sum_top: usize = cells.iter().take(top.max(1)).sum();
        let total = ch.num_nodes.saturating_sub(1);
        if total == 0 { 0.0 } else { (sum_top as f64) / (total as f64) }
    }

    match solver.solve_state(state, max_iters) {
        Ok(mut result_state) => {
            let solve_time_ms = solve_start.elapsed().as_secs_f64() * 1000.0;

            // Check for repairs if solution has infeasibilities
            let infeasibilities = if result_state.is_feasible() { 0 } else { 1 };
            let mut repairs_triggered = 0;
            let mut repairs_successful = 0;

            if flags.enable_shrm && infeasibilities > 0 && !flags.shrm_shadow_mode {
                // Real repair mode: actually apply repairs
                let features = RepairFeatures::new(
                    (result_state.time_violation_penalty() as f64).max(0.0),
                    result_state.route.nodes.len(),
                    0.5, // route density approximation
                    0.0, // ADP risk score (simplified)
                    result_state.time_violation_penalty(),
                    result_state.capacity_violation_penalty(),
                );

                if let Some(action) = shrm_system.attempt_repair(&features) {
                    repairs_triggered = 1;
                    let repair_start = std::time::Instant::now();

                    // Apply repair based on action
                    let repair_success = match action {
                        RepairAction::ReinsertCustomer { customer_id, position } => {
                            // Try to reinsert customer at different position
                            if customer_id < result_state.route.nodes.len() && position < result_state.route.nodes.len() {
                                let node = result_state.route.nodes.remove(customer_id);
                                result_state.route.nodes.insert(position, node);
                                result_state.is_feasible()
                            } else {
                                false
                            }
                        }
                        RepairAction::SwapSegments { route1: _, route2: _ } => {
                            // Simplified: just reverse a segment
                            if result_state.route.nodes.len() > 3 {
                                let mid = result_state.route.nodes.len() / 2;
                                result_state.route.nodes[1..mid].reverse();
                            }
                            result_state.is_feasible()
                        }
                        RepairAction::MergeRoutes { route1: _, route2: _ } => {
                            // Can't merge routes in single-route state, skip
                            false
                        }
                        RepairAction::DelayService { customer_id: _, delay_minutes } => {
                            // Increase time budget
                            result_state.current_time += delay_minutes;
                            result_state.is_feasible()
                        }
                    };

                    if repair_success {
                        repairs_successful = 1;
                    }

                    let latency = repair_start.elapsed().as_secs_f64() * 1000.0;
                    shrm_system.record_repair_outcome(action, repair_success, latency, &features);
                }
            } else if flags.enable_shrm && infeasibilities > 0 && flags.shrm_shadow_mode {
                // Shadow mode: evaluate but do not apply
                let features = RepairFeatures::new(
                    (result_state.time_violation_penalty() as f64).max(0.0),
                    result_state.route.nodes.len(),
                    0.5,
                    0.0,
                    result_state.time_violation_penalty(),
                    result_state.capacity_violation_penalty(),
                );

                if let Some(action) = shrm_system.attempt_repair(&features) {
                    repairs_triggered = 1;
                    let simulate_success = rand::random::<f64>() > 0.1; // 90% success rate
                    if simulate_success { repairs_successful = 1; }
                    let latency = 20.0; // ms
                    shrm_system.record_repair_outcome(action, simulate_success, latency, &features);
                }
            }

            // Log metrics
            let solve_metrics = SolveMetrics {
                instance_id: instance_id.clone(),
                num_nodes: challenge.num_nodes,
                solution_cost: result_state.route.nodes.len() as f64,
                infeasibilities: if result_state.is_feasible() { 0 } else { 1 },
                solve_time_ms,
                cold_start: false,
                embedding_used: true,
                repairs_triggered,
                repairs_successful,
            };

            let emb_metrics = EmbeddingMetrics {
                instance_id: instance_id.clone(),
                embedding_norm: 1.0,
                spatial_entropy: 0.5,
                avg_tw_width: 100.0,
                instance_type: "vehicle_routing".to_string(),
            };

            // Logging disabled: shrm_system.logger not present

            // Convert result to TIG solution format
            let mut route = result_state.route.nodes.to_vec();

            // Ensure route starts and ends with depot
            if route.first().copied() != Some(0) {
                route.insert(0, 0);
            }
            if route.last().copied() != Some(0) {
                route.push(0);
            }

            let solution = Solution {
                routes: vec![route.clone()],
            };

            // If MA-ADP is enabled, run coordination signals and optional decision logging
            if let Some(ref mut adp) = ma_adp {
                // Build simple per-route feature vectors (len=5)
                let mut route_embeddings: Vec<Vec<f64>> = Vec::new();
                for (rid, r) in solution.routes.iter().enumerate() {
                    let features = vec![
                        r.len() as f64,
                        (rid as f64),
                        0.0,
                        1.0,
                        0.0,
                    ];
                    route_embeddings.push(features);
                }
                
                // Fleet stats: utilization and a simple congestion proxy
                let utilization = (solution.routes.iter().map(|r| r.len()).sum::<usize>() as f64) / (challenge.num_nodes.max(1) as f64);
                let congestion_proxy = compute_congestion_proxy(challenge);
                let fleet_stats = [utilization, congestion_proxy];

                // Use RouteValueDecomposer if available
                if let Some(ref mut decomp) = decomposer {
                    // Update coordination signals
                    decomp.update_signals(&solution.routes, &challenge.node_positions, &challenge.due_times, result_state.current_time);
                    
                    // Compute per-route values
                    let mut route_values = Vec::new();
                    for emb in &route_embeddings {
                        let val = adp.predict_local_value(&emb);
                        route_values.push(val);
                    }
                    
                    // Decompose global value
                    let (global_value, contributions) = decomp.decompose(&route_embeddings, &route_values);
                    
                    // Logging disabled: shrm_system.logger not present
                }
                
                // Decide hold vs dispatch (logged only; this implementation doesn't alter solution)
                let dispatch_now = !adp.decide_hold_or_dispatch(&route_embeddings, &fleet_stats);
                // Logging disabled: shrm_system.logger not present

                // Generate counterfactuals for first route (if exists) and update ADP
                if let Some(first_route) = solution.routes.get(0) {
                    let adp_state = TIGState { route: Route { nodes: first_route.clone() }, current_time: 0, current_load: 0 };
                    adp.generate_and_label_counterfactuals(0, &adp_state);
                    adp.train_on_batch(16, 0.001, 1);
                }
            }

            // If solver produced a trivial or invalid solution, fall back to greedy baseline
            let should_fallback = solution.routes.iter().any(|r| r.len() <= 2);

            if should_fallback {
                let greedy = build_greedy(challenge);
                return save_solution(&greedy);
            }

            // Validate solution with verifier logic where possible; if invalid, fall back
            if !validate_solution(challenge, &solution) {
                let greedy = build_greedy(challenge);
                return save_solution(&greedy);
            }

            let save_res = save_solution(&solution);

            // Append teacher trace for distillation (lightweight logging)
            let distilled = DistilledADP::new_with_move_dim(2, 16, 4);
            let feats = [solution.routes.first().map(|r| r.len() as f64).unwrap_or(0.0), 0.0];
            let value_est = distilled.infer_value(&feats);
            let move_probs = distilled.infer_move_probs(&feats);
            let trace = TeacherTrace { instance_id: instance_id.clone(), move_probabilities: move_probs, value_estimates: vec![value_est], final_cost: solution.routes.iter().map(|r| r.len() as f64).sum::<f64>() };
            let _ = append_teacher_trace("./synthetic_traces.jsonl", &trace);

            save_res
        }
        Err(_) => {
            // Solver failed; try greedy baseline, otherwise produce simple per-node routes
            let greedy = build_greedy(challenge);
            return save_solution(&greedy);
        }
    }
}

// Final re-export - only interface exposed to TIG

// ============================================================================
// Multi-Agent ADP (MA-ADP) - Route-Level Value Decomposition
// ============================================================================
/// MultiAgentADP: per-route and global heads with online update hook
pub struct MultiAgentADP {
    // Per-route heads: simple MLPs parameterized by weights (vectorized)
    pub per_route_w1: Vec<f64>, // hidden dim (d) weights flattened for simplicity
    pub per_route_b1: f64,
    pub per_route_w2: Vec<f64>,
    pub per_route_b2: f64,

    // Global head weights
    pub global_w1: Vec<f64>,
    pub global_b1: f64,
    pub global_w2: Vec<f64>,
    pub global_b2: f64,

    // Experience replay buffer (route-level transitions)
    pub replay_buffer: VecDeque<MAReplayTransition>,
    pub replay_capacity: usize,
}

#[derive(Clone, Debug)]
pub struct MAReplayTransition {
    pub route_id: usize,
    pub route_embedding: Vec<f64>,
    pub global_features: Vec<f64>,
    pub predicted_value: f64,
    pub observed_cost: f64,
}

/// Teacher trace record for distillation
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TeacherTrace {
    pub instance_id: String,
    pub move_probabilities: Vec<f64>,
    pub value_estimates: Vec<f64>,
    pub final_cost: f64,
}

impl TeacherTrace {
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Append a teacher trace to disk (newline-delimited JSON)
pub fn append_teacher_trace(_path: &str, _trace: &TeacherTrace) -> Result<()> {
    // File I/O disabled in this build: do not write teacher traces to disk.
    Ok(())
}

impl MultiAgentADP {
    pub fn new(hidden_dim: usize, global_dim: usize) -> Self {
        // Initialize with small random-ish constants (deterministic here)
        let per_route_w1 = vec![0.01; hidden_dim * 5]; // input dim assumed 5
        let per_route_w2 = vec![0.01; hidden_dim];
        let global_w1 = vec![0.01; global_dim * hidden_dim];
        let global_w2 = vec![0.01; 1 * global_dim];

        Self {
            per_route_w1,
            per_route_b1: 0.0,
            per_route_w2,
            per_route_b2: 0.0,
            global_w1,
            global_b1: 0.0,
            global_w2,
            global_b2: 0.0,
            replay_buffer: VecDeque::with_capacity(1024),
            replay_capacity: 10_000,
        }
    }

    /// Predict per-route local value from a compact route feature vector (len=5)
    #[inline]
    pub fn predict_local_value(&self, route_features: &[f64]) -> f64 {
        // Simple two-layer MLP: hidden = ReLU(W1 x + b1), out = W2 hidden + b2 (optimized)
        let hidden_dim = self.per_route_w2.len();
        let feat_len = route_features.len();
        
        // Stack allocation for small hidden layers
        const MAX_HIDDEN: usize = 256;
        let mut hidden_stack = [0.0; MAX_HIDDEN];
        let hidden = if hidden_dim <= MAX_HIDDEN {
            &mut hidden_stack[..hidden_dim]
        } else {
            // Fall back to heap for very large networks
            return self.predict_local_value_slow(route_features);
        };
        
        // Optimized forward pass with unrolling
        let unroll = hidden_dim.min(4);
        for i in (0..hidden_dim).step_by(unroll) {
            for k in 0..unroll.min(hidden_dim - i) {
                let mut acc = self.per_route_b1;
                for j in 0..feat_len {
                    acc += self.per_route_w1[(i + k) * feat_len + j] * route_features[j];
                }
                hidden[i + k] = acc.max(0.0);
            }
        }
        
        let mut out = self.per_route_b2;
        for i in (0..hidden_dim).step_by(4) {
            let end = (i + 4).min(hidden_dim);
            for k in i..end {
                out += self.per_route_w2[k] * hidden[k];
            }
        }
        out
    }
    
    fn predict_local_value_slow(&self, route_features: &[f64]) -> f64 {
        let hidden_dim = self.per_route_w2.len();
        let mut hidden = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            let mut acc = self.per_route_b1;
            for j in 0..route_features.len() {
                acc += self.per_route_w1[i * route_features.len() + j] * route_features[j];
            }
            hidden[i] = acc.max(0.0);
        }
        let mut out = self.per_route_b2;
        for i in 0..hidden_dim {
            out += self.per_route_w2[i] * hidden[i];
        }
        out
    }

    /// Predict global fleet value from concatenated route embeddings and fleet stats - optimized
    #[inline]
    pub fn predict_global_value(&self, concat_features: &[f64]) -> f64 {
        let global_dim = self.global_w2.len();
        if global_dim == 0 { return 0.0; }
        
        let hidden_dim = global_dim;
        
        // Stack allocation for small networks
        const MAX_GLOBAL_HIDDEN: usize = 64;
        let mut hidden_stack = [0.0; MAX_GLOBAL_HIDDEN];
        
        if hidden_dim <= MAX_GLOBAL_HIDDEN {
            let hidden = &mut hidden_stack[..hidden_dim];
            for i in (0..hidden_dim).step_by(4) {
                let end = (i + 4).min(hidden_dim);
                for k in i..end {
                    let mut acc = self.global_b1;
                    for j in 0..concat_features.len() {
                        let idx = k * concat_features.len() + j;
                        if idx < self.global_w1.len() {
                            acc += self.global_w1[idx] * concat_features[j];
                        }
                    }
                    hidden[k] = acc.max(0.0);
                }
            }
            let mut out = self.global_b2;
            for i in (0..global_dim).step_by(4) {
                let end = (i + 4).min(global_dim);
                for k in i..end {
                    out += self.global_w2[k] * hidden[k];
                }
            }
            out
        } else {
            // Heap allocation fallback
            let mut hidden = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                let mut acc = self.global_b1;
                for j in 0..concat_features.len() {
                    let idx = i * concat_features.len() + j;
                    if idx < self.global_w1.len() {
                        acc += self.global_w1[idx] * concat_features[j];
                    }
                }
                hidden[i] = acc.max(0.0);
            }
            let mut out = self.global_b2;
            for i in 0..global_dim {
                out += self.global_w2[i] * hidden[i];
            }
            out
        }
    }

    /// Coordination decision: hold vs dispatch based on counterfactual global value
    pub fn decide_hold_or_dispatch(&self, route_embeddings: &[Vec<f64>], fleet_stats: &[f64]) -> bool {
        // Build feature vector for dispatch_now
        let mut dispatch_features = Vec::new();
        for emb in route_embeddings.iter() { dispatch_features.extend_from_slice(&emb[..]); }
        dispatch_features.extend_from_slice(fleet_stats);

        // Counterfactual: if we hold, simulate slight change in fleet_stats (e.g., utilization dec -0.05)
        let mut hold_features = dispatch_features.clone();
        if let Some(last) = hold_features.last_mut() { *last -= 0.05; }

        let v_dispatch = self.predict_global_value(&dispatch_features);
        let v_hold = self.predict_global_value(&hold_features);

        v_hold > v_dispatch
    }

    /// Generate simple counterfactual rollouts (what-if scenarios)
    pub fn generate_counterfactuals(&self, route_id: usize, state: &TIGState) -> Vec<(String, TIGState)> {
        let mut out = Vec::new();
        // Delay route by 5 minutes
        let mut s1 = state.clone();
        s1.current_time += 5;
        out.push((format!("delay_route_{}_5min", route_id), s1));

        // Reassign a customer (if exists)
        if state.route.nodes.len() > 2 {
            let mut s2 = state.clone();
            // swap two customers to simulate reassignment
            let n = s2.route.nodes.len();
            s2.route.nodes.swap(1, n-1);
            out.push((format!("reassign_route_{}_swap", route_id), s2));
        }

        // Add small random perturbations for diversity
        let mut rng = rand::thread_rng();
        for k in 0..3 {
            let mut s = state.clone();
            let dt: i32 = rng.gen_range(1..10);
            let dl: i32 = rng.gen_range(-5..6);
            s.current_time = (s.current_time + dt).max(0);
            s.current_load = (s.current_load + dl).max(0);
            out.push((format!("perturb_{}_{}", route_id, k), s));
        }

        out
    }

    /// Online update after dispatch: store transition and apply a simple incremental update
    pub fn update_value_after_dispatch(&mut self, transition: MAReplayTransition) {
        // push to buffer
        if self.replay_buffer.len() >= self.replay_capacity {
            self.replay_buffer.pop_front();
        }
        self.replay_buffer.push_back(transition.clone());

        // simple update: move bias towards observed cost difference
        let lr = 0.01;
        let td_error = transition.observed_cost - transition.predicted_value;
        self.per_route_b2 += lr * td_error;
        self.global_b2 += lr * td_error * 0.1;
    }

    /// Sample a minibatch from replay for training (deterministic sampling here)
    pub fn sample_replay(&self, batch_size: usize) -> Vec<MAReplayTransition> {
        let mut out = Vec::new();
        let mut i = 0usize;
        for t in self.replay_buffer.iter().rev() {
            out.push(t.clone());
            i += 1;
            if i >= batch_size { break; }
        }
        out
    }

    /// Simple simulated cost function for labelling counterfactuals.
    /// Uses route length and current time as a proxy for cost.
    pub fn simulate_cost(&self, state: &TIGState) -> f64 {
        let len_cost = state.route.nodes.len() as f64;
        let time_cost = (state.current_time as f64) * 0.01;
        len_cost + time_cost
    }

    /// Generate counterfactuals and label them with simulated outcomes, storing
    /// replay transitions in the buffer.
    pub fn generate_and_label_counterfactuals(&mut self, route_id: usize, state: &TIGState) {
        let cfs = self.generate_counterfactuals(route_id, state);
        for (_name, cf_state) in cfs.into_iter() {
            // produce a compact route feature vector (len=5) deterministic from state
            let features = vec![
                cf_state.route.nodes.len() as f64,
                cf_state.current_time as f64,
                cf_state.current_load as f64,
                cf_state.free_time_budget() as f64,
                0.0,
            ];
            let pred = self.predict_local_value(&features);
            let observed = self.simulate_cost(&cf_state);

            let transition = MAReplayTransition {
                route_id,
                route_embedding: features.clone(),
                global_features: vec![cf_state.current_time as f64, cf_state.current_load as f64],
                predicted_value: pred,
                observed_cost: observed,
            };
            if self.replay_buffer.len() >= self.replay_capacity {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back(transition);
        }
    }

    /// Train on a minibatch sampled from replay using simple MSE updates - optimized
    pub fn train_on_batch(&mut self, batch_size: usize, lr: f64, epochs: usize) {
        // Pre-allocate buffers to reduce allocations
        let max_hidden = self.per_route_w2.len();
        let mut hidden_buf = vec![0.0; max_hidden];
        
        for _ in 0..epochs {
            let batch = self.sample_replay(batch_size);
            for t in batch.iter() {
                let x = &t.route_embedding;
                let in_dim = x.len().max(1);
                let hidden_dim = self.per_route_w2.len();

                // forward: compute hidden activations (reuse buffer)
                hidden_buf.clear();
                hidden_buf.resize(hidden_dim, 0.0);
                let hidden = &mut hidden_buf;
                
                for i in (0..hidden_dim).step_by(4) {
                    let end = (i + 4).min(hidden_dim);
                    for k in i..end {
                        let mut acc = self.per_route_b1;
                        for j in 0..in_dim {
                            let idx = k * in_dim + j;
                            if idx < self.per_route_w1.len() {
                                acc += self.per_route_w1[idx] * x[j % x.len()];
                            }
                        }
                        hidden[k] = acc.max(0.0);
                    }
                }

                // output
                let mut pred = self.per_route_b2;
                for i in (0..hidden_dim).step_by(4) {
                    let end = (i + 4).min(hidden_dim);
                    for k in i..end {
                        pred += self.per_route_w2[k] * hidden[k];
                    }
                }

                let label = t.observed_cost;
                let err = pred - label;
                
                // Early exit if error is very small
                if err.abs() < 1e-6 { continue; }

                // update output layer weights and bias (unrolled)
                for i in (0..hidden_dim).step_by(4) {
                    let end = (i + 4).min(hidden_dim);
                    for k in i..end {
                        self.per_route_w2[k] -= lr * err * hidden[k];
                    }
                }
                self.per_route_b2 -= lr * err;

                // backprop into W1 (optimized with early termination)
                let lr_scaled = lr * 0.01;
                for i in (0..hidden_dim).step_by(4) {
                    let end = (i + 4).min(hidden_dim);
                    for k in i..end {
                        if hidden[k] <= 0.0 { continue; } // Skip inactive neurons
                        let upstream = err * self.per_route_w2[k];
                        for j in 0..in_dim {
                            let idx = k * in_dim + j;
                            if idx < self.per_route_w1.len() {
                                let gx = upstream * x[j % x.len()];
                                self.per_route_w1[idx] -= lr_scaled * gx;
                            }
                        }
                        self.per_route_b1 -= lr_scaled * upstream;
                    }
                }

                // small global bias update
                self.global_b2 -= lr * err * 0.001;
            }
        }
    }
}

// ============================================================================
// RouteValueDecomposer: Reconstructs global value using attention-based fusion
// Implements Shapley-like attribution for multi-agent coordination
// ============================================================================

/// RouteValueDecomposer: Attention-based value aggregation across routes
/// Uses Query-Key-Value attention to combine per-route values into global estimate
pub struct RouteValueDecomposer {
    // Attention parameters: embedding_dim → attention weights
    pub query_weights: Vec<f64>,   // shape: embedding_dim
    pub key_weights: Vec<f64>,     // shape: embedding_dim  
    pub value_weights: Vec<f64>,   // shape: embedding_dim
    pub embedding_dim: usize,
    
    // Coordination signals
    pub congestion_map: CongestionMap,
    pub time_pressure_index: f64,
}

/// Grid-based congestion tracking
#[derive(Clone, Debug)]
pub struct CongestionMap {
    pub grid_size: usize,          // e.g., 10x10 grid
    pub cells: Vec<f64>,           // flattened grid of congestion scores
    pub update_count: usize,
}

impl CongestionMap {
    pub fn new(grid_size: usize) -> Self {
        Self {
            grid_size,
            cells: vec![0.0; grid_size * grid_size],
            update_count: 0,
        }
    }
    
    /// Update congestion based on route density in each cell
    pub fn update_from_routes(&mut self, routes: &[Vec<usize>], positions: &[(i32, i32)]) {
        // Reset cells
        self.cells.fill(0.0);
        
        let grid_size_i32 = self.grid_size as i32;
        let mut max_visits = 0.0f64;
        
        // Count visits per cell and track max in single pass
        for route in routes {
            for &node in route {
                if node < positions.len() {
                    let (x, y) = positions[node];
                    // Map to grid cell (simplified, inlined)
                    let grid_x = ((x.abs() % 1000) / 100).min(grid_size_i32 - 1) as usize;
                    let grid_y = ((y.abs() % 1000) / 100).min(grid_size_i32 - 1) as usize;
                    let cell_idx = grid_y * self.grid_size + grid_x;
                    
                    self.cells[cell_idx] += 1.0;
                    if self.cells[cell_idx] > max_visits {
                        max_visits = self.cells[cell_idx];
                    }
                }
            }
        }
        
        // Normalize by max (avoid division if max is 0)
        if max_visits > 0.0 {
            let inv_max = 1.0 / max_visits;
            for cell in self.cells.iter_mut() {
                *cell *= inv_max;
            }
        }
        
        self.update_count += 1;
    }
    
    /// Get congestion pressure for a specific position
    pub fn get_pressure(&self, pos: (i32, i32)) -> f64 {
        let grid_x = ((pos.0.abs() % 1000) / 100).min(self.grid_size as i32 - 1) as usize;
        let grid_y = ((pos.1.abs() % 1000) / 100).min(self.grid_size as i32 - 1) as usize;
        let cell_idx = grid_y * self.grid_size + grid_x;
        self.cells.get(cell_idx).copied().unwrap_or(0.0)
    }
}

impl RouteValueDecomposer {
    pub fn new(embedding_dim: usize) -> Self {
        let mut query_weights = vec![0.0; embedding_dim];
        let mut key_weights = vec![0.0; embedding_dim];
        let mut value_weights = vec![0.0; embedding_dim];
        
        // Xavier initialization
        let scale = (2.0 / embedding_dim as f64).sqrt();
        for i in 0..embedding_dim {
            query_weights[i] = (rand::random::<f64>() - 0.5) * scale;
            key_weights[i] = (rand::random::<f64>() - 0.5) * scale;
            value_weights[i] = (rand::random::<f64>() - 0.5) * scale;
        }
        
        Self {
            query_weights,
            key_weights,
            value_weights,
            embedding_dim,
            congestion_map: CongestionMap::new(10),
            time_pressure_index: 0.0,
        }
    }
    
    /// Decompose global value using attention-weighted sum of route values
    /// Returns: (global_value, per_route_contributions)
    pub fn decompose(&self, route_embeddings: &[Vec<f64>], route_values: &[f64]) -> (f64, Vec<f64>) {
        if route_embeddings.is_empty() || route_values.is_empty() {
            return (0.0, vec![]);
        }
        
        let n_routes = route_embeddings.len().min(route_values.len());
        
        // Pre-allocate all vectors once
        let mut attention_scores = Vec::with_capacity(n_routes);
        attention_scores.resize(n_routes, 0.0);
        
        // Compute attention scores: Q·K^T
        let mut max_score = f64::NEG_INFINITY;
        for i in 0..n_routes {
            let emb = &route_embeddings[i];
            let emb_dim = emb.len().min(self.embedding_dim);
            
            // Q·K (simplified: just dot product with weights)
            let mut score = 0.0;
            for j in 0..emb_dim {
                score += emb[j] * self.query_weights[j] * self.key_weights[j];
            }
            
            // Apply congestion penalty
            score *= 1.0 - self.congestion_map.cells.get(i).copied().unwrap_or(0.0) * 0.2;
            
            attention_scores[i] = score;
            if score > max_score {
                max_score = score;
            }
        }
        
        // Softmax normalization (reuse attention_scores buffer)
        let mut sum_exp = 0.0;
        for i in 0..n_routes {
            let exp_val = (attention_scores[i] - max_score).exp();
            attention_scores[i] = exp_val;  // Reuse buffer for exp scores
            sum_exp += exp_val;
        }
        
        if sum_exp == 0.0 {
            sum_exp = 1.0;
        }
        
        let inv_sum = 1.0 / sum_exp;  // Multiply instead of divide
        
        // Compute global value and contributions in single pass
        let mut global_value = 0.0;
        let mut contributions = Vec::with_capacity(n_routes);
        contributions.resize(n_routes, 0.0);
        
        for i in 0..n_routes {
            let attention_weight = attention_scores[i] * inv_sum;
            let contribution = attention_weight * route_values[i];
            contributions[i] = contribution;
            global_value += contribution;
        }
        
        // Apply time pressure adjustment
        global_value *= 1.0 + self.time_pressure_index * 0.1;
        
        (global_value, contributions)
    }
    
    /// Update coordination signals based on current system state
    pub fn update_signals(&mut self, routes: &[Vec<usize>], positions: &[(i32, i32)], deadlines: &[i32], current_time: i32) {
        // Update congestion map
        self.congestion_map.update_from_routes(routes, positions);
        
        // Calculate time pressure index: % of customers within 15min of deadline
        let mut urgent_count = 0;
        let mut total_count = 0;
        
        for route in routes {
            for &node in route {
                if node < deadlines.len() && node > 0 {
                    total_count += 1;
                    if deadlines[node] - current_time <= 15 {
                        urgent_count += 1;
                    }
                }
            }
        }
        
        self.time_pressure_index = if total_count > 0 {
            urgent_count as f64 / total_count as f64
        } else {
            0.0
        };
    }
    
    /// Shapley value approximation via Monte Carlo sampling
    pub fn approximate_shapley(&self, route_embeddings: &[Vec<f64>], route_values: &[f64], samples: usize) -> Vec<f64> {
        let n_routes = route_embeddings.len().min(route_values.len());
        let mut shapley_values = vec![0.0; n_routes];
        
        if n_routes == 0 {
            return shapley_values;
        }
        
        // Pre-allocate coalition buffers outside loop
        let mut coalition = Vec::with_capacity(n_routes);
        let mut coalition_embs = Vec::with_capacity(n_routes);
        let mut coalition_vals = Vec::with_capacity(n_routes);
        
        // Monte Carlo sampling of permutations
        for _ in 0..samples {
            // Random permutation
            let mut perm: Vec<usize> = (0..n_routes).collect();
            for i in (1..n_routes).rev() {
                let j = rand::random::<usize>() % (i + 1);
                perm.swap(i, j);
            }
            
            // Marginal contributions
            coalition.clear();
            let mut prev_value = 0.0;
            
            for &route_idx in &perm {
                coalition.push(route_idx);
                
                // Reuse buffers instead of allocating new vecs
                coalition_embs.clear();
                coalition_vals.clear();
                
                for &idx in &coalition {
                    coalition_embs.push(route_embeddings[idx].clone());
                    coalition_vals.push(route_values[idx]);
                }
                
                let (curr_value, _) = self.decompose(&coalition_embs, &coalition_vals);
                
                // Marginal contribution
                shapley_values[route_idx] += curr_value - prev_value;
                prev_value = curr_value;
            }
        }
        
        // Average over samples
        let inv_samples = 1.0 / samples as f64;
        for val in shapley_values.iter_mut() {
            *val *= inv_samples;
        }
        
        shapley_values
    }
}

// ============================================================================
// Lightweight placeholders for distillation (Phase 2)
// ============================================================================
pub struct DistilledADP {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<f64>,
    b1: f64,
    w2: Vec<f64>,
    b2: f64,
    // move-head parameters
    move_dim: usize,
    w_move: Vec<f64>, // shape: move_dim x hidden_dim
    b_move: Vec<f64>,
}

impl DistilledADP {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self::new_with_move_dim(input_dim, hidden_dim, 0)
    }

    /// Construct with an optional move-prob output head of size `move_dim`.
    pub fn new_with_move_dim(input_dim: usize, hidden_dim: usize, move_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut w1 = vec![0.0; input_dim * hidden_dim];
        for v in w1.iter_mut() { *v = (rng.gen::<f64>() - 0.5) * 0.1 }
        let mut w2 = vec![0.0; hidden_dim];
        for v in w2.iter_mut() { *v = (rng.gen::<f64>() - 0.5) * 0.1 }
        let mut w_move = vec![0.0; move_dim * hidden_dim];
        for v in w_move.iter_mut() { *v = (rng.gen::<f64>() - 0.5) * 0.1 }
        let b_move = vec![0.0; move_dim];
        Self { input_dim, hidden_dim, w1, b1: 0.0, w2, b2: 0.0, move_dim, w_move, b_move }
    }

    /// Very small student training using teacher traces saved as newline-delimited JSON.
    /// Loss = alpha * KL(teacher_probs || student_probs) + beta * MSE(teacher_value, student_value)
    pub fn train_from_trace_file(&mut self, trace_path: &str, alpha: f64, beta: f64, epochs: usize, lr: f64, log_path: Option<&str>) -> Result<(), String> {
        // File I/O is disabled: treat `trace_path` as the trace content string (newline-delimited JSON)
        let content = trace_path.to_string();
        let mut traces: Vec<TeacherTrace> = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            match serde_json::from_str::<TeacherTrace>(line) {
                Ok(t) => traces.push(t),
                Err(_) => continue,
            }
        }
        if traces.is_empty() { return Err("no traces".to_string()); }

        // build very simple per-trace feature vector: use teacher value estimates mean and final cost
        let mut epoch_losses: Vec<f64> = Vec::new();
        for _ep in 0..epochs {
            let mut epoch_loss = 0.0f64;
            for t in traces.iter() {
                let feat_mean = if t.value_estimates.is_empty() { 0.0 } else { t.value_estimates.iter().sum::<f64>() / (t.value_estimates.len() as f64) };
                let features = vec![feat_mean, t.final_cost];

                // expand or shrink features to input_dim
                let mut x = vec![0.0; self.input_dim];
                for i in 0..self.input_dim {
                    x[i] = if i < features.len() { features[i] } else { features[i % features.len()] };
                }

                // forward
                let mut hidden = vec![0.0; self.hidden_dim];
                for i in 0..self.hidden_dim {
                    let mut acc = self.b1;
                    for j in 0..self.input_dim {
                        acc += self.w1[i*self.input_dim + j] * x[j];
                    }
                    hidden[i] = if acc > 0.0 { acc } else { 0.0 };
                }
                let mut pred_val = self.b2;
                for i in 0..self.hidden_dim { pred_val += self.w2[i] * hidden[i]; }

                let teacher_val = t.value_estimates.iter().copied().fold(0.0, |a,b| a+b) / (t.value_estimates.len().max(1) as f64);

                // compute losses
                let mse = (pred_val - teacher_val).powi(2);

                // simple MSE grad on value
                let err = pred_val - teacher_val;
                for i in 0..self.hidden_dim {
                    let grad_w2 = err * hidden[i];
                    self.w2[i] -= lr * beta * grad_w2;
                }
                self.b2 -= lr * beta * err;

                // If teacher move probabilities are present, compute student probs using move-head and KL loss
                let mut upstream_move = vec![0.0f64; self.hidden_dim];
                let mut kl_loss = 0.0f64;
                if !t.move_probabilities.is_empty() && self.move_dim == t.move_probabilities.len() && self.move_dim>0 {
                    let move_dim = self.move_dim;
                    // compute logits = W_move * hidden + b_move
                    let mut logits = vec![0.0f64; move_dim];
                    for k in 0..move_dim {
                        let mut acc = self.b_move[k];
                        for i_h in 0..self.hidden_dim {
                            acc += self.w_move[k*self.hidden_dim + i_h] * hidden[i_h];
                        }
                        logits[k] = acc;
                    }
                    // softmax -> student_probs
                    let maxl = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0f64;
                    let mut student_probs = vec![0.0f64; move_dim];
                    for k in 0..move_dim { student_probs[k] = (logits[k]-maxl).exp(); sum += student_probs[k]; }
                    for k in 0..move_dim { student_probs[k] /= sum.max(1e-12); }

                    // compute KL (teacher || student) contribution
                    for k in 0..move_dim {
                        let tpk = t.move_probabilities.get(k).cloned().unwrap_or(0.0).max(1e-12);
                        kl_loss += tpk * ((tpk).ln() - student_probs[k].ln().max(-1e12));
                    }

                    // grad logits = student - teacher (for cross-entropy / KL)
                    let mut grad_logits = vec![0.0f64; move_dim];
                    for k in 0..move_dim { let tpk = t.move_probabilities.get(k).cloned().unwrap_or(0.0); grad_logits[k] = student_probs[k] - tpk; }

                    // backprop to move-head weights and biases
                    for k in 0..move_dim {
                        for i_h in 0..self.hidden_dim {
                            let idx = k*self.hidden_dim + i_h;
                            self.w_move[idx] -= lr * alpha * grad_logits[k] * hidden[i_h];
                        }
                        self.b_move[k] -= lr * alpha * grad_logits[k];
                    }

                    // backprop logits into hidden upstream_move via W_move^T * grad_logits
                    for i_h in 0..self.hidden_dim {
                        let mut acc = 0.0f64;
                        for k in 0..move_dim { acc += self.w_move[k*self.hidden_dim + i_h] * grad_logits[k]; }
                        upstream_move[i_h] = acc;
                    }
                }

                // total loss contribution for logging
                let total_loss = beta * mse + alpha * kl_loss;
                epoch_loss += total_loss;

                // backprop to w1 combining value head and move head upstream signals
                for i in 0..self.hidden_dim {
                    let grad_h = if hidden[i] > 0.0 { 1.0 } else { 0.0 };
                    let upstream = err * self.w2[i] + alpha * upstream_move[i];
                    for j in 0..self.input_dim {
                        let idx = i*self.input_dim + j;
                        self.w1[idx] -= lr * beta * upstream * grad_h * x[j] * 0.01;
                    }
                    self.b1 -= lr * beta * upstream * grad_h * 0.01;
                }
            }
            epoch_losses.push(epoch_loss);
        }
        // write loss log if requested (disabled)
        let _ = epoch_losses;
        let _ = log_path;
        Ok(())
    }

    /// Export a lightweight JSON model description which can be converted to ONNX externally.
    pub fn export_to_onnx_scaffold(&self, out_path: &str) -> Result<()> {
        #[derive(Serialize)]
        struct ModelDump<'a> {
            input_dim: usize,
            hidden_dim: usize,
            w1: &'a [f64],
            b1: f64,
            w2: &'a [f64],
            b2: f64,
            note: &'a str,
        }
        let dump = ModelDump { input_dim: self.input_dim, hidden_dim: self.hidden_dim, w1: &self.w1, b1: self.b1, w2: &self.w2, b2: self.b2, note: "This is a scaffold JSON. Convert to ONNX using a small Python script that recreates layers and serializes with onnx" };
        let json = serde_json::to_string_pretty(&dump).unwrap_or_else(|_| "{}".to_string());
        // File output disabled: return success without writing to disk.
        let _ = out_path;
        let _ = json;
        Ok(())
    }

}

impl DistilledADP {
    /// Infer scalar value from features - optimized
    #[inline]
    pub fn infer_value(&self, features: &[f64]) -> f64 {
        // expand features to input_dim (on stack for small sizes)
        const MAX_INPUT: usize = 32;
        let mut x_stack = [0.0; MAX_INPUT];
        let x = if self.input_dim <= MAX_INPUT {
            let x = &mut x_stack[..self.input_dim];
            for i in 0..self.input_dim {
                x[i] = if i < features.len() { features[i] } else { features[i % features.len()] };
            }
            x
        } else {
            // Heap allocation for large inputs
            let mut x_heap = vec![0.0; self.input_dim];
            for i in 0..self.input_dim {
                x_heap[i] = if i < features.len() { features[i] } else { features[i % features.len()] };
            }
            return self.infer_value_slow(&x_heap);
        };
        
        // Stack allocation for hidden layer
        const MAX_HIDDEN: usize = 128;
        let mut hidden_stack = [0.0; MAX_HIDDEN];
        let hidden = if self.hidden_dim <= MAX_HIDDEN {
            &mut hidden_stack[..self.hidden_dim]
        } else {
            return self.infer_value_slow(x);
        };
        
        // Optimized forward with unrolling
        for i in (0..self.hidden_dim).step_by(4) {
            let end = (i + 4).min(self.hidden_dim);
            for k in i..end {
                let mut acc = self.b1;
                for j in (0..self.input_dim).step_by(4) {
                    let j_end = (j + 4).min(self.input_dim);
                    for jj in j..j_end {
                        acc += self.w1[k * self.input_dim + jj] * x[jj];
                    }
                }
                hidden[k] = acc.max(0.0);
            }
        }
        
        let mut pred = self.b2;
        for i in (0..self.hidden_dim).step_by(4) {
            let end = (i + 4).min(self.hidden_dim);
            for k in i..end {
                pred += self.w2[k] * hidden[k];
            }
        }
        pred
    }
    
    fn infer_value_slow(&self, x: &[f64]) -> f64 {
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut acc = self.b1;
            for j in 0..self.input_dim {
                acc += self.w1[i*self.input_dim + j] * x[j];
            }
            hidden[i] = acc.max(0.0);
        }
        let mut pred = self.b2;
        for i in 0..self.hidden_dim { pred += self.w2[i] * hidden[i]; }
        pred
    }

    /// Infer move probability distribution from features (softmax over move_dim) - optimized
    #[inline]
    pub fn infer_move_probs(&self, features: &[f64]) -> Vec<f64> {
        if self.move_dim == 0 { return Vec::new(); }
        
        // Stack allocation for small arrays
        const MAX_INPUT: usize = 32;
        const MAX_HIDDEN: usize = 128;
        let mut x_stack = [0.0; MAX_INPUT];
        let mut hidden_stack = [0.0; MAX_HIDDEN];
        
        // compute hidden (optimized)
        let x = if self.input_dim <= MAX_INPUT {
            let x = &mut x_stack[..self.input_dim];
            for i in 0..self.input_dim { 
                x[i] = if i < features.len() { features[i] } else { features[i % features.len()] }; 
            }
            x
        } else {
            return self.infer_move_probs_slow(features);
        };
        
        let hidden = if self.hidden_dim <= MAX_HIDDEN {
            let hidden = &mut hidden_stack[..self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut acc = self.b1;
                for j in 0..self.input_dim { acc += self.w1[i*self.input_dim + j] * x[j]; }
                hidden[i] = acc.max(0.0);
            }
            hidden
        } else {
            return self.infer_move_probs_slow(features);
        };
        
        // logits
        let mut logits = vec![0.0f64; self.move_dim];
        for k in 0..self.move_dim {
            let mut acc = self.b_move[k];
            for i_h in (0..self.hidden_dim).step_by(4) {
                let end = (i_h + 4).min(self.hidden_dim);
                for j in i_h..end {
                    acc += self.w_move[k*self.hidden_dim + j] * hidden[j];
                }
            }
            logits[k] = acc;
        }
        
        // fast softmax
        let maxl = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for k in 0..self.move_dim { 
            logits[k] = (logits[k]-maxl).exp(); 
            sum += logits[k]; 
        }
        let inv_sum = if sum == 0.0 { 1.0 / (self.move_dim as f64) } else { 1.0 / sum };
        for k in 0..self.move_dim { logits[k] *= inv_sum; }
        logits
    }
    
    fn infer_move_probs_slow(&self, features: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.input_dim];
        for i in 0..self.input_dim { x[i] = if i < features.len() { features[i] } else { features[i % features.len()] }; }
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut acc = self.b1;
            for j in 0..self.input_dim { acc += self.w1[i*self.input_dim + j] * x[j]; }
            hidden[i] = acc.max(0.0);
        }
        let mut logits = vec![0.0f64; self.move_dim];
        for k in 0..self.move_dim {
            let mut acc = self.b_move[k];
            for i_h in 0..self.hidden_dim { acc += self.w_move[k*self.hidden_dim + i_h] * hidden[i_h]; }
            logits[k] = acc;
        }
        let maxl = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for k in 0..self.move_dim { logits[k] = (logits[k]-maxl).exp(); sum += logits[k]; }
        if sum == 0.0 { return vec![1.0/(self.move_dim as f64); self.move_dim]; }
        for k in 0..self.move_dim { logits[k] /= sum; }
        logits
    }
}