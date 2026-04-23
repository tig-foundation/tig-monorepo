use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

// =============================================================================
// Operation types
// =============================================================================

/// Operation types in the circuit DAG.
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    /// Placeholder during DAG construction
    Undefined,
    /// Public input variable
    Input,
    /// Public output variable
    Output,
    /// Addition: out = left + right (1 constraint)
    Add(usize, usize),
    /// Multiplication: out = left * right (1 constraint, non-linear)
    Mul(usize, usize),
    /// Alias: out = source (1 constraint, optimization trap)
    Alias(usize),
    /// Scaling: out = k * source (1 constraint, linear trap)
    Scale(usize, u64),
    /// Fifth power: out = source^5 (3 constraints, algebraic trap)
    Pow5(usize),
}

/// A node in the circuit DAG.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub op: OpType,
}

impl Node {
    pub fn new(id: usize, op: OpType) -> Self {
        Self { id, op }
    }

    pub fn constraint_count(&self) -> usize {
        match self.op {
            OpType::Pow5(_) => 3,
            OpType::Add(_, _) | OpType::Mul(_, _) | OpType::Alias(_) | OpType::Scale(_, _) => 1,
            OpType::Undefined | OpType::Input | OpType::Output => 0,
        }
    }

    pub fn is_input(&self) -> bool {
        matches!(self.op, OpType::Input)
    }

    pub fn is_output(&self) -> bool {
        matches!(self.op, OpType::Output)
    }
}

// =============================================================================
// Circuit configuration
// =============================================================================

/// Configuration for circuit generation.
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    /// Target number of constraints
    pub num_constraints: usize,
    /// Probability of reusing existing nodes (shared subexpressions)
    pub redundancy_ratio: f64,
    /// Frequency of Pow5 operations (algebraic trap)
    pub power_map_ratio: f64,
    /// Frequency of alias operations (optimization trap)
    pub alias_ratio: f64,
    /// Frequency of linear scaling operations (linear trap)
    pub linear_ratio: f64,
}

impl CircuitConfig {
    /// Creates a configuration for `delta`: `num_constraints = delta * 1000`.
    pub fn from_delta(delta: usize) -> Self {
        Self {
            num_constraints: delta * 1000,
            redundancy_ratio: 0.25,
            power_map_ratio: 0.15,
            alias_ratio: 0.15,
            linear_ratio: 0.20,
        }
    }
}

// =============================================================================
// DAG
// =============================================================================

/// A directed acyclic graph representing a generated circuit.
#[derive(Debug, Clone)]
pub struct DAG {
    pub nodes: Vec<Node>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl DAG {
    pub fn total_constraints(&self) -> usize {
        self.nodes.iter().map(|n| n.constraint_count()).sum()
    }
}

// =============================================================================
// DAG generation
// =============================================================================

/// Generates a random circuit DAG deterministically from a seed and config.
///
/// Construction proceeds backwards from output nodes (low IDs) toward input
/// nodes (high IDs). Evaluating in reverse ID order gives correct topological
/// forward order. PRNG is `SHA256(seed) → ChaCha20`.
pub fn generate_dag(seed: &str, config: &CircuitConfig) -> DAG {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let mut rng = ChaCha20Rng::from_seed(hasher.finalize().into());

    let mut nodes: Vec<Node> = Vec::new();
    let mut frontier: VecDeque<usize> = VecDeque::new();

    // Create output nodes (DAG roots, IDs 0..num_outputs-1)
    let num_outputs = rng.gen_range(1..=3);
    for _ in 0..num_outputs {
        let id = nodes.len();
        nodes.push(Node::new(id, OpType::Output));
        frontier.push_back(id);
    }

    let mut constraint_count = 0;

    while !frontier.is_empty() {
        if constraint_count >= config.num_constraints {
            break;
        }

        let curr_id = frontier.pop_front().unwrap();
        let rand_val = rng.gen_range(0.0..1.0);

        let t_alias = config.alias_ratio;
        let t_lin = t_alias + config.linear_ratio;
        let t_pow = t_lin + config.power_map_ratio;

        if rand_val < t_alias {
            let src = pick_operand(curr_id, &mut nodes, &mut frontier, &mut rng, 0.0);
            nodes[curr_id].op = OpType::Alias(src);
            constraint_count += 1;
        } else if rand_val < t_lin {
            let src = pick_operand(curr_id, &mut nodes, &mut frontier, &mut rng, 0.0);
            let k = rng.gen_range(2..1000);
            nodes[curr_id].op = OpType::Scale(src, k);
            constraint_count += 1;
        } else if rand_val < t_pow && (constraint_count + 3 <= config.num_constraints) {
            let src = pick_operand(
                curr_id,
                &mut nodes,
                &mut frontier,
                &mut rng,
                config.redundancy_ratio,
            );
            nodes[curr_id].op = OpType::Pow5(src);
            constraint_count += 3;
        } else {
            let is_mul = rng.gen_bool(0.5);
            let l = pick_operand(
                curr_id,
                &mut nodes,
                &mut frontier,
                &mut rng,
                config.redundancy_ratio,
            );
            let r = pick_operand(
                curr_id,
                &mut nodes,
                &mut frontier,
                &mut rng,
                config.redundancy_ratio,
            );
            nodes[curr_id].op = if is_mul {
                OpType::Mul(l, r)
            } else {
                OpType::Add(l, r)
            };
            constraint_count += 1;
        }
    }

    // All undefined/output nodes become inputs
    for node in &mut nodes {
        if matches!(node.op, OpType::Undefined | OpType::Output) {
            node.op = OpType::Input;
        }
    }

    let num_inputs = nodes.iter().filter(|n| n.is_input()).count();

    DAG {
        nodes,
        num_inputs,
        num_outputs,
    }
}

fn pick_operand(
    parent_id: usize,
    nodes: &mut Vec<Node>,
    frontier: &mut VecDeque<usize>,
    rng: &mut ChaCha20Rng,
    redundancy_ratio: f64,
) -> usize {
    let valid_count = nodes.len().saturating_sub(parent_id + 1);
    if rng.gen_bool(redundancy_ratio) && valid_count > 0 {
        parent_id + 1 + rng.gen_range(0..valid_count)
    } else {
        let id = nodes.len();
        nodes.push(Node::new(id, OpType::Undefined));
        frontier.push_back(id);
        id
    }
}
