use crate::config::SolverConfig;
use crate::tig_adaptive::TIGState;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    pub name: String,
    pub num_nodes: usize,
    pub depot: usize,
    pub max_capacity: i32,
    pub initial_time: i32,
    
    pub time_windows: Vec<TimeWindow>,
    pub service_times: Vec<i32>,
    pub demands: Vec<i32>,
    pub distance_matrix: Vec<Vec<i32>>,
    
    #[serde(default)]
    pub initial_route: Option<Vec<usize>>,
    
    #[serde(default)]
    pub config: Option<SolverConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: i32,
    pub end: i32,
}

impl Problem {
    pub fn to_state(&self) -> TIGState {
        let route = self.initial_route.clone().unwrap_or_else(|| {
            (0..self.num_nodes).collect()
        });

        let tw_start: Vec<i32> = self.time_windows.iter().map(|tw| tw.start).collect();
        let tw_end: Vec<i32> = self.time_windows.iter().map(|tw| tw.end).collect();

        TIGState::new(
            route,
            self.initial_time,
            self.max_capacity,
            tw_start,
            tw_end,
            self.service_times.clone(),
            self.distance_matrix.clone(),
            self.demands.clone(),
        )
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.num_nodes > 0,
            "Number of nodes must be positive"
        );
        
        anyhow::ensure!(
            self.time_windows.len() == self.num_nodes,
            "Time windows length mismatch: expected {}, got {}",
            self.num_nodes,
            self.time_windows.len()
        );
        
        anyhow::ensure!(
            self.service_times.len() == self.num_nodes,
            "Service times length mismatch"
        );
        
        anyhow::ensure!(
            self.demands.len() == self.num_nodes,
            "Demands length mismatch"
        );
        
        anyhow::ensure!(
            self.distance_matrix.len() == self.num_nodes,
            "Distance matrix row count mismatch"
        );
        
        for (i, row) in self.distance_matrix.iter().enumerate() {
            anyhow::ensure!(
                row.len() == self.num_nodes,
                "Distance matrix row {} length mismatch",
                i
            );
        }
        
        Ok(())
    }
}

pub fn load_problem(path: &str) -> Result<Problem> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read problem file: {}", path))?;

    let problem: Problem = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON from: {}", path))?;

    problem.validate()
        .with_context(|| format!("Problem validation failed for: {}", path))?;

    Ok(problem)
}

pub fn save_problem(problem: &Problem, path: &str) -> Result<()> {
    let content = serde_json::to_string_pretty(problem)
        .context("Failed to serialize problem")?;

    fs::write(path, content)
        .with_context(|| format!("Failed to write problem file: {}", path))?;

    Ok(())
}

/// Simple solution schema written for TIG-compatible submission.
#[derive(Debug, Serialize, Deserialize)]
pub struct Solution {
    pub routes: Vec<Vec<usize>>,
    pub total_cost: i64,
    pub feasible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrival_times: Option<Vec<i32>>,
}

pub fn save_solution(state: &crate::tig_adaptive::TIGState, path: &str) -> Result<()> {
    let routes = vec![state.route.nodes.to_vec()];
    let total_cost = state.total_cost();
    let feasible = state.is_feasible();
    let arrival_times = if state.arrival_times.is_empty() { None } else { Some(state.arrival_times.clone()) };

    let sol = Solution { routes, total_cost, feasible, arrival_times };

    let content = serde_json::to_string_pretty(&sol)
        .context("Failed to serialize solution")?;

    fs::write(path, content)
        .with_context(|| format!("Failed to write solution file: {}", path))?;

    Ok(())
}
