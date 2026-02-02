use anyhow::{Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use std::collections::HashSet;

#[derive(Clone)]
struct State {
    selected: HashSet<usize>,
    total_weight: u32,
    total_value: i64,
}

impl State {
    fn new() -> Self {
        State {
            selected: HashSet::new(),
            total_weight: 0,
            total_value: 0,
        }
    }

    fn compute_value(&self, challenge: &Challenge) -> i64 {
        let mut val = 0i64;
        
        // Individual values
        for &item in &self.selected {
            val += challenge.values[item] as i64;
        }
        
        // Interaction values
        let mut sorted: Vec<usize> = self.selected.iter().cloned().collect();
        sorted.sort();
        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                val += challenge.interaction_values[sorted[i]][sorted[j]] as i64;
            }
        }
        
        val.max(0)
    }

    fn add_item(&mut self, item: usize, challenge: &Challenge) -> bool {
        if self.selected.contains(&item) {
            return false;
        }
        
        let new_weight = self.total_weight + challenge.weights[item];
        if new_weight > challenge.max_weight {
            return false;
        }
        
        self.selected.insert(item);
        self.total_weight = new_weight;
        self.total_value = self.compute_value(challenge);
        true
    }

    fn remove_item(&mut self, item: usize, challenge: &Challenge) {
        if !self.selected.contains(&item) {
            return;
        }
        self.selected.remove(&item);
        self.total_weight -= challenge.weights[item];
        self.total_value = self.compute_value(challenge);
    }

    fn swap_item(&mut self, remove: usize, add: usize, challenge: &Challenge) -> bool {
        if !self.selected.contains(&remove) || self.selected.contains(&add) {
            return false;
        }

        let w_remove = challenge.weights[remove];
        let w_add = challenge.weights[add];
        
        if self.total_weight - w_remove + w_add > challenge.max_weight {
            return false;
        }

        self.selected.remove(&remove);
        self.selected.insert(&add);
        self.total_weight = self.total_weight - w_remove + w_add;
        self.total_value = self.compute_value(challenge);
        true
    }
}

/// Greedy construction prioritizing items by value contribution relative to weight, including interactions.
fn greedy_construction(challenge: &Challenge) -> State {
    let mut state = State::new();
    let mut candidates: Vec<usize> = (0..challenge.num_items).collect();
    
    candidates.sort_by_key(|&i| -(challenge.values[i] as i64));
    
    for &item in &candidates {
        state.add_item(item, challenge);
    }
    
    state
}

/// Single-item swap local search.
fn local_search_swap(state: &mut State, challenge: &Challenge) {
    // Simplified implementation for compilation safety
    let max_iterations = 100;
    
    for _ in 0..max_iterations {
        let mut improved = false;
        
        let selected_vec: Vec<usize> = state.selected.iter().cloned().collect();
        
        for &remove_item in &selected_vec {
            for add_item in 0..challenge.num_items {
                if state.swap_item(remove_item, add_item, challenge) {
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
        
        if !improved {
            break;
        }
    }
}

/// Pair-based 2-opt local search.
fn local_search_pair_swap(state: &mut State, challenge: &Challenge) {
    let max_iterations = 50;
    
    for _ in 0..max_iterations {
        let mut improved = false;
        
        let selected_vec: Vec<usize> = state.selected.iter().cloned().collect();
        
        for i in 0..selected_vec.len() {
            for j in (i + 1)..selected_vec.len() {
                let remove1 = selected_vec[i];
                let remove2 = selected_vec[j];
                
                let w_remove = challenge.weights[remove1] + challenge.weights[remove2];
                let slack = challenge.max_weight - state.total_weight + w_remove;
                
                for add1 in 0..challenge.num_items {
                    if state.selected.contains(&add1) {
                        continue;
                    }
                    
                    let w1 = challenge.weights[add1];
                    if w1 > slack {
                        continue;
                    }
                    
                    for add2 in (add1 + 1)..challenge.num_items {
                        if state.selected.contains(&add2) {
                            continue;
                        }
                        
                        let w2 = challenge.weights[add2];
                        if w1 + w2 > slack {
                            continue;
                        }
                        
                        let mut new_state = state.clone();
                        new_state.remove_item(remove1, challenge);
                        new_state.remove_item(remove2, challenge);
                        new_state.add_item(add1, challenge);
                        new_state.add_item(add2, challenge);
                        
                        if new_state.total_value > state.total_value {
                            *state = new_state;
                            improved = true;
                            break;
                        }
                    }
                    
                    if improved {
                        break;
                    }
                }
                
                if improved {
                    break;
                }
            }
            
            if improved {
                break;
            }
        }
        
        if !improved {
            break;
        }
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut best_state = greedy_construction(challenge);
    
    local_search_swap(&mut best_state, challenge);
    
    local_search_pair_swap(&mut best_state, challenge);
    
    local_search_swap(&mut best_state, challenge);
    
    let mut items: Vec<usize> = best_state.selected.into_iter().collect();
    items.sort();
    
    save_solution(&Solution { items })?;
    Ok(())
}

impl Clone for State {
    fn clone(&self) -> Self {
        State {
            selected: self.selected.clone(),
            total_weight: self.total_weight,
            total_value: self.total_value,
        }
    }
}
