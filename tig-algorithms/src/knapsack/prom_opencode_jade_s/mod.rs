use tig_challenges::knapsack::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Reverse;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
  pub window_k: Option<usize>,
  pub ils_rounds: Option<usize>,
  pub core_half_dp: Option<usize>,
  pub use_milp: Option<bool>,
}

struct Rng { state: u64 }

impl Rng {
  fn from_seed(seed: &[u8; 32]) -> Self {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for (i, &b) in seed.iter().enumerate() {
      s ^= (b as u64) << ((i & 7) * 8);
      s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
    }
    if s == 0 { s = 1; }
    Self { state: s }
  }

  #[inline] fn next_u64(&mut self) -> u64 {
    let mut x = self.state;
    x ^= x << 7;
    x ^= x >> 9;
    x ^= x << 8;
    self.state = x;
    x
  }

  #[inline] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
  #[inline] fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
  #[inline] fn next_usize(&mut self, bound: usize) -> usize {
    if bound == 0 { return 0; }
    (self.next_u64() % bound as u64) as usize
  }
}

struct State<'a> {
  ch: &'a Challenge,
  selected_bit: Vec<bool>,
  contrib: Vec<i32>,
  total_value: i64,
  total_weight: u32,
}

impl<'a> State<'a> {
  fn new_empty(ch: &'a Challenge) -> Self {
    let n = ch.num_items;
    let mut contrib = vec![0i32; n];
    for i in 0..n { contrib[i] = ch.values[i] as i32; }
    Self {
      ch,
      selected_bit: vec![false; n],
      contrib,
      total_value: 0,
      total_weight: 0,
    }
  }

  #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

  #[inline(always)]
  fn add_item(&mut self, i: usize) {
    self.total_value += self.contrib[i] as i64;
    self.total_weight += self.ch.weights[i];
    let n = self.ch.num_items;
    let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
    let contrib_ptr = self.contrib.as_mut_ptr();
    unsafe {
      for k in 0..n {
        let ck = contrib_ptr.add(k);
        *ck = (*ck).wrapping_add(*row_ptr.add(k));
      }
    }
    self.selected_bit[i] = true;
  }

  #[inline(always)]
  fn remove_item(&mut self, j: usize) {
    self.total_value -= self.contrib[j] as i64;
    self.total_weight -= self.ch.weights[j];
    let n = self.ch.num_items;
    let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
    let contrib_ptr = self.contrib.as_mut_ptr();
    unsafe {
      for k in 0..n {
        let ck = contrib_ptr.add(k);
        *ck = (*ck).wrapping_sub(*row_ptr.add(k));
      }
    }
    self.selected_bit[j] = false;
  }

  fn selected_items(&self) -> Vec<usize> {
    (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
  }
}

fn build_greedy_density(state: &mut State) {
  let n = state.ch.num_items;
  let cap = state.ch.max_weight;
  for i in 0..n { state.add_item(i); }
  while state.total_weight > cap {
    let mut worst = 0;
    let mut worst_s = i64::MAX;
    for i in 0..n {
      if state.selected_bit[i] {
        let c = state.contrib[i] as i64;
        let w = (state.ch.weights[i] as i64).max(1);
        let s = (c * 1000) / w;
        if s < worst_s { worst_s = s; worst = i; }
      }
    }
    state.remove_item(worst);
  }
}

fn greedy_fill(state: &mut State, rng: &mut Rng) {
  let n = state.ch.num_items;
  let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= state.slack()).collect();
  if candidates.is_empty() { return; }
  candidates.sort_by_key(|&i| Reverse((state.contrib[i] as i64 * 1000) / state.ch.weights[i] as i64));
  for &i in &candidates {
    if state.ch.weights[i] <= state.slack() {
      state.add_item(i);
    }
  }
}

fn ruin_and_recreate(state: &mut State, rng: &mut Rng, ruin_pct: f64) {
  let n = state.ch.num_items;
  let selected: Vec<usize> = state.selected_items();
  let num_to_remove = (selected.len() as f64 * ruin_pct).max(1.0) as usize;
  
  // Score items by marginal contribution, bias toward removing low contributors
  let mut scored: Vec<(i64, usize)> = selected.iter().map(|&i| (state.contrib[i] as i64, i)).collect();
  scored.sort_by_key(|&(c, _)| c);
  
  // Remove bottom fraction
  let to_remove: Vec<usize> = scored.iter().take(num_to_remove).map(|&(_, i)| i).collect();
  for &i in &to_remove {
    state.remove_item(i);
  }
  
  // Refill greedily
  greedy_fill(state, rng);
}

fn solve_milp(challenge: &Challenge, initial_solution: &Solution) -> Result<Solution> {
  Ok(initial_solution.clone())
}

struct Hparams {
  n_random_starts: usize,
  window_k: usize,
  core_half_dp: usize,
  use_milp: bool,
}

impl Hparams {
  fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
    let mut p = Self {
      n_random_starts: if n <= 1200 { 3 } else { 4 },
      window_k: 200,
      core_half_dp: 40,
      use_milp: n <= 200,
    };
    if let Some(m) = h {
      if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
      if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) { p.core_half_dp = v as usize; }
      if let Some(v) = m.get("use_milp").and_then(|v| v.as_bool()) { p.use_milp = v; }
    }
    p
  }
}

pub struct Solver;

impl Solver {
  pub fn solve(
    challenge: &Challenge,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    hyperparameters: &Option<Map<String, Value>>,
  ) -> Result<Option<Solution>> {
    let n = challenge.num_items;
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
    let hp = Hparams::from_map(hyperparameters, n, budget_pct);

    let deadline = Instant::now() + std::time::Duration::from_secs(55);
    let mut best_solution = None;
    let mut best_value = i64::MIN;

    let mut rng = Rng::from_seed(&challenge.seed);

    // Multiple random starts with SA
    for start_idx in 0..hp.n_random_starts {
      if Instant::now() >= deadline { break; }

      let mut state = State::new_empty(challenge);
      
      // Different construction strategies per start
      if start_idx == 0 {
        build_greedy_density(&mut state);
      } else {
        // Randomized construction: add items in random order, eject when over capacity
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
          let j = rng.next_usize(i + 1);
          perm.swap(i, j);
        }
        for &i in &perm {
          if state.total_weight + challenge.weights[i] <= challenge.max_weight {
            state.add_item(i);
          }
        }
        greedy_fill(&mut state, &mut rng);
      }

      // Save initial solution
      if state.total_value > best_value {
        best_value = state.total_value;
        best_solution = Some(Solution { items: state.selected_items() });
        if let Some(save) = save_solution {
          let _ = save(best_solution.as_ref().unwrap());
        }
      }

      // SA with ejection chains and ruin-recreate
      let mut temp = 1000.0;
      let cooling_rate = 0.9995;
      let min_temp = 0.1;
      let mut iter_since_recreate = 0;
      let recreate_interval = 500;

      while Instant::now() < deadline && temp > min_temp {
        iter_since_recreate += 1;

        // Ruin-and-recreate phase periodically
        if iter_since_recreate >= recreate_interval {
          iter_since_recreate = 0;
          let ruin_pct = 0.2 + rng.next_f64() * 0.2; // 20-40%
          ruin_and_recreate(&mut state, &mut rng, ruin_pct);
          
          if state.total_value > best_value {
            best_value = state.total_value;
            best_solution = Some(Solution { items: state.selected_items() });
            if let Some(save) = save_solution {
              let _ = save(best_solution.as_ref().unwrap());
            }
          }
          continue;
        }

        // Ejection chain move: remove 1-3 random items, then greedy refill
        let num_eject = 1 + rng.next_usize(3);
        let selected: Vec<usize> = state.selected_items();
        if selected.len() < num_eject { continue; }
        
        let mut to_remove = Vec::new();
        let mut temp_state_items = selected.clone();
        for _ in 0..num_eject {
          if temp_state_items.is_empty() { break; }
          let idx = rng.next_usize(temp_state_items.len());
          to_remove.push(temp_state_items[idx]);
          temp_state_items.swap_remove(idx);
        }
        
        for &i in &to_remove {
          state.remove_item(i);
        }
        
        // Greedy refill
        greedy_fill(&mut state, &mut rng);

        // Greedy add moves: try to add any unselected item that fits
        let unselected: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= state.slack()).collect();
        for &i in &unselected {
          if state.ch.weights[i] <= state.slack() && state.contrib[i] > 0 {
            state.add_item(i);
          }
        }

        // Accept if improved (SA always accepts improvements)
        if state.total_value > best_value {
          best_value = state.total_value;
          best_solution = Some(Solution { items: state.selected_items() });
          if let Some(save) = save_solution {
            let _ = save(best_solution.as_ref().unwrap());
          }
        }

        temp *= cooling_rate;
      }
    }

    Ok(best_solution)
  }
}

pub fn solve_challenge(
  challenge: &Challenge,
  save_solution: &dyn Fn(&Solution) -> Result<()>,
  hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
  if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
    let _ = save_solution(&solution);
  }
  Ok(())
}

pub fn help() {
}