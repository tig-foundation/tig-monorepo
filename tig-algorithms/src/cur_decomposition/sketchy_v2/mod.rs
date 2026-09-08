// Quality-first CUR selector: preserve the original sketchy result, add deeper
// block-Krylov candidates, then jointly search their row/column cross-product
// with the verifier's canonical fast-U residual.

use anyhow::{anyhow, Result};
use cudarc::{
    cublas::CudaBlas,
    cusolver::DnHandle,
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{cell::RefCell, sync::Arc};
use tig_challenges::cur_decomposition::*;

#[path = "../sketchy/mod.rs"]
mod sketchy;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    /// Trials used by the original sketchy configuration. Keeping the default
    /// at three makes its previous answer a member of the v2 search space.
    pub baseline_trials: usize,
    /// Additional candidates generated with the stronger configuration.
    pub quality_trials: usize,
    /// Oversampling used by the stronger block-Krylov range finder.
    pub sketch_extra: usize,
    /// Power/subspace iterations used by the stronger range finder.
    pub power_iters: usize,
    /// Maximum max-volume refinement swaps for each index set.
    pub maxvol_swaps: usize,
    /// Max-volume interpolation-coefficient stopping threshold.
    pub maxvol_tolerance: f32,
    /// Number of best paired candidates whose column and row sets are
    /// recombined and scored jointly.
    pub crossover_pool: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            baseline_trials: 3,
            quality_trials: 10,
            sketch_extra: 64,
            power_iters: 4,
            maxvol_swaps: 48,
            maxvol_tolerance: 1.001,
            crossover_pool: 10,
        }
    }
}

pub fn help() {
    println!("Quality-first block-Krylov/max-volume CUR decomposition (GPU).");
    println!(
        "Preserves sketchy v1, adds deeper candidates, and jointly recombines row/column sets."
    );
    println!(
        "Every candidate is ranked with the verifier's canonical fast U; only indices are saved."
    );
    println!("Hyperparameters:");
    println!("  baseline_trials: v1-compatible fallback trials (default: 3)");
    println!("  quality_trials:  deeper block-Krylov candidates (default: 10)");
    println!("  sketch_extra:    deeper sketch dimension is k + this value (default: 64)");
    println!("  power_iters:     deeper subspace iterations (default: 4)");
    println!("  maxvol_swaps:    max-volume swaps per side (default: 48)");
    println!("  maxvol_tolerance:max-volume stopping threshold (default: 1.001)");
    println!("  crossover_pool:  paired candidates used for joint row/column search (default: 10)");
}

#[derive(Clone)]
struct ScoredCandidate {
    solution: Solution,
    fnorm: f32,
}

fn contains_solution(candidates: &[ScoredCandidate], solution: &Solution) -> bool {
    candidates.iter().any(|candidate| {
        candidate.solution.c_idxs == solution.c_idxs && candidate.solution.r_idxs == solution.r_idxs
    })
}

fn insert_unique(index_sets: &mut Vec<Vec<i32>>, indices: &[i32]) {
    if !index_sets.iter().any(|existing| existing == indices) {
        index_sets.push(indices.to_vec());
    }
}

#[allow(clippy::too_many_arguments)]
fn score_and_record(
    challenge: &Challenge,
    solution: Solution,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prop: &cudaDeviceProp,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    candidates: &mut Vec<ScoredCandidate>,
    best_fnorm: &mut f32,
    best_solution: &mut Option<Solution>,
) -> Result<()> {
    if contains_solution(candidates, &solution) {
        return Ok(());
    }

    let fnorm = challenge.evaluate_fast_fnorm(
        &solution.c_idxs,
        &solution.r_idxs,
        module.clone(),
        stream.clone(),
        prop,
    )?;
    candidates.push(ScoredCandidate {
        solution: solution.clone(),
        fnorm,
    });
    if fnorm < *best_fnorm {
        *best_fnorm = fnorm;
        save_solution(&solution)?;
        *best_solution = Some(solution);
    }
    Ok(())
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let hp = match hyperparameters {
        Some(hp) => serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
            .map_err(|error| anyhow!("Failed to parse hyperparameters: {}", error))?,
        None => Hyperparameters::default(),
    };
    if hp.baseline_trials == 0
        || hp.quality_trials == 0
        || hp.crossover_pool == 0
        || !hp.maxvol_tolerance.is_finite()
        || hp.maxvol_tolerance < 1.0
    {
        return Err(anyhow!("invalid sketchy_v2 hyperparameters"));
    }

    let mut candidates = Vec::<ScoredCandidate>::new();
    let mut best_fnorm = f32::INFINITY;
    let mut best_solution = None;

    // First reproduce v1's search. Its save callback is forwarded so a valid
    // incumbent survives even if the more expensive quality stages exhaust
    // the runtime's fuel budget.
    let emitted = RefCell::new(Vec::<Solution>::new());
    let collect_baseline = |solution: &Solution| -> Result<()> {
        emitted.borrow_mut().push(solution.clone());
        save_solution(solution)
    };
    let mut baseline_hp = Map::new();
    baseline_hp.insert("num_trials".into(), Value::from(hp.baseline_trials));
    baseline_hp.insert("sketch_extra".into(), Value::from(32));
    baseline_hp.insert("power_iters".into(), Value::from(2));
    baseline_hp.insert("maxvol_swaps".into(), Value::from(16));
    baseline_hp.insert("maxvol_tolerance".into(), Value::from(1.01));
    let baseline_result = sketchy::solve_challenge(
        challenge,
        &collect_baseline,
        &Some(baseline_hp),
        module.clone(),
        stream.clone(),
        prop,
    );

    let mut baseline_solutions = emitted.into_inner();
    if let Ok(Some(solution)) = baseline_result {
        if !baseline_solutions.iter().any(|existing| {
            existing.c_idxs == solution.c_idxs && existing.r_idxs == solution.r_idxs
        }) {
            baseline_solutions.push(solution);
        }
    }
    for solution in baseline_solutions {
        // A numerical failure for one candidate should not discard the valid
        // incumbent already emitted by v1.
        let _ = score_and_record(
            challenge,
            solution,
            &module,
            &stream,
            prop,
            save_solution,
            &mut candidates,
            &mut best_fnorm,
            &mut best_solution,
        );
    }

    // Generate more accurate subspaces. These trials use independent sketch
    // seeds but retain the same deterministic seed derivation as v1.
    let cublas = CudaBlas::new(stream.clone())?;
    let cusolver = DnHandle::new(stream.clone())?;
    for trial in 0..hp.quality_trials {
        let candidate = sketchy::block_krylov_maxvol_candidate(
            challenge,
            hp.sketch_extra,
            hp.power_iters,
            hp.maxvol_swaps,
            hp.maxvol_tolerance,
            trial,
            &module,
            &stream,
            &cublas,
            &cusolver,
        );
        let Ok((c_idxs, r_idxs)) = candidate else {
            continue;
        };
        let _ = score_and_record(
            challenge,
            Solution { c_idxs, r_idxs },
            &module,
            &stream,
            prop,
            save_solution,
            &mut candidates,
            &mut best_fnorm,
            &mut best_solution,
        );
    }

    // C and R are selected independently but v1 evaluates only same-trial
    // pairs. Recombining the strongest column and row sets exposes a small,
    // high-value joint search space without repeating the expensive SVDs.
    candidates.sort_by(|left, right| left.fnorm.total_cmp(&right.fnorm));
    let pool_len = hp.crossover_pool.min(candidates.len());
    let pool = candidates[..pool_len].to_vec();
    let mut column_sets = Vec::<Vec<i32>>::new();
    let mut row_sets = Vec::<Vec<i32>>::new();
    for candidate in &pool {
        insert_unique(&mut column_sets, &candidate.solution.c_idxs);
        insert_unique(&mut row_sets, &candidate.solution.r_idxs);
    }
    for c_idxs in &column_sets {
        for r_idxs in &row_sets {
            let _ = score_and_record(
                challenge,
                Solution {
                    c_idxs: c_idxs.clone(),
                    r_idxs: r_idxs.clone(),
                },
                &module,
                &stream,
                prop,
                save_solution,
                &mut candidates,
                &mut best_fnorm,
                &mut best_solution,
            );
        }
    }

    Ok(best_solution)
}

// Important! Do not include tests in an algorithm submission file.
