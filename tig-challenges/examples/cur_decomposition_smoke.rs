use anyhow::{ensure, Result};
use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};
use std::{env, sync::Arc};
use tig_challenges::cur_decomposition::{
    aggregate_sub_scores, sample_design_metadata, verifier_fast_u_mask, Challenge,
    DesignGenerationConfig, Solution, Track, DESIGN_DELTA, DESIGN_SPECTRUM_A, NUM_SUB_INSTANCES,
};

fn main() -> Result<()> {
    let ptx_path = env::args()
        .nth(1)
        .expect("usage: cur_decomposition_smoke <challenge PTX>");
    let device = 0;
    let context = CudaContext::new(device)?;
    context.set_blocking_synchronize()?;
    let module = context.load_module(Ptx::from_file(ptx_path))?;
    let stream = context.default_stream();
    let prop = get_device_prop(device as i32)?;

    let track = Track {
        m: 512,
        n: 640,
        poly: true,
    };
    let seed = [23u8; 32];
    let metadata = sample_design_metadata(
        &seed,
        &DesignGenerationConfig {
            m: track.m,
            n: track.n,
            delta: DESIGN_DELTA,
            poly: track.poly,
            spectrum_a: DESIGN_SPECTRUM_A,
        },
    )?;
    let challenges = Challenge::generate_multiple_instances(
        &seed,
        &track,
        module.clone(),
        stream.clone(),
        &prop,
    )?;

    ensure!(challenges.len() == NUM_SUB_INSTANCES);
    ensure!(metadata.len() == NUM_SUB_INSTANCES);
    let expected_fast_u = verifier_fast_u_mask(
        &metadata
            .iter()
            .map(|sub_instance| sub_instance.target_k)
            .collect::<Vec<_>>(),
    )?;
    ensure!(challenges
        .iter()
        .map(|challenge| challenge.verifier_computes_u)
        .eq(expected_fast_u.iter().copied()));
    let mut scores = Vec::with_capacity(NUM_SUB_INSTANCES);
    let mut solutions = Vec::with_capacity(NUM_SUB_INSTANCES);
    let mut verifier_fast_u_count = 0;
    let mut checked_shared_fast_u = false;
    for (challenge, expected) in challenges.iter().zip(metadata.iter()) {
        ensure!(challenge.target_k == expected.target_k);
        let k = challenge.target_k as usize;
        if challenge.verifier_computes_u {
            verifier_fast_u_count += 1;
        }
        let solution = Solution {
            c_idxs: (0..challenge.target_k).collect(),
            u_mat: if challenge.verifier_computes_u {
                Vec::new()
            } else {
                vec![0.0; k * k]
            },
            r_idxs: (0..challenge.target_k).collect(),
        };
        if challenge.verifier_computes_u && !checked_shared_fast_u {
            let explicit_u = challenge.fast_linking_matrix(
                &solution.c_idxs,
                &solution.r_idxs,
                Arc::clone(&module),
                Arc::clone(&stream),
            )?;
            let explicit_solution = Solution {
                c_idxs: solution.c_idxs.clone(),
                u_mat: explicit_u,
                r_idxs: solution.r_idxs.clone(),
            };
            let innovator_fnorm = challenge.evaluate_fnorm(
                &explicit_solution,
                Arc::clone(&module),
                Arc::clone(&stream),
                &prop,
            )?;
            let verifier_fnorm = challenge.evaluate_fast_fnorm(
                &solution.c_idxs,
                &solution.r_idxs,
                Arc::clone(&module),
                Arc::clone(&stream),
                &prop,
            )?;
            let scale = innovator_fnorm.abs().max(verifier_fnorm.abs()).max(1.0);
            ensure!((innovator_fnorm - verifier_fnorm).abs() <= 5e-5 * scale);
            checked_shared_fast_u = true;
        }
        let score = challenge.evaluate_solution(
            &solution,
            Arc::clone(&module),
            Arc::clone(&stream),
            &prop,
        )?;
        ensure!(score.is_finite() && (0.0..=1.0).contains(&score));
        let mut invalid_format = solution.clone();
        invalid_format.u_mat = if challenge.verifier_computes_u {
            vec![0.0; k * k]
        } else {
            Vec::new()
        };
        ensure!(challenge
            .evaluate_solution(
                &invalid_format,
                Arc::clone(&module),
                Arc::clone(&stream),
                &prop,
            )
            .is_err());
        scores.push(score);
        solutions.push(solution);
    }
    ensure!(verifier_fast_u_count == NUM_SUB_INSTANCES / 2);
    ensure!(checked_shared_fast_u);

    let encoded = serde_json::to_string(&solutions)?;
    let decoded: Vec<Solution> = serde_json::from_str(&encoded)?;
    ensure!(decoded.len() == NUM_SUB_INSTANCES);
    let quality = aggregate_sub_scores(&scores)?;
    println!(
        "CUR production smoke passed: sub_instances={}, quality={}, serialized_bytes={}",
        NUM_SUB_INSTANCES,
        quality,
        encoded.len()
    );
    Ok(())
}
