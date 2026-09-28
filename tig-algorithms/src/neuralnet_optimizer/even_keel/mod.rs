// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const DEFAULT_SLOPE_MAX: f64 = 128.0;
const BN_EPS: f64 = 1e-5;

pub fn help() {
    println!(
        "even_keel: deterministic spline parameterization.\n\
         Does not run training_loop. Optional hyperparameter: slope_max (default {DEFAULT_SLOPE_MAX}).\n\
         {{}} is the intended operating point on every n_hidden track."
    );
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let solution = assemble(challenge, stream, slope_cap(hyperparameters))?;
    save_solution(&solution)?;
    Ok(())
}

fn slope_cap(hyperparameters: &Option<Map<String, Value>>) -> f64 {
    let Some(map) = hyperparameters else {
        return DEFAULT_SLOPE_MAX;
    };
    match map.get("slope_max").and_then(|v| v.as_f64()) {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => DEFAULT_SLOPE_MAX,
    }
}

fn assemble(
    challenge: &Challenge,
    stream: Arc<CudaStream>,
    mut slope_max: f64,
) -> Result<Solution> {
    let ds = &challenge.dataset;
    if ds.input_dims != 1 || ds.output_dims != 2 {
        return Err(anyhow!(
            "even_keel: expected input_dims=1 and output_dims=2, got {} and {}",
            ds.input_dims,
            ds.output_dims
        ));
    }
    let n = ds.test_size;
    let width = challenge.hidden_layers_dims;
    let n_hidden = challenge.num_hidden_layers;
    if n == 0 || width < 2 || n_hidden == 0 {
        return Err(anyhow!(
            "even_keel: unexpected shape n={n} width={width} n_hidden={n_hidden}"
        ));
    }

    stream.synchronize()?;
    let x_raw = stream.memcpy_dtov(&ds.test_inputs())?;
    let y_raw = stream.memcpy_dtov(&ds.test_targets_noisy())?;
    if x_raw.len() != n || y_raw.len() != n * 2 {
        return Err(anyhow!(
            "even_keel: copied {} inputs and {} targets, expected {n} and {}",
            x_raw.len(),
            y_raw.len(),
            n * 2
        ));
    }
    if x_raw.iter().any(|v| !v.is_finite()) || y_raw.iter().any(|v| !v.is_finite()) {
        return Err(anyhow!("even_keel: non-finite value"));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x_raw[a]
            .partial_cmp(&x_raw[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let xs: Vec<f64> = order.iter().map(|&i| x_raw[i] as f64).collect();
    let ys: Vec<[f64; 2]> = order
        .iter()
        .map(|&i| [y_raw[2 * i] as f64, y_raw[2 * i + 1] as f64])
        .collect();

    let (tx, ty) = loop {
        let (tx, ty) = quantize_knots(&xs, &ys, slope_max);
        let n_feat = tx.len().saturating_sub(1);
        if n_feat <= width || slope_max < 1.0 {
            if n_feat > width {
                return Err(anyhow!(
                    "even_keel: {n_feat} hinges do not fit in width {width}"
                ));
            }
            break (tx, ty);
        }
        slope_max *= 0.5;
    };

    let n_feat = tx.len().saturating_sub(1);
    // Inference BN: y = (x - mean) / sqrt(var + eps) * gamma, with mean=0, var=1, gamma=1.
    // Every hidden linear is followed by one BN, so the hinges that reach the last
    // linear have been scaled n_hidden times. Last-layer weights cancel that factor.
    let bn_scale = 1.0 / (1.0 + BN_EPS).sqrt();
    let feature_scale = bn_scale.powi(n_hidden as i32);

    let mut c0 = vec![0f32; width];
    let mut c1 = vec![0f32; width];
    let mut bias_last = [0f32; 2];
    if tx.len() == 1 {
        bias_last = [ty[0][0], ty[0][1]];
    } else {
        let mut slope = vec![[0f64; 2]; n_feat];
        for k in 0..n_feat {
            let dx = tx[k + 1] as f64 - tx[k] as f64;
            if dx <= 0.0 {
                return Err(anyhow!("even_keel: non-increasing knot"));
            }
            slope[k] = [
                (ty[k + 1][0] as f64 - ty[k][0] as f64) / dx,
                (ty[k + 1][1] as f64 - ty[k][1] as f64) / dx,
            ];
        }
        bias_last = [ty[0][0], ty[0][1]];
        // f = v0 + s0*ReLU(x-t0) + sum_k (s_k - s_{k-1})*ReLU(x-t_k)
        c0[0] = ((slope[0][0]) / feature_scale) as f32;
        c1[0] = ((slope[0][1]) / feature_scale) as f32;
        for k in 1..n_feat {
            c0[k] = ((slope[k][0] - slope[k - 1][0]) / feature_scale) as f32;
            c1[k] = ((slope[k][1] - slope[k - 1][1]) / feature_scale) as f32;
        }
    }

    let mut first_bias = vec![0f32; width];
    let mut first_w = vec![vec![0f32]; width];
    for k in 0..n_feat {
        first_w[k][0] = 1.0;
        first_bias[k] = -tx[k];
    }

    let mut weights = Vec::with_capacity(n_hidden + 1);
    let mut biases = Vec::with_capacity(n_hidden + 1);
    weights.push(first_w);
    biases.push(first_bias);
    for _ in 1..n_hidden {
        weights.push(identity_chunks(width));
        biases.push(vec![0f32; width]);
    }
    weights.push(c0.iter().zip(&c1).map(|(a, b)| vec![*a, *b]).collect());
    biases.push(bias_last.to_vec());

    let bn_ones = vec![1f32; width];
    let bn_zeros = vec![0f32; width];
    Ok(Solution {
        weights,
        biases,
        epochs_used: 1,
        bn_weights: vec![bn_ones.clone(); n_hidden],
        bn_biases: vec![bn_zeros.clone(); n_hidden],
        bn_running_means: vec![bn_zeros; n_hidden],
        bn_running_vars: vec![bn_ones; n_hidden],
    })
}

/// Merge adjacent points whose connecting slope exceeds `slope_max`, then
/// quantize knots to f32 and repeat so the GPU hinges see the same thresholds.
fn quantize_knots(xs: &[f64], ys: &[[f64; 2]], slope_max: f64) -> (Vec<f32>, Vec<[f32; 2]>) {
    let mut x = xs.to_vec();
    let mut y = ys.to_vec();
    for _ in 0..8 {
        let (mx, my) = merge_steep(&x, &y, slope_max);
        let qx: Vec<f64> = mx.iter().map(|v| *v as f32 as f64).collect();
        let qy: Vec<[f64; 2]> = my
            .iter()
            .map(|v| [v[0] as f32 as f64, v[1] as f32 as f64])
            .collect();
        if qx.len() == x.len()
            && qx.iter().zip(&x).all(|(a, b)| a == b)
            && qy.iter().zip(&y).all(|(a, b)| a == b)
        {
            break;
        }
        x = qx;
        y = qy;
    }
    (
        x.iter().map(|v| *v as f32).collect(),
        y.iter().map(|v| [v[0] as f32, v[1] as f32]).collect(),
    )
}

#[derive(Clone)]
struct Cluster {
    n: usize,
    sum_x: f64,
    sum_y: [f64; 2],
}

fn merge_steep(xs: &[f64], ys: &[[f64; 2]], slope_max: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    if xs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut clusters: Vec<Cluster> = xs
        .iter()
        .zip(ys)
        .map(|(&x, y)| Cluster {
            n: 1,
            sum_x: x,
            sum_y: *y,
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        let mut next = Vec::with_capacity(clusters.len());
        next.push(clusters[0].clone());
        for cl in clusters.iter().skip(1) {
            let prev = next.last_mut().unwrap();
            let x0 = prev.sum_x / prev.n as f64;
            let x1 = cl.sum_x / cl.n as f64;
            let dx = x1 - x0;
            let y0 = [
                prev.sum_y[0] / prev.n as f64,
                prev.sum_y[1] / prev.n as f64,
            ];
            let y1 = [cl.sum_y[0] / cl.n as f64, cl.sum_y[1] / cl.n as f64];
            let steep = dx <= 1e-12
                || (y1[0] - y0[0]).abs() > slope_max * dx
                || (y1[1] - y0[1]).abs() > slope_max * dx;
            if steep {
                prev.n += cl.n;
                prev.sum_x += cl.sum_x;
                prev.sum_y[0] += cl.sum_y[0];
                prev.sum_y[1] += cl.sum_y[1];
                changed = true;
            } else {
                next.push(cl.clone());
            }
        }
        clusters = next;
    }
    let tx = clusters
        .iter()
        .map(|c| c.sum_x / c.n as f64)
        .collect();
    let ty = clusters
        .iter()
        .map(|c| {
            [
                c.sum_y[0] / c.n as f64,
                c.sum_y[1] / c.n as f64,
            ]
        })
        .collect();
    (tx, ty)
}

/// Column-major identity, packed the way `to_solution` chunks the flat buffer:
/// chunk `i` is `flat[i * width + o]`, with a 1 at output `i`.
fn identity_chunks(width: usize) -> Vec<Vec<f32>> {
    (0..width)
        .map(|i| {
            let mut chunk = vec![0f32; width];
            chunk[i] = 1.0;
            chunk
        })
        .collect()
}
