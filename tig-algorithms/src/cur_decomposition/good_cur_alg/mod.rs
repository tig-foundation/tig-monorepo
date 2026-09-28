//! Quality-first CUR: a sketchy incumbent, rank-revealing pivots, and
//! alternating conditional subset selection in an oversampled SVD space.
use anyhow::{anyhow, Result};
use cudarc::{
    cublas::{
        sys::{self, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    cusolver::DnHandle,
    driver::{
        CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
    },
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{cell::RefCell, sync::Arc};
use tig_challenges::cur_decomposition::*;

// Reuse the original solver verbatim as a quality floor. Its CUDA kernels are
// included in our kernels.cu because build_ptx only collects this directory.
#[path = "../sketchy/mod.rs"]
mod baseline;
mod gpu;
use gpu::{gpu_qr, gpu_svd_thin};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Hyperparameters {
    pub num_trials: usize,
    pub sketch_extra: usize,
    /// Also oversample by ceil(k * sketch_ratio); use the larger allowance.
    pub sketch_ratio: f64,
    pub power_iters: usize,
    pub refinement_rounds: usize,
    pub baseline_trials: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            num_trials: 2,
            sketch_extra: 64,
            sketch_ratio: 0.5,
            power_iters: 3,
            refinement_rounds: 3,
            baseline_trials: 3,
        }
    }
}

pub fn help() {
    println!("good_cur_alg: quality-first GPU CUR with verifier-scored candidates.");
    println!("num_trials=2, sketch_extra=64, sketch_ratio=0.5, power_iters=3");
    println!("refinement_rounds=3, baseline_trials=3 (0 disables sketchy baseline)");
    println!("Sketch size: k + max(sketch_extra, ceil(k * sketch_ratio)), capped at min(m,n).");
    println!("Each saved solution improves the exact canonical fast-U residual.");
    println!("Allow more runtime/fuel than sketchy; larger trials/rounds search longer.");
}

fn grid(size: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((size + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn blocks(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct Search<'a> {
    challenge: &'a Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    solver: DnHandle,
    prop: &'a cudaDeviceProp,
    save: &'a dyn Fn(&Solution) -> Result<()>,
    best: Option<Solution>,
    error: f32,
}

impl Search<'_> {
    // Candidate generation is approximate; acceptance always uses the exact
    // same linking matrix and residual as verification, including roundoff.
    fn consider(&mut self, cols: Vec<i32>, rows: Vec<i32>) -> Result<bool> {
        let value = match self.challenge.evaluate_fast_fnorm(
            &cols,
            &rows,
            self.module.clone(),
            self.stream.clone(),
            self.prop,
        ) {
            Ok(value) if value.is_finite() && value >= 0.0 => value,
            _ => return Ok(false),
        };
        if value >= self.error {
            return Ok(false);
        }
        let solution = Solution {
            c_idxs: cols,
            r_idxs: rows,
        };
        (self.save)(&solution)?;
        self.error = value;
        self.best = Some(solution);
        Ok(true)
    }

    fn transpose(&self, a: &CudaSlice<f32>, rows: i32, cols: i32) -> Result<CudaSlice<f32>> {
        let mut out = self
            .stream
            .alloc_zeros::<f32>(rows as usize * cols as usize)?;
        unsafe {
            sys::cublasSgeam(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                cols,
                rows,
                &1.0f32,
                a.device_ptr(&self.stream).0 as *const f32,
                rows,
                &0.0f32,
                a.device_ptr(&self.stream).0 as *const f32,
                rows,
                out.device_ptr_mut(&self.stream).0 as *mut f32,
                cols,
            )
            .result()?;
        }
        Ok(out)
    }

    fn multiply(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: i32,
        n: i32,
        k: i32,
        transpose_a: bool,
    ) -> Result<CudaSlice<f32>> {
        let mut out = self.stream.alloc_zeros::<f32>(m as usize * n as usize)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: if transpose_a {
                        cublasOperation_t::CUBLAS_OP_T
                    } else {
                        cublasOperation_t::CUBLAS_OP_N
                    },
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    alpha: 1.0,
                    lda: if transpose_a { k } else { m },
                    ldb: k,
                    beta: 0.0,
                    ldc: m,
                },
                a,
                b,
                &mut out,
            )?;
        }
        Ok(out)
    }

    fn scale(
        &self,
        a: &mut CudaSlice<f32>,
        values: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let kernel = self.module.load_function("scale_rows_kernel")?;
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(a)
                .arg(values)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .launch(grid(rows * cols))?;
        }
        Ok(())
    }

    fn leading(
        &self,
        a: &CudaSlice<f32>,
        dim: usize,
        points: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>> {
        let indices = self
            .stream
            .memcpy_stod(&(0..k as i32).collect::<Vec<_>>())?;
        let mut out = self.stream.alloc_zeros::<f32>(k * points)?;
        let kernel = self.module.load_function("extract_rows_kernel")?;
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(a)
                .arg(&mut out)
                .arg(&(dim as i32))
                .arg(&(points as i32))
                .arg(&(k as i32))
                .arg(&indices)
                .launch(grid(k * points))?;
        }
        Ok(out)
    }

    /// Return U_s^T, V_s^T and normalized singular values. Both sides come
    /// from the same approximate SVD. QR between every power multiplication
    /// prevents the dominant singular directions from wiping out the tail.
    fn embeddings(
        &self,
        dim: usize,
        power_iters: usize,
        trial: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
        let m = self.challenge.m;
        let n = self.challenge.n;
        let s = dim as i32;
        let seed = u64::from_le_bytes(self.challenge.seed[..8].try_into()?)
            ^ (trial as u64 + 1).wrapping_mul(0xA076_1D64_78BD_642F);
        let mut omega = self.stream.alloc_zeros::<f32>(n as usize * dim)?;
        let gaussian = self.module.load_function("standard_gaussian_kernel")?;
        unsafe {
            self.stream
                .launch_builder(&gaussian)
                .arg(&mut omega)
                .arg(&(n * s))
                .arg(&(1.0 / (s as f32).sqrt()))
                .arg(&seed)
                .launch(grid(n as usize * dim))?;
        }
        let mut q = self.multiply(&self.challenge.d_a_mat, &omega, m, s, n, false)?;
        drop(omega);
        gpu_qr(&self.solver, &self.stream, &mut q, m, s)?;
        for _ in 0..power_iters {
            let mut z = self.multiply(&self.challenge.d_a_mat, &q, n, s, m, true)?;
            gpu_qr(&self.solver, &self.stream, &mut z, n, s)?;
            q = self.multiply(&self.challenge.d_a_mat, &z, m, s, n, false)?;
            gpu_qr(&self.solver, &self.stream, &mut q, m, s)?;
        }
        let mut projected = self.multiply(&q, &self.challenge.d_a_mat, s, n, m, true)?;
        let (small_u, mut sigma, vt) =
            gpu_svd_thin(&self.solver, &self.stream, &mut projected, s, n)?;
        drop(projected);
        let u = self.multiply(&q, &small_u, m, s, s, false)?;
        let ut = self.transpose(&u, m, s)?;
        let largest = sigma[0];
        if !largest.is_finite()
            || largest <= 0.0
            || sigma.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(anyhow!("non-finite or zero projected spectrum"));
        }
        for value in &mut sigma {
            *value /= largest;
        }
        Ok((ut, vt, self.stream.memcpy_stod(&sigma)?))
    }

    /// CPQR if target=None; otherwise simultaneous orthogonal matching
    /// pursuit, maximizing reduction in ||(I-P_selected) target||_F^2.
    /// All pivot/deflation work stays on the GPU, including selected masks.
    fn select(
        &self,
        dictionary: &CudaSlice<f32>,
        dim: usize,
        points: usize,
        k: usize,
        target: Option<&CudaSlice<f32>>,
    ) -> Result<Vec<i32>> {
        let mut x = dictionary.try_clone()?;
        let targets = if target.is_some() { k } else { 0 };
        let mut y = match target {
            Some(t) => t.try_clone()?,
            None => self.stream.alloc_zeros::<f32>(1)?,
        };
        let mut cross = self.stream.alloc_zeros::<f32>((targets * points).max(1))?;
        let mut used = self.stream.alloc_zeros::<i32>(points)?;
        let mut indices = self.stream.alloc_zeros::<i32>(k)?;
        let mut scores = self.stream.alloc_zeros::<f64>(points)?;
        let mut q = self.stream.alloc_zeros::<f32>(dim)?;
        let score_kernel = self.module.load_function("gc_scores")?;
        let pick_kernel = self.module.load_function("gc_pick")?;
        let direction_kernel = self.module.load_function("gc_direction")?;
        let deflate_kernel = self.module.load_function("gc_deflate")?;
        for step in 0..k {
            unsafe {
                // Recompute from the residuals at EVERY pivot. Rank-one
                // downdates of the original cross-products lose the weak
                // singular directions through catastrophic cancellation.
                if targets > 0 {
                    self.blas.gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_T,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: targets as i32,
                            n: points as i32,
                            k: dim as i32,
                            alpha: 1.0,
                            lda: dim as i32,
                            ldb: dim as i32,
                            beta: 0.0,
                            ldc: targets as i32,
                        },
                        &y,
                        &x,
                        &mut cross,
                    )?;
                }
                self.stream
                    .launch_builder(&score_kernel)
                    .arg(&x)
                    .arg(&cross)
                    .arg(&used)
                    .arg(&mut scores)
                    .arg(&(dim as i32))
                    .arg(&(points as i32))
                    .arg(&(targets as i32))
                    .launch(blocks(points))?;
                self.stream
                    .launch_builder(&pick_kernel)
                    .arg(&scores)
                    .arg(&mut used)
                    .arg(&mut indices)
                    .arg(&(points as i32))
                    .arg(&(step as i32))
                    .launch(blocks(1))?;
                self.stream
                    .launch_builder(&direction_kernel)
                    .arg(&x)
                    .arg(&indices)
                    .arg(&mut q)
                    .arg(&(dim as i32))
                    .arg(&(step as i32))
                    .launch(blocks(1))?;
                for _ in 0..2 {
                    self.stream
                        .launch_builder(&deflate_kernel)
                        .arg(&mut x)
                        .arg(&q)
                        .arg(&(dim as i32))
                        .arg(&(points as i32))
                        .launch(blocks(points))?;
                    if targets > 0 {
                        self.stream
                            .launch_builder(&deflate_kernel)
                            .arg(&mut y)
                            .arg(&q)
                            .arg(&(dim as i32))
                            .arg(&(targets as i32))
                            .launch(blocks(targets))?;
                    }
                }
            }
        }
        Ok(self.stream.memcpy_dtov(&indices)?)
    }

    // In A ≈ U Σ V^T, selected columns have coordinates Σ V[J,:]^T.
    // Their orthonormal span Q defines the row target Σ Q; selecting rows
    // to approximate that target improves the coupled CUR objective. The
    // reverse update is identical with U and V interchanged.
    fn conditional_target(
        &self,
        opposite: &CudaSlice<f32>,
        sigma: &CudaSlice<f32>,
        selected: &[i32],
        dim: usize,
        points: usize,
    ) -> Result<CudaSlice<f32>> {
        let k = selected.len();
        let indices = self.stream.memcpy_stod(selected)?;
        let mut target = self.stream.alloc_zeros::<f32>(dim * k)?;
        let extract = self.module.load_function("extract_columns_kernel")?;
        unsafe {
            self.stream
                .launch_builder(&extract)
                .arg(opposite)
                .arg(&mut target)
                .arg(&(dim as i32))
                .arg(&(points as i32))
                .arg(&(k as i32))
                .arg(&indices)
                .launch(grid(dim * k))?;
        }
        gpu_qr(
            &self.solver,
            &self.stream,
            &mut target,
            dim as i32,
            k as i32,
        )?;
        self.scale(&mut target, sigma, dim, k)?;
        Ok(target)
    }

    fn refine(
        &mut self,
        left: &CudaSlice<f32>,
        right: &CudaSlice<f32>,
        sigma: &CudaSlice<f32>,
        dim: usize,
        rounds: usize,
    ) -> Result<()> {
        let m = self.challenge.m as usize;
        let n = self.challenge.n as usize;
        let k = self.challenge.target_k as usize;
        for _ in 0..rounds {
            let Some(best) = &self.best else {
                break;
            };
            let cols = best.c_idxs.clone();
            let target = self.conditional_target(right, sigma, &cols, dim, n)?;
            let rows = self.select(left, dim, m, k, Some(&target))?;
            let mut improved = self.consider(cols, rows)?;
            let rows = self.best.as_ref().unwrap().r_idxs.clone();
            let target = self.conditional_target(left, sigma, &rows, dim, m)?;
            let cols = self.select(right, dim, n, k, Some(&target))?;
            improved |= self.consider(cols, rows)?;
            if !improved {
                break;
            }
        }
        Ok(())
    }
}

/// Submit improvements through `save_solution`; the TIG entry point expects
/// only success/failure as the return value.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let hp = match hyperparameters {
        Some(values) => serde_json::from_value::<Hyperparameters>(Value::Object(values.clone()))?,
        None => Hyperparameters::default(),
    };
    if hp.num_trials == 0 || !hp.sketch_ratio.is_finite() || hp.sketch_ratio < 0.0 {
        return Err(anyhow!(
            "num_trials must be positive and sketch_ratio finite and nonnegative"
        ));
    }
    if challenge.target_k <= 0 || challenge.target_k > challenge.m.min(challenge.n) {
        return Err(anyhow!("invalid CUR dimensions/target rank"));
    }
    let k = challenge.target_k as usize;
    let m = challenge.m as usize;
    let n = challenge.n as usize;
    // All kernels use i32 flat indices, matching the challenge kernels.
    if m.checked_mul(n)
        .map_or(true, |size| size > i32::MAX as usize)
    {
        return Err(anyhow!("CUR matrix exceeds supported index range"));
    }
    let extra = hp
        .sketch_extra
        .max((k as f64 * hp.sketch_ratio).ceil() as usize);
    let dim = k.saturating_add(extra).min(m).min(n);
    let mut search = Search {
        challenge,
        blas: CudaBlas::new(stream.clone())?,
        solver: DnHandle::new(stream.clone())?,
        module: module.clone(),
        stream: stream.clone(),
        prop,
        save: save_solution,
        best: None,
        error: f32::INFINITY,
    };

    if hp.baseline_trials > 0 {
        let parameters = serde_json::json!({"num_trials": hp.baseline_trials})
            .as_object()
            .unwrap()
            .clone();
        // Keep every baseline improvement, including saved candidates when
        // a later restart fails. Never mask a save_solution callback error.
        let incumbent = RefCell::new(None);
        let callback_error = RefCell::new(None);
        let save_baseline = |solution: &Solution| -> Result<()> {
            if let Err(error) = save_solution(solution) {
                *callback_error.borrow_mut() = Some(error);
                return Err(anyhow!("save_solution failed"));
            }
            *incumbent.borrow_mut() = Some(Solution {
                c_idxs: solution.c_idxs.clone(),
                r_idxs: solution.r_idxs.clone(),
            });
            Ok(())
        };
        let result = baseline::solve_challenge(
            challenge,
            &save_baseline,
            &Some(parameters),
            module,
            stream,
            prop,
        );
        if let Some(error) = callback_error.into_inner() {
            return Err(error);
        }
        if let Some(solution) = incumbent.into_inner().or_else(|| result.ok().flatten()) {
            let value = challenge.evaluate_fast_fnorm(
                &solution.c_idxs,
                &solution.r_idxs,
                search.module.clone(),
                search.stream.clone(),
                prop,
            )?;
            if value.is_finite() && value >= 0.0 {
                search.error = value;
                search.best = Some(solution);
            }
        }
    }

    for trial in 0..hp.num_trials {
        let (mut left, mut right, sigma) = match search.embeddings(dim, hp.power_iters, trial) {
            Ok(value) => value,
            // SVD convergence failure should preserve any existing solution.
            Err(error) if search.best.is_none() => return Err(error),
            Err(_) => continue,
        };
        let spectral_left = search.leading(&left, dim, m, k)?;
        let spectral_right = search.leading(&right, dim, n, k)?;
        let rows = search.select(&spectral_left, k, m, k, None)?;
        let cols = search.select(&spectral_right, k, n, k, None)?;
        search.consider(cols, rows)?;
        drop(spectral_left);
        drop(spectral_right);

        search.scale(&mut left, &sigma, dim, m)?;
        search.scale(&mut right, &sigma, dim, n)?;
        let rows = search.select(&left, dim, m, k, None)?;
        let cols = search.select(&right, dim, n, k, None)?;
        search.consider(cols.clone(), rows.clone())?;
        // A good column set can complement a row set from another candidate.
        if let Some(best) = &search.best {
            let best_rows = best.r_idxs.clone();
            search.consider(cols, best_rows)?;
            let best_cols = search.best.as_ref().unwrap().c_idxs.clone();
            search.consider(best_cols, rows)?;
        }
        search.refine(&left, &right, &sigma, dim, hp.refinement_rounds)?;
    }
    search
        .best
        .map(|_| ())
        .ok_or_else(|| anyhow!("no finite CUR candidate found"))
}

// Keep tests outside the submitted algorithm sources.
