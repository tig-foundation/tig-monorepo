use super::{anyhow, fuel_remaining, Arc, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, Result};
use cudarc::cublas::{result as blas, sys::{cublasHandle_t, cublasOperation_t}, CudaBlas};
use cudarc::cudnn::{sys::*, Cudnn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use std::ffi::c_void;
use std::ptr::null_mut;
use tig_challenges::neuralnet_optimizer::{Challenge, Solution};

const OUT_DIMS: usize = 2;
const BN_EPS: f64 = 1e-5;
const MAX_FIT_ERR: f64 = 3e-3;
const MAX_STEPS: u64 = 20_000;
const SHALLOW_LAYERS: usize = 5;
const SHALLOW_STEPS: usize = 600;
const DEEP_STEPS: usize = 300;
const FUEL_SHARE: f64 = 0.8;
const MIN_AFFORDABLE_STEPS: usize = 12;
const RESERVED_STEPS: usize = 2;
const MIN_HALF_WIDTH: f64 = 1.05;
const MIN_SPECTRAL_DENSITY: f64 = 1e-280;
const MAX_PRIOR_PRECISION: f64 = 1e30;
const LIBRARY_FUEL_WEIGHT: u64 = 5;
const ADAM_FUEL_PER_THREAD: u64 = 60;
const GPU_FUEL_SCALE: u64 = 20;
const INIT_THREADS: u32 = 1024;
const ADAM_THREADS: u32 = 256;
const PI: f64 = std::f64::consts::PI;

pub(super) struct GpCfg {
    pub(super) enabled: bool,
    steps: Option<usize>,
    lr: f32,
    lr_min: f32,
    warmup: usize,
    beta1: f32,
    beta2: f32,
    eps: f32,
    grid_size: usize,
    anchor_count: usize,
    anchor_lo: f64,
    anchor_hi: f64,
    basis_size: usize,
    half_width: f64,
    length_scale: f64,
    signal_var: f64,
    noise_std: f64,
    use_validation: bool,
    kink: f32,
    bias_lr: f32,
    bn_lr: f32,
    save_initial: bool,
}

impl Default for GpCfg {
    fn default() -> Self {
        GpCfg {
            enabled: true,
            steps: None,
            lr: 2e-3,
            lr_min: 1e-5,
            warmup: 30,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            grid_size: 512,
            anchor_count: 128,
            anchor_lo: 1.25,
            anchor_hi: 1.75,
            basis_size: 80,
            half_width: 1.5,
            length_scale: 0.09,
            signal_var: 1.0,
            noise_std: 0.2,
            use_validation: true,
            kink: 1.5,
            bias_lr: 1.0,
            bn_lr: 1.0,
            save_initial: true,
        }
    }
}

pub(super) fn parse(hp: &Option<Map<String, Value>>) -> GpCfg {
    let mut c = GpCfg::default();
    let Some(m) = hp else { return c };
    let real = |k: &str, ok: fn(f64) -> bool| {
        m.get(k).and_then(Value::as_f64).filter(|v| v.is_finite() && ok(*v))
    };
    let double = |k: &str, d: f64, ok: fn(f64) -> bool| real(k, ok).unwrap_or(d);
    let single = |k: &str, d: f32, ok: fn(f64) -> bool| {
        real(k, ok).map(|v| v as f32).filter(|v| v.is_finite() && ok(*v as f64)).unwrap_or(d)
    };
    let integer = |k: &str, d: usize, lo: u64, hi: u64| {
        m.get(k).and_then(Value::as_u64).filter(|v| (lo..=hi).contains(v)).map(|v| v as usize).unwrap_or(d)
    };
    let flag = |k: &str, d: bool| m.get(k).and_then(Value::as_u64).filter(|v| *v <= 1).map(|v| v == 1).unwrap_or(d);
    let positive: fn(f64) -> bool = |v| v > 0.0;
    let nonnegative: fn(f64) -> bool = |v| v >= 0.0;
    let unit: fn(f64) -> bool = |v| (0.0..1.0).contains(&v);
    let coordinate: fn(f64) -> bool = |v| (1.0..=f32::MAX as f64).contains(&v);
    c.enabled = flag("gp_mode", c.enabled);
    c.steps = m.get("gp_steps").and_then(Value::as_u64).filter(|v| (1..=MAX_STEPS).contains(v)).map(|v| v as usize);
    c.lr = single("gp_lr", c.lr, positive);
    c.lr_min = single("gp_lr_min", c.lr_min, nonnegative);
    c.warmup = integer("gp_warm", c.warmup, 0, u64::MAX);
    c.beta1 = single("gp_b1", c.beta1, unit);
    c.beta2 = single("gp_b2", c.beta2, unit);
    c.eps = single("gp_eps", c.eps, positive);
    c.grid_size = integer("gp_n", c.grid_size, 64, 8192);
    c.anchor_count = m.get("gp_na").and_then(Value::as_u64).filter(|v| (2..=4096).contains(v) && v % 2 == 0)
        .map(|v| v as usize).unwrap_or(c.anchor_count);
    c.anchor_lo = double("gp_alo", c.anchor_lo, coordinate);
    c.anchor_hi = double("gp_ahi", c.anchor_hi, coordinate).max(c.anchor_lo);
    c.basis_size = integer("gp_m", c.basis_size, 8, 256);
    c.half_width = double("gp_lb", c.half_width, |v| v >= MIN_HALF_WIDTH);
    c.length_scale = double("gp_ell", c.length_scale, positive);
    c.signal_var = double("gp_var", c.signal_var, positive);
    c.noise_std = double("gp_noise", c.noise_std, positive);
    c.use_validation = flag("gp_val", c.use_validation);
    c.kink = single("gp_kink", c.kink, nonnegative);
    c.bias_lr = single("gp_blr", c.bias_lr, nonnegative);
    c.bn_lr = single("gp_bnlr", c.bn_lr, nonnegative);
    c.save_initial = flag("gp_save0", c.save_initial);
    c
}

pub(super) fn supported(challenge: &Challenge) -> bool {
    let dims = challenge.layer_dims();
    let ds = &challenge.dataset;
    dims.len() >= 4
        && ds.input_dims == 1
        && ds.output_dims == OUT_DIMS
        && challenge.num_frozen_layers == 2
        && ds.train_size > 0
        && dims[1..dims.len() - 1].iter().all(|&d| d == dims[1])
}

pub(super) fn try_solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    c: &GpCfg,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<bool> {
    let dims = challenge.layer_dims();
    let width = dims[1];
    let (xs, ys) = observations(challenge, c, &stream)?;
    let Some(teacher) = Teacher::fit(&xs, &ys, c) else { return Ok(false) };
    let Some(steps) = affordable_steps(&dims, c) else { return Ok(false) };
    let rows = c.grid_size + c.anchor_count;
    let (inputs, targets) = distillation_set(&teacher, c);

    let cublas = CudaBlas::new(stream.clone())?;
    let cudnn = Cudnn::new(stream.clone())?;
    let cx = Handles {
        stream: stream.clone(),
        blas: *cublas.handle(),
        dnn: cudnn.handle,
        relu: ActivationDesc::relu()?,
        add: OpTensorDesc::add()?,
        norm: TensorDesc::new(1, width)?,
    };
    let batch = BatchDescs::new(rows, width)?;

    let mut net = Net::new(&dims, rows, &stream)?;
    stream.memcpy_htod(&inputs, &mut net.inputs)?;
    stream.memcpy_htod(&targets, &mut net.targets)?;
    initialize(&mut net, challenge.seed, c, &module, &stream)?;
    if c.save_initial {
        save_solution(&net.initial_solution(&stream)?)?;
    }
    train(&net, &cx, &batch, c, steps, &module)?;
    net.forward(&cx, &batch)?;
    let running = net.running_stats(&stream)?;
    let fits = fit_error(&net, &targets, c.grid_size, &stream)? < MAX_FIT_ERR;
    if !fits {
        return Ok(false);
    }
    save_solution(&net.to_solution(&stream, running, steps)?)?;
    Ok(true)
}

fn observations(challenge: &Challenge, c: &GpCfg, st: &Arc<CudaStream>) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    let ds = &challenge.dataset;
    let train_x = st.memcpy_dtov(&ds.train_inputs())?;
    let train_y = st.memcpy_dtov(&ds.train_targets_noisy())?;
    let (val_x, val_y) = if c.use_validation && ds.validation_size > 0 {
        (st.memcpy_dtov(&ds.validation_inputs())?, st.memcpy_dtov(&ds.validation_targets_noisy())?)
    } else {
        (Vec::new(), Vec::new())
    };
    st.synchronize()?;
    let mut xs: Vec<f64> = train_x.iter().take(ds.train_size).map(|&v| v as f64).collect();
    let mut ys: Vec<[f64; 2]> = (0..ds.train_size).map(|i| [train_y[2 * i] as f64, train_y[2 * i + 1] as f64]).collect();
    for (i, &x) in val_x.iter().enumerate() {
        xs.push(x as f64);
        ys.push([val_y[2 * i] as f64, val_y[2 * i + 1] as f64]);
    }
    Ok((xs, ys))
}

fn affordable_steps(dims: &[usize], c: &GpCfg) -> Option<usize> {
    let default_steps = if dims.len() - 1 <= SHALLOW_LAYERS { SHALLOW_STEPS } else { DEEP_STEPS };
    let wanted = c.steps.unwrap_or(default_steps);
    let per_step = fuel_per_step(dims, c.grid_size + c.anchor_count).max(1);
    let affordable = ((fuel_remaining() as f64 * FUEL_SHARE) as u64 / per_step) as usize;
    (affordable >= MIN_AFFORDABLE_STEPS).then(|| wanted.min(affordable - RESERVED_STEPS))
}

fn fuel_per_step(dims: &[usize], rows: usize) -> u64 {
    let layers = dims.len() - 1;
    let trainable = layers - 2;
    let rows = rows as u64;
    let (mut products, mut norms, mut params) = (0u64, 0u64, 0u64);
    for i in 0..layers {
        let (fan_in, fan_out) = (dims[i] as u64, dims[i + 1] as u64);
        products += fan_in * fan_out * rows;
        if i > 0 {
            products += fan_in * fan_out * rows;
        }
        if i < trainable {
            products += fan_in * fan_out * rows + 2 * fan_out * rows;
            params += fan_in * fan_out + 3 * fan_out;
        }
        if i + 1 < layers {
            norms += 6 * fan_out * rows + 2;
        }
    }
    (LIBRARY_FUEL_WEIGHT * (products + norms) + ADAM_FUEL_PER_THREAD * (params + ADAM_THREADS as u64)) / GPU_FUEL_SCALE + 1
}

fn distillation_set(teacher: &Teacher, c: &GpCfg) -> (Vec<f32>, Vec<f32>) {
    let (grid, anchors) = (c.grid_size, c.anchor_count);
    let rows = grid + anchors;
    let mut row = vec![0.0f64; c.basis_size];
    let mut inputs = vec![0.0f32; rows];
    let mut targets = vec![0.0f32; OUT_DIMS * rows];
    let mut mean = [0.0f64; 2];
    for i in 0..grid {
        let x = -1.0 + (i as f64 + 0.5) * 2.0 / grid as f64;
        let o = teacher.eval(x, &mut row);
        inputs[i] = x as f32;
        targets[2 * i] = o[0] as f32;
        targets[2 * i + 1] = o[1] as f32;
        mean[0] += o[0];
        mean[1] += o[1];
    }
    mean[0] /= grid as f64;
    mean[1] /= grid as f64;
    let half = anchors / 2;
    let scale = grid as f64 / anchors as f64;
    for j in 0..anchors {
        let k = j % half;
        let pos = c.anchor_lo + (c.anchor_hi - c.anchor_lo) * (k as f64 + 0.5) / half as f64;
        let x = if j < half { -pos } else { pos };
        inputs[grid + j] = x as f32;
        targets[2 * (grid + j)] = (-scale * mean[0]) as f32;
        targets[2 * (grid + j) + 1] = (-scale * mean[1]) as f32;
    }
    (inputs, targets)
}

fn initialize(net: &mut Net, seed: [u8; 32], c: &GpCfg, module: &Arc<CudaModule>, st: &Arc<CudaStream>) -> Result<()> {
    let init = module.load_function("init_linear_layer")?;
    let mut rng = StdRng::from_seed(seed);
    for layer in 0..net.layers {
        let bytes: [u8; 32] = rng.r#gen();
        let layer_seed = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]);
        let (fan_out, fan_in) = (net.fan_out[layer], net.fan_in[layer]);
        let (weights, biases) = (net.weight_addr(layer, st), net.bias_addr(layer, st));
        let count = (fan_out * fan_in) as u32;
        unsafe {
            st.launch_builder(&init)
                .arg(&layer_seed)
                .arg(&(fan_out as i32))
                .arg(&(fan_in as i32))
                .arg(&weights)
                .arg(&biases)
                .launch(LaunchConfig {
                    grid_dim: ((count + INIT_THREADS - 1) / INIT_THREADS, 1, 1),
                    block_dim: (INIT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }
    net.spread_first_biases(c.kink, st)?;
    net.upload_lr_scale(c, st)
}

fn learning_rate(c: &GpCfg, steps: usize, s: usize) -> f32 {
    if s < c.warmup {
        return c.lr * (s + 1) as f32 / c.warmup.max(1) as f32;
    }
    let span = steps.saturating_sub(c.warmup).max(1) as f32;
    let p = ((s - c.warmup) as f32 / span).min(1.0);
    c.lr_min + 0.5 * (c.lr - c.lr_min) * (1.0 + (std::f32::consts::PI * p).cos())
}

fn train(net: &Net, cx: &Handles, batch: &BatchDescs, c: &GpCfg, steps: usize, module: &Arc<CudaModule>) -> Result<()> {
    let st = &cx.stream;
    let adam = module.load_function("gp_adam")?;
    let count = net.param_count as i32;
    let cfg = LaunchConfig {
        grid_dim: (((net.param_count + ADAM_THREADS as usize - 1) / ADAM_THREADS as usize) as u32, 1, 1),
        block_dim: (ADAM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    for s in 0..steps {
        net.forward(cx, batch)?;
        net.output_residual(cx, batch)?;
        net.backward(cx, batch)?;
        let lr = learning_rate(c, steps, s);
        let bc1 = 1.0 - c.beta1.powi((s + 1) as i32);
        let bc2 = 1.0 - c.beta2.powi((s + 1) as i32);
        let params = addr(&net.params, st);
        let grads = addr(&net.grads, st);
        let first = addr(&net.first_moment, st);
        let second = addr(&net.second_moment, st);
        let scale = addr(&net.lr_scale, st);
        unsafe {
            st.launch_builder(&adam)
                .arg(&params)
                .arg(&grads)
                .arg(&first)
                .arg(&second)
                .arg(&scale)
                .arg(&lr)
                .arg(&c.beta1)
                .arg(&c.beta2)
                .arg(&bc1)
                .arg(&bc2)
                .arg(&c.eps)
                .arg(&count)
                .launch(cfg)?;
        }
    }
    Ok(())
}

fn fit_error(net: &Net, targets: &[f32], grid: usize, st: &Arc<CudaStream>) -> Result<f64> {
    let outputs = st.memcpy_dtov(&net.outputs)?;
    let mut se = 0.0f64;
    for i in 0..grid {
        for o in 0..OUT_DIMS {
            let d = outputs[2 * i + o] as f64 - targets[2 * i + o] as f64;
            se += d * d;
        }
    }
    Ok(se / (2 * grid) as f64)
}

struct Teacher {
    basis_size: usize,
    half_width: f64,
    inv_sqrt_half_width: f64,
    coef: [Vec<f64>; 2],
}

impl Teacher {
    fn basis(&self, x: f64, row: &mut [f64]) {
        let th = PI * (x + self.half_width) / (2.0 * self.half_width);
        let (s1, c1) = th.sin_cos();
        let two_c = 2.0 * c1;
        let mut prev = 0.0f64;
        let mut cur = s1;
        for r in row.iter_mut().take(self.basis_size) {
            *r = cur * self.inv_sqrt_half_width;
            let next = two_c * cur - prev;
            prev = cur;
            cur = next;
        }
    }

    fn eval(&self, x: f64, row: &mut [f64]) -> [f64; 2] {
        self.basis(x, row);
        let mut o = [0.0f64; 2];
        for j in 0..self.basis_size {
            o[0] += row[j] * self.coef[0][j];
            o[1] += row[j] * self.coef[1][j];
        }
        o
    }

    fn fit(xs: &[f64], ys: &[[f64; 2]], c: &GpCfg) -> Option<Teacher> {
        let m = c.basis_size;
        let mut t = Teacher {
            basis_size: m,
            half_width: c.half_width,
            inv_sqrt_half_width: 1.0 / c.half_width.sqrt(),
            coef: [vec![0.0; m], vec![0.0; m]],
        };
        let (mut gram, rhs) = t.normal_equations(xs, ys);
        add_prior(&mut gram, m, c);
        cholesky(&mut gram, m)?;
        for o in 0..2 {
            t.coef[o] = cholesky_solve(&gram, m, &rhs[o])?;
        }
        Some(t)
    }

    fn normal_equations(&self, xs: &[f64], ys: &[[f64; 2]]) -> (Vec<f64>, [Vec<f64>; 2]) {
        let m = self.basis_size;
        let mut gram = vec![0.0f64; m * m];
        let mut rhs = [vec![0.0f64; m], vec![0.0f64; m]];
        let mut row = vec![0.0f64; m];
        for (&x, y) in xs.iter().zip(ys) {
            self.basis(x, &mut row);
            for p in 0..m {
                let rp = row[p];
                rhs[0][p] += rp * y[0];
                rhs[1][p] += rp * y[1];
                let base = p * m;
                for q in 0..=p {
                    gram[base + q] += rp * row[q];
                }
            }
        }
        (gram, rhs)
    }
}

fn add_prior(gram: &mut [f64], m: usize, c: &GpCfg) {
    let noise2 = c.noise_std * c.noise_std;
    let amp = c.signal_var * (2.0 * PI).sqrt() * c.length_scale;
    for j in 0..m {
        let w = PI * (j + 1) as f64 / (2.0 * c.half_width);
        let s = amp * (-0.5 * w * w * c.length_scale * c.length_scale).exp();
        let precision = if s > MIN_SPECTRAL_DENSITY { (noise2 / s).min(MAX_PRIOR_PRECISION) } else { MAX_PRIOR_PRECISION };
        gram[j * m + j] += precision;
    }
}

fn cholesky(a: &mut [f64], m: usize) -> Option<()> {
    for p in 0..m {
        for q in 0..=p {
            let mut sum = a[p * m + q];
            for k in 0..q {
                sum -= a[p * m + k] * a[q * m + k];
            }
            if p == q {
                if !(sum > 0.0) || !sum.is_finite() {
                    return None;
                }
                a[p * m + p] = sum.sqrt();
            } else {
                a[p * m + q] = sum / a[q * m + q];
            }
        }
    }
    Some(())
}

fn cholesky_solve(l: &[f64], m: usize, rhs: &[f64]) -> Option<Vec<f64>> {
    let mut z = rhs.to_vec();
    for p in 0..m {
        let mut s = z[p];
        for k in 0..p {
            s -= l[p * m + k] * z[k];
        }
        z[p] = s / l[p * m + p];
    }
    for p in (0..m).rev() {
        let mut s = z[p];
        for k in p + 1..m {
            s -= l[k * m + p] * z[k];
        }
        z[p] = s / l[p * m + p];
    }
    z.iter().all(|v| v.is_finite()).then_some(z)
}

fn cudnn_check(status: cudnnStatus_t) -> Result<()> {
    if status == cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("cudnn status {:?}", status))
    }
}

struct TensorDesc(cudnnTensorDescriptor_t);

impl TensorDesc {
    fn new(rows: usize, channels: usize) -> Result<Self> {
        let mut d: cudnnTensorDescriptor_t = null_mut();
        unsafe {
            cudnn_check(cudnnCreateTensorDescriptor(&mut d))?;
            let desc = TensorDesc(d);
            cudnn_check(cudnnSetTensor4dDescriptor(
                d,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                rows as i32,
                channels as i32,
                1,
                1,
            ))?;
            Ok(desc)
        }
    }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.0);
        }
    }
}

struct ActivationDesc(cudnnActivationDescriptor_t);

impl ActivationDesc {
    fn relu() -> Result<Self> {
        let mut d: cudnnActivationDescriptor_t = null_mut();
        unsafe {
            cudnn_check(cudnnCreateActivationDescriptor(&mut d))?;
            let desc = ActivationDesc(d);
            cudnn_check(cudnnSetActivationDescriptor(
                d,
                cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                0.0,
            ))?;
            Ok(desc)
        }
    }
}

impl Drop for ActivationDesc {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyActivationDescriptor(self.0);
        }
    }
}

struct OpTensorDesc(cudnnOpTensorDescriptor_t);

impl OpTensorDesc {
    fn add() -> Result<Self> {
        let mut d: cudnnOpTensorDescriptor_t = null_mut();
        unsafe {
            cudnn_check(cudnnCreateOpTensorDescriptor(&mut d))?;
            let desc = OpTensorDesc(d);
            cudnn_check(cudnnSetOpTensorDescriptor(
                d,
                cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            ))?;
            Ok(desc)
        }
    }
}

impl Drop for OpTensorDesc {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyOpTensorDescriptor(self.0);
        }
    }
}

struct BatchDescs {
    rows: usize,
    hidden: TensorDesc,
    output: TensorDesc,
}

impl BatchDescs {
    fn new(rows: usize, width: usize) -> Result<Self> {
        Ok(BatchDescs { rows, hidden: TensorDesc::new(rows, width)?, output: TensorDesc::new(rows, OUT_DIMS)? })
    }
}

struct Handles {
    stream: Arc<CudaStream>,
    blas: cublasHandle_t,
    dnn: cudnnHandle_t,
    relu: ActivationDesc,
    add: OpTensorDesc,
    norm: TensorDesc,
}

fn addr<T>(buf: &CudaSlice<T>, st: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = buf.device_ptr(st);
    ptr
}

fn const_ptr(ptr: u64) -> *const c_void {
    ptr as *const c_void
}

fn mut_ptr(ptr: u64) -> *mut c_void {
    ptr as *mut c_void
}

fn scalar(v: &f32) -> *const c_void {
    v as *const f32 as *const c_void
}

fn gemm(
    cx: &Handles, ta: bool, tb: bool, m: usize, n: usize, k: usize,
    a: u64, lda: usize, b: u64, ldb: usize, beta: f32, c: u64, ldc: usize,
) -> Result<()> {
    let alpha = 1.0f32;
    let op = |t: bool| if t { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
    unsafe {
        blas::sgemm(
            cx.blas, op(ta), op(tb), m as i32, n as i32, k as i32,
            &alpha as *const f32, a as *const f32, lda as i32,
            b as *const f32, ldb as i32, &beta as *const f32, c as *mut f32, ldc as i32,
        )
    }
    .map_err(|e| anyhow!("sgemm: {e:?}"))
}

#[derive(Clone, Copy)]
struct Offsets {
    weight: usize,
    bias: usize,
    gamma: usize,
    beta: usize,
}

struct Net {
    layers: usize,
    width: usize,
    trainable: usize,
    fan_in: Vec<usize>,
    fan_out: Vec<usize>,
    params: CudaSlice<f32>,
    grads: CudaSlice<f32>,
    first_moment: CudaSlice<f32>,
    second_moment: CudaSlice<f32>,
    lr_scale: CudaSlice<f32>,
    param_count: usize,
    offsets: Vec<Offsets>,
    frozen_weights: Vec<CudaSlice<f32>>,
    frozen_biases: Vec<CudaSlice<f32>>,
    frozen_gamma: CudaSlice<f32>,
    frozen_beta: CudaSlice<f32>,
    frozen_norm_grads: CudaSlice<f32>,
    saved_mean: Vec<CudaSlice<f32>>,
    saved_inv_std: Vec<CudaSlice<f32>>,
    inputs: CudaSlice<f32>,
    targets: CudaSlice<f32>,
    outputs: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    activations: Vec<CudaSlice<f32>>,
    hidden: Vec<CudaSlice<f32>>,
    upstream: CudaSlice<f32>,
    local: CudaSlice<f32>,
}

impl Net {
    fn new(dims: &[usize], rows: usize, st: &Arc<CudaStream>) -> Result<Self> {
        let layers = dims.len() - 1;
        let width = dims[1];
        let trainable = layers - 2;
        let mut offsets = Vec::with_capacity(trainable);
        let mut acc = 0usize;
        for i in 0..trainable {
            let weight = acc;
            acc += dims[i + 1] * dims[i];
            let bias = acc;
            acc += dims[i + 1];
            let gamma = acc;
            acc += dims[i + 1];
            let beta = acc;
            acc += dims[i + 1];
            offsets.push(Offsets { weight, bias, gamma, beta });
        }
        let param_count = acc;
        let zeros = |len: usize| st.alloc_zeros::<f32>(len).map_err(|e| anyhow!("alloc: {e}"));
        let per_norm = |len: usize| (0..layers - 1).map(|_| zeros(len)).collect::<Result<Vec<_>>>();
        let mut params = zeros(param_count)?;
        let mut initial = vec![0.0f32; param_count];
        for (i, o) in offsets.iter().enumerate() {
            initial[o.gamma..o.gamma + dims[i + 1]].fill(1.0);
        }
        st.memcpy_htod(&initial, &mut params)?;
        Ok(Net {
            layers,
            width,
            trainable,
            fan_in: dims[..layers].to_vec(),
            fan_out: dims[1..].to_vec(),
            params,
            grads: zeros(param_count)?,
            first_moment: zeros(param_count)?,
            second_moment: zeros(param_count)?,
            lr_scale: zeros(param_count)?,
            param_count,
            offsets,
            frozen_weights: vec![zeros(dims[layers - 1] * dims[layers - 2])?, zeros(dims[layers] * dims[layers - 1])?],
            frozen_biases: vec![zeros(dims[layers - 1])?, zeros(dims[layers])?],
            frozen_gamma: st.memcpy_stod(&vec![1.0f32; width])?,
            frozen_beta: zeros(width)?,
            frozen_norm_grads: zeros(2 * width)?,
            saved_mean: per_norm(width)?,
            saved_inv_std: per_norm(width)?,
            inputs: zeros(rows)?,
            targets: zeros(OUT_DIMS * rows)?,
            outputs: zeros(OUT_DIMS * rows)?,
            residual: zeros(OUT_DIMS * rows)?,
            ones: st.memcpy_stod(&vec![1.0f32; rows])?,
            activations: per_norm(width * rows)?,
            hidden: per_norm(width * rows)?,
            upstream: zeros(width * rows)?,
            local: zeros(width * rows)?,
        })
    }

    fn weight_addr(&self, layer: usize, st: &Arc<CudaStream>) -> u64 {
        if layer < self.trainable {
            addr(&self.params, st) + 4 * self.offsets[layer].weight as u64
        } else {
            addr(&self.frozen_weights[layer - self.trainable], st)
        }
    }

    fn bias_addr(&self, layer: usize, st: &Arc<CudaStream>) -> u64 {
        if layer < self.trainable {
            addr(&self.params, st) + 4 * self.offsets[layer].bias as u64
        } else {
            addr(&self.frozen_biases[layer - self.trainable], st)
        }
    }

    fn norm_addrs(&self, layer: usize, st: &Arc<CudaStream>) -> (u64, u64) {
        if layer < self.trainable {
            let base = addr(&self.params, st);
            let o = self.offsets[layer];
            (base + 4 * o.gamma as u64, base + 4 * o.beta as u64)
        } else {
            (addr(&self.frozen_gamma, st), addr(&self.frozen_beta, st))
        }
    }

    fn spread_first_biases(&mut self, spread: f32, st: &Arc<CudaStream>) -> Result<()> {
        let o = self.offsets[0];
        let width = self.width;
        let weights = st.memcpy_dtov(&self.params.slice(o.weight..o.weight + width))?;
        let biases: Vec<f32> = (0..width)
            .map(|i| {
                let t = spread * (-1.0 + 2.0 * (i as f32 + 0.5) / width as f32);
                -weights[i] * t
            })
            .collect();
        st.memcpy_htod(&biases, &mut self.params.slice_mut(o.bias..o.bias + width))?;
        Ok(())
    }

    fn upload_lr_scale(&mut self, c: &GpCfg, st: &Arc<CudaStream>) -> Result<()> {
        let mut scale = vec![1.0f32; self.param_count];
        for o in &self.offsets {
            scale[o.bias..o.bias + self.width].fill(c.bias_lr);
            scale[o.gamma..o.gamma + self.width].fill(c.bn_lr);
            scale[o.beta..o.beta + self.width].fill(c.bn_lr);
        }
        st.memcpy_htod(&scale, &mut self.lr_scale)?;
        Ok(())
    }

    fn forward(&self, cx: &Handles, d: &BatchDescs) -> Result<()> {
        let st = &cx.stream;
        let (width, rows) = (self.width, d.rows);
        let (one, zero) = (1.0f32, 0.0f32);
        for l in 0..self.layers - 1 {
            let (input, k) = if l == 0 { (addr(&self.inputs, st), 1) } else { (addr(&self.hidden[l - 1], st), width) };
            let act = addr(&self.activations[l], st);
            gemm(cx, false, false, width, rows, k, self.weight_addr(l, st), width, input, k, 0.0, act, width)?;
            if l < self.trainable {
                gemm(cx, false, false, width, rows, 1, self.bias_addr(l, st), width, addr(&self.ones, st), 1, 1.0, act, width)?;
            }
            unsafe {
                cudnn_check(cudnnActivationForward(
                    cx.dnn, cx.relu.0, scalar(&one), d.hidden.0, const_ptr(act), scalar(&zero), d.hidden.0, mut_ptr(act),
                ))?;
            }
            let (gamma, beta) = self.norm_addrs(l, st);
            let out = addr(&self.hidden[l], st);
            unsafe {
                cudnn_check(cudnnBatchNormalizationForwardTraining(
                    cx.dnn,
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                    scalar(&one),
                    scalar(&zero),
                    d.hidden.0,
                    const_ptr(act),
                    d.hidden.0,
                    mut_ptr(out),
                    cx.norm.0,
                    const_ptr(gamma),
                    const_ptr(beta),
                    1.0,
                    null_mut(),
                    null_mut(),
                    BN_EPS,
                    mut_ptr(addr(&self.saved_mean[l], st)),
                    mut_ptr(addr(&self.saved_inv_std[l], st)),
                ))?;
            }
        }
        let last = self.layers - 1;
        let input = addr(&self.hidden[last - 1], st);
        gemm(cx, false, false, OUT_DIMS, rows, width, self.weight_addr(last, st), OUT_DIMS, input, width, 0.0, addr(&self.outputs, st), OUT_DIMS)
    }

    fn output_residual(&self, cx: &Handles, d: &BatchDescs) -> Result<()> {
        let st = &cx.stream;
        let (one, minus, zero) = (1.0f32, -1.0f32, 0.0f32);
        unsafe {
            cudnn_check(cudnnOpTensor(
                cx.dnn,
                cx.add.0,
                scalar(&one),
                d.output.0,
                const_ptr(addr(&self.outputs, st)),
                scalar(&minus),
                d.output.0,
                const_ptr(addr(&self.targets, st)),
                scalar(&zero),
                d.output.0,
                mut_ptr(addr(&self.residual, st)),
            ))
        }
    }

    fn backward(&self, cx: &Handles, d: &BatchDescs) -> Result<()> {
        let st = &cx.stream;
        let (width, rows) = (self.width, d.rows);
        let (one, zero) = (1.0f32, 0.0f32);
        let last = self.layers - 1;
        let (upstream, local) = (addr(&self.upstream, st), addr(&self.local, st));
        let grads = addr(&self.grads, st);
        gemm(cx, true, false, width, rows, OUT_DIMS, self.weight_addr(last, st), OUT_DIMS, addr(&self.residual, st), OUT_DIMS, 0.0, upstream, width)?;
        for l in (0..last).rev() {
            let act = addr(&self.activations[l], st);
            let (gamma, _) = self.norm_addrs(l, st);
            let (d_gamma, d_beta) = if l < self.trainable {
                let o = self.offsets[l];
                (grads + 4 * o.gamma as u64, grads + 4 * o.beta as u64)
            } else {
                let scratch = addr(&self.frozen_norm_grads, st);
                (scratch, scratch + 4 * width as u64)
            };
            unsafe {
                cudnn_check(cudnnBatchNormalizationBackward(
                    cx.dnn,
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                    scalar(&one),
                    scalar(&zero),
                    scalar(&one),
                    scalar(&zero),
                    d.hidden.0,
                    const_ptr(act),
                    d.hidden.0,
                    const_ptr(upstream),
                    d.hidden.0,
                    mut_ptr(local),
                    cx.norm.0,
                    const_ptr(gamma),
                    mut_ptr(d_gamma),
                    mut_ptr(d_beta),
                    BN_EPS,
                    const_ptr(addr(&self.saved_mean[l], st)),
                    const_ptr(addr(&self.saved_inv_std[l], st)),
                ))?;
                cudnn_check(cudnnActivationBackward(
                    cx.dnn, cx.relu.0, scalar(&one), d.hidden.0, const_ptr(act), d.hidden.0, const_ptr(local),
                    d.hidden.0, const_ptr(act), scalar(&zero), d.hidden.0, mut_ptr(local),
                ))?;
            }
            let (input, k) = if l == 0 { (addr(&self.inputs, st), 1) } else { (addr(&self.hidden[l - 1], st), width) };
            if l < self.trainable {
                let o = self.offsets[l];
                gemm(cx, false, true, width, k, rows, local, width, input, k, 0.0, grads + 4 * o.weight as u64, width)?;
                gemm(cx, false, false, width, 1, rows, local, width, addr(&self.ones, st), rows, 0.0, grads + 4 * o.bias as u64, width)?;
            }
            if l > 0 {
                gemm(cx, true, false, width, rows, width, self.weight_addr(l, st), width, local, width, 0.0, upstream, width)?;
            }
        }
        Ok(())
    }

    fn running_stats(&self, st: &Arc<CudaStream>) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let mut means = Vec::with_capacity(self.layers - 1);
        let mut vars = Vec::with_capacity(self.layers - 1);
        for l in 0..self.layers - 1 {
            let mean = st.memcpy_dtov(&self.saved_mean[l])?;
            let inv_std = st.memcpy_dtov(&self.saved_inv_std[l])?;
            let var = inv_std
                .iter()
                .map(|&s| {
                    let s = s as f64;
                    if s > 0.0 && s.is_finite() { ((1.0 / (s * s)) - BN_EPS).max(0.0) as f32 } else { 0.0 }
                })
                .collect();
            means.push(mean);
            vars.push(var);
        }
        Ok((means, vars))
    }

    fn initial_solution(&self, st: &Arc<CudaStream>) -> Result<Solution> {
        let means = (0..self.layers - 1).map(|_| vec![0.0f32; self.width]).collect();
        let vars = (0..self.layers - 1).map(|_| vec![1.0f32; self.width]).collect();
        self.to_solution(st, (means, vars), 0)
    }

    fn to_solution(&self, st: &Arc<CudaStream>, running: (Vec<Vec<f32>>, Vec<Vec<f32>>), epochs: usize) -> Result<Solution> {
        let host = st.memcpy_dtov(&self.params)?;
        let mut weights = Vec::with_capacity(self.layers);
        let mut biases = Vec::with_capacity(self.layers);
        for l in 0..self.layers {
            let (w, b) = if l < self.trainable {
                let o = self.offsets[l];
                (
                    host[o.weight..o.weight + self.fan_out[l] * self.fan_in[l]].to_vec(),
                    host[o.bias..o.bias + self.fan_out[l]].to_vec(),
                )
            } else {
                (
                    st.memcpy_dtov(&self.frozen_weights[l - self.trainable])?,
                    st.memcpy_dtov(&self.frozen_biases[l - self.trainable])?,
                )
            };
            weights.push(w.chunks(self.fan_in[l]).map(|c| c.to_vec()).collect());
            biases.push(b);
        }
        let mut bn_weights = Vec::with_capacity(self.layers - 1);
        let mut bn_biases = Vec::with_capacity(self.layers - 1);
        for l in 0..self.layers - 1 {
            if l < self.trainable {
                let o = self.offsets[l];
                bn_weights.push(host[o.gamma..o.gamma + self.width].to_vec());
                bn_biases.push(host[o.beta..o.beta + self.width].to_vec());
            } else {
                bn_weights.push(vec![1.0f32; self.width]);
                bn_biases.push(vec![0.0f32; self.width]);
            }
        }
        Ok(Solution {
            weights,
            biases,
            epochs_used: epochs,
            bn_weights,
            bn_biases,
            bn_running_means: running.0,
            bn_running_vars: running.1,
        })
    }
}
