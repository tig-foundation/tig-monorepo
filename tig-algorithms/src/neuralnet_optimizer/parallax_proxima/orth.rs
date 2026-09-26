use super::{Arc, Cfg, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, Result, Role, State};
use cudarc::driver::CudaView;

const SIDE: u32 = 256;
const MATRIX: usize = (SIDE * SIDE) as usize;
const TILE: u32 = 16;
const QUINTIC: [f32; 3] = [3.4445, -4.7750, 2.0315];
const CUBIC: [f32; 2] = [1.5, -0.5];

fn gemm(
    stream: &Arc<CudaStream>, kernel: &CudaFunction, cfg: LaunchConfig,
    a: &CudaSlice<f32>, b: &CudaSlice<f32>, out: &mut CudaSlice<f32>, transpose: i32,
) -> Result<()> {
    unsafe {
        stream.launch_builder(kernel).arg(a).arg(b).arg(out).arg(&transpose).launch(cfg)?;
    }
    Ok(())
}

fn combine(
    stream: &Arc<CudaStream>, kernel: &CudaFunction, cfg: LaunchConfig,
    x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, a: f32, b: f32,
) -> Result<()> {
    unsafe {
        stream.launch_builder(kernel).arg(x).arg(y).arg(&a).arg(&b).launch(cfg)?;
    }
    Ok(())
}

fn per_matrix(blocks: (u32, u32), threads: (u32, u32), count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks.0, blocks.1, count as u32),
        block_dim: (threads.0, threads.1, 1),
        shared_mem_bytes: 0,
    }
}

fn all_identical(
    weights: &CudaView<'_, u64>, count: usize,
    stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
) -> Result<bool> {
    let same = module.load_function("orth_same")?;
    let reduce = module.load_function("orth_same_reduce")?;
    let mut flags = unsafe { stream.alloc::<u32>(count * SIDE as usize)? };
    let mut result = unsafe { stream.alloc::<u32>(count)? };
    unsafe {
        stream.launch_builder(&same).arg(weights).arg(&mut flags).launch(per_matrix((SIDE, 1), (SIDE, 1), count))?;
        stream.launch_builder(&reduce).arg(&flags).arg(&mut result).launch(per_matrix((1, 1), (SIDE, 1), count))?;
    }
    let identical = stream.memcpy_dtov(&result)?;
    Ok(identical.iter().all(|&v| v == 1))
}

pub(super) fn apply(
    s: &State, c: &Cfg, params: &[CudaSlice<f32>], updates: &mut [CudaSlice<f32>],
    stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
) -> Result<()> {
    let indices: Vec<usize> = s.active.iter().copied().filter(|&i| {
        i >= 2 && i % 2 == 0 && params[i].len() == MATRIX
            && (s.roles[i] == Role::Weight || s.roles[i] == Role::HeadWeight)
    }).collect();
    if indices.is_empty() {
        return Ok(());
    }
    let count = indices.len();
    let mut ptrs = Vec::with_capacity(2 * count);
    for &i in &indices {
        ptrs.push(params[i].device_ptr(stream).0);
    }
    for &i in &indices {
        ptrs.push(updates[i].device_ptr(stream).0);
    }
    let table = stream.memcpy_stod(&ptrs)?;
    let wp = table.slice(..count);
    let up = table.slice(count..);
    let shared: i32 = if count > 1 && all_identical(&wp, count, stream, module)? { 1 } else { 0 };
    let work_count = if shared != 0 { 1 } else { count };
    let size = work_count * MATRIX;
    let mut x0 = unsafe { stream.alloc::<f32>(size)? };
    let mut x1 = unsafe { stream.alloc::<f32>(size)? };
    let mut gram = unsafe { stream.alloc::<f32>(size)? };
    let mut square = unsafe { stream.alloc::<f32>(size)? };
    let mut norm_w = unsafe { stream.alloc::<f32>(work_count)? };
    let mut norm_x = unsafe { stream.alloc::<f32>(work_count)? };
    let pack = module.load_function("orth_pack")?;
    let norm = module.load_function("orth_norm")?;
    let normalize = module.load_function("orth_normalize")?;
    let product = module.load_function("orth_gemm")?;
    let mix = module.load_function("orth_combine")?;
    let finish = module.load_function("orth_finish")?;
    let vector_cfg = per_matrix((SIDE, 1), (SIDE, 1), work_count);
    let norm_cfg = per_matrix((1, 1), (SIDE, 1), work_count);
    let matrix_cfg = per_matrix((SIDE / TILE, SIDE / TILE), (TILE, TILE), work_count);
    unsafe {
        stream.launch_builder(&pack).arg(&wp).arg(&mut x0).launch(vector_cfg)?;
        stream.launch_builder(&norm).arg(&x0).arg(&mut norm_w).launch(norm_cfg)?;
        stream.launch_builder(&normalize).arg(&mut x0).arg(&norm_w).launch(vector_cfg)?;
    }
    let mut current_is_zero = true;
    for _ in 0..c.orth_quintic {
        let (x, y) = if current_is_zero { (&x0, &mut x1) } else { (&x1, &mut x0) };
        gemm(stream, &product, matrix_cfg, x, x, &mut gram, 1)?;
        gemm(stream, &product, matrix_cfg, &gram, &gram, &mut square, 0)?;
        combine(stream, &mix, vector_cfg, &gram, &mut square, QUINTIC[1], QUINTIC[2])?;
        gemm(stream, &product, matrix_cfg, &square, x, y, 0)?;
        combine(stream, &mix, vector_cfg, x, y, QUINTIC[0], 1.0)?;
        current_is_zero = !current_is_zero;
    }
    for _ in 0..c.orth_cubic {
        let (x, y) = if current_is_zero { (&x0, &mut x1) } else { (&x1, &mut x0) };
        gemm(stream, &product, matrix_cfg, x, x, &mut gram, 1)?;
        gemm(stream, &product, matrix_cfg, &gram, x, y, 0)?;
        combine(stream, &mix, vector_cfg, x, y, CUBIC[0], CUBIC[1])?;
        current_is_zero = !current_is_zero;
    }
    let x = if current_is_zero { &x0 } else { &x1 };
    let finish_cfg = per_matrix((SIDE, 1), (SIDE, 1), count);
    unsafe {
        stream.launch_builder(&norm).arg(x).arg(&mut norm_x).launch(norm_cfg)?;
        stream.launch_builder(&finish).arg(&wp).arg(&up).arg(x)
            .arg(&norm_w).arg(&norm_x).arg(&c.orth_scale).arg(&shared).launch(finish_cfg)?;
    }
    Ok(())
}
