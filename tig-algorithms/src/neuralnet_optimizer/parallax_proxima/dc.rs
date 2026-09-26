use super::{target_mean, vec_launch, Arc, Cfg, CudaModule, CudaSlice, CudaStream, PushKernelArg, Result, State};

const OD: usize = 2;
const BN_EPS: f64 = 1e-5;
const HALF: f64 = 0.5;

#[derive(Clone, Copy)]
pub(super) struct DcIdx {
    bias: usize,
    w3: usize,
    w4: usize,
    rv: usize,
}

pub(super) fn indices(param_count: usize) -> Option<DcIdx> {
    if (param_count + 4) % 6 != 0 {
        return None;
    }
    let l = (param_count + 4) / 6;
    if l < 4 {
        return None;
    }
    let base = 2 * l;
    let bn_frozen = l - 2;
    let bn_shifted = l - 3;
    Some(DcIdx {
        bias: base + 4 * bn_shifted + 1,
        w3: 2 * (l - 2),
        w4: 2 * (l - 1),
        rv: base + 4 * bn_frozen + 3,
    })
}

pub(super) fn apply(
    s: &mut State, c: &Cfg, epoch: usize, batch_idx: usize, model_params: &[CudaSlice<f32>],
    updates: &mut [CudaSlice<f32>], stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
) -> Result<()> {
    let Some(ix) = s.dc_idx else { return Ok(()) };
    if s.batches_per_epoch == 0 {
        return Ok(());
    }
    if batch_idx == 0 && s.dc_applied != 0.0 {
        let gain = -s.dc_applied;
        s.dc_applied = 0.0;
        if let Some(offset) = s.dc_dev.as_ref() {
            add_offset(stream, module, &mut updates[ix.bias], offset, gain)?;
        }
    }
    if batch_idx == 0 && epoch + 1 >= c.dc_start && epoch >= s.dc_next_refresh {
        s.dc_next_refresh = epoch.saturating_add(c.dc_refresh);
        solve(s, c, ix, model_params, stream)?;
    }
    let gain = s.dc_gain_eff;
    if batch_idx + 1 >= s.batches_per_epoch && epoch >= c.dc_start && s.dc_ready && gain != 0.0 {
        if let Some(offset) = s.dc_dev.as_ref() {
            add_offset(stream, module, &mut updates[ix.bias], offset, gain)?;
        }
        s.dc_applied = gain;
    }
    Ok(())
}

fn add_offset(
    stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
    update: &mut CudaSlice<f32>, offset: &CudaSlice<f32>, gain: f32,
) -> Result<()> {
    let n = update.len();
    unsafe {
        stream
            .launch_builder(&module.load_function("sk_offset_vec")?)
            .arg(update)
            .arg(offset)
            .arg(&gain)
            .arg(&(n as i32))
            .launch(vec_launch(n))?;
    }
    Ok(())
}

fn solve(s: &mut State, c: &Cfg, ix: DcIdx, model_params: &[CudaSlice<f32>], stream: &Arc<CudaStream>) -> Result<()> {
    if ix.rv >= model_params.len() {
        return Ok(());
    }
    let h = model_params[ix.bias].len();
    if h == 0
        || model_params[ix.w3].len() != h * h
        || model_params[ix.w4].len() != OD * h
        || model_params[ix.rv].len() != h
    {
        return Ok(());
    }

    if !s.dc_frozen_read {
        s.dc_w3 = stream.memcpy_dtov(&model_params[ix.w3])?;
        s.dc_w4 = stream.memcpy_dtov(&model_params[ix.w4])?;
        stream.synchronize()?;
        s.dc_frozen_read = true;
    }
    let rv = stream.memcpy_dtov(&model_params[ix.rv])?;
    stream.synchronize()?;

    let mut c0 = vec![0.0f64; h];
    let mut c1 = vec![0.0f64; h];
    for i in 0..h {
        let sig = ((rv[i] as f64) + BN_EPS).max(1e-12).sqrt();
        c0[i] = (s.dc_w4[i * OD] as f64) * HALF / sig;
        c1[i] = (s.dc_w4[i * OD + 1] as f64) * HALF / sig;
    }

    let mut a0 = vec![0.0f64; h];
    let mut a1 = vec![0.0f64; h];
    for k in 0..h {
        let base = k * h;
        let mut t0 = 0.0f64;
        let mut t1 = 0.0f64;
        for i in 0..h {
            let w = s.dc_w3[base + i] as f64;
            t0 += c0[i] * w;
            t1 += c1[i] * w;
        }
        a0[k] = t0;
        a1[k] = t1;
    }

    let (mut g00, mut g01, mut g11) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..h {
        g00 += a0[k] * a0[k];
        g01 += a0[k] * a1[k];
        g11 += a1[k] * a1[k];
    }
    let lam = 1e-6 * (g00 + g11) + 1e-30;
    let (g00r, g11r) = (g00 + lam, g11 + lam);
    let det = g00r * g11r - g01 * g01;
    if !det.is_finite() || det.abs() < 1e-30 {
        return Ok(());
    }
    let m = target_mean();
    let y0 = (g11r * m[0] - g01 * m[1]) / det;
    let y1 = (-g01 * m[0] + g00r * m[1]) / det;

    let mut d = vec![0.0f32; h];
    let mut mx = 0.0f64;
    for k in 0..h {
        let v = a0[k] * y0 + a1[k] * y1;
        if !v.is_finite() {
            return Ok(());
        }
        mx = mx.max(v.abs());
        d[k] = v as f32;
    }
    if mx > c.dc_max_abs as f64 {
        let sc = (c.dc_max_abs as f64) / mx;
        for x in d.iter_mut() {
            *x = ((*x as f64) * sc) as f32;
        }
    }

    if s.dc_dev.as_ref().map(|b| b.len()) != Some(h) {
        s.dc_dev = Some(unsafe { stream.alloc::<f32>(h)? });
    }
    if let Some(buf) = s.dc_dev.as_mut() {
        stream.memcpy_htod(&d[..], buf)?;
    }
    s.dc_ready = true;
    Ok(())
}
