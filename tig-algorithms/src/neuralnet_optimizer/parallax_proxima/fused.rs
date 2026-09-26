use super::{anyhow, Result, Cfg, State, Role, CudaSlice, CudaStream, CudaModule,
    CudaFunction, DevicePtr, PushKernelArg, LaunchConfig, Arc, VECTOR_THREADS};

const BIAS_MULTIPLIER: f32 = 1.25;
const FUSED_BLOCKS: u32 = 82;

fn role_lr_wd(role: Role, lr: f32, wd: f32, c: &Cfg) -> (f32, f32) {
    match role {
        Role::Weight => (lr, wd),
        Role::Bias => (lr * BIAS_MULTIPLIER, 0.0),
        Role::HeadWeight => (lr * c.head_mult, wd),
        Role::HeadBias => (lr * c.head_mult * BIAS_MULTIPLIER, 0.0),
        Role::BnWeight => (lr * c.bn_w, 0.0),
        Role::BnBias => (lr * c.bn_b, 0.0),
        Role::Skip => (0.0, 0.0),
    }
}

pub(super) struct FusedCache {
    ptrs_h: Vec<u64>,
    pg: CudaSlice<u64>,
    pp: CudaSlice<u64>,
    pd: CudaSlice<u64>,
    pm: CudaSlice<u64>,
    pv: CudaSlice<u64>,
    pn: CudaSlice<u64>,
    pgp: CudaSlice<u64>,
    lrs_h: Vec<f32>,
    wds_h: Vec<f32>,
    d_lr: CudaSlice<f32>,
    d_wd: CudaSlice<f32>,
    lr_bits: Option<u32>,
    kernel: CudaFunction,
}

pub(super) fn dispatch(
    s: &mut State, c: &Cfg, model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>], updates: &mut [CudaSlice<f32>],
    stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
    lr: f32, bc1: f32, bc2: f32, bc3: f32, first_step: i32,
) -> Result<()> {
    if s.active.is_empty() {
        return Ok(());
    }
    let t = s.active.len();
    if s.fz_prefix.is_none() {
        let mut pre = Vec::with_capacity(t + 1);
        let mut acc = 0i32;
        for &i in s.active.iter() {
            pre.push(acc);
            acc += model_params[i].len() as i32;
        }
        pre.push(acc);
        s.fz_total = acc as usize;
        s.fz_prefix = Some(stream.memcpy_stod(&pre)?);
    }
    let ptr = |v: &[CudaSlice<f32>], idx: &[usize]| -> Vec<u64> {
        idx.iter()
            .map(|&i| {
                let (p, _guard) = v[i].device_ptr(stream);
                p
            })
            .collect()
    };
    if s.fz_cache.is_none() {
        s.fz_cache = Some(FusedCache {
            ptrs_h: vec![0u64; t],
            pg: unsafe { stream.alloc::<u64>(t)? },
            pp: unsafe { stream.alloc::<u64>(t)? },
            pd: unsafe { stream.alloc::<u64>(t)? },
            pm: stream.memcpy_stod(&ptr(&s.m, &s.active))?,
            pv: stream.memcpy_stod(&ptr(&s.v, &s.active))?,
            pn: stream.memcpy_stod(&ptr(&s.nsq, &s.active))?,
            pgp: stream.memcpy_stod(&ptr(&s.gprev, &s.active))?,
            lrs_h: vec![0.0f32; t],
            wds_h: vec![0.0f32; t],
            d_lr: unsafe { stream.alloc::<f32>(t)? },
            d_wd: unsafe { stream.alloc::<f32>(t)? },
            lr_bits: None,
            kernel: module.load_function("sk_adan_fused")?,
        });
    }
    let cache = s.fz_cache.as_mut().ok_or_else(|| anyhow!("missing fused cache"))?;
    for (j, &i) in s.active.iter().enumerate() {
        cache.ptrs_h[j] = gradients[i].device_ptr(stream).0;
    }
    stream.memcpy_htod(&cache.ptrs_h, &mut cache.pg)?;
    for (j, &i) in s.active.iter().enumerate() {
        cache.ptrs_h[j] = model_params[i].device_ptr(stream).0;
    }
    stream.memcpy_htod(&cache.ptrs_h, &mut cache.pp)?;
    for (j, &i) in s.active.iter().enumerate() {
        cache.ptrs_h[j] = updates[i].device_ptr(stream).0;
    }
    stream.memcpy_htod(&cache.ptrs_h, &mut cache.pd)?;
    if cache.lr_bits != Some(lr.to_bits()) {
        for (j, &i) in s.active.iter().enumerate() {
            let (l, w) = role_lr_wd(s.roles[i], lr, c.wd, c);
            let frozen = i < 64 && (s.freeze_eff >> i) & 1 == 1;
            cache.lrs_h[j] = if frozen { 0.0 } else { l * s.depth_mult[i] };
            cache.wds_h[j] = if frozen { 0.0 } else { w };
        }
        stream.memcpy_htod(&cache.lrs_h, &mut cache.d_lr)?;
        stream.memcpy_htod(&cache.wds_h, &mut cache.d_wd)?;
        cache.lr_bits = Some(lr.to_bits());
    }
    let total = s.fz_total as i32;
    let nt = t as i32;
    unsafe {
        stream
            .launch_builder(&cache.kernel)
            .arg(&cache.pg).arg(&cache.pp).arg(&cache.pm).arg(&cache.pv).arg(&cache.pn).arg(&cache.pgp).arg(&cache.pd)
            .arg(s.fz_prefix.as_ref().ok_or_else(|| anyhow!("missing tensor offsets"))?)
            .arg(&cache.d_lr).arg(&cache.d_wd).arg(&nt)
            .arg(&c.beta1).arg(&c.beta2).arg(&c.b3).arg(&c.eps)
            .arg(&bc1).arg(&bc2).arg(&bc3)
            .arg(&first_step).arg(&c.cautious).arg(&c.ab_blend).arg(&c.wd_gate).arg(&total)
            .launch(LaunchConfig {
                grid_dim: (((s.fz_total as u32 + (VECTOR_THREADS - 1)) / VECTOR_THREADS).min(FUSED_BLOCKS), 1, 1),
                block_dim: (VECTOR_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}
