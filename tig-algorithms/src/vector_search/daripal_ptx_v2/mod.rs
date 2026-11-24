use anyhow::{anyhow, Result};
use cudarc::driver::{
    safe::{CudaFunction, CudaModule, CudaStream, LaunchConfig},
    sys, CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::{cell::RefCell,collections::HashMap};
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const Q_TILE_ROWS: usize = 32;
const DB_PAD_ROWS: usize = 64;
const QUERIES_PER_BLOCK: usize = 256;
const CHUNK_DB: usize = 6_144;
const SCAN_SMEM: u32 = 4 * 32 * 16 * 24 * 2;
const TAU_HALF: f32 = 0.5;

struct KernelTable {
    module: Weak<CudaModule>,
    pack: CudaFunction,
    scan: CudaFunction,
    merge: CudaFunction,
    rescan: CudaFunction,
}
fn kernel_cache() -> &'static Mutex<HashMap<usize, Arc<KernelTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<KernelTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}
fn get_kernel_table(module: &Arc<CudaModule>, stream: &Arc<CudaStream>) -> Result<Arc<KernelTable>> {
    let mut cache = kernel_cache().lock().unwrap();
    cache.retain(|_, entry| entry.module.upgrade().is_some());
    let key = Arc::as_ptr(module) as usize;
    if let Some(entry) = cache.get(&key) {
        if entry.module.upgrade().is_some_and(|m| Arc::ptr_eq(&m, module)) {
            return Ok(entry.clone());
        }
    }
    // Function handles own the module. Bound retention rather than relying on
    // Weak expiry while the table is cached. Active calls keep their own Arc.
    if cache.len() >= 8 && !cache.contains_key(&key) {
        cache.clear();
        // Module destruction during eviction can bind a different context.
        stream.context().bind_to_thread()?;
    }
    let table = Arc::new(KernelTable {
        module: Arc::downgrade(module),
        pack: module.load_function("daripal_pack_fold")?,
        scan: module.load_function("daripal_scan")?,
        merge: module.load_function("daripal_merge")?,
        rescan: module.load_function("daripal_rescan")?,
    });
    table.scan.set_attribute(
        sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SCAN_SMEM as i32,
    )?;
    cache.insert(key, table.clone());
    Ok(table)
}
#[derive(Default)]struct Bufs {ctx_ptr:usize,q:Option<CudaSlice<u16>>,db:Option<CudaSlice<u16>>,s1:Option<CudaSlice<f32>>,i1:Option<CudaSlice<i32>>,s2:Option<CudaSlice<f32>>,best:Option<CudaSlice<u64>>,rescan:Option<CudaSlice<i32>>,count:Option<CudaSlice<i32>>}
thread_local! { static BUFS:RefCell<Bufs>=RefCell::new(Bufs::default()); }
fn reuse<T>(slot:&mut Option<CudaSlice<T>>,stream:&Arc<CudaStream>,n:usize)->Result<CudaSlice<T>> {
 match slot.take(){Some(v) if v.len()>=n=>Ok(v),_=>unsafe{stream.alloc::<T>(n)}}
}
fn round_up(n: usize, m: usize) -> usize {
    (n + m - 1) / m * m
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!("daripal expects {} dimensions, got {}", DIMS, challenge.vector_dims));
    }
    let nq = challenge.num_queries as usize;
    let db = challenge.database_size as usize;
    if nq == 0 {
        return save_solution(&Solution { indexes: Vec::new() });
    }
    if db == 0 {
        return save_solution(&Solution { indexes: vec![0usize; nq] });
    }

    stream.context().bind_to_thread()?;
    let n_sm=(prop.multiProcessorCount as u32).max(1);
    let kernels=get_kernel_table(&module, &stream)?;
    let ctx_ptr=Arc::as_ptr(stream.context()) as usize;
    let nq_pad = round_up(nq, Q_TILE_ROWS);
    let db_pad = round_up(db, DB_PAD_ROWS);
    let nchunks = (db + CHUNK_DB - 1) / CHUNK_DB;
    let nqb = (nq_pad + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
    let n_out = nchunks * nq_pad;
    let (nq_i, nq_pad_i, db_i, db_pad_i) = (nq as i32, nq_pad as i32, db as i32, db_pad as i32);
    let (chunk_i, nchunks_i, nqb_i) = (CHUNK_DB as i32, nchunks as i32, nqb as i32);
    let pack_blocks = (n_sm * 6).min((((nq_pad + db_pad) * 32 + 255) / 256) as u32).max(1);

    let pack=&kernels.pack;
    let (mut d_q,mut d_db,mut d_s1,mut d_i1,mut d_s2,mut d_best,mut d_rescan,mut d_count)=BUFS.with(|cell|->Result<_>{
      let mut b=cell.borrow_mut();
      if b.ctx_ptr != ctx_ptr {
          *b=Bufs{ctx_ptr,..Bufs::default()};
          // Dropping old slices may bind their old context.
          stream.context().bind_to_thread()?;
      }
      Ok((reuse(&mut b.q,&stream,nq_pad*DIMS_PAD)?,reuse(&mut b.db,&stream,db_pad*DIMS_PAD)?,reuse(&mut b.s1,&stream,n_out)?,reuse(&mut b.i1,&stream,n_out)?,reuse(&mut b.s2,&stream,n_out)?,reuse(&mut b.best,&stream,nq)?,reuse(&mut b.rescan,&stream,2*nq*nchunks)?,reuse(&mut b.count,&stream,1)?))
    })?;
    stream.memset_zeros(&mut d_count)?;
    unsafe {
        stream
            .launch_builder(&pack)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_q)
            .arg(&mut d_db)
            .arg(&nq_i)
            .arg(&nq_pad_i)
            .arg(&db_i)
            .arg(&db_pad_i)
            .launch(LaunchConfig {
                grid_dim: (pack_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let scan=&kernels.scan;
    unsafe {
        stream
            .launch_builder(&scan)
            .arg(&d_q)
            .arg(&d_db)
            .arg(&mut d_s1)
            .arg(&mut d_i1)
            .arg(&mut d_s2)
            .arg(&db_i)
            .arg(&nq_pad_i)
            .arg(&chunk_i)
            .arg(&nqb_i)
            .arg(&nchunks_i)
            .launch(LaunchConfig {
                grid_dim: (n_sm.min((nqb * nchunks) as u32), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: SCAN_SMEM,
            })?;
    }

    let merge=&kernels.merge;
    let rescan=&kernels.rescan;
    let tau = TAU_HALF;
    unsafe {
        stream
            .launch_builder(&merge)
            .arg(&d_s1)
            .arg(&d_i1)
            .arg(&d_s2)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_best)
            .arg(&mut d_rescan)
            .arg(&mut d_count)
            .arg(&nq_i)
            .arg(&nchunks_i)
            .arg(&nq_pad_i)
            .arg(&db_i)
            .arg(&tau)
            .launch(LaunchConfig {
                grid_dim: (((nq * 32 + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        stream
            .launch_builder(&rescan)
            .arg(&d_rescan)
            .arg(&d_count)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_best)
            .arg(&chunk_i)
            .arg(&db_i)
            .launch(LaunchConfig {
                grid_dim: (n_sm * 4, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    stream.synchronize()?;
    let best: Vec<u64> = stream.memcpy_dtov(&d_best)?;
    let indexes = best
        .iter().take(nq)
        .map(|&k| {
            let idx = (k & 0xffff_ffff) as i64;
            if idx < 0 || idx >= db as i64 { 0usize } else { idx as usize }
        })
        .collect();
    let saved=save_solution(&Solution { indexes });
    BUFS.with(|cell|*cell.borrow_mut()=Bufs{ctx_ptr,q:Some(d_q),db:Some(d_db),s1:Some(d_s1),i1:Some(d_i1),s2:Some(d_s2),best:Some(d_best),rescan:Some(d_rescan),count:Some(d_count)});
    saved
}

pub fn help() {
    println!("daripal_ptx_v2 - exact GPU 1-NN with immediate-offset WMMA loads and lop3 selection, no hyperparameters.");
}

