use super::{Job, Result};
use crate::future_utils::Mutex;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use tig_utils::get;

static CACHE: OnceCell<Mutex<HashMap<String, Vec<u8>>>> = OnceCell::new();

pub async fn execute(job: &Job) -> Result<Vec<u8>> {
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .await;
    if let Some(wasm_blob) = cache.get(&job.settings.algorithm_id) {
        Ok(wasm_blob.clone())
    } else {
        let wasm = get::<Vec<u8>>(&job.download_url, None)
            .await
            .map_err(|e| format!("Failed to download wasm from {}: {:?}", job.download_url, e))?;
        (*cache).insert(job.settings.algorithm_id.clone(), wasm.clone());
        Ok(wasm)
    }
}
