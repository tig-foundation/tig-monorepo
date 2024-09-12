use super::{Job, Result};
use moka::future::{Cache, CacheBuilder};
use once_cell::sync::OnceCell;
use tig_utils::get;
use tokio::sync::Mutex;

static CACHE: OnceCell<Mutex<Cache<String, Vec<u8>>>> = OnceCell::new();

pub async fn execute(job: &Job) -> Result<Vec<u8>> {
    let cache = CACHE
        .get_or_init(|| {
            Mutex::new(
                CacheBuilder::new(100)
                    .time_to_live(std::time::Duration::from_secs(120))
                    .build(),
            )
        })
        .lock()
        .await;
    if let Some(wasm_blob) = cache.get(&job.settings.algorithm_id).await {
        Ok(wasm_blob)
    } else {
        let wasm = get::<Vec<u8>>(&job.download_url, None)
            .await
            .map_err(|e| format!("Failed to download wasm from {}: {:?}", job.download_url, e))?;
        (*cache)
            .insert(job.settings.algorithm_id.clone(), wasm.clone())
            .await;
        Ok(wasm)
    }
}
