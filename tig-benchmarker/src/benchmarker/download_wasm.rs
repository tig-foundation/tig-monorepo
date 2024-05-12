use super::{blobs, Job, Result};
use tig_utils::get;

pub async fn execute(job: &Job) -> Result<Vec<u8>> {
    let mut blobs = blobs().lock().await;
    if let Some(wasm_blob) = blobs.get(&job.settings.algorithm_id) {
        Ok(wasm_blob.clone())
    } else {
        let wasm = get::<Vec<u8>>(&job.download_url, None)
            .await
            .map_err(|e| format!("Failed to download wasm from {}: {:?}", job.download_url, e))?;
        (*blobs).insert(job.settings.algorithm_id.clone(), wasm.clone());
        Ok(wasm)
    }
}
