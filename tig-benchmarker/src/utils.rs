use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time;

pub type Result<T> = std::result::Result<T, String>;

pub async fn sleep(ms: u64) {
    time::sleep(time::Duration::from_millis(ms)).await;
}

pub fn time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
