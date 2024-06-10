#[cfg(all(feature = "standalone", feature = "browser"))]
compile_error!("features `standalone` and `browser` are mutually exclusive");

use serde::{de::DeserializeOwned, Serialize};
use std::future::Future;

#[cfg(feature = "standalone")]
mod utils {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    pub use tokio::sync::Mutex;
    use tokio::{join, task, time};

    pub async fn join<T, U, V, W>(
        a: impl Future<Output = T> + 'static,
        b: impl Future<Output = U> + 'static,
        c: impl Future<Output = V> + 'static,
        d: impl Future<Output = W> + 'static,
    ) -> Result<(T, U, V, W), String>
    where
        T: Serialize + DeserializeOwned + 'static,
        U: Serialize + DeserializeOwned + 'static,
        V: Serialize + DeserializeOwned + 'static,
        W: Serialize + DeserializeOwned + 'static,
    {
        Ok(join!(a, b, c, d))
    }

    pub fn spawn(f: impl Future<Output = ()> + 'static + Send) {
        tokio::spawn(f);
    }

    pub async fn yield_now() {
        task::yield_now().await
    }

    pub async fn sleep(ms: u32) {
        time::sleep(time::Duration::from_millis(ms as u64)).await;
    }

    pub fn time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

#[cfg(feature = "browser")]
mod utils {
    use super::*;
    pub use futures::lock::Mutex;
    use gloo_timers::future::TimeoutFuture;
    use js_sys::{Array, Date, Promise};
    use serde_wasm_bindgen::{from_value, to_value};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::future_to_promise;
    use wasm_bindgen_futures::JsFuture;

    fn to_string<T: std::fmt::Debug>(e: T) -> String {
        format!("{:?}", e)
    }
    pub async fn join<T, U, V, W>(
        a: impl Future<Output = T> + 'static,
        b: impl Future<Output = U> + 'static,
        c: impl Future<Output = V> + 'static,
        d: impl Future<Output = W> + 'static,
    ) -> Result<(T, U, V, W), String>
    where
        T: Serialize + DeserializeOwned + 'static,
        U: Serialize + DeserializeOwned + 'static,
        V: Serialize + DeserializeOwned + 'static,
        W: Serialize + DeserializeOwned + 'static,
    {
        let a = future_to_promise(async move { Ok(to_value(&a.await)?) });
        let b = future_to_promise(async move { Ok(to_value(&b.await)?) });
        let c = future_to_promise(async move { Ok(to_value(&c.await)?) });
        let d = future_to_promise(async move { Ok(to_value(&d.await)?) });

        let promises = Array::new();
        promises.push(&a);
        promises.push(&b);
        promises.push(&c);
        promises.push(&d);

        let js_promise = Promise::all(&promises);
        let js_values = JsFuture::from(js_promise).await.map_err(to_string)?;

        let values = js_values.dyn_into::<Array>().map_err(to_string)?;
        let results = (
            from_value(values.get(0)).map_err(to_string)?,
            from_value(values.get(1)).map_err(to_string)?,
            from_value(values.get(2)).map_err(to_string)?,
            from_value(values.get(3)).map_err(to_string)?,
        );

        Ok(results)
    }

    pub fn spawn(f: impl Future<Output = ()> + 'static) {
        // Convert the Rust Future into a JavaScript Promise
        let _ = future_to_promise(async move {
            f.await;
            Ok(JsValue::undefined())
        });
    }

    pub async fn yield_now() {
        TimeoutFuture::new(0).await;
    }

    pub async fn sleep(ms: u32) {
        TimeoutFuture::new(ms).await;
    }

    pub fn time() -> u64 {
        Date::now() as u64
    }
}

pub use utils::*;
