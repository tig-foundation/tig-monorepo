#[cfg(all(feature = "request", feature = "request-js"))]
compile_error!("features `request` and `request-js` are mutually exclusive");

use anyhow::{anyhow, Result};

#[cfg(feature = "request-js")]
mod request {
    use super::*;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Headers, Request, RequestInit, RequestMode, Response};

    #[allow(async_fn_in_trait)]
    pub trait FromResponse: Sized {
        async fn from_response(response: Response) -> Result<Self>;
    }

    async fn check_status(response: Response) -> Result<Response> {
        let status = response.status();
        if !(200..=299).contains(&status) {
            let msg = match response.text() {
                Ok(promise) => match JsFuture::from(promise).await {
                    Ok(value) => value.as_string().unwrap_or("".to_string()),
                    Err(_) => "".to_string(),
                },
                Err(_) => "".to_string(),
            };
            return Err(anyhow!("Request error (status: {}, body: {})", status, msg));
        }
        Ok(response)
    }

    impl FromResponse for Vec<u8> {
        async fn from_response(response: Response) -> Result<Self> {
            let promise = check_status(response).await?.array_buffer().unwrap();
            let future = JsFuture::from(promise);
            let buffer = future
                .await
                .map_err(|_| anyhow!("Failed to read response body as array buffer"))?;
            let uint8_array = js_sys::Uint8Array::new(&buffer);
            Ok(uint8_array.to_vec())
        }
    }

    impl FromResponse for String {
        async fn from_response(response: Response) -> Result<Self> {
            let promise = check_status(response)
                .await?
                .text()
                .map_err(|_| anyhow!("Failed to read response body as text"))?;
            JsFuture::from(promise)
                .await
                .map_err(|_| anyhow!("Failed to read response body as text"))?
                .as_string()
                .ok_or_else(|| anyhow!("Failed to convert JsValue to String"))
        }
    }

    async fn call<T>(
        method: &str,
        url: &str,
        body: Option<&JsValue>,
        headers: Option<Headers>,
    ) -> Result<T>
    where
        T: FromResponse,
    {
        let mut opts = RequestInit::new();
        opts.method(method);
        opts.mode(RequestMode::Cors);

        if let Some(b) = body {
            opts.body(Some(b));
        }

        if let Some(h) = headers {
            opts.headers(&h);
        }

        let request = Request::new_with_str_and_init(url, &opts)
            .map_err(|_| anyhow!("Failed to create request"))?;

        let window = web_sys::window().ok_or_else(|| anyhow!("No global `window` exists"))?;
        let response_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|_| anyhow!("Failed to fetch"))?;

        let response: Response = response_value
            .dyn_into()
            .map_err(|_| anyhow!("Failed to cast to Response"))?;

        T::from_response(response).await
    }

    pub async fn get<T>(url: &str, headers: Option<Vec<(String, String)>>) -> Result<T>
    where
        T: FromResponse,
    {
        let headers = convert_headers(headers)?;
        call::<T>("GET", url, None, headers).await
    }

    pub async fn post<T>(url: &str, body: &str, headers: Option<Vec<(String, String)>>) -> Result<T>
    where
        T: FromResponse,
    {
        let headers = convert_headers(headers)?;
        let body_value = Some(JsValue::from_str(body));
        call::<T>("POST", url, body_value.as_ref(), headers).await
    }

    fn convert_headers(headers_option: Option<Vec<(String, String)>>) -> Result<Option<Headers>> {
        headers_option
            .map(|headers_map| {
                let headers = Headers::new().map_err(|_| anyhow!("Failed to create Headers"))?;
                for (key, value) in headers_map {
                    headers
                        .set(&key, &value)
                        .map_err(|_| anyhow!("Failed to set header"))?;
                }
                Ok(headers)
            })
            .transpose()
    }
}

#[cfg(feature = "request")]
mod request {
    use super::*;
    use reqwest::{
        header::{HeaderMap, HeaderName, HeaderValue},
        Response,
    };

    #[allow(async_fn_in_trait)]
    pub trait FromResponse: Sized {
        async fn from_response(response: Response) -> Result<Self>;
    }

    async fn check_status(response: Response) -> Result<Response> {
        let status = response.status().as_u16();
        if !(200..=299).contains(&status) {
            let msg = match response.text().await {
                Ok(msg) => msg.clone(),
                Err(_) => "".to_string(),
            };
            return Err(anyhow!("Request error (status: {}, body: {})", status, msg));
        }
        Ok(response)
    }

    impl FromResponse for Vec<u8> {
        async fn from_response(response: Response) -> Result<Self> {
            Ok(check_status(response).await?.bytes().await?.to_vec())
        }
    }

    impl FromResponse for String {
        async fn from_response(response: Response) -> Result<Self> {
            Ok(check_status(response).await?.text().await?)
        }
    }

    async fn call<T: FromResponse>(
        method: &str,
        url: &str,
        body: Option<String>,
        headers: Option<HeaderMap>,
    ) -> Result<T> {
        let client = reqwest::Client::new();
        let mut request_builder = client.request(method.parse().unwrap(), url);

        if let Some(b) = body {
            request_builder = request_builder.body(b);
        }

        if let Some(h) = headers {
            request_builder = request_builder.headers(h);
        }

        let response = request_builder.send().await?;
        T::from_response(response).await
    }

    pub async fn get<T: FromResponse>(
        url: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Result<T> {
        let headers = convert_headers(headers)?;
        call::<T>("GET", url, None, headers).await
    }

    pub async fn post<T: FromResponse>(
        url: &str,
        body: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Result<T> {
        let headers = convert_headers(headers)?;
        let body_value = Some(body.to_string());
        call::<T>("POST", url, body_value, headers).await
    }

    fn convert_headers(headers_option: Option<Vec<(String, String)>>) -> Result<Option<HeaderMap>> {
        headers_option
            .map(|headers_map| {
                let mut headers = HeaderMap::new();
                for (key, value) in headers_map {
                    let header_name = HeaderName::from_bytes(key.as_bytes())
                        .map_err(|_| anyhow!("Invalid header name"))?;
                    let header_value = HeaderValue::from_str(&value)
                        .map_err(|_| anyhow!("Invalid header value"))?;
                    headers.insert(header_name, header_value);
                }
                Ok(headers)
            })
            .transpose()
    }
}

pub use request::*;
