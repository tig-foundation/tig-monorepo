#[cfg(not(any(feature = "request", feature = "request-js")))]
compile_error!("Either feature `request` or `request-js` must be enabled");
#[cfg(all(feature = "request", feature = "request-js"))]
compile_error!("features `request` and `request-js` are mutually exclusive");

use anyhow::{anyhow, Result};
use query_map::QueryMap;
use serde::de::DeserializeOwned;
use std::{collections::HashMap, vec};
pub use tig_structs::api::*;
use tig_utils::{dejsonify, get, jsonify, post};

pub struct Api {
    api_url: String,
    api_key: String,
}

impl Api {
    pub fn new(api_url: String, api_key: String) -> Self {
        Self { api_url, api_key }
    }

    async fn get<T>(&self, path: String) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let resp = get::<String>(
            format!("{}/{}", self.api_url, path).as_str(),
            Some(
                vec![
                    ("x-api-key".to_string(), self.api_key.clone()),
                    ("user-agent".to_string(), "TIG API".to_string()),
                ]
                .into_iter()
                .collect(),
            ),
        )
        .await?;
        dejsonify::<T>(&resp).map_err(|e| anyhow!("Failed to dejsonify: {}", e))
    }
    async fn post<T>(&self, path: String, body: String) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let resp = post::<String>(
            format!("{}/{}", self.api_url, path).as_str(),
            body.as_str(),
            Some(
                vec![
                    ("x-api-key".to_string(), self.api_key.clone()),
                    ("user-agent".to_string(), "TIG API".to_string()),
                ]
                .into_iter()
                .collect(),
            ),
        )
        .await?;
        dejsonify::<T>(&resp).map_err(|e| anyhow!("Failed to dejsonify: {}", e))
    }

    pub async fn get_challenges(&self, req: GetChallengesReq) -> Result<GetChallengesResp> {
        let mut query = HashMap::<String, String>::new();
        if let Some(block_id) = req.block_id {
            query.insert("block_id".to_string(), block_id);
        }
        let query = QueryMap::from(query);
        self.get(format!("get-challenges?{}", query.to_query_string()))
            .await
    }

    pub async fn get_algorithms(&self, req: GetAlgorithmsReq) -> Result<GetAlgorithmsResp> {
        let mut query = HashMap::<String, String>::new();
        if let Some(block_id) = req.block_id {
            query.insert("block_id".to_string(), block_id);
        }
        let query = QueryMap::from(query);
        self.get(format!("get-algorithms?{}", query.to_query_string()))
            .await
    }

    pub async fn get_players(&self, req: GetPlayersReq) -> Result<GetPlayersResp> {
        let mut query = HashMap::<String, String>::new();
        if let Some(block_id) = req.block_id {
            query.insert("block_id".to_string(), block_id);
        }
        query.insert("player_type".to_string(), req.player_type.to_string());
        let query = QueryMap::from(query);
        self.get(format!("get-players?{}", query.to_query_string()))
            .await
    }

    pub async fn get_benchmarks(&self, req: GetBenchmarksReq) -> Result<GetBenchmarksResp> {
        let mut query = HashMap::<String, String>::new();
        if let Some(block_id) = req.block_id {
            query.insert("block_id".to_string(), block_id);
        }
        query.insert("player_id".to_string(), req.player_id);
        let query = QueryMap::from(query);
        self.get(format!("get-benchmarks?{}", query.to_query_string()))
            .await
    }

    pub async fn get_benchmark_data(
        &self,
        req: GetBenchmarkDataReq,
    ) -> Result<GetBenchmarkDataResp> {
        let mut query = HashMap::<String, String>::new();
        query.insert("benchmark_id".to_string(), req.benchmark_id);
        let query = QueryMap::from(query);
        self.get(format!("get-benchmark-data?{}", query.to_query_string()))
            .await
    }

    pub async fn get_block(&self, req: GetBlockReq) -> Result<GetBlockResp> {
        let mut query = HashMap::<String, String>::new();
        if let Some(id) = req.id {
            query.insert("id".to_string(), id);
        }
        if let Some(height) = req.height {
            query.insert("height".to_string(), height.to_string());
        }
        if let Some(round) = req.round {
            query.insert("round".to_string(), round.to_string());
        }
        let query = QueryMap::from(query);
        self.get(format!("get-block?{}", query.to_query_string()))
            .await
    }

    pub async fn submit_algorithm(&self, req: SubmitAlgorithmReq) -> Result<SubmitAlgorithmResp> {
        self.post("submit-algorithm".to_string(), jsonify(&req))
            .await
    }

    pub async fn submit_benchmark(&self, req: SubmitBenchmarkReq) -> Result<SubmitBenchmarkResp> {
        self.post("submit-benchmark".to_string(), jsonify(&req))
            .await
    }

    pub async fn submit_proof(&self, req: SubmitProofReq) -> Result<SubmitProofResp> {
        self.post("submit-proof".to_string(), jsonify(&req)).await
    }
}
