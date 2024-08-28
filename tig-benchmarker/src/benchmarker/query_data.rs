use super::{api, player_id, QueryData, Result};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use tig_api::*;
use tig_structs::core::*;
use tokio::{join, sync::Mutex};

static CACHE: OnceCell<Mutex<HashMap<String, QueryData>>> = OnceCell::new();

pub async fn execute() -> Result<QueryData> {
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let latest_block = query_latest_block().await?;
    let latest_block_id = latest_block.id.clone();
    let mut cache = cache.lock().await;
    if !cache.contains_key(&latest_block.id) {
        cache.clear();
        let results = join!(
            query_algorithms(latest_block.id.clone()),
            query_player_data(latest_block.id.clone()),
            query_benchmarks(latest_block.id.clone()),
            query_challenges(latest_block.id.clone()),
        );
        let (algorithms_by_challenge, download_urls) = results.0?;
        let player_data = results.1?;
        let (benchmarks, proofs, frauds) = results.2?;
        let challenges = results
            .3?
            .into_iter()
            .filter(|c| {
                c.state()
                    .round_active
                    .as_ref()
                    .is_some_and(|r| *r <= latest_block.details.round)
            })
            .collect();
        cache.insert(
            latest_block.id.clone(),
            QueryData {
                latest_block,
                algorithms_by_challenge,
                player_data,
                download_urls,
                benchmarks,
                proofs,
                frauds,
                challenges,
            },
        );
    }
    Ok(cache.get(&latest_block_id).unwrap().clone())
}

async fn query_latest_block() -> Result<Block> {
    let GetBlockResp { block, .. } = api()
        .get_block(GetBlockReq {
            id: None,
            round: None,
            height: None,
            include_data: false,
        })
        .await
        .map_err(|e| format!("Failed to query latest block: {:?}", e))?;
    Ok(block.ok_or_else(|| format!("Expecting latest block to exist"))?)
}

async fn query_benchmarks(
    block_id: String,
) -> Result<(
    HashMap<String, Benchmark>,
    HashMap<String, Proof>,
    HashMap<String, Fraud>,
)> {
    let GetBenchmarksResp {
        benchmarks,
        proofs,
        frauds,
        ..
    } = api()
        .get_benchmarks(GetBenchmarksReq {
            block_id: block_id.clone(),
            player_id: player_id().clone(),
        })
        .await
        .map_err(|e| format!("Failed to get benchmarks: {:?}", e))?;
    Ok((
        benchmarks.into_iter().map(|x| (x.id.clone(), x)).collect(),
        proofs
            .into_iter()
            .map(|x| (x.benchmark_id.clone(), x))
            .collect(),
        frauds
            .into_iter()
            .map(|x| (x.benchmark_id.clone(), x))
            .collect(),
    ))
}

async fn query_player_data(block_id: String) -> Result<Option<PlayerBlockData>> {
    let GetPlayersResp { players, .. } = api()
        .get_players(GetPlayersReq {
            block_id: block_id.clone(),
            player_type: PlayerType::Benchmarker,
        })
        .await
        .map_err(|e| format!("Failed to query players: {:?}", e))?;
    let player_id = player_id().clone();
    match players.into_iter().find(|x| x.id == player_id) {
        Some(player) => {
            Ok(Some(player.block_data.ok_or_else(|| {
                format!("Expecting player to have block_data")
            })?))
        }
        None => Ok(None),
    }
}

async fn query_challenges(block_id: String) -> Result<Vec<Challenge>> {
    let GetChallengesResp { challenges, .. } = api()
        .get_challenges(GetChallengesReq {
            block_id: block_id.clone(),
        })
        .await
        .map_err(|e| format!("Failed to query challenges: {:?}", e))?;
    Ok(challenges)
}

async fn query_algorithms(
    block_id: String,
) -> Result<(HashMap<String, Vec<Algorithm>>, HashMap<String, String>)> {
    let GetAlgorithmsResp {
        algorithms, wasms, ..
    } = api()
        .get_algorithms(GetAlgorithmsReq {
            block_id: block_id.clone(),
        })
        .await
        .map_err(|e| format!("Failed to query algorithms: {:?}", e))?;
    let algorithms_by_challenge: HashMap<String, Vec<Algorithm>> =
        algorithms.into_iter().fold(HashMap::new(), |mut acc, x| {
            acc.entry(x.details.challenge_id.clone())
                .or_default()
                .push(x.clone());
            acc
        });
    let download_urls = wasms
        .into_iter()
        .filter(|x| x.details.download_url.is_some())
        .map(|x| (x.algorithm_id, x.details.download_url.unwrap()))
        .collect();
    Ok((algorithms_by_challenge, download_urls))
}
