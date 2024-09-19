use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    benchmark_id: &String,
    merkle_root: MerkleHash,
    solution_nonces: HashSet<u64>,
) -> ProtocolResult<()> {
    let precommit = get_precommit_by_id(ctx, benchmark_id).await?;
    verify_benchmark_ownership(player, &precommit.settings)?;
    verify_nonces(&precommit, &solution_nonces)?;
    ctx.add_benchmark_to_mempool(
        benchmark_id,
        BenchmarkDetails {
            num_solutions: solution_nonces.len() as u32,
            merkle_root: Some(merkle_root),
        },
        solution_nonces,
    )
    .await
    .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e));
    Ok(())
}

#[time]
async fn get_precommit_by_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Precommit> {
    ctx.get_precommits(PrecommitsFilter::BenchmarkId(benchmark_id.clone()))
        .await
        .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
        .pop()
        .filter(|p| p.state.is_some())
        .ok_or_else(|| ProtocolError::InvalidPrecommit {
            benchmark_id: benchmark_id.clone(),
        })
}

#[time]
fn verify_benchmark_ownership(player: &Player, settings: &BenchmarkSettings) -> ProtocolResult<()> {
    if player.id != settings.player_id {
        return Err(ProtocolError::InvalidSubmittingPlayer {
            actual_player_id: player.id.clone(),
            expected_player_id: settings.player_id.clone(),
        });
    }
    Ok(())
}

#[time]
fn verify_nonces(precommit: &Precommit, solution_nonces: &HashSet<u64>) -> ProtocolResult<()> {
    for n in solution_nonces.iter() {
        if *n >= precommit.details.num_nonces {
            return Err(ProtocolError::InvalidBenchmarkNonce { nonce: *n });
        }
    }
    Ok(())
}
