use tig_algorithms::satisfiability::schnoing;
use tig_challenges::satisfiability::*;
use std::panic::catch_unwind;


#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: Challenge) -> Result<Option<Solution>, String>
{
    return catch_unwind(|| {
        schnoing::solve_challenge(&challenge).map_err(|e| e.to_string())
    }).unwrap_or_else(|_| {
        Err("Panic occurred calling solve_challenge".to_string())
    });
}