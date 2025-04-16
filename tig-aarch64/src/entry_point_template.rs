use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{CHALLENGE}::*;
use std::panic::catch_unwind;


#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: Challenge) -> Option<Solution>
{
    return catch_unwind(|| {
        {ALGORITHM}::solve_challenge(&challenge).unwrap_or(None)
    }).unwrap_or(None);
}