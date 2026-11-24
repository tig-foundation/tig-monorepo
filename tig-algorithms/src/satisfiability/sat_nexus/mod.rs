#![allow(dead_code,unused_variables,unused_mut,unused_imports,non_snake_case,clippy::all,dangerous_implicit_autorefs)]
mod tekinon;
mod n3_small;
mod n3;
mod v8;
use tig_challenges::satisfiability::{Challenge,Solution};
pub use n3::solve_challenge;
pub fn verified(ch:&Challenge,s:&Solution)->bool {
    s.variables.len()==ch.num_variables && ch.clauses.iter().all(|cl|cl.iter().any(|&lit| {
        let v=lit.unsigned_abs() as usize;
        v>0 && v<=ch.num_variables && s.variables[v-1]==(lit>0)
    }))
}
pub fn help(){println!("sat_nexus - bounded bitwise probSAT and reinforced SP with verified rescue");}
