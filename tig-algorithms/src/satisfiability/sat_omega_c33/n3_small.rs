use tig_challenges::satisfiability::{Challenge,Solution};
use super::tekinon;
pub const FRONT_CAP:u64=1<<22;
pub fn params(ch:&Challenge)->tekinon::P {
    let mut p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());
    p.sp=false;p.attempts=1;p.max_flips=FRONT_CAP;
    p.giveup=vec![(1<<21,8,0,0),(FRONT_CAP,2,0,0)];p
}
pub fn run_probe(ch:&Challenge)->(Solution,bool,tekinon::Stats){
    assert!(ch.num_variables<=10_000,"small-track probe only");
    let p=params(ch);let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());
    let(x,ok,stats)=tekinon::run(ch,&p,seed);
    assert!(stats.flips<=FRONT_CAP);
    (Solution{variables:x},ok,stats)
}
#[cfg(test)]mod tests{
    use super::*;
    fn c(n:usize)->Challenge{Challenge{seed:[0;32],num_variables:n,clauses:vec![vec![1,2,3]]}}
    #[test]fn exact_small_probe_contract(){for n in [5000,7500,10000]{let p=params(&c(n));assert!(!p.sp);assert_eq!(p.attempts,1);assert_eq!(p.max_flips,4194304);assert_eq!(p.giveup,vec![(2097152,8,0,0),(4194304,2,0,0)]);}}
    #[test]fn real_unsat_probe_stops_at_4194304_flips(){let mut ch=c(3);ch.clauses=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();let(_,ok,s)=run_probe(&ch);assert!(!ok);assert_eq!(s.flips,4194304);}
    #[test]fn verified_easy_sat_probe_returns_early(){let ch=c(3);let(sol,ok,s)=run_probe(&ch);assert!(ok);assert_eq!(sol.variables.len(),3);assert!(ch.clauses.iter().all(|cl|cl.iter().any(|&l|sol.variables[l.unsigned_abs() as usize-1]==(l>0))));assert!(s.flips<FRONT_CAP);}
}
