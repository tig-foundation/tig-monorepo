use anyhow::Result;
use serde_json::{Map,Value};
use tig_challenges::satisfiability::{Challenge,Solution};
use super::{tekinon,v8,n3_small};
pub use super::verified;
pub struct Outcome {pub solution:Option<Solution>,pub ok:bool,pub flips:Option<u64>,pub us:u128,pub attempts:usize,pub sp_iterations:usize}
fn accept(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,_name:&str,out:Outcome)->Result<bool>{
 let valid=out.ok && out.solution.as_ref().map_or(false,|s|verified(ch,s));
 if valid {save(out.solution.as_ref().unwrap())?;}Ok(valid)
}
fn dispatch<P,V>(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,primary:P,rescue:V)->Result<()>
where P:FnOnce()->Result<Outcome>,V:FnOnce()->Result<Outcome>{
 if ch.num_variables<=10_000 {
  if accept(ch,save,"bitwise_probe",primary()?)?{return Ok(());}
  let _=accept(ch,save,"v8_6b_rescue",rescue()?)?;
 }else{
  if accept(ch,save,"sp_primary",primary()?)?{return Ok(());}
  let _=accept(ch,save,"v8_2b_fallback",rescue()?)?;
 }Ok(())
}
type Solve=fn(&Challenge,&dyn Fn(&Solution)->Result<()>,&Option<Map<String,Value>>)->Result<()>;
fn capture(ch:&Challenge,hp:&Option<Map<String,Value>>,solve:Solve)->Result<Outcome>{
 let saved=std::cell::RefCell::new(None);let cb=|s:&Solution|->Result<()>{*saved.borrow_mut()=Some(s.clone());Ok(())};
 solve(ch,&cb,hp)?;
 let solution=saved.into_inner();let ok=solution.as_ref().map_or(false,|s|verified(ch,s));
 Ok(Outcome{solution,ok,flips:None,us:0,attempts:0,sp_iterations:0})
}
pub fn solve_challenge(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,_hp:&Option<Map<String,Value>>)->Result<()>{
 dispatch(ch,save,||{
  if ch.num_variables<=10_000 {
   let(sol,ok,s)=n3_small::run_probe(ch);
   Ok(Outcome{solution:Some(sol),ok,flips:Some(s.flips),us:0,attempts:s.attempts,sp_iterations:s.sp_it})
  }else{
   let p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());
   let(x,ok,s)=tekinon::run(ch,&p,seed);
   Ok(Outcome{solution:Some(Solution{variables:x}),ok,flips:Some(s.flips),us:0,attempts:s.attempts,sp_iterations:s.sp_it})
  }
 },||capture(ch,&None,v8::n3_bounded_rescue))
}
#[cfg(test)]mod tests{
 use super::*;use std::cell::{Cell,RefCell};
 fn c(n:usize)->Challenge{Challenge{seed:[0;32],num_variables:n,clauses:vec![vec![1,2,3]]}}
 fn out(n:usize,ok:bool,valid:bool)->Outcome{Outcome{solution:Some(Solution{variables:vec![valid;n]}),ok,flips:Some(0),us:0,attempts:1,sp_iterations:0}}
 #[test]fn small_failure_calls_rescue(){let ch=c(3);let saved=Cell::new(0);dispatch(&ch,&|s|{assert!(verified(&ch,s));saved.set(1);Ok(())},||Ok(out(3,false,false)),||Ok(out(3,true,true))).unwrap();assert_eq!(saved.get(),1);}
 #[test]fn small_verified_probe_skips_fallback(){let ch=c(3);dispatch(&ch,&|_|Ok(()),||Ok(out(3,true,true)),||panic!()).unwrap();}
 #[test]fn large_verified_sp_skips_v8(){let ch=c(100_000);dispatch(&ch,&|_|Ok(()),||Ok(out(100_000,true,true)),||panic!()).unwrap();}
 #[test]fn large_invalid_sp_calls_verified_rescue(){let ch=c(100_000);let order=RefCell::new(Vec::new());let saved=Cell::new(0);dispatch(&ch,&|s|{assert!(verified(&ch,s));saved.set(1);Ok(())},||{order.borrow_mut().push(1);Ok(out(100_000,true,false))},||{order.borrow_mut().push(2);Ok(out(100_000,true,true))}).unwrap();assert_eq!(*order.borrow(),vec![1,2]);assert_eq!(saved.get(),1);}
 #[test]fn both_invalid_stages_save_nothing(){let ch=c(100_000);dispatch(&ch,&|_|panic!("invalid solution saved"),||Ok(out(100_000,true,false)),||Ok(out(100_000,true,false))).unwrap();}
}
