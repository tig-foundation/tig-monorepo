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
const T1_NATIVE_CAP:u64=1<<28;
fn t1_native_params(ch:&Challenge)->tekinon::P {
 let mut p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());
 p.max_flips=p.max_flips.min(T1_NATIVE_CAP);p
}
const SMALL_RESTART_TOTAL:u64=1<<27;
fn small_restart_params(ch:&Challenge)->tekinon::P {
 let mut p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());
 p.attempts=16;p.max_flips=1<<25;p
}
fn small_primary(ch:&Challenge)->Outcome {
 let(sol,ok,s)=if (ch.num_variables,ch.clauses.len())==(5000,21335) {
  // Extend the same majority/ProbSAT prefix; successful planted cases exit
  // before any changed giveup boundary. Failed prefixes retain the old rescue.
  let p=t1_native_params(ch);let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());
  let(x,ok,s)=tekinon::run(ch,&p,seed);(Solution{variables:x},ok,s)
 }else if (ch.num_variables,ch.clauses.len())==(7500,32002) {
  let p=small_restart_params(ch);let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());
  // Preserve the entire C7 restart prefix. Only its failed, promising first
  // attempt is then resumed; the verified original rescue remains below.
  let(x,ok,s)=tekinon::run_restarts_resume(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6);(Solution{variables:x},ok,s)
 }else{n3_small::run_probe(ch)};
 Outcome{solution:Some(sol),ok,flips:Some(s.flips),us:0,attempts:s.attempts,sp_iterations:s.sp_it}
}
fn solve_c8_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,_hp:&Option<Map<String,Value>>)->Result<()>{
 dispatch(ch,save,||{
  if ch.num_variables<=10_000 {
   Ok(small_primary(ch))
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
 #[test]fn t3_probe_budget_on_canonical_unsat_instance(){
  let mut ch=c(10000);
  ch.clauses=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
  // Eight sign patterns force one false clause for every assignment. Positive
  // auxiliary clauses start satisfied by majority initialization; all degrees
  // remain small, matching the u8 kernel's supported count range.
  while ch.clauses.len()<42670{let i=ch.clauses.len()-8;ch.clauses.push((0..3).map(|k|4+((i+k)%9997)as i32).collect());}
  let result=small_primary(&ch);assert!(!result.ok);assert_eq!(result.flips,Some(n3_small::FRONT_CAP));assert_eq!(result.attempts,1);
 }
 #[test]fn small_restart_policy_retains_fitted_checkpoints_with_finite_shared_cap(){
  for n in [7500,10000]{let ch=c(n);let p=small_restart_params(&ch);let original=tekinon::P::for_instance(n,ch.clauses.len());assert!(!p.sp);assert_eq!(p.giveup,original.giveup);assert_eq!(p.cb,original.cb);assert_eq!(p.attempts,16);assert_eq!(p.max_flips,33554432);assert_eq!(SMALL_RESTART_TOTAL,134217728);}
 }
 #[test]fn native_extension_has_a_finite_cap_and_preserves_fitted_checkpoints(){
  let ch=c(5000);let p=t1_native_params(&ch);let original=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());
  assert!(!p.sp);assert_eq!(p.attempts,1);assert_eq!(p.max_flips,268435456);assert_eq!(p.giveup,original.giveup);assert_eq!(p.cb,original.cb);
 }
 #[test]fn extended_t1_preserves_successful_probe_assignment_and_rng_prefix(){
  use rand::{Rng,SeedableRng};
  for seed in 0..3u64{
   let mut r=rand::rngs::SmallRng::seed_from_u64(seed);let planted:Vec<bool>=(0..5000).map(|_|r.gen()).collect();
   let mut ch=c(5000);ch.seed[..8].copy_from_slice(&seed.to_le_bytes());ch.clauses.clear();
   for _ in 0..21335{
    let mut cl:Vec<i32>=(0..3).map(|_|{let v=r.gen_range(1..=5000i32);if r.gen::<bool>(){v}else{-v}}).collect();
    if !cl.iter().any(|&l|planted[l.unsigned_abs()as usize-1]==(l>0)){cl[0] = -cl[0];}ch.clauses.push(cl);
   }
   let(old,ok,s)=n3_small::run_probe(&ch);assert!(ok);assert!(s.flips<1<<21);
   let new=small_primary(&ch);assert!(new.ok);assert_eq!(new.solution.unwrap().variables,old.variables);assert_eq!(new.flips,Some(s.flips));assert_eq!(new.attempts,1);
  }
 }
 #[test]fn small_failure_calls_rescue(){let ch=c(3);let saved=Cell::new(0);dispatch(&ch,&|s|{assert!(verified(&ch,s));saved.set(1);Ok(())},||Ok(out(3,false,false)),||Ok(out(3,true,true))).unwrap();assert_eq!(saved.get(),1);}
 #[test]fn small_verified_probe_skips_fallback(){let ch=c(3);dispatch(&ch,&|_|Ok(()),||Ok(out(3,true,true)),||panic!()).unwrap();}
 #[test]fn large_verified_sp_skips_v8(){let ch=c(100_000);dispatch(&ch,&|_|Ok(()),||Ok(out(100_000,true,true)),||panic!()).unwrap();}
 #[test]fn large_invalid_sp_calls_verified_rescue(){let ch=c(100_000);let order=RefCell::new(Vec::new());let saved=Cell::new(0);dispatch(&ch,&|s|{assert!(verified(&ch,s));saved.set(1);Ok(())},||{order.borrow_mut().push(1);Ok(out(100_000,true,false))},||{order.borrow_mut().push(2);Ok(out(100_000,true,true))}).unwrap();assert_eq!(*order.borrow(),vec![1,2]);assert_eq!(saved.get(),1);}
 #[test]fn both_invalid_stages_save_nothing(){let ch=c(100_000);dispatch(&ch,&|_|panic!("invalid solution saved"),||Ok(out(100_000,true,false)),||Ok(out(100_000,true,false))).unwrap();}
}

// C15: the sole selected change is the canonical T3 gate/extension arm.
fn solve_c15_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 if (ch.num_variables,ch.clauses.len())!=(10000,42670){return solve_c8_pipeline(ch,save,hp);}
 let solved=std::cell::Cell::new(false);let accept_solution=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};
 let(sol,ok,_)=n3_small::run_probe(ch);if ok{accept_solution(&sol)?;}if solved.get(){return Ok(());}
 let gates=v8::c15_gated_t3(ch,&accept_solution,true)?;if solved.get(){return Ok(());}
 let seeds=gates.into_iter().filter(|g|!g.near_x.is_empty()).map(|g|(g.near_u,g.near_x)).collect();let _=v8::c15_jump(ch,&accept_solution,seeds)?;Ok(())
}

// C16: bindings selected exclusively by the registered B0+B2 screen.
fn solve_c16_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 let key=(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept_solution=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};
 if key==(5000,21335){let mut p=t1_native_params(ch);p.max_flips=400_000_000;p.giveup=vec![(1<<25,9,0,0),(1<<26,4,0,0),(1<<27,2,0,0),(1<<28,1,0,0)];let(x,ok,st,near)=tekinon::run_near(ch,&p,seed,1);if ok{accept_solution(&Solution{variables:x})?;}if solved.get(){return Ok(());}let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}if let Some(n)=v8::c16_rescue_near(ch,&accept_solution,1)?{seeds.push(n);}if !solved.get()&&st.flips<=268_435_456{v8::c16_jump_one(ch,&accept_solution,seeds,1,5_000_000_000)?;}return Ok(());}
 solve_c15_pipeline(ch,save,hp)
}

// C18: only the preregistered B0+B2-selected dispatch is active.
fn solve_c18_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 let key=(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};
 if key==(7500,32002){let p=small_restart_params(ch);let(x,ok,_,near)=tekinon::run_restarts_resume_near(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6,2);if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}let(near,_)=v8::c18_t2_rescue(ch,&accept,Some((1,18)),false)?;if let Some(n)=near{seeds.push(n);}if !solved.get(){v8::c16_jump_one(ch,&accept,seeds,2,12_000_000_000)?;}return Ok(());}
 if key==(10000,42670){let gates=v8::c15_gated_t3(ch,&accept,true)?;if !solved.get(){let seeds=gates.into_iter().filter(|g|!g.near_x.is_empty()).map(|g|(g.near_u,g.near_x)).collect();v8::c15_jump(ch,&accept,seeds)?;}return Ok(());}
 solve_c16_pipeline(ch,save,hp)
}

#[cfg(test)]mod c18_tests{use super::*;
 fn planted()->Challenge{use rand::{Rng,SeedableRng};let mut r=rand::rngs::SmallRng::seed_from_u64(0);let x:Vec<bool>=(0..10000).map(|_|r.gen()).collect();let clauses=(0..42670).map(|_|{let mut cl:Vec<i32>=(0..3).map(|_|{let v=r.gen_range(1..=10000i32);if r.gen::<bool>(){v}else{-v}}).collect();if !cl.iter().any(|&l|x[l.unsigned_abs()as usize-1]==(l>0)){cl[0] = -cl[0];}cl}).collect();Challenge{seed:[0;32],num_variables:10000,clauses}}
 #[test]fn canonical_planted_dispatch_solves(){let ch=planted();let saved=std::cell::Cell::new(false);solve_challenge(&ch,&|s|{assert!(verified(&ch,s));saved.set(true);Ok(())},&None).unwrap();assert!(saved.get());}}

pub fn c21_large_policy(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,full:bool)->Result<()>{
 dispatch(ch,save,||{let p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let(x,ok,s)=if full{tekinon::run_fast(ch,&p,seed)}else{tekinon::run_spr_simd(ch,&p,seed)};Ok(Outcome{solution:Some(Solution{variables:x}),ok,flips:Some(s.flips),us:0,attempts:s.attempts,sp_iterations:s.sp_it})},||capture(ch,&None,v8::n3_bounded_rescue))
}

// C21: frozen per-track choices; original C18 remains the fallback.
fn solve_c21_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let key=(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};
if key==(5000,21335){let mut p=t1_native_params(ch);p.max_flips=400_000_000;p.giveup=vec![(1<<25,9,0,0),(1<<26,4,0,0),(1<<27,2,0,0),(1<<28,1,0,0)];let(x,ok,st,near)=tekinon::run_near_fast(ch,&p,seed,1);if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}if if true{st.flips==67_108_864}else{(67_108_864..=134_217_728).contains(&st.flips)}{if let Some(n)=v8::c20_t1_rescue(ch,&accept,true)?{seeds.push(n);}}if !solved.get()&&st.flips<=268_435_456{let _=v8::c16_jump_one(ch,&accept,seeds,1,5_000_000_000)?;}return Ok(());}
if key==(7500,32002){let p=small_restart_params(ch);let(x,ok,st,near)=tekinon::run_restarts_resume_near_fast(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6,2);if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}let(near,g)=v8::c20_t2_rescue(ch,&accept,32,false)?;if let Some(n)=near{seeds.push(n);}if !solved.get(){let _=v8::c16_jump_one(ch,&accept,seeds,2,12_000_000_000)?;}return Ok(());}
if key==(100000,415000){return c21_large_policy(ch,save,true);}
if key==(100000,420000){return c21_large_policy(ch,save,true);}
solve_c18_pipeline(ch,save,hp)}

/// Screen-only Cycle 26 policy; the C21 public entry remains unchanged.


fn solve_c26_pipeline(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 let key=(ch.num_variables,ch.clauses.len());let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);
 let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};
if key==(5000,21335){
  let mut p=t1_native_params(ch);p.max_flips=392_000_000;
  p.giveup=vec![(4_194_304,22,0,0),(8_388_608,18,0,0),(16_777_216,10,0,0),(33_554_432,8,0,0),(67_108_864,4,0,0),(100_663_296,2,0,0),(134_217_728,2,0,0),(167_772_160,1,0,0),(268_435_456,1,0,0)];
  let(x,ok,st,near,final_minlen)=tekinon::run_near_fast_minlen(ch,&p,seed,1);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  if st.flips==67_108_864 && final_minlen<=6{if let Some(n)=v8::c20_t1_rescue(ch,&accept,true)?{seeds.push(n);}}
  if !solved.get() && st.flips<=268_435_456{let _=v8::c16_jump_one(ch,&accept,seeds,1,5_000_000_000)?;}return Ok(());
 }
if key==(7500,32002){
  let mut p=small_restart_params(ch);p.attempts=4;p.giveup=vec![(1<<21,usize::MAX,0,0),(1<<23,27,1<<21,47),(1<<25,8,0,0),(1<<26,5,0,0),(1<<27,2,0,0),(201_326_592,1,0,0),(1<<28,2,0,0)];
  let(x,ok,_,near,tail)=tekinon::run_restarts_resume_with_uniform_tail(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6,2,4);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  let(near,_)=v8::c26_t2_rescue(ch,&accept,23,false)?;if let Some(n)=near{seeds.push(n);}
  if !solved.get(){let _=v8::c16_jump_one(ch,&accept,seeds,2,12_000_000_000)?;}if solved.get(){return Ok(());}
  if let Some(tail)=tail{let mut q=p.clone();q.max_flips=(1<<25)+80_000_000;q.giveup=vec![((1<<25)+24_000_000,3,0,0)];let(x,ok,_,_)=tekinon::resume_uniform_tail(tail,&q,2,true);if ok{accept(&Solution{variables:x})?;}}return Ok(());
 }
if key==(10000,42670){let gates=v8::c26_gated_t3(ch,&accept,true)?;if !solved.get(){let seeds=gates.into_iter().filter(|g|!g.near_x.is_empty()).map(|g|(g.near_u,g.near_x)).collect();let _=v8::c15_jump(ch,&accept,seeds)?;}return Ok(());}
if key==(100000,420000){return dispatch(ch,save,||{let p=tekinon::P::for_instance(ch.num_variables,ch.clauses.len());let(x,ok,s)=tekinon::c25_run_fast(ch,&p,seed);Ok(Outcome{solution:Some(Solution{variables:x}),ok,flips:Some(s.flips),us:0,attempts:s.attempts,sp_iterations:s.sp_it})},||capture(ch,&None,v8::n3_bounded_rescue));}
solve_c21_pipeline(ch,save,hp)}

/// Isolated C27 screen dispatcher; the C26 public dispatch remains unchanged.
fn c27_fallback(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>)->Result<()>{
 let out=if ch.clauses.iter().all(|c|c.len()>=3){capture(ch,&None,v8::n3_bounded_rescue)?}else{let mut padded=ch.clone();for c in &mut padded.clauses{let first=c[0];while c.len()<3{c.push(first);}}capture(&padded,&None,v8::n3_bounded_rescue)?};
 let _=accept(ch,save,"v8_2b_fallback",out)?;Ok(())
}



fn solve_routes(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 let key=(ch.num_variables,ch.clauses.len());
if key==(5000,21335){return c31_t1(ch,save,hp);}
if key==(7500,32002){return c31_t2(ch,save,hp);}
if key==(10000,42670){return c32_t3_selected(ch,save,hp);}

if key==(100000,415000){let status=tekinon::c32_solve_large::<3,true>(ch,save)?;return match status{tekinon::C27Status::Malformed|tekinon::C27Status::Solved=>Ok(()),tekinon::C27Status::NeedsFallback|tekinon::C27Status::Unsolved=>c27_fallback(ch,save)};}
if key==(100000,420000){let status=tekinon::c32_solve_large::<2,true>(ch,save)?;return match status{tekinon::C27Status::Malformed|tekinon::C27Status::Solved=>Ok(()),tekinon::C27Status::NeedsFallback|tekinon::C27Status::Unsolved=>c27_fallback(ch,save)};}
solve_c26_pipeline(ch,save,hp)}

fn finish_with_length_fallback<F>(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,run:F)->Result<()>
where F:FnOnce(&dyn Fn(&Solution)->Result<()>)->Result<()>{
 // Upstream tig-verifier rejects empty variables:[] (Solution::new()) with exit 1;
 // saving vec![false; num_variables] when unsolved guarantees quality: 0 (exit 0).
 let saved=std::cell::Cell::new(false);
 let tracked=|s:&Solution|->Result<()>{save(s)?;saved.set(true);Ok(())};
 run(&tracked)?;
 if !saved.get(){save(&Solution{variables:vec![false;ch.num_variables]})?;}
 Ok(())
}

pub fn solve_challenge(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 finish_with_length_fallback(ch,save,|tracked|solve_routes(ch,tracked,hp))
}

#[cfg(test)]mod r142_save_fallback_tests{
 use super::*;use std::cell::RefCell;
 #[test]fn unsolved_route_saves_exact_length_solution(){
  let ch=Challenge{seed:[0;32],num_variables:5000,clauses:vec![vec![1,2,3]]};
  let saved=RefCell::new(Vec::new());
  finish_with_length_fallback(&ch,&|s|{saved.borrow_mut().push(s.variables.clone());Ok(())},|_|Ok(())).unwrap();
  let out=saved.into_inner();assert_eq!(out.len(),1);assert_eq!(out[0].len(),5000);
 }
 #[test]fn solved_route_saves_only_once_without_overwrite(){
  let ch=Challenge{seed:[0;32],num_variables:3,clauses:vec![vec![1,2,3]]};
  let saved=RefCell::new(Vec::new());
  finish_with_length_fallback(&ch,&|s|{saved.borrow_mut().push(s.variables.clone());Ok(())},|cb|cb(&Solution{variables:vec![true,false,true]})).unwrap();
  let out=saved.into_inner();assert_eq!(out,vec![vec![true,false,true]]);
 }
 #[test]fn unsolved_t3_solve_challenge_saves_exact_length_solution(){
  let mut clauses:Vec<Vec<i32>>=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
  while clauses.len()<42670{let i=clauses.len()-8;clauses.push((0..3).map(|k|4+((i+k)%9997)as i32).collect());}
  let ch=Challenge{seed:[0;32],num_variables:10000,clauses};
  let saved=RefCell::new(Vec::new());
  solve_challenge(&ch,&|s|{saved.borrow_mut().push(s.variables.clone());Ok(())},&None).unwrap();
  let out=saved.into_inner();assert_eq!(out.len(),1);assert_eq!(out[0].len(),10000);
 }
}

fn c29_t1_checkpoints()->Vec<(u64,usize,u64,usize)>{vec![(4_194_304,21,0,0),(8_388_608,18,0,0),(12_582_912,11,0,0),(16_777_216,10,0,0),(25_165_824,8,0,0),(33_554_432,7,0,0),(67_108_864,4,0,0),(100_663_296,2,0,0),(134_217_728,2,0,0),(167_772_160,1,0,0),(268_435_456,1,0,0)]}

#[cfg(test)]mod c29_selected_t1_test{use super::*;#[test]fn checkpoint_order_is_strict(){let p=c29_t1_checkpoints();assert!(p.windows(2).all(|w|w[0].0<w[1].0));}}

fn c29_t1_selected(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{
  let mut p=t1_native_params(ch);p.max_flips=392_000_000;
  p.giveup=c29_t1_checkpoints();
  let(x,ok,st,near,final_minlen)=tekinon::run_near_fast_minlen(ch,&p,seed,1);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  if st.flips==67_108_864 && final_minlen<=6{if let Some(n)=v8::c20_t1_rescue(ch,&accept,true)?{seeds.push(n);}}
  if !solved.get() && st.flips<=268_435_456{let _=v8::c16_jump_one(ch,&accept,seeds,1,5_000_000_000)?;}return Ok(());
 }}

fn c29_t2_checkpoints()->Vec<(u64,usize,u64,usize)>{vec![(2_097_152,68,0,0),(4_194_304,43,0,0),(8_388_608,27,2_097_152,47),(12_582_912,25,0,0),(16_777_216,17,0,0),(25_165_824,16,0,0),(29_360_128,13,0,0),(33_554_432,8,0,0),(67_108_864,5,0,0),(100_663_296,5,0,0),(134_217_728,2,0,0),(201_326_592,1,0,0),(268_435_456,2,0,0)]}

#[cfg(test)]mod c29_selected_t2_test{use super::*;#[test]fn two_point_checkpoint_slot_does_not_collide(){let p=c29_t2_checkpoints();assert!(p.windows(2).all(|w|w[0].0<w[1].0));assert!(p.iter().filter(|x|x.0>2_097_152&&x.0<8_388_608).all(|x|x.0.trailing_zeros()!=21));assert!(p.contains(&(8_388_608,27,2_097_152,47)));}}

fn c29_t2_selected(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{
  let mut p=small_restart_params(ch);p.attempts=4;p.giveup=c29_t2_checkpoints();
  let(x,ok,_,near,tail)=tekinon::run_restarts_resume_with_uniform_tail(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6,2,4);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  let(near,_)=v8::c26_t2_rescue(ch,&accept,23,false)?;if let Some(n)=near{seeds.push(n);}
  if !solved.get(){let _=v8::c16_jump_one(ch,&accept,seeds,2,12_000_000_000)?;}if solved.get(){return Ok(());}
  if let Some(tail)=tail{let mut q=p.clone();q.max_flips=(1<<25)+80_000_000;q.giveup=vec![((1<<25)+24_000_000,3,0,0)];let(x,ok,_,_)=tekinon::resume_uniform_tail(tail,&q,2,true);if ok{accept(&Solution{variables:x})?;}}return Ok(());
 }}

fn c29_t3_selected(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{let gates=v8::c29_gated_t3(ch,&accept,true)?;if !solved.get(){let seeds=gates.into_iter().filter(|g|!g.near_x.is_empty()).map(|g|(g.near_u,g.near_x)).collect();let _=v8::c16_jump_one(ch,&accept,seeds,3,12_000_000_000)?;}return Ok(());}}

fn c31_t1_checkpoints()->Vec<(u64,usize,u64,usize)>{vec![(4_194_304,21,0,0),(8_388_608,18,0,0),(12_582_912,11,0,0),(16_777_216,10,0,0),(25_165_824,8,0,0),(33_554_432,7,0,0),(50_331_648,4,0,0),(67_108_864,3,0,0),(83_886_080,3,0,0),(100_663_296,2,0,0),(134_217_728,2,0,0),(155_189_248,1,0,0),(167_772_160,1,0,0),(268_435_456,1,0,0)]}
fn c31_t1(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{
  let mut p=t1_native_params(ch);p.max_flips=392_000_000;
  p.giveup=c31_t1_checkpoints();
  let(x,ok,st,near,final_minlen)=tekinon::c31_run_near_fast_minlen::<false>(ch,&p,seed,1);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  if ((st.flips==50_331_648 && final_minlen==6) || (st.flips==67_108_864 && final_minlen<=6)) && final_minlen<=6{if let Some(n)=v8::c20_t1_rescue(ch,&accept,true)?{seeds.push(n);}}
  if !solved.get() && st.flips<=268_435_456{let _=v8::c16_jump_one(ch,&accept,seeds,1,5_000_000_000)?;}return Ok(());
 }}

#[cfg(test)]mod c32_selected_t1_tests{use super::*;#[test]fn prefix_and_new_gate_are_preserved(){let p=c31_t1_checkpoints();assert_eq!(&p[..6],&c29_t1_checkpoints()[..6]);assert!(p.windows(2).all(|w|w[0].0<w[1].0));assert!(p.contains(&(155_189_248,1,0,0)));}}

fn c31_t2_checkpoints()->Vec<(u64,usize,u64,usize)>{vec![(2_097_152,68,0,0),(4_194_304,43,0,0),(8_388_608,27,2_097_152,47),(12_582_912,25,0,0),(16_777_216,17,0,0),(20_971_520,15,0,0),(25_165_824,15,0,0),(29_360_128,11,0,0),(33_554_432,8,0,0),(67_108_864,5,0,0),(100_663_296,5,0,0),(109_051_904,3,0,0),(134_217_728,2,0,0),(201_326_592,1,0,0),(268_435_456,2,0,0)]}
fn c31_t2(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{
  let mut p=small_restart_params(ch);p.attempts=4;p.giveup=c31_t2_checkpoints();
  let(x,ok,_,near,tail,best_minlen)=tekinon::c31_run_restarts_resume_with_uniform_tail_minlen::<false>(ch,&p,seed,SMALL_RESTART_TOTAL,1<<25,1<<28,6,2,4);
  if ok{accept(&Solution{variables:x})?;}if solved.get(){return Ok(());}
  let mut seeds=Vec::new();if !near.x.is_empty(){seeds.push((near.u,near.x.iter().map(|&v|v==1).collect()));}
  if best_minlen<=4{let(near,_)=v8::c26_t2_rescue(ch,&accept,23,false)?;if let Some(n)=near{seeds.push(n);}}
  if !solved.get(){let _=v8::c16_jump_one(ch,&accept,seeds,2,12_000_000_000)?;}if solved.get(){return Ok(());}
  if let Some(tail)=tail{let mut q=p.clone();q.max_flips=(1<<25)+80_000_000;q.giveup=vec![((1<<25)+24_000_000,3,0,0)];let(x,ok,_,_)=tekinon::c31_resume_uniform_tail::<false>(tail,&q,2,true);if ok{accept(&Solution{variables:x})?;}}return Ok(());
 }}

#[cfg(test)]mod c32_selected_t2_tests{use super::*;#[test]fn prefix_and_new_gate_are_preserved(){let p=c31_t2_checkpoints();assert_eq!(&p[..5],&c29_t2_checkpoints()[..5]);assert!(p.windows(2).all(|w|w[0].0<w[1].0));assert!(p.contains(&(109_051_904,3,0,0)));}}

fn c32_t3_selected(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if verified(ch,s){save(s)?;solved.set(true);}Ok(())};{let gates=v8::c32_gated_t3(ch,&accept,true)?;if !solved.get(){let seeds=gates.into_iter().filter(|g|!g.near_x.is_empty()).map(|g|(g.near_u,g.near_x)).collect();let _=v8::c16_jump_one(ch,&accept,seeds,3,12_000_000_000)?;}return Ok(());}}
