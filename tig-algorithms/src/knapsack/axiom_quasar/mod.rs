#![allow(dead_code,unused_imports,unused_variables,unused_mut,non_snake_case,clippy::all,dangerous_implicit_autorefs)]
mod axiom;
mod donor;
use anyhow::Result;
use serde_json::{Map,Value};
use tig_challenges::knapsack::{Challenge,Solution};
pub fn solve_challenge(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()>{
 let mut params=hp.clone().unwrap_or_default();
 for (key,value) in [("lookahead_relink",1u64),("ils_rounds_t42",0),("t42_elites",4),("t43_elites",3)] {
  params.entry(key.to_owned()).or_insert(Value::from(value));
 }
 let stored=std::cell::RefCell::new(None);
 axiom::solve_challenge(ch,&|s|{*stored.borrow_mut()=Some(s.clone());save(s)},&Some(params))?;
 let initial=stored.into_inner().ok_or_else(||anyhow::anyhow!("base solver saved no solution"))?;
 if ch.num_items==5000 {
  let sum_w:u64=ch.weights.iter().map(|&w|w as u64).sum();
  let budget=if sum_w>0{(ch.max_weight as u64*100/sum_w) as u32}else{10};
  let cand=if budget<=17 {
   donor::block_search::improve(ch,&initial)
  }else{
   let(q,_,min_q)=axiom::build_q_flat(ch);
   donor::block_search::improve_with(ch,&initial,|c,s|{
    let mut st=axiom::FlatState::from_items(c,&q,&s.items);
    axiom::polish_stage_b2_hybrid(&mut st,c,&q,min_q,48,38,18,30);
    st.to_solution()
   })
  };
  if cand.items != initial.items && donor::aim::feasible(ch,&cand.items) && donor::aim::value(ch,&cand.items)>donor::aim::value(ch,&initial.items){save(&cand)?;}
 }
 Ok(())
}
pub fn help(){println!("axiom_quasar - elite relinking with monotonic five-candidate block search");}
