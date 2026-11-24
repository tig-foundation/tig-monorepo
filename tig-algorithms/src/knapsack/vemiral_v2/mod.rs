// packrun - quadratic knapsack solver. Dispatch by (num_items, budget fraction) to a
// per-track engine: iterated local search + DP refinement, population search and an
// exact branch-and-bound core on hard tracks. Randomisation seeded only from the public
// instance seed. Tuning read in from_map; defaults ship.

mod aim;
mod block_search;
mod flow_bound;
mod core_search;
pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Clone, Debug, Default)]
pub struct Hparams {}

impl Hparams {
    pub fn for_track1() -> Self { Self::default() }
    pub fn for_track2() -> Self { Self::default() }
    pub fn for_track4() -> Self { Self::default() }
    pub fn for_track3() -> Self { Self::default() }
    pub fn for_track5() -> Self { Self::default() }
}

pub fn help() {
    println!("Send {{\"aim_target\": T}} on every track, T = the track's qualifying quality as a fraction (164722 on chain = 0.164722)");
    println!("{{\"aim\": 0}} = packrun");
}

fn compute_budget_pct(challenge: &Challenge) -> u32 {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save:      &dyn Fn(&Solution) -> Result<()>,
    hp:        &Option<Map<String, Value>>,
) -> Result<()> {
    if challenge.num_items==5000 && compute_budget_pct(challenge)<=17{
        let first=std::cell::RefCell::new(None);
        dispatch(challenge,&|s|{*first.borrow_mut()=Some(s.clone());Ok(())},hp)?;
        if let Some(mut best)=first.into_inner(){
            let value=aim::value(challenge,&best.items);
            if aim::near_target(challenge,hp,value,0.005){best=block_search::improve(challenge,&best);}
            let _=save(&best);
        }
        return Ok(());
    }
    if challenge.num_items != 1000 {
        return dispatch(challenge, save, hp);
    }
    let mut flow = None;
    if let Some((lower, above)) = aim::window25(challenge, hp) {
        if let Some((seed, bound, breakpoints)) = track1::flow_seed(challenge) {
            if aim::value(challenge, &seed.items) >= above || bound < lower {
                return save(&seed);
            }
            track3::note_flow_breakpoints(challenge, track1::FLOW_LAMBDAS, breakpoints, seed.items.clone());
            flow = Some(seed);
        }
    }
    let last = std::cell::RefCell::new(None);
    let keep = |s: &Solution| -> Result<()> {
        *last.borrow_mut() = Some(s.clone());
        Ok(())
    };
    dispatch(challenge, &keep, hp)?;
    let first = match last.borrow_mut().take() {
        Some(s) => s,
        None => return Ok(()),
    };
    let mut best = first;
    if let Some(seed) = &flow {
        if aim::value(challenge, &seed.items) > aim::value(challenge, &best.items) {
            best = seed.clone();
        }
    }
    if compute_budget_pct(challenge) <= 17 && aim::out_of_reach(challenge, hp) {
        return save(&best);
    }
    let mut extra:Option<Solution>=None;
    if compute_budget_pct(challenge)<=17 {
        if let Some((baseline,target))=aim::small_target(challenge,hp) {
            let v=aim::value(challenge,&best.items);
            let heavy=aim::Boost::new(challenge,hp,compute_budget_pct(challenge)).map_or(false,|b|b.heavy_hp(v).is_some());
            if v<target && (heavy || v.saturating_add(baseline/200)>=target) {
                let (proven,found)=settle(challenge,&best,baseline,target,2000000);
                if proven {return save(&best);}
                extra=found;
            }
        }
    }
    if let Some(boost) = aim::Boost::new(challenge, hp, compute_budget_pct(challenge)) {
        let v0 = aim::value(challenge, &best.items);
        if let Some(heavy) = boost.heavy_hp(v0) {
            dispatch(challenge, &keep, &Some(heavy))?;
            if let Some(s) = last.borrow_mut().take() {
                if aim::feasible(challenge, &s.items) && aim::value(challenge, &s.items) > v0 {
                    best = s;
                }
            }
        }
    }
    if compute_budget_pct(challenge)>17 && aim::near_target(challenge,hp,aim::value(challenge,&best.items),0.003){
        let mut heavy=hp.clone().unwrap_or_default();
        if heavy.get("ils_rounds").and_then(Value::as_u64).unwrap_or(24)<135{
            heavy.insert("ils_rounds".into(),Value::from(135u64));
            dispatch(challenge,&keep,&Some(heavy))?;
            if let Some(s)=last.borrow_mut().take(){
                if aim::feasible(challenge,&s.items) && aim::value(challenge,&s.items)>aim::value(challenge,&best.items){best=s;}
            }
        }
    }
    if compute_budget_pct(challenge)<=17 {
        if let Some((baseline,target))=aim::small_target(challenge,hp) {
            let initial_value=aim::value(challenge,&best.items);
            if initial_value<target && initial_value.saturating_add(baseline/200)>=target {
                best=block_search::improve(challenge,&best);
                let value=aim::value(challenge,&best.items);
                if value<target {
                    let work=if compute_budget_pct(challenge)<=7 {400000000u64}else{1000000000u64};
                    let midpoint=value+(target-value+1)/2;
                    let (first,spent,_)=core_search::improve_to_target(challenge,&best,work/4,midpoint);
                    best=first;
                    let (second,_,_)=core_search::improve_to_target(challenge,&best,work.saturating_sub(spent),target);
                    best=second;
                }
                let mut late=false;
                if compute_budget_pct(challenge)>7 && aim::value(challenge,&best.items)<target {
                    let (proven,found)=settle(challenge,&best,baseline,target,4000000);
                    late=proven;
                    if found.is_some() {extra=found;}
                }
                if !late && compute_budget_pct(challenge)>7 && aim::value(challenge,&best.items)<target {
                    let mut field=hp.clone().unwrap_or_default();
                    if field.get("ils_rounds").and_then(Value::as_u64).unwrap_or(9)<135 {
                        field.insert("ils_rounds".into(),Value::from(135u64));
                        field.insert("aim".into(),Value::from(0u64));
                        dispatch(challenge,&keep,&Some(field))?;
                        if let Some(solution)=last.borrow_mut().take() {
                            if aim::feasible(challenge,&solution.items) && aim::value(challenge,&solution.items)>aim::value(challenge,&best.items) {best=solution;}
                        }
                    }
                }
            }
        }
    }
    if let Some(s)=extra {
        if aim::value(challenge,&s.items)>aim::value(challenge,&best.items) {best=s;}
    }
    save(&best)
}

fn settle(challenge:&Challenge,best:&Solution,baseline:i64,target:i64,work:u64)->(bool,Option<Solution>) {
    let floor=target-(baseline*25+9999)/10000;
    if aim::value(challenge,&best.items)>=floor {return (false,None);}
    let (sol,spent,_)=core_search::improve_to_target(challenge,best,work,floor);
    if aim::value(challenge,&sol.items)>=floor {(false,Some(sol))} else {(spent<work,None)}
}

fn dispatch(
    challenge: &Challenge,
    save:      &dyn Fn(&Solution) -> Result<()>,
    hp:        &Option<Map<String, Value>>,
) -> Result<()> {

    let n          = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);

    match (n, budget_pct) {
        (1000, b) if b <= 7  => track1::solve(challenge, save, hp),
        (1000, b) if b <= 17 => track2::solve(challenge, save, hp),
        (1000, _)            => track3::solve(challenge, save, hp),

        (5000, b) if b <= 17 => track4::solve(challenge, save, hp),
        (5000, _)            => track5::solve(challenge, save, hp),

        _ => Err(anyhow::anyhow!(
            "QKBP dispatcher: unknown track config \
             (num_items={}, budget_pct={}). \
             Expected num_items in {{1000, 5000}}.",
            n, budget_pct
        )),
    }

}
