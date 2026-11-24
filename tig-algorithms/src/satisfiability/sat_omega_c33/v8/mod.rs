// sat_tailwalk_v8 derivative/composite source; constituent origins and retained copyrights are listed in README.md.
mod donor_kernel;
mod a115;
mod a130;
use self::a130::{t4 as a130_t4, t5 as a130_t5};
use self::engine_iv::{sat as engine_iv_sat, rescue as engine_iv_rescue, rescue_sat as engine_iv_rescue_sat};
use self::track1::{cached as track1_cached, as_tail as track1_as_tail};
mod engine_iv;
mod engine_b;

mod track1;
mod track2;
mod track3;
mod track4;
mod track5;
mod track_fallback;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverRoute {
    Track1,
    Track2,
    Track3,
    Track4,
    Track5,
    Fallback,
}

fn select_route(
    raw_num_variables: usize,
    raw_num_clauses: usize,
    num_variables: usize,
    density: f64,
) -> SolverRoute {
    if (raw_num_variables, raw_num_clauses) == (100_000, 420_000) {
        return SolverRoute::Track5;
    }
    if density >= 4.25 {
        if num_variables <= 5000 {
            return SolverRoute::Track1;
        }
        if num_variables <= 7500 {
            return SolverRoute::Track2;
        }
        return SolverRoute::Track3;
    }
    if density < 4.18 {
        return SolverRoute::Track4;
    }
    SolverRoute::Fallback
}

pub fn help() {
    println!("{}", r#"sat_tailwalk_v8 - C001 satisfiability
Per-track composite with cached search, bounded rescue, and adapted a130 T4/T5 routes.
Starting hyperparameters from local calibration:
T1 n_vars=5000,ratio=4267: {"max_fuel_high":400000000000}
T2 n_vars=7500,ratio=4267: {"max_fuel_high":400000000000}
T3 n_vars=10000,ratio=4267: {"max_fuel_high":125000000000}
T4 n_vars=100000,ratio=4150: {"max_fuel_low":150000000000}
T5 n_vars=100000,ratio=4200: {"max_fuel_low":150000000000}
Use an empty object or omit hyperparameters for inherited defaults.
The explicit settings above are not all identical to the defaults.
For T1's optimized composite, supply only max_fuel_high or use defaults.
Internal stage budgets are estimates; the official runtime enforces the outer fuel allowance.
See README.md for configuration, method attribution and limitations."#);
}

use self::a115 as reference_a115;

/// Per-track composite derived from v6 and the public solvers acknowledged in README.md.
/// Alternate routes forward complete SAT assignments; failed alternates retain the original fallback seed.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // The online a128 T1 configuration supplies only max_fuel_high. Route this
    // supported override through our existing composite and apply it to its
    // cached R5 primary search, instead of silently falling back to a115.
    // Other explicit configurations retain the previous compatibility path.
    let t1_primary_fuel = hyperparameters.as_ref().and_then(|m| {
        if m.len() == 1 {
            m.get("max_fuel_high").and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v >= 0.0)
        } else { None }
    });
    let t1_composite_hp = hyperparameters.as_ref().map_or(true, |m| m.is_empty())
        || t1_primary_fuel.is_some();
    if (challenge.num_variables, challenge.clauses.len()) == (100000, 415000) {
        let solved=std::cell::Cell::new(false);
        let accept_sat=|solution:&Solution| {
            if solution.variables.len()==challenge.num_variables
                && challenge.clauses.iter().all(|clause|clause.iter().any(|&lit| {
                    solution.variables[(lit.abs()-1) as usize]==(lit>0)
                })) {
                save_solution(solution)?;
                solved.set(true);
            }
            Ok(())
        };
        a130_t4::solve(challenge,&accept_sat,hyperparameters)?;
        if solved.get() {return Ok(());}
        // Preserve the original T4 route below if the donor does not solve.
    }
    if (challenge.num_variables, challenge.clauses.len()) == (100000, 420000) {
        // Preserve initial/partial saves and use the maintained SAT status.
        // This avoids rescanning the original formula after a130 succeeds.
        if a130_t5::solve(challenge, save_solution, hyperparameters)? {
            return Ok(());
        }
        // Accept only a complete SAT assignment from the SP portfolio. If it
        // fails, enter the unchanged R5 path below with a fresh original RNG.
        let solved = std::cell::Cell::new(false);
        let accept_sat = |solution: &Solution| {
            if solution.variables.len() == challenge.num_variables
                && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                    solution.variables[(lit.abs() - 1) as usize] == (lit > 0)
                }))
            {
                save_solution(solution)?;
                solved.set(true);
            }
            Ok(())
        };
        engine_b::solve(challenge, &accept_sat, hyperparameters)?;
        if solved.get() { return Ok(()); }
    }
    if (challenge.num_variables, challenge.clauses.len()) == (5000, 21335)
        && !t1_composite_hp {
        // The complement and original R5 each seed their own RNG. Forward
        // complete SAT results only, otherwise use the unchanged R5 below.
        let solved = std::cell::Cell::new(false);
        let accept_sat = |solution: &Solution| {
            if solution.variables.len() == challenge.num_variables
                && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                    solution.variables[(lit.abs()-1) as usize] == (lit > 0)
                }))
            {
                save_solution(solution)?;
                solved.set(true);
            }
            Ok(())
        };
        let budget = hyperparameters.as_ref()
            .and_then(|m| m.get("t4_fuel_b")).and_then(Value::as_u64)
            .unwrap_or(50_000_000_000);
        let mut complement_hp = Map::new();
        complement_hp.insert("max_fuel_high".into(), Value::from(budget));
        engine_iv::solve(challenge, &accept_sat, &Some(complement_hp))?;
        if solved.get() { return Ok(()); }
    }
    if (challenge.num_variables, challenge.clauses.len()) == (5000, 21335) {
        // t4_fuel_b configures only the preceding complement. The original
        // a115 fallback ignores it. Other keys retain a115's complete parser,
        // diagnostics and defaults, including whole-object parse failures.
        let default_fallback = hyperparameters.as_ref()
            .map_or(true, |m| m.keys().all(|key| key == "t4_fuel_b"))
            || t1_primary_fuel.is_some();
        if default_fallback {
            if t1_composite_hp {
                let solved = std::cell::Cell::new(false);
                let best_seed = std::cell::RefCell::new(None::<Vec<bool>>);
                let original_fallback = |solution: &Solution| -> Result<()> {
                    save_solution(solution)?;
                    let complete = solution.variables.len()==challenge.num_variables
                        && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                            solution.variables[(lit.abs()-1) as usize] == (lit>0)
                        }));
                    if complete { solved.set(true); }
                    else if solution.variables.len()==challenge.num_variables {
                        *best_seed.borrow_mut() = Some(solution.variables.clone());
                    }
                    Ok(())
                };
                // Try R5 first. Each route retains its original fresh seed and
                // full effort budget. On two failures, preserve R5's saved result.
                track1_cached::solve_with_fuel(challenge, &original_fallback, t1_primary_fuel)?;
                if !solved.get() {
                    let accept_iv = |solution: &Solution| -> Result<()> {
                        if solution.variables.len()==challenge.num_variables
                            && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                                solution.variables[(lit.abs()-1) as usize] == (lit>0)
                            })) {save_solution(solution)?;solved.set(true);}
                        Ok(())
                    };
                    let mut complement_hp=Map::new();
                    complement_hp.insert("max_fuel_high".into(),Value::from(50_000_000_000u64));
                    engine_iv_sat::solve(challenge,&accept_iv,&Some(complement_hp))?;
                }
                if !solved.get() {
                    let seed = best_seed.borrow().clone();
                    if let Some(seed) = seed {
                        let unsat=challenge.clauses.iter().filter(|clause| !clause.iter().any(|&lit| {
                            seed[(lit.abs()-1) as usize] == (lit>0)
                        })).count();
                        if unsat==1 {
                            let accept_complete = |solution: &Solution| -> Result<()> {
                                if solution.variables.len()==challenge.num_variables
                                    && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                                        solution.variables[(lit.abs()-1) as usize] == (lit>0)
                                    })) { save_solution(solution)?; solved.set(true); }
                                Ok(())
                            };
                            let mut rescue_hp=Map::new();
                            rescue_hp.insert("max_fuel_high".into(),Value::from(10_000_000_000u64));
                            engine_iv_rescue_sat::solve(challenge,&accept_complete,&Some(rescue_hp),&seed)?;
                        }
                    }
                }
                // Complementary AS restart uses only the global allowance left after
                // every original stage failed. Partial tail outputs never overwrite
                // the saved original result. A successful original rescue stops here.
                if !solved.get() {
                    let accept_tail = |solution: &Solution| -> Result<()> {
                        if solution.variables.len()==challenge.num_variables
                            && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                                solution.variables[(lit.abs()-1) as usize] == (lit>0)
                            })) { save_solution(solution)?; solved.set(true); }
                        Ok(())
                    };
                    track1_as_tail::solve(challenge, &accept_tail)?;
                }
                return Ok(());
            }
            return track1_cached::solve(challenge, save_solution);
        }
        return reference_a115::solve_challenge(challenge, save_solution, hyperparameters);
    }
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });

    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;
    let density = nc as f64 / nv as f64;
    let route = select_route(
        challenge.num_variables,
        challenge.clauses.len(),
        nv,
        density,
    );

    if route == SolverRoute::Track2 {
        return track2::solve(challenge, &hp, save_solution);
    }

    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    match route {
        SolverRoute::Track1 => {
            track1::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track2 => {
            unreachable!("Track2 returns before the shared index build")
        }
        SolverRoute::Track3 => {
            track3::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track4 => {
            track4::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track5 => {
            track5::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Fallback => {
            track_fallback::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
    }
}

#[repr(C, align(16))]
struct SearchClause {
    literals: [i32; 3],
    len: u32,
}

fn pack_search_clauses(flat: &[i32], offsets: &[u32]) -> Vec<SearchClause> {
    offsets.windows(2).map(|pair| {
        let slice = &flat[pair[0] as usize..pair[1] as usize];
        let mut literals = [0i32; 3];
        literals[..slice.len()].copy_from_slice(slice);
        SearchClause { literals, len: slice.len() as u32 }
    }).collect()
}

// Normalization produces one to three distinct literals. Keep the generic case
// explicit so this helper also retains modulo semantics for other lengths.
#[inline(always)]
fn short_clause_index(value: usize, len: usize) -> usize {
    match len {
        1 => 0,
        2 => value & 1,
        3 => value % 3,
        _ => value % len,
    }
}

/// Scratch fusion adapter: bounded subset, not the full default portfolio.
pub fn n2_bounded_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,_hp:&Option<Map<String,Value>>)->Result<()> {
    use std::cell::Cell;
    let solved=Cell::new(false);
    let accept=|s:&Solution|->Result<()> {
        if super::verified(ch,s) {save(s)?;solved.set(true);}
        Ok(())
    };
    if ch.num_variables==5000 && ch.clauses.len()==21335 {
        track1_cached::solve_with_fuel(ch,&accept,Some(1_000_000_000.0))?;
        if solved.get(){return Ok(());}
        let mut m=Map::new();m.insert("max_fuel_high".into(),Value::from(1_000_000_000u64));
        engine_iv_sat::solve(ch,&accept,&Some(m))?;
        return Ok(());
    }
    let mut m=Map::new();
    for key in ["max_fuel_high","max_fuel_low","t1_fuel_a","t5_fuel_a"] {m.insert(key.into(),Value::from(2_000_000_000u64));}
    for key in ["t1_fuel_b","t4_fuel_b","t5_fuel_c"] {m.insert(key.into(),Value::from(1_000_000_000u64));}
    solve_challenge(ch,&accept,&Some(m))
}

/// N3: six-billion small-track rescue allowance; T1 splits it 4.5B+1.5B.
pub fn n3_bounded_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,hp:&Option<Map<String,Value>>)->Result<()> {
    if ch.num_variables>10_000 {return n2_bounded_rescue(ch,save,hp);}
    if ch.num_variables==5000 && ch.clauses.len()==21335 {
        let solved=std::cell::Cell::new(false);
        let accept=|s:&Solution|->Result<()> {if super::verified(ch,s){save(s)?;solved.set(true);}Ok(())};
        track1_cached::solve_with_fuel(ch,&accept,Some(4_500_000_000.0))?;
        if solved.get(){return Ok(());}
        let mut m=Map::new();m.insert("max_fuel_high".into(),Value::from(1_500_000_000u64));
        engine_iv_sat::solve(ch,&accept,&Some(m))?;
        return Ok(());
    }
    let mut m=Map::new();m.insert("max_fuel_high".into(),Value::from(6_000_000_000u64));
    solve_challenge(ch,save,&Some(m))
}

/// Opt-in canonical T3 tail; existing solve_challenge and bounded rescue retain their routing.
pub fn c15_gated_t3(challenge:&Challenge,save_solution:&dyn Fn(&Solution)->Result<()>,extend:bool)->Result<Vec<track3::GateStats>>{
 assert_eq!((challenge.num_variables,challenge.clauses.len()),(10000,42670));
 let mut out=Vec::new();let solved=std::cell::Cell::new(false);
 let accept=|s:&Solution|->Result<()>{if super::verified(challenge,s){save_solution(s)?;solved.set(true);}Ok(())};
 for attempt in 0..1{
  let mut other=challenge.clone();
  let challenge=&other;let nv=challenge.num_variables;let mut rng=SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;
    let density = nc as f64 / nv as f64;
    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

  let st=track3::solve_c15_gated(&mut rng,nv,nc,density,p_cnt,n_cnt,&all_off,&p_bound,&all_data,&mut cl,&co,&accept,extend)?;
  let extended=st.extended;out.push(st);if solved.get()||extended{break;}
 }Ok(out)
}

/// Preserve the original rescue routing and callback order, recording only
/// raw verified near assignments outside the original search kernels.
pub fn c16_rescue_near(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,max_u:usize)->Result<Option<(usize,Vec<bool>)>>{
 let solved=std::cell::Cell::new(false);let near=std::cell::RefCell::new(None::<(usize,Vec<bool>)>);
 let accept=|s:&Solution|->Result<()>{
  if super::verified(ch,s){save(s)?;solved.set(true);}else{let u=raw_u(ch,&s.variables);let mut slot=near.borrow_mut();if u>0&&u<=max_u&&slot.as_ref().map_or(true,|p|u<p.0){*slot=Some((u,s.variables.clone()));}}Ok(())
 };
 if ch.num_variables==5000&&ch.clauses.len()==21335{
  track1_cached::solve_with_fuel(ch,&accept,Some(4_500_000_000.0))?;
  if !solved.get(){let mut hp=Map::new();hp.insert("max_fuel_high".into(),Value::from(1_500_000_000u64));engine_iv_sat::solve(ch,&accept,&Some(hp))?;}
 }else{let mut hp=Map::new();hp.insert("max_fuel_high".into(),Value::from(6_000_000_000u64));solve_challenge(ch,&accept,&Some(hp))?;}
 Ok(near.into_inner())
}
pub fn c16_jump_one(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,seeds:Vec<(usize,Vec<bool>)>,max_u:usize,fuel:u64)->Result<usize>{
 jump_one_with(ch,save,seeds,max_u,|x,accept|{let mut hp=Map::new();hp.insert("max_fuel_high".into(),Value::from(fuel));engine_iv_rescue_sat::solve(ch,accept,&Some(hp),x)})
}
fn jump_one_with<F>(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,seeds:Vec<(usize,Vec<bool>)>,max_u:usize,mut run:F)->Result<usize>
where F:FnMut(&[bool],&dyn Fn(&Solution)->Result<()>)->Result<()>{
 let best=seeds.into_iter().filter(|(u,x)|*u>0&&*u<=max_u&&raw_u(ch,x)==*u).min_by_key(|s|s.0);
 if let Some((_,x))=best{let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;}Ok(())};run(&x,&accept)?;Ok(1)}else{Ok(0)}
}

fn raw_u(ch:&Challenge,x:&[bool])->usize{if x.len()!=ch.num_variables{return usize::MAX;}ch.clauses.iter().filter(|c|!c.iter().any(|&l|x[l.unsigned_abs()as usize-1]==(l>0))).count()}
pub fn c15_jump(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,seeds:Vec<(usize,Vec<bool>)>)->Result<usize>{
 jump_with(ch,save,seeds,|seed,accept|{let mut hp=Map::new();hp.insert("max_fuel_high".into(),Value::from(12_000_000_000u64));engine_iv_rescue_sat::solve(ch,accept,&Some(hp),seed)})
}
fn jump_with<F>(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,mut seeds:Vec<(usize,Vec<bool>)>,mut solve:F)->Result<usize>
where F:FnMut(&[bool],&dyn Fn(&Solution)->Result<()>)->Result<()>{
 seeds.retain(|(u,x)|(1..=3).contains(u)&&raw_u(ch,x)==*u);seeds.sort_by_key(|s|s.0);let mut unique=Vec::<(usize,Vec<bool>)>::new();for seed in seeds{if !unique.iter().any(|s|s.1==seed.1){unique.push(seed);}}
 if unique.len()==1&&unique[0].0<=2{use rand::RngCore;let mut x=unique[0].1.clone();let base=u64::from_le_bytes(ch.seed[..8].try_into().unwrap())^0x4e4541524a554d50;let mut rng=super::tekinon::PortableRng64::seed_from_u64(base);let mut changed=Vec::new();let clauses:Vec<_>=ch.clauses.iter().filter(|c|!c.iter().any(|&l|x[l.unsigned_abs()as usize-1]==(l>0))).collect();for _ in 0..64{if changed.len()==4{break;}let c=clauses[(rng.next_u64()%clauses.len()as u64)as usize];let v=c[(rng.next_u64()%c.len()as u64)as usize].unsigned_abs()as usize-1;if !changed.contains(&v){changed.push(v);}}if changed.len()<4{'all:for clause in &ch.clauses{for &l in clause{let v=l.unsigned_abs()as usize-1;if !changed.contains(&v){changed.push(v);}if changed.len()==4{break 'all;}}}}for v in changed{x[v]=!x[v];}if x!=unique[0].1{unique.push((raw_u(ch,&x),x));}}
 let solved=std::cell::Cell::new(false);let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;solved.set(true);}Ok(())};let mut calls=0;
 for(_,x)in unique.into_iter().take(2){solve(&x,&accept)?;calls+=1;if solved.get(){break;}}Ok(calls)
}
#[cfg(test)]mod c15_near_tests{
 use super::*;use std::cell::Cell;
 #[test]fn jump_wrapper_filters_partial_proposals_and_skips_non_near_or_solved_seeds(){let ch=Challenge{seed:[0;32],num_variables:4,clauses:vec![vec![1,2,3]]};let calls=Cell::new(0);let saved=Cell::new(0);
  let n=jump_with(&ch,&|s|{assert!(super::super::verified(&ch,s));saved.set(saved.get()+1);Ok(())},vec![(1,vec![false;4])],|_,accept|{calls.set(calls.get()+1);accept(&Solution{variables:vec![false;4]})?;accept(&Solution{variables:vec![true;4]})}).unwrap();assert_eq!((n,calls.get(),saved.get()),(1,1,1));
  let original=vec![true;4];let n=jump_with(&ch,&|_|panic!(),vec![(4,vec![false;4]),(0,original.clone())],|_,_|panic!("jumper on non-near miss or solved prefix")).unwrap();assert_eq!(n,0);assert_eq!(original,vec![true;4]);
 }
 #[test]fn real_engine_rescue_solves_a_near_seed_through_verified_only_wrapper(){let ch=Challenge{seed:[0;32],num_variables:4,clauses:vec![vec![1,2,3]]};let saved=Cell::new(0);let calls=c15_jump(&ch,&|s|{assert!(super::super::verified(&ch,s));saved.set(saved.get()+1);Ok(())},vec![(1,vec![false;4])]).unwrap();assert_eq!(calls,1);assert!(saved.get()>0);}
}

pub use track2::GateStats as T2GateStats;
/// Canonical T2 only: original track2 search, with an opt-in boundary gate.
pub fn c18_t2_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,gate:Option<(usize,usize)>,diagnostic:bool)->Result<(Option<(usize,Vec<bool>)>,T2GateStats)> {
 assert_eq!((ch.num_variables,ch.clauses.len()),(7500,32002));let near=std::cell::RefCell::new(None::<(usize,Vec<bool>)>);
 let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;}else{let u=raw_u(ch,&s.variables);let mut slot=near.borrow_mut();if u>0&&u<=2&&slot.as_ref().map_or(true,|n|u<n.0){*slot=Some((u,s.variables.clone()));}}Ok(())};
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.0),max_fuel_low:None});
 let stats=if diagnostic{track2::solve_gated_diagnostic(ch,&hp,&accept,gate)?}else{track2::solve_gated(ch,&hp,&accept,gate)?;T2GateStats::default()};Ok((near.into_inner(),stats))
}

pub fn c20_t1_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,narrow:bool)->Result<Option<(usize,Vec<bool>)>>{let near=std::cell::RefCell::new(None::<(usize,Vec<bool>)>);let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;}else{let u=raw_u(ch,&s.variables);let mut n=near.borrow_mut();if u==1&&n.is_none(){*n=Some((u,s.variables.clone()));}}Ok(())};let mut trace=Vec::new();if narrow{track1_cached::solve_with_fuel_gated::<3,12,false>(ch,&accept,Some(4_500_000_000.0),&mut trace)?;}else{track1_cached::solve_with_fuel_gated::<3,13,false>(ch,&accept,Some(4_500_000_000.0),&mut trace)?;}Ok(near.into_inner())}

pub fn c20_t2_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,limit0:usize,diagnostic:bool)->Result<(Option<(usize,Vec<bool>)>,T2GateStats)> {
 assert_eq!((ch.num_variables,ch.clauses.len()),(7500,32002));let near=std::cell::RefCell::new(None::<(usize,Vec<bool>)>);
 let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;}else{let u=raw_u(ch,&s.variables);let mut slot=near.borrow_mut();if u>0&&u<=2&&slot.as_ref().map_or(true,|n|u<n.0){*slot=Some((u,s.variables.clone()));}}Ok(())};
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.0),max_fuel_low:None});
 let mut stats=T2GateStats::default();match(limit0,diagnostic){(28,false)=>track2::solve_two_stage::<28,false>(ch,&hp,&accept,&mut stats)?,(32,false)=>track2::solve_two_stage::<32,false>(ch,&hp,&accept,&mut stats)?,(28,true)=>track2::solve_two_stage::<28,true>(ch,&hp,&accept,&mut stats)?,(32,true)=>track2::solve_two_stage::<32,true>(ch,&hp,&accept,&mut stats)?,_=>anyhow::bail!("unregistered C20 gate")};Ok((near.into_inner(),stats))
}

pub fn c26_gated_t3(challenge:&Challenge,save_solution:&dyn Fn(&Solution)->Result<()>,extend:bool)->Result<Vec<track3::GateStats>>{
 assert_eq!((challenge.num_variables,challenge.clauses.len()),(10000,42670));
 let mut out=Vec::new();let solved=std::cell::Cell::new(false);
 let accept=|s:&Solution|->Result<()>{if super::verified(challenge,s){save_solution(s)?;solved.set(true);}Ok(())};
 for attempt in 0..1{
  let nv=challenge.num_variables;let mut rng=SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;
    let density = nc as f64 / nv as f64;
    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

  let st=track3::solve_c26_gated(&mut rng,nv,nc,density,p_cnt,n_cnt,&all_off,&p_bound,&all_data,&mut cl,&co,&accept,extend)?;
  let extended=st.extended;out.push(st);if solved.get()||extended{break;}
 }Ok(out)
}
pub fn c26_t2_rescue(ch:&Challenge,save:&dyn Fn(&Solution)->Result<()>,limit0:usize,diagnostic:bool)->Result<(Option<(usize,Vec<bool>)>,T2GateStats)> {
 assert_eq!((ch.num_variables,ch.clauses.len()),(7500,32002));let near=std::cell::RefCell::new(None::<(usize,Vec<bool>)>);
 let accept=|s:&Solution|->Result<()>{if super::verified(ch,s){save(s)?;}else{let u=raw_u(ch,&s.variables);let mut slot=near.borrow_mut();if u>0&&u<=2&&slot.as_ref().map_or(true,|n|u<n.0){*slot=Some((u,s.variables.clone()));}}Ok(())};
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.0),max_fuel_low:None});
 let mut stats=T2GateStats::default();if diagnostic{track2::solve_c26_two_stage::<23,true>(ch,&hp,&accept,&mut stats)?}else{track2::solve_c26_two_stage::<23,false>(ch,&hp,&accept,&mut stats)?};Ok((near.into_inner(),stats))
}

pub fn c29_gated_t3(challenge:&Challenge,save_solution:&dyn Fn(&Solution)->Result<()>,extend:bool)->Result<Vec<track3::GateStats>>{
 assert_eq!((challenge.num_variables,challenge.clauses.len()),(10000,42670));
 let mut out=Vec::new();let solved=std::cell::Cell::new(false);
 let accept=|s:&Solution|->Result<()>{if super::verified(challenge,s){save_solution(s)?;solved.set(true);}Ok(())};
 for attempt in 0..1{
  let nv=challenge.num_variables;let mut rng=SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;
    let density = nc as f64 / nv as f64;
    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

  let st=track3::solve_c29_gated(&mut rng,nv,nc,density,p_cnt,n_cnt,&all_off,&p_bound,&all_data,&mut cl,&co,&accept,extend)?;
  let extended=st.extended;out.push(st);if solved.get()||extended{break;}
 }Ok(out)
}

pub fn c32_gated_t3(challenge:&Challenge,save_solution:&dyn Fn(&Solution)->Result<()>,extend:bool)->Result<Vec<track3::GateStats>>{
 assert_eq!((challenge.num_variables,challenge.clauses.len()),(10000,42670));
 let mut out=Vec::new();let solved=std::cell::Cell::new(false);
 let accept=|s:&Solution|->Result<()>{if super::verified(challenge,s){save_solution(s)?;solved.set(true);}Ok(())};
 for attempt in 0..1{
  let nv=challenge.num_variables;let mut rng=SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;
    let density = nc as f64 / nv as f64;
    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

  let st=track3::solve_c32_gated(&mut rng,nv,nc,density,p_cnt,n_cnt,&all_off,&p_bound,&all_data,&mut cl,&co,&accept,extend)?;
  let extended=st.extended;out.push(st);if solved.get()||extended{break;}
 }Ok(out)
}
