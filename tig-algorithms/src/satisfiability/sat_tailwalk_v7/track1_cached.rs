// Track1 cached fallback adapted from a125 with a115 preprocessing; v7 changes are described in README.md.
// Track1 default fallback derived from c001_a125 hybrid_engine_b::track_t4.
// Upstream commit b3c056ad909a65243314cbae24347cc4d0e9bdd6.
// Keep the a115 preprocessor and 15-reinitialization wrapper default exactly.
// This module is entered only for default fallback parameters; arbitrary user
// parameters continue to use reference_a115 with its original serde semantics.
use anyhow::Result;
use tig_challenges::satisfiability::*;
use super::reference_a115::{Prepared,preprocess};

struct Hparams {
    check_interval: Option<usize>,
    stagnation_limit: Option<usize>,
    max_fuel_high: Option<f64>,
    max_reinits: Option<usize>,
}
pub(super) fn solve(challenge:&Challenge,save_solution:&dyn Fn(&Solution)->Result<()>)->Result<()> {
    debug_assert_eq!((challenge.num_variables,challenge.clauses.len()),(5000,21335));
    let hp=Hparams {check_interval:None,stagnation_limit:None,max_fuel_high:None,max_reinits:Some(15)};
    track_t4::solve(challenge,save_solution,&hp)
}
pub(super) mod explicit_roulette {
// Positive integer scores use the exact baseline weights. Pack a 32-bit
// reciprocal plus three cumulative-weight fields in one u64. The numerator
// remains the original low 32 bits; no RNG draw or distribution is changed.
const W:[u32;16]=[2535,551,233,127,80,55,41,30,24,19,16,13,11,9,8,7];
const fn make_integer()->[u64;4096]{
 let mut tab=[0u64;4096];let mut a=1;
 while a<16{let mut b=1;while b<16{let mut c=1;while c<16{
  let w0=W[a];let w01=w0+W[b];let total=w01+W[c];
  let m=u32::MAX/total;
  tab[(a<<8)|(b<<4)|c]=(m as u64)|((w0 as u64)<<32)|((w01 as u64)<<42)|((total as u64)<<53);
  c+=1;}b+=1;}a+=1;}tab
}
static INTEGER:[u64;4096]=make_integer();
#[inline(always)]pub(crate) fn integer_three(a:usize,b:usize,c:usize,r:u32)->usize{
 debug_assert!(a>0&&b>0&&c>0);
 let entry=unsafe{*INTEGER.get_unchecked((a.min(15)<<8)|(b.min(15)<<4)|c.min(15))};
 let m=entry as u32;let total=(entry>>53)as u32;
 let q=(((r as u64)*(m as u64))>>32)as u32;
 let raw=r-q*total;let rem=if raw>=total{raw-total}else{raw};
 let w0=((entry>>32)&1023)as u32;let w01=((entry>>42)&2047)as u32;
 (rem>=w0)as usize+(rem>=w01)as usize
}
}
pub(super) mod signed_order {
// A chosen clause is unsatisfied. Thus every selected literal is false and
// its sign gives the current assignment exactly: a negative literal implies
// variable=true; a positive one implies variable=false. No truth-array read or
// mutable range orientation is necessary to choose gain/loss occurrence lists.
pub(crate) struct SignedOrder{words:Vec<u64>}
pub(crate) struct LiteralRanges{ranges:Vec<u64>}
impl LiteralRanges{
 pub(crate) fn new(off:&[u32],mid:&[u32])->Self{
  let mut ranges=Vec::with_capacity(2*mid.len());
  for v in 0..mid.len(){
   ranges.push((off[v]as u64)|(((mid[v]-off[v])as u64)<<32));
   ranges.push((mid[v]as u64)|(((off[v+1]-mid[v])as u64)<<32));
  }
  Self{ranges}
 }
 #[inline(always)]pub(crate) unsafe fn of(&self,code:usize)->(usize,usize,usize,usize){
  let gain=*self.ranges.get_unchecked(code);let loss=*self.ranges.get_unchecked(code^1);
  let a=gain as u32 as usize;let b=loss as u32 as usize;
  (a,a+(gain>>32)as usize,b,b+(loss>>32)as usize)
 }
}
impl SignedOrder{
 pub(crate) fn new(cl:&[i32],co:&[u32])->Self{
  let mut words=Vec::with_capacity(co.len()-1);
  for c in 0..co.len()-1{
   let lits=&cl[co[c]as usize..co[c+1]as usize];assert!(!lits.is_empty()&&lits.len()<=3);
   let mut word=(lits.len()as u64)<<54;
   for(j,&l)in lits.iter().enumerate(){let v=(l.abs()-1)as u64;assert!(v<(1<<17));word|=((v<<1)|((l<0)as u64))<<(18*j);}
   words.push(word);
  }
  Self{words}
 }
 #[inline(always)]pub(crate) unsafe fn len(&self,c:usize)->usize{(*self.words.get_unchecked(c)>>54)as usize}
 #[inline(always)]pub(crate) unsafe fn len_bounded(&self,c:usize)->usize{((*self.words.get_unchecked(c)>>54)&3)as usize}
 #[inline(always)]pub(crate) unsafe fn codes(&self,c:usize)->[usize;3]{
  let w=*self.words.get_unchecked(c);[(w&262143)as usize,((w>>18)&262143)as usize,((w>>36)&262143)as usize]
 }
 #[inline(always)]pub(crate) unsafe fn lit(&self,c:usize,j:usize)->i32{
  let code=((*self.words.get_unchecked(c)>>(18*j))&262143)as i32;
  let v=(code>>1)+1;if code&1!=0{-v}else{v}
 }
 #[inline(always)]pub(crate) unsafe fn swap(&mut self,c:usize,a:usize,b:usize){
  let w=self.words.get_unchecked_mut(c);let d=((*w>>(18*a))^(*w>>(18*b)))&262143;
  *w^=(d<<(18*a))|(d<<(18*b));
 }
 #[inline(always)]pub(crate) unsafe fn choose_zero(v:[usize;3],mask:usize,r:usize)->usize{
  super::super::donor_kernel::clause_order::ClauseOrder::choose_zero(v,mask,r)
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn signs_order_and_static_range_polarity(){
  let cl=[1,-100000,34567,-4,7,-99];let co=[0,3,5,6];let mut order=SignedOrder::new(&cl,&co);
  let mut expected=vec![vec![1,-100000,34567],vec![-4,7],vec![-99]];
  for step in 0..10000{let c=step%3;let n=expected[c].len();let r=(step*11+7)%n;
   expected[c].swap(0,r);unsafe{order.swap(c,0,r);assert_eq!(order.len(c),n);
    for j in 0..n{assert_eq!(order.lit(c,j),expected[c][j]);let code=order.codes(c)[j];assert_eq!(code>>1,(expected[c][j].abs()-1)as usize);assert_eq!(code&1!=0,expected[c][j]<0);}
   }
  }
  let off=[0u32,7,11,19];let mid=[3u32,8,15];let ranges=LiteralRanges::new(&off,&mid);
  for v in 0..3{for negative in [false,true]{let got=unsafe{ranges.of((v<<1)|(negative as usize))};let p=(off[v]as usize,mid[v]as usize);let n=(mid[v]as usize,off[v+1]as usize);assert_eq!(got,if negative{(n.0,n.1,p.0,p.1)}else{(p.0,p.1,n.0,n.1)});}}
 }
}
}
// Exact three-literal specialization shared by the two Track1 engines.
// The short-clause path retains the original implementation and RNG word.
pub(crate) mod shaped_choice {
    use super::signed_order::SignedOrder;
    use super::super::donor_kernel::{exact_div,subset_cache::CacheOps};
    #[inline(always)]
    pub(crate) unsafe fn choose_three<C:CacheOps>(order:&mut SignedOrder,cid:usize,cache:&C,word:usize)->usize {
        order.swap(cid,0,exact_div::clause_rem(word,3));
        let ll=order.codes(cid);
        let b0=cache.breaks_of(ll[0]>>1);
        let b1=cache.breaks_of(ll[1]>>1);
        let b2=cache.breaks_of(ll[2]>>1);
        let mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
        if mask!=0 {SignedOrder::choose_zero(ll,mask,word)}
        else {ll[super::explicit_roulette::integer_three(b0,b1,b2,word as u32)]}
    }
}
mod track_t4 {
use super::explicit_roulette;
use super::signed_order::{SignedOrder as ClauseOrder,LiteralRanges};

use super::super::donor_kernel::exact_div;

use super::super::donor_kernel::subset_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};

use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;

use super::Hparams;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let prepared=super::preprocess(challenge,save_solution);
    let narrow=prepared.nv<=8192 && prepared.p_cnt.iter().zip(&prepared.n_cnt).all(|(&p,&n)|p+n<=255);
    if narrow {solve_impl::<Byte13BreakCache>(prepared,save_solution,hp)}
    else {solve_impl::<SmallBreakCache>(prepared,save_solution,hp)}
}
fn solve_impl<C:CacheOps>(prepared:super::Prepared,save_solution:&dyn Fn(&Solution)->Result<()>,hp:&Hparams)->Result<()> {
    let super::Prepared {
        mut rng,
        nv,
        nc,
        density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        mut cl,
        co,
    } = prepared;

    let max_fuel = hp.max_fuel_high.unwrap_or(160_000_000_000.0);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let nad = 1.0;
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 { vars[v] = true; continue; }
        if np == 0 && nn > 0 { continue; }
        let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
        if vad <= nad {
            vars[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            vars[v] = rng.gen_bool(prob);
        }
    }

    let mut num_good = vec![0u8; nc];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let mut g = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
        }
        num_good[c] = g;
        if g == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }


    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.check_interval
        .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.stagnation_limit.unwrap_or(3);

    let probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    const REINIT_STAGNATION: usize = 2_000_000;
    const REINIT_MIN_UNSAT: usize = 10;
    let max_reinits = hp.max_reinits.unwrap_or(5);

    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;

    const N_BON_RESTARTS: usize = 5;
    let mut bon_candidate = vec![false; nv];
    let mut bon_num_good = vec![0u8; nc];

    let mut order = ClauseOrder::new(&cl, &co);
    let ranges=LiteralRanges::new(&all_off,&p_bound);
    // The scratch capacity covers every occurrence of any variable. The
    // initialized prefix is exactly the clauses whose membership changes.
    let max_degree=all_off.windows(2).map(|w|(w[1]-w[0])as usize).max().unwrap_or(0);
    let mut changed=Vec::<u32>::with_capacity(max_degree);
    let changed_ptr=changed.as_mut_ptr();
    let mut cache = C::new(nv, &cl, &co, &vars);
    let mut next_check_interval = check_interval;
    unsafe {
        // Best count is monotone after the last allowed reinitialization.
        // Inactive restart bookkeeping cannot become observable again.
        if best_unsat>=REINIT_MIN_UNSAT && reinit_count<max_reinits{
        loop {
            if rounds >= max_flips { break; }
            if unsat_list.is_empty() { break; }

            if stagnation_count >= REINIT_STAGNATION && best_unsat >= REINIT_MIN_UNSAT && reinit_count < max_reinits {
                reinit_count += 1;

                let mut best_cand_unsat = usize::MAX;
                for _ in 0..N_BON_RESTARTS {
                    for v in 0..nv { bon_candidate[v] = false; }
                    for v in 0..nv {
                        let np = p_cnt[v] as usize;
                        let nn = n_cnt[v] as usize;
                        if nn == 0 && np > 0 { bon_candidate[v] = true; continue; }
                        if np == 0 && nn > 0 { continue; }
                        let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
                        if vad <= nad {
                            bon_candidate[v] = rng.gen_bool(random_threshold);
                        } else {
                            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                            bon_candidate[v] = rng.gen_bool(prob);
                        }
                    }
                    bon_num_good.fill(0);
                    for c in 0..nc {
                        let s = co[c] as usize;
                        let e = co[c + 1] as usize;
                        let mut g = 0u8;
                        for j in s..e {
                            let l = cl[j];
                            let v = (l.abs() - 1) as usize;
                            if (l > 0 && bon_candidate[v]) || (l < 0 && !bon_candidate[v]) { g += 1; }
                        }
                        bon_num_good[c] = g;
                    }
                    let cand_unsat = bon_num_good.iter().filter(|&&x| x == 0).count();
                    if cand_unsat < best_cand_unsat {
                        best_cand_unsat = cand_unsat;
                        vars.copy_from_slice(&bon_candidate);
                    }
                }

                num_good.fill(0);
                for c in 0..nc {
                    let s = co[c] as usize;
                    let e = co[c + 1] as usize;
                    let mut g = 0u8;
                    for j in s..e {
                        let l = cl[j];
                        let v = (l.abs() - 1) as usize;
                        if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
                    }
                    num_good[c] = g;
                }

                unsat_list.clear();
                unsat_pos.fill(u32::MAX);
                for c in 0..nc {
                    if num_good[c] == 0 {
                        unsat_pos[c] = unsat_list.len() as u32;
                        unsat_list.push(c as u32);
                    }
                }

                cache = C::new(nv, &cl, &co, &vars);
                best_unsat = unsat_list.len();
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            }

            if (rounds == next_check_interval || check_interval == 0) && rounds % check_interval == 0 && rounds > 0 {
                    next_check_interval = next_check_interval.wrapping_add(check_interval);
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() { break; }

                            let rid = exact_div::rem(rng.gen::<usize>(), unsat_list.len());
                            let pcid = *unsat_list.get_unchecked(rid) as usize;
                            let pcs = 0usize;
                            let pce = order.len(pcid);
                            if pcs == pce { continue; }
                            let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            let was_true=lit<0;
    let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
    let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v,changed_ptr);
    for i in 0..gains {
        let c=*changed_ptr.add(i)as usize;
        let pos=*unsat_pos.get_unchecked(c)as usize;
        let last_idx=unsat_list.len()-1;
        let last_c=*unsat_list.get_unchecked(last_idx)as usize;
        *unsat_list.get_unchecked_mut(pos)=last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c)=pos as u32;
        unsat_list.set_len(last_idx);
        // A satisfied clause's position is unobserved until its next insertion,
        // which overwrites it. No need to clear that dead value to u32::MAX.
    }
    let losses=cache.lose_many(all_data.get_unchecked(ds..de),v,changed_ptr);
    let len=unsat_list.len();
    // There is at most one live entry per clause, so len+losses<=nc and the
    // original with_capacity(nc) allocation always has enough space.
    for i in 0..losses {
        let c=*changed_ptr.add(i);
        *unsat_pos.get_unchecked_mut(c as usize)=(len+i)as u32;
        unsat_list.as_mut_ptr().add(len+i).write(c);
    }
    unsat_list.set_len(len+losses);
    cache.set_break(v,gains as u32);

    *vars.get_unchecked_mut(v) = !was_true;
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
            }

            if unsat_list.is_empty() { break; }

            let rand_val = rng.gen::<usize>();

            // Unit clause weights are immutable, so the original tournament
            // always selected c1. Preserve the same low-32-bit index and RNG draw.
            let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32, unsat_list.len())) as usize;

            let cs = 0usize;
            let ce = order.len(cid);
            let clen = ce - cs;

            let chosen_code=if clen==3 {
                super::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
            } else {
            if clen > 1 {
                let ri = exact_div::clause_rem(rand_val, clen);
                order.swap(cid, 0, ri);
            }


            let ll=order.codes(cid);
            let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
            let b0=cache.breaks_of(vv[0]);
            let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
            let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
            let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
            let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}else{
                let w0=*probs_break.get_unchecked(b0.min(15));
                let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
                let w2=if clen>2{*probs_break.get_unchecked(b2.min(15))}else{0};
                let total=w0+w1+w2;
                let r=(rand_val as u32)%total.max(1);
                if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else if clen>2 && r-w0-w1<w2{ll[2]}else{ll[0]}
            };

                chosen_code
            };
            let v_idx=chosen_code>>1;
            let was_true=chosen_code&1!=0;
    let (is,ie,ds,de)=ranges.of(chosen_code);
    let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v_idx,changed_ptr);
    for i in 0..gains {
        let c=*changed_ptr.add(i)as usize;
        let pos=*unsat_pos.get_unchecked(c)as usize;
        let last_idx=unsat_list.len()-1;
        let last_c=*unsat_list.get_unchecked(last_idx)as usize;
        *unsat_list.get_unchecked_mut(pos)=last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c)=pos as u32;
        unsat_list.set_len(last_idx);
        // A satisfied clause's position is unobserved until its next insertion,
        // which overwrites it. No need to clear that dead value to u32::MAX.
    }
    let losses=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,changed_ptr);
    let len=unsat_list.len();
    // There is at most one live entry per clause, so len+losses<=nc and the
    // original with_capacity(nc) allocation always has enough space.
    for i in 0..losses {
        let c=*changed_ptr.add(i);
        *unsat_pos.get_unchecked_mut(c as usize)=(len+i)as u32;
        unsat_list.as_mut_ptr().add(len+i).write(c);
    }
    unsat_list.set_len(len+losses);
    cache.set_break(v_idx,gains as u32);

    *vars.get_unchecked_mut(v_idx) = !was_true;
            rounds += 1;

            let cur = unsat_list.len();
            if cur < best_unsat {
                best_unsat = cur;
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
            if best_unsat<REINIT_MIN_UNSAT || reinit_count>=max_reinits{break;}
        }
        }
        loop {
            if rounds >= max_flips { break; }
            if unsat_list.is_empty() { break; }



            if (rounds == next_check_interval || check_interval == 0) && rounds % check_interval == 0 && rounds > 0 {
                    next_check_interval = next_check_interval.wrapping_add(check_interval);
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() { break; }

                            let rid = exact_div::rem(rng.gen::<usize>(), unsat_list.len());
                            let pcid = *unsat_list.get_unchecked(rid) as usize;
                            let pcs = 0usize;
                            let pce = order.len(pcid);
                            if pcs == pce { continue; }
                            let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            let was_true=lit<0;
    let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
    let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v,changed_ptr);
    for i in 0..gains {
        let c=*changed_ptr.add(i)as usize;
        let pos=*unsat_pos.get_unchecked(c)as usize;
        let last_idx=unsat_list.len()-1;
        let last_c=*unsat_list.get_unchecked(last_idx)as usize;
        *unsat_list.get_unchecked_mut(pos)=last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c)=pos as u32;
        unsat_list.set_len(last_idx);
        // A satisfied clause's position is unobserved until its next insertion,
        // which overwrites it. No need to clear that dead value to u32::MAX.
    }
    let losses=cache.lose_many(all_data.get_unchecked(ds..de),v,changed_ptr);
    let len=unsat_list.len();
    // There is at most one live entry per clause, so len+losses<=nc and the
    // original with_capacity(nc) allocation always has enough space.
    for i in 0..losses {
        let c=*changed_ptr.add(i);
        *unsat_pos.get_unchecked_mut(c as usize)=(len+i)as u32;
        unsat_list.as_mut_ptr().add(len+i).write(c);
    }
    unsat_list.set_len(len+losses);
    cache.set_break(v,gains as u32);

    // Assignment is reconstructed only at snapshot and final-output events.
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
            }

            if unsat_list.is_empty() { break; }

            // The restart bookkeeping is permanently inactive in this tail.
            // Periodic kicks still execute at their original round; batch only
            // the main flips before the next such event or the absolute budget.
            let phase_end=if next_check_interval>rounds {
                next_check_interval.min(max_flips)
            }else{rounds.saturating_add(1).min(max_flips)};
            while rounds<phase_end && !unsat_list.is_empty() {
            let rand_val = rng.gen::<usize>();

            // Unit clause weights are immutable, so the original tournament
            // always selected c1. Preserve the same low-32-bit index and RNG draw.
            let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32, unsat_list.len())) as usize;

            let cs = 0usize;
            let ce = order.len(cid);
            let clen = ce - cs;

            let chosen_code=if clen==3 {
                super::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
            } else {
            if clen > 1 {
                let ri = exact_div::clause_rem(rand_val, clen);
                order.swap(cid, 0, ri);
            }


            let ll=order.codes(cid);
            let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
            let b0=cache.breaks_of(vv[0]);
            let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
            let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
            let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
            let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}else{
                let w0=*probs_break.get_unchecked(b0.min(15));
                let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
                let w2=if clen>2{*probs_break.get_unchecked(b2.min(15))}else{0};
                let total=w0+w1+w2;
                let r=(rand_val as u32)%total.max(1);
                if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else if clen>2 && r-w0-w1<w2{ll[2]}else{ll[0]}
            };

                chosen_code
            };
            let v_idx=chosen_code>>1;
            let was_true=chosen_code&1!=0;
    let (is,ie,ds,de)=ranges.of(chosen_code);
    let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v_idx,changed_ptr);
    for i in 0..gains {
        let c=*changed_ptr.add(i)as usize;
        let pos=*unsat_pos.get_unchecked(c)as usize;
        let last_idx=unsat_list.len()-1;
        let last_c=*unsat_list.get_unchecked(last_idx)as usize;
        *unsat_list.get_unchecked_mut(pos)=last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c)=pos as u32;
        unsat_list.set_len(last_idx);
        // A satisfied clause's position is unobserved until its next insertion,
        // which overwrites it. No need to clear that dead value to u32::MAX.
    }
    let losses=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,changed_ptr);
    let len=unsat_list.len();
    // There is at most one live entry per clause, so len+losses<=nc and the
    // original with_capacity(nc) allocation always has enough space.
    for i in 0..losses {
        let c=*changed_ptr.add(i);
        *unsat_pos.get_unchecked_mut(c as usize)=(len+i)as u32;
        unsat_list.as_mut_ptr().add(len+i).write(c);
    }
    unsat_list.set_len(len+losses);
    cache.set_break(v_idx,gains as u32);


            rounds += 1;

            let cur = unsat_list.len();
            if cur < best_unsat {
                best_unsat = cur;
                cache.restore_variables(&cl,&co,&mut vars);
                best_vars.copy_from_slice(&vars);

            } else {

            }
            }
        }
    }

    if unsat_list.is_empty() {cache.restore_variables(&cl,&co,&mut vars);}
    let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
    let _ = save_solution(&Solution { variables: final_vars });

    Ok(())
}

}
