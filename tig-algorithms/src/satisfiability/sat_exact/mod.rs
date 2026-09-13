mod paged_flip_ranges {
use super::large_buffer::LargeBuf;
// Pre-oriented gaining/losing occurrence slices. A logical flip exchanges
// the two packed descriptors. Counts and iteration order within a slice do not
// change, eliminating repeated polarity-dependent CSR offset selection.
pub(crate) struct FlipRanges {data:LargeBuf<[u64;2]>}
impl FlipRanges {
    pub(crate) fn restore_variables(&self,off:&[u32],mid:&[u32],vars:&mut[bool]){
        assert_eq!(vars.len(),self.data.len());
        for v in 0..vars.len(){
            if off[v+1]!=off[v]{
                // Nonempty positive/negative descriptors are distinct, even
                // when either polarity is empty. Zero-degree variables never
                // flip and keep their initialization value.
                let negative=(mid[v]as u64)|(((off[v+1]-mid[v])as u64)<<32);
                vars[v]=self.data[v][0]==negative;
            }
        }
    }

    pub(crate) fn new(off:&[u32],mid:&[u32],vars:&[bool])->Self {
        let mut data=Vec::with_capacity(vars.len());
        for v in 0..vars.len() {
            let p=(off[v]as u64)|(((mid[v]-off[v])as u64)<<32);
            let n=(mid[v]as u64)|(((off[v+1]-mid[v])as u64)<<32);
            data.push(if vars[v]{[n,p]}else{[p,n]});
        }
        Self{data:super::large_buffer::from_slice(&data)}
    }
    #[inline(always)] pub(crate) unsafe fn flip(&mut self,v:usize)->(usize,usize,usize,usize) {
        let pair=self.data.get_unchecked_mut(v);
        let gain=pair[0];let lose=pair[1];*pair=[lose,gain];
        let is=gain as u32 as usize;let ds=lose as u32 as usize;
        (is,is+(gain>>32)as usize,ds,ds+(lose>>32)as usize)
    }
}

#[cfg(test)]mod snapshot_tests{
 use super::*;
 #[test]fn every_polarity_including_empty_and_unused(){let off=[0u32,0,4,9,16,20];let mid=[0u32,4,4,12,20];for bits in 0..32{let mut wanted:Vec<bool>=(0..5).map(|v|bits&(1<<v)!=0).collect();let mut stored=wanted.clone();let mut r=FlipRanges::new(&off,&mid,&wanted);for step in 0..1000{let v=1+(step*7)%4;unsafe{r.flip(v);}wanted[v]=!wanted[v];if step%13==0{r.restore_variables(&off,&mid,&mut stored);assert_eq!(stored,wanted);}}r.restore_variables(&off,&mid,&mut stored);assert_eq!(stored,wanted);}}
}
}
mod paged_small_order {
// Small-track-only helper; variable IDs fit exactly in u16.
use super::large_buffer::LargeBuf;
// Search reads only variable IDs after exact break caching. Literal signs stay
// in the immutable preprocessing arrays used for initialization/reinitialization.
// The mutable permutation of each clause is represented by three 17-bit IDs and
// a two-bit length. Every baseline swap is mirrored, including across restarts.
pub(crate) struct ClauseOrder { words:LargeBuf<u64> }
impl ClauseOrder {
    pub(crate) fn new(cl:&[i32],co:&[u32])->Self {
        let mut words=Vec::with_capacity(co.len()-1);
        for c in 0..co.len()-1 {
            let lits=&cl[co[c] as usize..co[c+1] as usize];
            assert!(!lits.is_empty() && lits.len()<=3);
            let mut w=(lits.len() as u64)<<48;
            for (j,&l) in lits.iter().enumerate() {
                let v=(l.abs()-1) as u64;
                assert!(v<(1<<16));
                w |= v<<(16*j);
            }
            words.push(w);
        }
        Self{words:super::large_buffer::from_slice(&words)}
    }
    #[inline(always)] pub(crate) unsafe fn variables(&self,c:usize)->[usize;3] {
        let w=*self.words.get_unchecked(c);
        [(w&65535)as usize,((w>>16)&65535)as usize,((w>>32)&65535)as usize]
    }
    // A two-member zero set needs only the original random parity. A
    // singleton is fixed. For all three zero candidates, the ordinal is exactly
    // the random%3 value ALREADY computed for this clause's first-literal swap.
    #[inline(always)] pub(crate) fn choose_zero_rank(v:[usize;3],mask:usize,random:usize,rank:usize)->usize{
        let shift=(mask<<2)|((random&1)<<1);
        let table=((160056576u32>>shift)&3)as usize;
        let k=if mask==7{rank}else{table};
        if k==0{v[0]}else if k==1{v[1]}else{v[2]}
    }
    #[inline(always)] pub(crate) unsafe fn choose_zero(v:[usize;3],mask:usize,random:usize)->usize {
        const COUNTS:[usize;8]=[0,1,1,2,1,2,2,3];
        const ORDER:[[usize;3];8]=[[0,0,0],[0,0,0],[1,0,0],[0,1,0],[2,0,0],[0,2,0],[1,2,0],[0,1,2]];
        let count=*COUNTS.get_unchecked(mask);
        let k=super::exact_div::clause_rem(random,count);
        *v.get_unchecked(*ORDER.get_unchecked(mask).get_unchecked(k))
    }
    #[inline(always)] pub(crate) unsafe fn len_bounded(&self,c:usize)->usize {
        ((*self.words.get_unchecked(c)>>48)&3) as usize
    }
    #[inline(always)] pub(crate) unsafe fn len(&self,c:usize)->usize {
        (*self.words.get_unchecked(c)>>48) as usize
    }
    // Synthetic positive literal: only abs(lit)-1 is observed in the hot search.
    #[inline(always)] pub(crate) unsafe fn lit(&self,c:usize,j:usize)->i32 {
        (((*self.words.get_unchecked(c)>>(16*j))&((1<<16)-1)) as i32)+1
    }
    #[inline(always)] pub(crate) unsafe fn swap(&mut self,c:usize,a:usize,b:usize) {
        let w=self.words.get_unchecked_mut(c);
        let d=((*w>>(16*a))^(*w>>(16*b)))&((1<<16)-1);
        *w ^= (d<<(16*a))|(d<<(16*b));
    }
}

#[cfg(test)]mod rank_tests{
 use super::*;
 #[test]fn exact_zero_choice_all_masks_lengths_and_random_words(){
  let mut r=0x91827364deadbeefusize;let vv=[197,731,19];
  for len in 1..=3{for mask in 1usize..1<<len{for _ in 0..60000{
   r=r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
   assert_eq!(ClauseOrder::choose_zero_rank(vv,mask,r,r%len),unsafe{ClauseOrder::choose_zero(vv,mask,r)});
  }}}
 }
}
}
mod paged_small_signed {
// T3-only helper; signed codes (v<<1)|sign fit exactly in u16.
use super::large_buffer::LargeBuf;
// A chosen clause is unsatisfied. Thus every selected literal is false and
// its sign gives the current assignment exactly: a negative literal implies
// variable=true; a positive one implies variable=false. No truth-array read or
// mutable range orientation is necessary to choose gain/loss occurrence lists.
pub(crate) struct SignedOrder{words:LargeBuf<u64>}
pub(crate) struct LiteralRanges{ranges:LargeBuf<u64>}
impl LiteralRanges{
 pub(crate) fn new(off:&[u32],mid:&[u32])->Self{
  let mut ranges=Vec::with_capacity(2*mid.len());
  for v in 0..mid.len(){
   ranges.push((off[v]as u64)|(((mid[v]-off[v])as u64)<<32));
   ranges.push((mid[v]as u64)|(((off[v+1]-mid[v])as u64)<<32));
  }
  Self{ranges:super::large_buffer::from_slice(&ranges)}
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
   let mut word=(lits.len()as u64)<<48;
   for(j,&l)in lits.iter().enumerate(){let v=(l.abs()-1)as u64;assert!(v<(1<<15));word|=((v<<1)|((l<0)as u64))<<(16*j);}
   words.push(word);
  }
  Self{words:super::large_buffer::from_slice(&words)}
 }
 #[inline(always)]pub(crate) unsafe fn len(&self,c:usize)->usize{(*self.words.get_unchecked(c)>>48)as usize}
 #[inline(always)]pub(crate) unsafe fn len_bounded(&self,c:usize)->usize{((*self.words.get_unchecked(c)>>48)&3)as usize}
 #[inline(always)]pub(crate) unsafe fn codes(&self,c:usize)->[usize;3]{
  let w=*self.words.get_unchecked(c);[(w&65535)as usize,((w>>16)&65535)as usize,((w>>32)&65535)as usize]
 }
 #[inline(always)]pub(crate) unsafe fn lit(&self,c:usize,j:usize)->i32{
  let code=((*self.words.get_unchecked(c)>>(16*j))&65535)as i32;
  let v=(code>>1)+1;if code&1!=0{-v}else{v}
 }
 #[inline(always)]pub(crate) unsafe fn swap(&mut self,c:usize,a:usize,b:usize){
  let w=self.words.get_unchecked_mut(c);let d=((*w>>(16*a))^(*w>>(16*b)))&65535;
  *w^=(d<<(16*a))|(d<<(16*b));
 }
 #[inline(always)]pub(crate) unsafe fn choose_zero(v:[usize;3],mask:usize,r:usize)->usize{
  super::clause_order::ClauseOrder::choose_zero(v,mask,r)
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn signs_order_and_static_range_polarity(){
  let cl=[1,-30000,24567,-4,7,-99];let co=[0,3,5,6];let mut order=SignedOrder::new(&cl,&co);
  let mut expected=vec![vec![1,-30000,24567],vec![-4,7],vec![-99]];
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
mod large_buffer {
// Plain Vec buffers, no standard-library module paths. Every element is written at construction,
// as the original explicit zeroing did. Contents equal an explicitly
// zeroed/copied buffer, so the solver path is unchanged.
pub(crate) type LargeBuf<T>=Vec<T>;
pub(crate) fn zeros<T:Copy+Default>(len:usize)->Vec<T>{let mut buf=Vec::with_capacity(len);buf.resize(len,T::default());buf}
pub(crate) fn from_slice<T:Copy>(values:&[T])->Vec<T>{let mut buf=Vec::with_capacity(values.len());buf.extend_from_slice(values);buf}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn initialized_elements_and_copy(){for n in [0usize,1,513,32768,131072,524289]{let mut a:LargeBuf<u32>=zeros(n);assert_eq!(a.len(),n);assert!(a.iter().all(|&x|x==0));if n>0{for i in 0..n{a[i]=(i as u32).wrapping_mul(17);}let b=from_slice(&a);assert_eq!(a,b);}}}
}
}
mod paged_cache {
use super::large_buffer::LargeBuf;
// REPRESENTATION PROOF: for a set of <=3 distinct variables, two subsets
// with equal cardinality parity can differ only in 0 or 2 variables. XOR of
// two distinct IDs is nonzero, so (parity,XOR) uniquely identifies the subset.
// Let TAG=1<<BITS. Store XOR(true variable IDs) | TAG if count is EVEN,
// and XOR(true variable IDs) otherwise. Empty is exactly TAG; two true IDs
// have nonzero XOR. A flip is therefore ONE XOR by (TAG|variable), no add/sub.
// On gain OLD count is <=2; on loss NEW count is <=2. An odd state then means
// exactly one true variable, whose ID indexes live score bank0. Even states
// index a disjoint unobserved bank1. No score-address XOR or count shift needed.
// Dummy counters wrap by definition; live counts are degree-bounded and exact.
pub(crate) trait CacheOps:Sized{
 fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self;
 fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]);
 unsafe fn good(&self,c:usize)->u8; // SAT predicate, not the literal count
 unsafe fn breaks_of(&self,v:usize)->usize;
 unsafe fn set_break(&mut self,v:usize,n:u32);
 unsafe fn inc(&mut self,c:usize,v:usize)->u8;
 unsafe fn dec(&mut self,c:usize,v:usize)->u8;
 unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4];
 unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4];
 unsafe fn gain_many(&mut self,data:&[u32],v:usize)->u32;
 unsafe fn gain_collect(&mut self,data:&[u32],v:usize,out:*mut u32)->usize;
 unsafe fn lose_many(&mut self,data:&[u32],v:usize,out:*mut u32)->usize;
}
macro_rules! define_cache{($name:ident,$state:ty,$count:ty,$bits:expr)=>{
pub(crate) struct $name{states:LargeBuf<$state>,breaks:LargeBuf<$count>}
impl $name{
 pub(crate) fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self{
  let nc=co.len()-1;assert!(nv<=1<<$bits);
  if <$count>::BITS==8{let mut degree=vec![0u32;nv];for &l in cl{degree[(l.abs()-1)as usize]+=1;}assert!(degree.iter().all(|&d|d<=255));}
  else{assert!(nc<=<$count>::MAX as usize);}
  let mut states=super::large_buffer::zeros::<$state>(nc);let mut breaks=super::large_buffer::zeros::<$count>(2<<$bits);
  for c in 0..nc{
   let mut owner=0usize;let mut n=0usize;
   for &l in &cl[co[c]as usize..co[c+1]as usize]{let v=(l.abs()-1)as usize;if vars[v]==(l>0){owner^=v;n+=1;}}
   states[c]=(owner|(((n&1)^1)<<$bits))as $state;
   if n==1{breaks[owner]+=1;}
  }
  Self{states,breaks}
 }
 pub(crate) fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]){
  for c in 0..co.len()-1{
   let state=self.states[c]as usize;let owner=state&((1<<$bits)-1);
   let lits=&cl[co[c]as usize..co[c+1]as usize];let mut full_xor=0usize;
   for &l in lits{full_xor^=(l.abs()-1)as usize;}
   let n=if state==(1<<$bits){0}else if state&(1<<$bits)!=0{2}else if lits.len()==3 && owner==full_xor{3}else{1};
   for &l in lits{let v=(l.abs()-1)as usize;let truth=if n==0{false}else if n==lits.len(){true}else if n==1{v==owner}else{v!=(full_xor^owner)};vars[v]=truth==(l>0);}
  }
 }
 #[inline(always)]pub(crate) unsafe fn good(&self,c:usize)->u8{(*self.states.get_unchecked(c)!=(1<<$bits))as u8}
 #[inline(always)]pub(crate) unsafe fn breaks_of(&self,v:usize)->usize{*self.breaks.get_unchecked(v)as usize}
 #[inline(always)]pub(crate) unsafe fn set_break(&mut self,v:usize,n:u32){*self.breaks.get_unchecked_mut(v)=n as $count;}
 #[inline(always)]pub(crate) unsafe fn inc(&mut self,c:usize,v:usize)->u8{
  let old=*self.states.get_unchecked(c);*self.states.get_unchecked_mut(c)=old^((v as $state)|(1<<$bits));
  let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  if old==(1<<$bits){0}else if old&(1<<$bits)==0{1}else{2}
 }
 #[inline(always)]pub(crate) unsafe fn dec(&mut self,c:usize,v:usize)->u8{
  let next=*self.states.get_unchecked(c)^((v as $state)|(1<<$bits));*self.states.get_unchecked_mut(c)=next;
  let p=self.breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  if next==(1<<$bits){1}else if next&(1<<$bits)==0{2}else{3}
 }
 #[inline(always)]pub(crate) unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4]{[self.inc(c[0],v),self.inc(c[1],v),self.inc(c[2],v),self.inc(c[3],v)]}
 #[inline(always)]pub(crate) unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4]{[self.dec(c[0],v),self.dec(c[1],v),self.dec(c[2],v),self.dec(c[3],v)]}
 #[inline(always)]pub(crate) unsafe fn gain_many(&mut self,data:&[u32],v:usize)->u32{
  match data.len(){
   0=>Self::gain_0(&mut self.states,&mut self.breaks,data,v),
   1=>Self::gain_1(&mut self.states,&mut self.breaks,data,v),
   2=>Self::gain_2(&mut self.states,&mut self.breaks,data,v),
   3=>Self::gain_3(&mut self.states,&mut self.breaks,data,v),
   4=>Self::gain_4(&mut self.states,&mut self.breaks,data,v),
   5=>Self::gain_5(&mut self.states,&mut self.breaks,data,v),
   6=>Self::gain_6(&mut self.states,&mut self.breaks,data,v),
   7=>Self::gain_7(&mut self.states,&mut self.breaks,data,v),
   8=>Self::gain_8(&mut self.states,&mut self.breaks,data,v),
   9=>Self::gain_9(&mut self.states,&mut self.breaks,data,v),
   10=>Self::gain_10(&mut self.states,&mut self.breaks,data,v),
   11=>Self::gain_11(&mut self.states,&mut self.breaks,data,v),
   12=>Self::gain_12(&mut self.states,&mut self.breaks,data,v),
   13=>Self::gain_13(&mut self.states,&mut self.breaks,data,v),
   14=>Self::gain_14(&mut self.states,&mut self.breaks,data,v),
   15=>Self::gain_15(&mut self.states,&mut self.breaks,data,v),
   16=>Self::gain_16(&mut self.states,&mut self.breaks,data,v),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
    n+=(old==(1<<$bits))as u32;
   }n}
  }
 }
 #[inline(always)]unsafe fn gain_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn gain_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]pub(crate) unsafe fn gain_collect(&mut self,data:&[u32],v:usize,out:*mut u32)->usize{
  match data.len(){
   0=>Self::collect_0(&mut self.states,&mut self.breaks,data,v,out),
   1=>Self::collect_1(&mut self.states,&mut self.breaks,data,v,out),
   2=>Self::collect_2(&mut self.states,&mut self.breaks,data,v,out),
   3=>Self::collect_3(&mut self.states,&mut self.breaks,data,v,out),
   4=>Self::collect_4(&mut self.states,&mut self.breaks,data,v,out),
   5=>Self::collect_5(&mut self.states,&mut self.breaks,data,v,out),
   6=>Self::collect_6(&mut self.states,&mut self.breaks,data,v,out),
   7=>Self::collect_7(&mut self.states,&mut self.breaks,data,v,out),
   8=>Self::collect_8(&mut self.states,&mut self.breaks,data,v,out),
   9=>Self::collect_9(&mut self.states,&mut self.breaks,data,v,out),
   10=>Self::collect_10(&mut self.states,&mut self.breaks,data,v,out),
   11=>Self::collect_11(&mut self.states,&mut self.breaks,data,v,out),
   12=>Self::collect_12(&mut self.states,&mut self.breaks,data,v,out),
   13=>Self::collect_13(&mut self.states,&mut self.breaks,data,v,out),
   14=>Self::collect_14(&mut self.states,&mut self.breaks,data,v,out),
   15=>Self::collect_15(&mut self.states,&mut self.breaks,data,v,out),
   16=>Self::collect_16(&mut self.states,&mut self.breaks,data,v,out),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
    out.add(n).write(c);
    n+=(old==(1<<$bits))as usize;
   }n}
  }
 }
 #[inline(always)]unsafe fn collect_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn collect_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]pub(crate) unsafe fn lose_many(&mut self,data:&[u32],v:usize,out:*mut u32)->usize{
  match data.len(){
   0=>Self::lose_0(&mut self.states,&mut self.breaks,data,v,out),
   1=>Self::lose_1(&mut self.states,&mut self.breaks,data,v,out),
   2=>Self::lose_2(&mut self.states,&mut self.breaks,data,v,out),
   3=>Self::lose_3(&mut self.states,&mut self.breaks,data,v,out),
   4=>Self::lose_4(&mut self.states,&mut self.breaks,data,v,out),
   5=>Self::lose_5(&mut self.states,&mut self.breaks,data,v,out),
   6=>Self::lose_6(&mut self.states,&mut self.breaks,data,v,out),
   7=>Self::lose_7(&mut self.states,&mut self.breaks,data,v,out),
   8=>Self::lose_8(&mut self.states,&mut self.breaks,data,v,out),
   9=>Self::lose_9(&mut self.states,&mut self.breaks,data,v,out),
   10=>Self::lose_10(&mut self.states,&mut self.breaks,data,v,out),
   11=>Self::lose_11(&mut self.states,&mut self.breaks,data,v,out),
   12=>Self::lose_12(&mut self.states,&mut self.breaks,data,v,out),
   13=>Self::lose_13(&mut self.states,&mut self.breaks,data,v,out),
   14=>Self::lose_14(&mut self.states,&mut self.breaks,data,v,out),
   15=>Self::lose_15(&mut self.states,&mut self.breaks,data,v,out),
   16=>Self::lose_16(&mut self.states,&mut self.breaks,data,v,out),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
    out.add(n).write(c);
    n+=(next==(1<<$bits))as usize;
   }n}
  }
 }
 #[inline(always)]unsafe fn lose_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn lose_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
}
impl CacheOps for $name{
 fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self{$name::new(nv,cl,co,vars)}
 fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]){$name::restore_variables(self,cl,co,vars)}
 #[inline(always)]unsafe fn good(&self,c:usize)->u8{$name::good(self,c)}
 #[inline(always)]unsafe fn breaks_of(&self,v:usize)->usize{$name::breaks_of(self,v)}
 #[inline(always)]unsafe fn set_break(&mut self,v:usize,n:u32){$name::set_break(self,v,n)}
 #[inline(always)]unsafe fn inc(&mut self,c:usize,v:usize)->u8{$name::inc(self,c,v)}
 #[inline(always)]unsafe fn dec(&mut self,c:usize,v:usize)->u8{$name::dec(self,c,v)}
 #[inline(always)]unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4]{$name::inc4(self,c,v)}
 #[inline(always)]unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4]{$name::dec4(self,c,v)}
 #[inline(always)]unsafe fn gain_many(&mut self,d:&[u32],v:usize)->u32{$name::gain_many(self,d,v)}
 #[inline(always)]unsafe fn gain_collect(&mut self,d:&[u32],v:usize,o:*mut u32)->usize{$name::gain_collect(self,d,v,o)}
 #[inline(always)]unsafe fn lose_many(&mut self,d:&[u32],v:usize,o:*mut u32)->usize{$name::lose_many(self,d,v,o)}
}
};}
define_cache!(ExactBreakCache,u32,u32,17);
define_cache!(SmallBreakCache,u16,u16,14);
define_cache!(NarrowLargeBreakCache,u32,u8,17);
define_cache!(Byte13BreakCache,u16,u8,13);
define_cache!(Byte14BreakCache,u16,u8,14);
#[cfg(test)]mod tests{
 use super::*;
 fn exercise<C:CacheOps>(nv:usize,cl:&[i32],co:&[u32],initial:&[bool],steps:usize){
  let mut vars=initial.to_vec();let mut cache=C::new(nv,cl,co,&vars);let nc=co.len()-1;
  for step in 0..steps{
   let v=step%nv;let mut gain=Vec::new();let mut loss=Vec::new();let mut eg=Vec::new();let mut el=Vec::new();
   for c in 0..nc{let lits=&cl[co[c]as usize..co[c+1]as usize];let n=lits.iter().filter(|&&l|vars[(l.abs()-1)as usize]==(l>0)).count();
    for &l in lits{if(l.abs()-1)as usize==v{if vars[v]!=(l>0){gain.push(c as u32);if n==0{eg.push(c as u32);}}else{loss.push(c as u32);if n==1{el.push(c as u32);}}}}
   }
   if gain.is_empty()&&loss.is_empty(){continue;}
   let mut gained=vec![u32::MAX;gain.len()+4];let mut lost=vec![u32::MAX;loss.len()+4];
   unsafe{
    let g=if step&1==0{let n=cache.gain_collect(&gain,v,gained.as_mut_ptr());assert_eq!(&gained[..n],eg.as_slice());n as u32}else{let n=cache.gain_many(&gain,v);assert_eq!(n,eg.len()as u32);n};
    let l=cache.lose_many(&loss,v,lost.as_mut_ptr());assert_eq!(&lost[..l],el.as_slice());
    assert!(gained[gain.len()..].iter().all(|&n|n==u32::MAX));assert!(lost[loss.len()..].iter().all(|&n|n==u32::MAX));
    cache.set_break(v,g);
   }
   vars[v]=!vars[v];let mut br=vec![0usize;nv];
   for c in 0..nc{let mut n=0;let mut owner=0;for &l in &cl[co[c]as usize..co[c+1]as usize]{let u=(l.abs()-1)as usize;if vars[u]==(l>0){n+=1;owner=u;}}
    assert_eq!(unsafe{cache.good(c)!=0},n!=0);if n==1{br[owner]+=1;}
   }
   for u in 0..nv{assert_eq!(unsafe{cache.breaks_of(u)},br[u]);}
   let mut restored=initial.to_vec();cache.restore_variables(cl,co,&mut restored);assert_eq!(restored,vars);
  }
 }
 #[test]fn every_small_subset_including_xor_zero_triples(){
  let mut cl=Vec::new();let mut co=vec![0u32];
  for mask in 1usize..16{let vs:Vec<_>=(0..4).filter(|&v|mask&(1<<v)!=0).collect();if vs.len()>3{continue;}
   for bits in 0..1<<vs.len(){for(j,&v)in vs.iter().enumerate(){cl.push(if bits&(1<<j)!=0{(v+1)as i32}else{-((v+1)as i32)});}co.push(cl.len()as u32);}
  }
  for bits in 0..32{let vars:Vec<_>=(0..5).map(|v|bits&(1<<v)!=0).collect();
   exercise::<ExactBreakCache>(5,&cl,&co,&vars,100);exercise::<SmallBreakCache>(5,&cl,&co,&vars,100);
   exercise::<NarrowLargeBreakCache>(5,&cl,&co,&vars,100);exercise::<Byte13BreakCache>(5,&cl,&co,&vars,100);exercise::<Byte14BreakCache>(5,&cl,&co,&vars,100);
  }
 }
 #[test]fn every_kernel_length_and_wide_fallback(){
  for n in 0..=40{let mut cl=Vec::new();let mut co=vec![0u32];for i in 0..n{cl.extend_from_slice(&[if i&1==0{1}else{-1},2,3]);co.push(cl.len()as u32);}
   for bits in 0..8{let vars:Vec<_>=(0..3).map(|v|bits&(1<<v)!=0).collect();exercise::<NarrowLargeBreakCache>(3,&cl,&co,&vars,24);}
  }
  let mut cl=Vec::new();let mut co=vec![0u32];for i in 0..600{cl.push(if i<300{1}else{-1});co.push(cl.len()as u32);}
  exercise::<ExactBreakCache>(1,&cl,&co,&[false],12);exercise::<SmallBreakCache>(1,&cl,&co,&[false],12);
 }
}
}
mod paged_order {
use super::large_buffer::LargeBuf;
// A chosen clause is unsatisfied. Thus every selected literal is false and
// its sign gives the current assignment exactly: a negative literal implies
// variable=true; a positive one implies variable=false. No truth-array read or
// mutable range orientation is necessary to choose gain/loss occurrence lists.
pub(crate) struct SignedOrder{words:LargeBuf<u64>}
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
  Self{words:super::large_buffer::from_slice(&words)}
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
  super::clause_order::ClauseOrder::choose_zero(v,mask,r)
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
mod list_events {
// Every gain event names a DISTINCT currently unsatisfied clause. Removing
// them in supplied incidence order exactly preserves the original swap-removal
// permutation, even when later events are moved by earlier removals. Positions
// are read freshly on each step. Only the Vec length store is delayed to the end.
#[inline(always)]pub(crate) unsafe fn remove_events(list:&mut Vec<u32>,pos:&mut[u32],ids:*const u32,n:usize){
 let old=list.len();debug_assert!(n<=old);let slots=list.as_mut_slice();
 match n{
  0=>remove_0(slots,pos,ids),
  1=>remove_1(slots,pos,ids),
  2=>remove_2(slots,pos,ids),
  3=>remove_3(slots,pos,ids),
  4=>remove_4(slots,pos,ids),
  5=>remove_5(slots,pos,ids),
  6=>remove_6(slots,pos,ids),
  7=>remove_7(slots,pos,ids),
  8=>remove_8(slots,pos,ids),
  _=>{for i in 0..n{let c=*ids.add(i)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-i-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;}}
 }
 list.set_len(old-n);
}
#[inline(always)]pub(crate) unsafe fn insert_events(list:&mut Vec<u32>,pos:&mut[u32],ids:*const u32,n:usize){
 let old=list.len();debug_assert!(old+n<=list.capacity());
 let slots=list.as_mut_ptr().add(old);
 match n{
  0=>insert_0(slots,pos,ids,old),
  1=>insert_1(slots,pos,ids,old),
  2=>insert_2(slots,pos,ids,old),
  3=>insert_3(slots,pos,ids,old),
  4=>insert_4(slots,pos,ids,old),
  5=>insert_5(slots,pos,ids,old),
  6=>insert_6(slots,pos,ids,old),
  7=>insert_7(slots,pos,ids,old),
  8=>insert_8(slots,pos,ids,old),
  _=>{for i in 0..n{let c=*ids.add(i);*pos.get_unchecked_mut(c as usize)=(old+i)as u32;slots.add(i).write(c);}}
 }
 list.set_len(old+n);
}
#[inline(always)]unsafe fn remove_0(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
}
#[inline(always)]unsafe fn insert_0(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
}
#[inline(always)]unsafe fn remove_1(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_1(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
}
#[inline(always)]unsafe fn remove_2(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_2(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
}
#[inline(always)]unsafe fn remove_3(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_3(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
}
#[inline(always)]unsafe fn remove_4(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(3)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-4);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_4(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
 let c=*ids.add(3);*pos.get_unchecked_mut(c as usize)=(old+3)as u32;slots.add(3).write(c);
}
#[inline(always)]unsafe fn remove_5(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(3)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-4);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(4)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-5);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_5(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
 let c=*ids.add(3);*pos.get_unchecked_mut(c as usize)=(old+3)as u32;slots.add(3).write(c);
 let c=*ids.add(4);*pos.get_unchecked_mut(c as usize)=(old+4)as u32;slots.add(4).write(c);
}
#[inline(always)]unsafe fn remove_6(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(3)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-4);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(4)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-5);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(5)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-6);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_6(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
 let c=*ids.add(3);*pos.get_unchecked_mut(c as usize)=(old+3)as u32;slots.add(3).write(c);
 let c=*ids.add(4);*pos.get_unchecked_mut(c as usize)=(old+4)as u32;slots.add(4).write(c);
 let c=*ids.add(5);*pos.get_unchecked_mut(c as usize)=(old+5)as u32;slots.add(5).write(c);
}
#[inline(always)]unsafe fn remove_7(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(3)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-4);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(4)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-5);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(5)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-6);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(6)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-7);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_7(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
 let c=*ids.add(3);*pos.get_unchecked_mut(c as usize)=(old+3)as u32;slots.add(3).write(c);
 let c=*ids.add(4);*pos.get_unchecked_mut(c as usize)=(old+4)as u32;slots.add(4).write(c);
 let c=*ids.add(5);*pos.get_unchecked_mut(c as usize)=(old+5)as u32;slots.add(5).write(c);
 let c=*ids.add(6);*pos.get_unchecked_mut(c as usize)=(old+6)as u32;slots.add(6).write(c);
}
#[inline(always)]unsafe fn remove_8(slots:&mut[u32],pos:&mut[u32],ids:*const u32){
 let old=slots.len();
 let c=*ids.add(0)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-1);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(1)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-2);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(2)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-3);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(3)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-4);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(4)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-5);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(5)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-6);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(6)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-7);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
 let c=*ids.add(7)as usize;let hole=*pos.get_unchecked(c)as usize;let last=*slots.get_unchecked(old-8);*slots.get_unchecked_mut(hole)=last;*pos.get_unchecked_mut(last as usize)=hole as u32;
}
#[inline(always)]unsafe fn insert_8(slots:*mut u32,pos:&mut[u32],ids:*const u32,old:usize){
 let c=*ids.add(0);*pos.get_unchecked_mut(c as usize)=(old+0)as u32;slots.add(0).write(c);
 let c=*ids.add(1);*pos.get_unchecked_mut(c as usize)=(old+1)as u32;slots.add(1).write(c);
 let c=*ids.add(2);*pos.get_unchecked_mut(c as usize)=(old+2)as u32;slots.add(2).write(c);
 let c=*ids.add(3);*pos.get_unchecked_mut(c as usize)=(old+3)as u32;slots.add(3).write(c);
 let c=*ids.add(4);*pos.get_unchecked_mut(c as usize)=(old+4)as u32;slots.add(4).write(c);
 let c=*ids.add(5);*pos.get_unchecked_mut(c as usize)=(old+5)as u32;slots.add(5).write(c);
 let c=*ids.add(6);*pos.get_unchecked_mut(c as usize)=(old+6)as u32;slots.add(6).write(c);
 let c=*ids.add(7);*pos.get_unchecked_mut(c as usize)=(old+7)as u32;slots.add(7).write(c);
}

#[cfg(test)]mod tests{
 use super::*;
 #[test]fn every_small_order_and_reinsert(){
  fn perm(a:&mut[usize],i:usize,f:&mut dyn FnMut(&[usize])){if i==a.len(){f(a);return;}for j in i..a.len(){a.swap(i,j);perm(a,i+1,f);a.swap(i,j);}}
  for size in 1..=7{let mut seq:Vec<_>=(0..size).collect();perm(&mut seq,0,&mut|p|{
   for n in 0..=size{
    let mut a:Vec<u32>=(0..size as u32).collect();let mut expected=a.clone();let mut pos:Vec<u32>=(0..size as u32).collect();let ids:Vec<_>=p[..n].iter().map(|&x|x as u32).collect();
    for &c in &ids{let i=expected.iter().position(|&x|x==c).unwrap();expected.swap_remove(i);}
    unsafe{remove_events(&mut a,&mut pos,ids.as_ptr(),n);}assert_eq!(a,expected);for(i,&c)in a.iter().enumerate(){assert_eq!(pos[c as usize],i as u32);}
    a.reserve(n);unsafe{insert_events(&mut a,&mut pos,ids.as_ptr(),n);}expected.extend_from_slice(&ids);assert_eq!(a,expected);for(i,&c)in a.iter().enumerate(){assert_eq!(pos[c as usize],i as u32);}
   }
  });}
 }
 #[test]fn longer_fallback_lengths(){for size in 0..100{for n in 0..=size{let mut a:Vec<u32>=(0..size as u32).collect();let mut expected=a.clone();let mut pos=a.clone();let ids:Vec<_>=(0..n as u32).rev().collect();for &c in &ids{let i=expected.iter().position(|&x|x==c).unwrap();expected.swap_remove(i);}unsafe{remove_events(&mut a,&mut pos,ids.as_ptr(),n);}assert_eq!(a,expected);a.reserve(n);unsafe{insert_events(&mut a,&mut pos,ids.as_ptr(),n);}expected.extend_from_slice(&ids);assert_eq!(a,expected);}}}
}
}
mod one_load_ranges {
use super::signed_order::LiteralRanges;
pub(crate) trait Ranges:Sized{fn new(off:&[u32],mid:&[u32])->Self;unsafe fn of(&self,code:usize)->(usize,usize,usize,usize);}
impl Ranges for LiteralRanges{
 fn new(off:&[u32],mid:&[u32])->Self{LiteralRanges::new(off,mid)}
 #[inline(always)]unsafe fn of(&self,c:usize)->(usize,usize,usize,usize){LiteralRanges::of(self,c)}
}
pub(crate) struct PackedRanges{words:Vec<u64>}
impl Ranges for PackedRanges{
 fn new(off:&[u32],mid:&[u32])->Self{
  let mut words=Vec::with_capacity(2*mid.len());assert!(off[mid.len()]<(1<<21));
  for v in 0..mid.len(){let a=off[v];let b=mid[v];let p=b-a;let n=off[v+1]-b;assert!(p<=255&&n<=255);
   words.push((a as u64)|((b as u64)<<21)|((p as u64)<<42)|((n as u64)<<50));
   words.push((b as u64)|((a as u64)<<21)|((n as u64)<<42)|((p as u64)<<50));
  }Self{words}
 }
 #[inline(always)]unsafe fn of(&self,c:usize)->(usize,usize,usize,usize){
  let w=*self.words.get_unchecked(c);let a=(w&2097151)as usize;let b=((w>>21)&2097151)as usize;
  (a,a+((w>>42)&255)as usize,b,b+(w>>50)as usize)
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn all_count_pairs_and_signs(){for p in 0u32..=255{for n in 0u32..=255{
  let off=[1500000,1500000+p+n];let mid=[1500000+p];let packed=PackedRanges::new(&off,&mid);let base=LiteralRanges::new(&off,&mid);
  unsafe{assert_eq!(packed.of(0),base.of(0));assert_eq!(packed.of(1),base.of(1));}
 }}}
}
}
mod short_mask {
// Retained short clauses have <=64 distinct participating variables.
// Map those variables to dense mask codes; code zero toggles nothing. On the
// guarded path only a real short-variable flip updates the cached predicate.
// For >32 short clauses the baseline fallback stays selected permanently.
pub(crate) struct ShortMask{codes:Vec<u8>,masks:Vec<u64>,truth:u64,lanes:u64,enabled:bool,satisfied:bool}
impl ShortMask{
 pub(crate) fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self{
  let short:Vec<_>=(0..co.len()-1).filter(|&c|co[c+1]-co[c]<3).collect();let enabled=short.len()<=32;
  let mut toggle=vec![0u64;nv];let mut truth=0u64;let mut lanes=0u64;
  if enabled{for(i,&c)in short.iter().enumerate(){lanes|=1u64<<(2*i);for(j,&l)in cl[co[c]as usize..co[c+1]as usize].iter().enumerate(){let v=(l.abs()-1)as usize;let bit=1u64<<(2*i+j);toggle[v]|=bit;if vars[v]==(l>0){truth|=bit;}}}}
  let mut codes=vec![0u8;nv];let mut masks=vec![0u64];for(v,&mask)in toggle.iter().enumerate(){if mask!=0{assert!(masks.len()<=64);codes[v]=masks.len()as u8;masks.push(mask);}}
  let satisfied=enabled && ((truth|(truth>>1))&lanes)==lanes;
  Self{codes,masks,truth,lanes,enabled,satisfied}
 }
 #[inline(always)]pub(crate) fn all_satisfied(&self)->bool{self.satisfied}
 #[inline(always)]pub(crate) unsafe fn flip(&mut self,v:usize){
  let code=*self.codes.get_unchecked(v)as usize;
  if code!=0{self.truth^=*self.masks.get_unchecked(code);self.satisfied=self.enabled && ((self.truth|(self.truth>>1))&self.lanes)==self.lanes;}
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn all_signs_truth_states_and_lane_boundary(){
  for size in 0..=33{let mut cl=Vec::new();let mut co=vec![0u32];for i in 0..size{cl.push(if i&1!=0{1}else{-1});if i%3!=0{cl.push(if i&2!=0{2}else{-2});}co.push(cl.len()as u32);}
   for bits in 0..4{let mut vars=vec![bits&1!=0,bits&2!=0];let mut m=ShortMask::new(2,&cl,&co,&vars);
    for step in 0..100{let expected=(0..size).all(|c|cl[co[c]as usize..co[c+1]as usize].iter().any(|&l|vars[(l.abs()-1)as usize]==(l>0)));
     assert_eq!(m.all_satisfied(),size<=32&&expected);let v=step%2;unsafe{m.flip(v);}vars[v]=!vars[v];
    }
   }
  }
 }
}
}
mod subset_cache {
// REPRESENTATION PROOF: for a set of <=3 distinct variables, two subsets
// with equal cardinality parity can differ only in 0 or 2 variables. XOR of
// two distinct IDs is nonzero, so (parity,XOR) uniquely identifies the subset.
// Let TAG=1<<BITS. Store XOR(true variable IDs) | TAG if count is EVEN,
// and XOR(true variable IDs) otherwise. Empty is exactly TAG; two true IDs
// have nonzero XOR. A flip is therefore ONE XOR by (TAG|variable), no add/sub.
// On gain OLD count is <=2; on loss NEW count is <=2. An odd state then means
// exactly one true variable, whose ID indexes live score bank0. Even states
// index a disjoint unobserved bank1. No score-address XOR or count shift needed.
// Dummy counters wrap by definition; live counts are degree-bounded and exact.
pub(crate) trait CacheOps:Sized{
 fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self;
 fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]);
 unsafe fn good(&self,c:usize)->u8; // SAT predicate, not the literal count
 unsafe fn breaks_of(&self,v:usize)->usize;
 unsafe fn set_break(&mut self,v:usize,n:u32);
 unsafe fn inc(&mut self,c:usize,v:usize)->u8;
 unsafe fn dec(&mut self,c:usize,v:usize)->u8;
 unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4];
 unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4];
 unsafe fn gain_many(&mut self,data:&[u32],v:usize)->u32;
 unsafe fn gain_collect(&mut self,data:&[u32],v:usize,out:*mut u32)->usize;
 unsafe fn lose_many(&mut self,data:&[u32],v:usize,out:*mut u32)->usize;
}
macro_rules! define_cache{($name:ident,$state:ty,$count:ty,$bits:expr)=>{
pub(crate) struct $name{states:Vec<$state>,breaks:Vec<$count>}
impl $name{
 pub(crate) fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self{
  let nc=co.len()-1;assert!(nv<=1<<$bits);
  if <$count>::BITS==8{let mut degree=vec![0u32;nv];for &l in cl{degree[(l.abs()-1)as usize]+=1;}assert!(degree.iter().all(|&d|d<=255));}
  else{assert!(nc<=<$count>::MAX as usize);}
  let mut states=vec![0 as $state;nc];let mut breaks=vec![0 as $count;2<<$bits];
  for c in 0..nc{
   let mut owner=0usize;let mut n=0usize;
   for &l in &cl[co[c]as usize..co[c+1]as usize]{let v=(l.abs()-1)as usize;if vars[v]==(l>0){owner^=v;n+=1;}}
   states[c]=(owner|(((n&1)^1)<<$bits))as $state;
   if n==1{breaks[owner]+=1;}
  }
  Self{states,breaks}
 }
 pub(crate) fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]){
  for c in 0..co.len()-1{
   let state=self.states[c]as usize;let owner=state&((1<<$bits)-1);
   let lits=&cl[co[c]as usize..co[c+1]as usize];let mut full_xor=0usize;
   for &l in lits{full_xor^=(l.abs()-1)as usize;}
   let n=if state==(1<<$bits){0}else if state&(1<<$bits)!=0{2}else if lits.len()==3 && owner==full_xor{3}else{1};
   for &l in lits{let v=(l.abs()-1)as usize;let truth=if n==0{false}else if n==lits.len(){true}else if n==1{v==owner}else{v!=(full_xor^owner)};vars[v]=truth==(l>0);}
  }
 }
 #[inline(always)]pub(crate) unsafe fn good(&self,c:usize)->u8{(*self.states.get_unchecked(c)!=(1<<$bits))as u8}
 #[inline(always)]pub(crate) unsafe fn breaks_of(&self,v:usize)->usize{*self.breaks.get_unchecked(v)as usize}
 #[inline(always)]pub(crate) unsafe fn set_break(&mut self,v:usize,n:u32){*self.breaks.get_unchecked_mut(v)=n as $count;}
 #[inline(always)]pub(crate) unsafe fn inc(&mut self,c:usize,v:usize)->u8{
  let old=*self.states.get_unchecked(c);*self.states.get_unchecked_mut(c)=old^((v as $state)|(1<<$bits));
  let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  if old==(1<<$bits){0}else if old&(1<<$bits)==0{1}else{2}
 }
 #[inline(always)]pub(crate) unsafe fn dec(&mut self,c:usize,v:usize)->u8{
  let next=*self.states.get_unchecked(c)^((v as $state)|(1<<$bits));*self.states.get_unchecked_mut(c)=next;
  let p=self.breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  if next==(1<<$bits){1}else if next&(1<<$bits)==0{2}else{3}
 }
 #[inline(always)]pub(crate) unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4]{[self.inc(c[0],v),self.inc(c[1],v),self.inc(c[2],v),self.inc(c[3],v)]}
 #[inline(always)]pub(crate) unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4]{[self.dec(c[0],v),self.dec(c[1],v),self.dec(c[2],v),self.dec(c[3],v)]}
 #[inline(always)]pub(crate) unsafe fn gain_many(&mut self,data:&[u32],v:usize)->u32{
  match data.len(){
   0=>Self::gain_0(&mut self.states,&mut self.breaks,data,v),
   1=>Self::gain_1(&mut self.states,&mut self.breaks,data,v),
   2=>Self::gain_2(&mut self.states,&mut self.breaks,data,v),
   3=>Self::gain_3(&mut self.states,&mut self.breaks,data,v),
   4=>Self::gain_4(&mut self.states,&mut self.breaks,data,v),
   5=>Self::gain_5(&mut self.states,&mut self.breaks,data,v),
   6=>Self::gain_6(&mut self.states,&mut self.breaks,data,v),
   7=>Self::gain_7(&mut self.states,&mut self.breaks,data,v),
   8=>Self::gain_8(&mut self.states,&mut self.breaks,data,v),
   9=>Self::gain_9(&mut self.states,&mut self.breaks,data,v),
   10=>Self::gain_10(&mut self.states,&mut self.breaks,data,v),
   11=>Self::gain_11(&mut self.states,&mut self.breaks,data,v),
   12=>Self::gain_12(&mut self.states,&mut self.breaks,data,v),
   13=>Self::gain_13(&mut self.states,&mut self.breaks,data,v),
   14=>Self::gain_14(&mut self.states,&mut self.breaks,data,v),
   15=>Self::gain_15(&mut self.states,&mut self.breaks,data,v),
   16=>Self::gain_16(&mut self.states,&mut self.breaks,data,v),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
    n+=(old==(1<<$bits))as u32;
   }n}
  }
 }
 #[inline(always)]unsafe fn gain_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn gain_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]unsafe fn gain_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize)->u32{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  n+=(old==(1<<$bits))as u32;
  n
 }
 #[inline(always)]pub(crate) unsafe fn gain_collect(&mut self,data:&[u32],v:usize,out:*mut u32)->usize{
  match data.len(){
   0=>Self::collect_0(&mut self.states,&mut self.breaks,data,v,out),
   1=>Self::collect_1(&mut self.states,&mut self.breaks,data,v,out),
   2=>Self::collect_2(&mut self.states,&mut self.breaks,data,v,out),
   3=>Self::collect_3(&mut self.states,&mut self.breaks,data,v,out),
   4=>Self::collect_4(&mut self.states,&mut self.breaks,data,v,out),
   5=>Self::collect_5(&mut self.states,&mut self.breaks,data,v,out),
   6=>Self::collect_6(&mut self.states,&mut self.breaks,data,v,out),
   7=>Self::collect_7(&mut self.states,&mut self.breaks,data,v,out),
   8=>Self::collect_8(&mut self.states,&mut self.breaks,data,v,out),
   9=>Self::collect_9(&mut self.states,&mut self.breaks,data,v,out),
   10=>Self::collect_10(&mut self.states,&mut self.breaks,data,v,out),
   11=>Self::collect_11(&mut self.states,&mut self.breaks,data,v,out),
   12=>Self::collect_12(&mut self.states,&mut self.breaks,data,v,out),
   13=>Self::collect_13(&mut self.states,&mut self.breaks,data,v,out),
   14=>Self::collect_14(&mut self.states,&mut self.breaks,data,v,out),
   15=>Self::collect_15(&mut self.states,&mut self.breaks,data,v,out),
   16=>Self::collect_16(&mut self.states,&mut self.breaks,data,v,out),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
    out.add(n).write(c);
    n+=(old==(1<<$bits))as usize;
   }n}
  }
 }
 #[inline(always)]unsafe fn collect_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn collect_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn collect_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(old as usize);*p=p.wrapping_sub(1);
  out.add(n).write(c);
  n+=(old==(1<<$bits))as usize;
  n
 }
 #[inline(always)]pub(crate) unsafe fn lose_many(&mut self,data:&[u32],v:usize,out:*mut u32)->usize{
  match data.len(){
   0=>Self::lose_0(&mut self.states,&mut self.breaks,data,v,out),
   1=>Self::lose_1(&mut self.states,&mut self.breaks,data,v,out),
   2=>Self::lose_2(&mut self.states,&mut self.breaks,data,v,out),
   3=>Self::lose_3(&mut self.states,&mut self.breaks,data,v,out),
   4=>Self::lose_4(&mut self.states,&mut self.breaks,data,v,out),
   5=>Self::lose_5(&mut self.states,&mut self.breaks,data,v,out),
   6=>Self::lose_6(&mut self.states,&mut self.breaks,data,v,out),
   7=>Self::lose_7(&mut self.states,&mut self.breaks,data,v,out),
   8=>Self::lose_8(&mut self.states,&mut self.breaks,data,v,out),
   9=>Self::lose_9(&mut self.states,&mut self.breaks,data,v,out),
   10=>Self::lose_10(&mut self.states,&mut self.breaks,data,v,out),
   11=>Self::lose_11(&mut self.states,&mut self.breaks,data,v,out),
   12=>Self::lose_12(&mut self.states,&mut self.breaks,data,v,out),
   13=>Self::lose_13(&mut self.states,&mut self.breaks,data,v,out),
   14=>Self::lose_14(&mut self.states,&mut self.breaks,data,v,out),
   15=>Self::lose_15(&mut self.states,&mut self.breaks,data,v,out),
   16=>Self::lose_16(&mut self.states,&mut self.breaks,data,v,out),
   _=>{let code=(v as $state)|(1<<$bits);let mut n=0;for &c in data{
    let old=*self.states.get_unchecked(c as usize);let next=old^code;*self.states.get_unchecked_mut(c as usize)=next;
    let p=self.breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
    out.add(n).write(c);
    n+=(next==(1<<$bits))as usize;
   }n}
  }
 }
 #[inline(always)]unsafe fn lose_0(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  n
 }
 #[inline(always)]unsafe fn lose_1(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_2(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_3(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_4(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_5(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_6(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_7(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_8(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_9(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_10(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_11(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_12(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_13(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_14(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_15(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
 #[inline(always)]unsafe fn lose_16(states:&mut[$state],breaks:&mut[$count],data:&[u32],v:usize,out:*mut u32)->usize{
  let code=(v as $state)|(1<<$bits);let mut n=0;
  let c=*data.get_unchecked(0);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(1);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(2);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(3);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(4);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(5);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(6);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(7);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(8);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(9);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(10);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(11);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(12);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(13);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(14);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  let c=*data.get_unchecked(15);
    let old=*states.get_unchecked(c as usize);let next=old^code;*states.get_unchecked_mut(c as usize)=next;
    let p=breaks.get_unchecked_mut(next as usize);*p=p.wrapping_add(1);
  out.add(n).write(c);
  n+=(next==(1<<$bits))as usize;
  n
 }
}
impl CacheOps for $name{
 fn new(nv:usize,cl:&[i32],co:&[u32],vars:&[bool])->Self{$name::new(nv,cl,co,vars)}
 fn restore_variables(&self,cl:&[i32],co:&[u32],vars:&mut[bool]){$name::restore_variables(self,cl,co,vars)}
 #[inline(always)]unsafe fn good(&self,c:usize)->u8{$name::good(self,c)}
 #[inline(always)]unsafe fn breaks_of(&self,v:usize)->usize{$name::breaks_of(self,v)}
 #[inline(always)]unsafe fn set_break(&mut self,v:usize,n:u32){$name::set_break(self,v,n)}
 #[inline(always)]unsafe fn inc(&mut self,c:usize,v:usize)->u8{$name::inc(self,c,v)}
 #[inline(always)]unsafe fn dec(&mut self,c:usize,v:usize)->u8{$name::dec(self,c,v)}
 #[inline(always)]unsafe fn inc4(&mut self,c:[usize;4],v:usize)->[u8;4]{$name::inc4(self,c,v)}
 #[inline(always)]unsafe fn dec4(&mut self,c:[usize;4],v:usize)->[u8;4]{$name::dec4(self,c,v)}
 #[inline(always)]unsafe fn gain_many(&mut self,d:&[u32],v:usize)->u32{$name::gain_many(self,d,v)}
 #[inline(always)]unsafe fn gain_collect(&mut self,d:&[u32],v:usize,o:*mut u32)->usize{$name::gain_collect(self,d,v,o)}
 #[inline(always)]unsafe fn lose_many(&mut self,d:&[u32],v:usize,o:*mut u32)->usize{$name::lose_many(self,d,v,o)}
}
};}
define_cache!(ExactBreakCache,u32,u32,17);
define_cache!(SmallBreakCache,u16,u16,14);
define_cache!(NarrowLargeBreakCache,u32,u8,17);
define_cache!(Byte13BreakCache,u16,u8,13);
define_cache!(Byte14BreakCache,u16,u8,14);
#[cfg(test)]mod tests{
 use super::*;
 fn exercise<C:CacheOps>(nv:usize,cl:&[i32],co:&[u32],initial:&[bool],steps:usize){
  let mut vars=initial.to_vec();let mut cache=C::new(nv,cl,co,&vars);let nc=co.len()-1;
  for step in 0..steps{
   let v=step%nv;let mut gain=Vec::new();let mut loss=Vec::new();let mut eg=Vec::new();let mut el=Vec::new();
   for c in 0..nc{let lits=&cl[co[c]as usize..co[c+1]as usize];let n=lits.iter().filter(|&&l|vars[(l.abs()-1)as usize]==(l>0)).count();
    for &l in lits{if(l.abs()-1)as usize==v{if vars[v]!=(l>0){gain.push(c as u32);if n==0{eg.push(c as u32);}}else{loss.push(c as u32);if n==1{el.push(c as u32);}}}}
   }
   if gain.is_empty()&&loss.is_empty(){continue;}
   let mut gained=vec![u32::MAX;gain.len()+4];let mut lost=vec![u32::MAX;loss.len()+4];
   unsafe{
    let g=if step&1==0{let n=cache.gain_collect(&gain,v,gained.as_mut_ptr());assert_eq!(&gained[..n],eg.as_slice());n as u32}else{let n=cache.gain_many(&gain,v);assert_eq!(n,eg.len()as u32);n};
    let l=cache.lose_many(&loss,v,lost.as_mut_ptr());assert_eq!(&lost[..l],el.as_slice());
    assert!(gained[gain.len()..].iter().all(|&n|n==u32::MAX));assert!(lost[loss.len()..].iter().all(|&n|n==u32::MAX));
    cache.set_break(v,g);
   }
   vars[v]=!vars[v];let mut br=vec![0usize;nv];
   for c in 0..nc{let mut n=0;let mut owner=0;for &l in &cl[co[c]as usize..co[c+1]as usize]{let u=(l.abs()-1)as usize;if vars[u]==(l>0){n+=1;owner=u;}}
    assert_eq!(unsafe{cache.good(c)!=0},n!=0);if n==1{br[owner]+=1;}
   }
   for u in 0..nv{assert_eq!(unsafe{cache.breaks_of(u)},br[u]);}
   let mut restored=initial.to_vec();cache.restore_variables(cl,co,&mut restored);assert_eq!(restored,vars);
  }
 }
 #[test]fn every_small_subset_including_xor_zero_triples(){
  let mut cl=Vec::new();let mut co=vec![0u32];
  for mask in 1usize..16{let vs:Vec<_>=(0..4).filter(|&v|mask&(1<<v)!=0).collect();if vs.len()>3{continue;}
   for bits in 0..1<<vs.len(){for(j,&v)in vs.iter().enumerate(){cl.push(if bits&(1<<j)!=0{(v+1)as i32}else{-((v+1)as i32)});}co.push(cl.len()as u32);}
  }
  for bits in 0..32{let vars:Vec<_>=(0..5).map(|v|bits&(1<<v)!=0).collect();
   exercise::<ExactBreakCache>(5,&cl,&co,&vars,100);exercise::<SmallBreakCache>(5,&cl,&co,&vars,100);
   exercise::<NarrowLargeBreakCache>(5,&cl,&co,&vars,100);exercise::<Byte13BreakCache>(5,&cl,&co,&vars,100);exercise::<Byte14BreakCache>(5,&cl,&co,&vars,100);
  }
 }
 #[test]fn every_kernel_length_and_wide_fallback(){
  for n in 0..=40{let mut cl=Vec::new();let mut co=vec![0u32];for i in 0..n{cl.extend_from_slice(&[if i&1==0{1}else{-1},2,3]);co.push(cl.len()as u32);}
   for bits in 0..8{let vars:Vec<_>=(0..3).map(|v|bits&(1<<v)!=0).collect();exercise::<NarrowLargeBreakCache>(3,&cl,&co,&vars,24);}
  }
  let mut cl=Vec::new();let mut co=vec![0u32];for i in 0..600{cl.push(if i<300{1}else{-1});co.push(cl.len()as u32);}
  exercise::<ExactBreakCache>(1,&cl,&co,&[false],12);exercise::<SmallBreakCache>(1,&cl,&co,&[false],12);
 }
}
}
mod explicit_roulette {
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
// Boundaries are derived from the ORIGINAL additive floating comparisons on
// the exact 53-bit rand grid. Do not substitute subtractive roulette: its
// rounding boundaries need not agree with this track's accum>=threshold rule.
const D:usize=8;const GRID:u64=1u64<<53;
pub(crate) struct AdditiveRoulette{cuts:Vec<[u64;2]>}
impl AdditiveRoulette{
 pub(crate) fn new(weights:&[f64])->Self{
  assert!(weights.len()>=D);
  // Solver weights are fixed nonnegative finite powers. Preserve the exact
  // first two boundaries; total is accumulated in the original order.
  assert!(weights[..D].iter().all(|&w|w>=0.0&&w.is_finite()));
  let mut cuts=vec![[0u64;2];D*D*D];
  for a in 0..D{for b in 0..D{for c in 0..D{
   let w=[weights[a],weights[b],weights[c]];let mut total=0.0;for &x in &w{total+=x;}
   assert!(total.is_finite());
   let mut accum=0.0;
   for stage in 0..2{
    accum+=w[stage];let(mut lo,mut hi)=(0u64,GRID);
    while lo<hi{let mid=lo+(hi-lo)/2;let threshold=(mid as f64)*(1.0/(GRID as f64))*total;
     if accum>=threshold{lo=mid+1;}else{hi=mid;}}
    assert!(lo>0);cuts[(a*D+b)*D+c][stage]=(((lo as u128)<<11)-1)as u64;
   }
  }}}
  Self{cuts}
 }
 #[inline(always)]pub(crate) fn choose(&self,a:usize,b:usize,c:usize,word:u64)->usize{
  debug_assert!((a|b|c)<D);let cc=unsafe{*self.cuts.get_unchecked((a*D+b)*D+c)};
  // The last accumulated prefix is bit-identical to total. Since
  // 0<=sample<1 and total is finite/nonnegative, rounded sample*total<=total.
  // Therefore the third comparison always succeeds; the fallback is unreachable.
  (word>cc[0])as usize+(word>cc[1])as usize
 }
 #[inline(always)]pub(crate) fn sample(word:u64)->f64{((word>>11)as f64)*(1.0/(GRID as f64))}
}
#[cfg(test)]mod tests{
 use super::*;use rand::{Rng,SeedableRng,rngs::SmallRng};
 #[test]fn exact_integer_weights_all_positive_triples_and_edges(){
  let mut rng=SmallRng::seed_from_u64(94756);
  for a in 1..=16{for b in 1..=16{for c in 1..=16{
   let w=[W[a.min(15)],W[b.min(15)],W[c.min(15)]];let total=w[0]+w[1]+w[2];
   let mut words=vec![0,1,u32::MAX,u32::MAX-1,total,total-1,w[0]-1,w[0],w[0]+w[1]-1,w[0]+w[1]];
   for _ in 0..64{words.push(rng.gen());}
   for r in words{let x=r%total;let old=if x<w[0]{0}else if x-w[0]<w[1]{1}else{2};assert_eq!(integer_three(a,b,c,r),old);}
  }}}
 }
 fn check(weights:&[f64]){
  let t=AdditiveRoulette::new(weights);let mut rng=SmallRng::seed_from_u64(818823);
  for a in 0..D{for b in 0..D{for c in 0..D{
   let w=[weights[a],weights[b],weights[c]];let mut total=0.0;for &x in &w{total+=x;}
   let mut words=vec![0,1,u64::MAX];
   for &cut in &t.cuts[(a*D+b)*D+c]{for d in [0,1,2,2047,2048,2049]{words.push(cut.saturating_sub(d));words.push(cut.saturating_add(d));}}
   for _ in 0..64{words.push(rng.gen());}
   for word in words{let threshold=AdditiveRoulette::sample(word)*total;let mut accum=0.0;let mut old=0;
    for j in 0..3{accum+=w[j];if accum>=threshold{old=j;break;}}assert_eq!(t.choose(a,b,c,word),old);}
  }}}
 }
 #[test]fn exact_additive_rounding_edges(){check(&(0..8).map(|i|2.06f64.powf(-(i as f64))).collect::<Vec<_>>());check(&[0.0,f64::from_bits(1),f64::MIN_POSITIVE,1e-250,1e-20,0.001,1.0,2.0]);}
 #[test]fn same_random_word_consumption(){let mut a=SmallRng::seed_from_u64(93425);let mut b=a.clone();for _ in 0..100000{assert_eq!(AdditiveRoulette::sample(a.gen()).to_bits(),b.gen::<f64>().to_bits());}}
}
}
mod phase_div {
// For every positive d, M=floor((2^64-1)/d) fits u64, including d=1.
// q=floor(x*M/2^64) underestimates floor(x/d) by at most one; hence a single
// correction returns x%d exactly. The caller proves 1<=d<LIMIT for the entire
// phase: residual length never exceeds its initial value plus reserved appends.
pub(crate) const LIMIT:usize=65536;
const fn build()->[u64;LIMIT]{let mut out=[0u64;LIMIT];let mut d=1;while d<LIMIT{out[d]=u64::MAX/(d as u64);d+=1;}out}
static M:[u64;LIMIT]=build();
#[inline(always)]pub(crate) unsafe fn rem_nonzero_small(x:usize,d:usize)->usize{
 debug_assert!(d>0 && d<LIMIT);
 let m=*M.get_unchecked(d);
 let q=(((x as u128)*(m as u128))>>64)as usize;
 let r=x-q*d;
 if r>=d{r-d}else{r}
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn all_divisors_and_boundary_random_words(){
  let mut x=0x37a01abc76543210usize;
  for d in 1..LIMIT{
   for v in [0,1,d-1,d,d+1,2*d-1,2*d,usize::MAX,usize::MAX-1,1usize<<63]{assert_eq!(unsafe{rem_nonzero_small(v,d)},v%d);}
   for _ in 0..48{x^=x<<13;x^=x>>7;x^=x<<17;assert_eq!(unsafe{rem_nonzero_small(x,d)},x%d);}
  }
 }
}
}
mod compact_ranges {
// One word holds the start and both signed counts when each fits u16.
// COMPACT is selected only under the existing <=255 degree guard. The general
// representation retains two full u32 start/length words for every other case.
pub(crate) struct CompactRanges<const COMPACT:bool>{data:Vec<u64>}
impl<const COMPACT:bool> CompactRanges<COMPACT>{
 pub(crate) fn new(off:&[u32],mid:&[u32])->Self{
  let mut data=Vec::with_capacity(mid.len()*if COMPACT{1}else{2});
  for v in 0..mid.len(){
   let p=mid[v]-off[v];let n=off[v+1]-mid[v];
   if COMPACT{assert!(p<=65535&&n<=65535);data.push((off[v]as u64)|((p as u64)<<32)|((n as u64)<<48));}
   else{data.push((off[v]as u64)|((p as u64)<<32));data.push((mid[v]as u64)|((n as u64)<<32));}
  }Self{data}
 }
 #[inline(always)]pub(crate) unsafe fn of(&self,code:usize)->(usize,usize,usize,usize){
  if COMPACT{
   let w=*self.data.get_unchecked(code>>1);let start=w as u32 as usize;let mid=start+(((w>>32)&65535)as usize);let end=mid+((w>>48)as usize);
   if code&1==0{(start,mid,mid,end)}else{(mid,end,start,mid)}
  }else{
   let a=*self.data.get_unchecked(code);let b=*self.data.get_unchecked(code^1);let i=a as u32 as usize;let j=b as u32 as usize;
   (i,i+(a>>32)as usize,j,j+(b>>32)as usize)
  }
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn ranges_at_all_count_boundaries_and_polarities(){
  for p in [0u32,1,2,16,255,256,65534,65535]{for n in [0u32,1,2,16,255,256,65534,65535]{
   let off=[17u32,17+p+n];let mid=[17+p];let a=CompactRanges::<true>::new(&off,&mid);let b=CompactRanges::<false>::new(&off,&mid);
   for c in 0..2{assert_eq!(unsafe{a.of(c)},unsafe{b.of(c)});}
  }}
  let off=[0u32,200000,800000];let mid=[150000u32,300000];let b=CompactRanges::<false>::new(&off,&mid);
  assert_eq!(unsafe{b.of(0)},(0,150000,150000,200000));assert_eq!(unsafe{b.of(3)},(300000,800000,200000,300000));
 }
}
}
mod exact_coin {
use rand::Rng;
// Standard<f64> consumes one u64 and returns its upper 53 bits times 2^-53.
// sample<p iff raw_word <= (ceil(p*2^53)<<11)-1. Handle empty/full intervals
// explicitly, and consume the same word even when the event is impossible.
pub(crate) struct ExactCoin {last:u64,enabled:bool}
impl ExactCoin {
 pub(crate) fn new(p:f64)->Self {
  if !(p>0.0){return Self{last:0,enabled:false};}
  if p>=1.0{return Self{last:u64::MAX,enabled:true};}
  let k=(p*9007199254740992.0).ceil()as u64;
  Self{last:(((k as u128)<<11)-1)as u64,enabled:true}
 }
 #[inline(always)]pub(crate) fn draw<R:Rng+?Sized>(&self,rng:&mut R)->bool{
  let word=rng.gen::<u64>();self.enabled & (word<=self.last)
 }
 #[cfg(test)]fn test_word(&self,word:u64)->bool{self.enabled & (word<=self.last)}
}
#[cfg(test)]mod tests{
 use super::*;use rand::{SeedableRng,rngs::SmallRng};
 #[test]fn all_boundary_neighbors_and_special_probabilities(){
  let mut probs=vec![f64::NAN,f64::NEG_INFINITY,-1.0,-0.0,0.0,f64::from_bits(1),1e-100,1.0/9007199254740992.0,0.003,0.45,0.52,0.9,1.0,2.0,f64::INFINITY];
  let mut rng=SmallRng::seed_from_u64(98364571);
  for _ in 0..10000{probs.push(rng.gen::<f64>());}
  for p in probs{
   let coin=ExactCoin::new(p);let mut words=vec![0,1,u64::MAX,coin.last,coin.last.saturating_sub(1),coin.last.saturating_add(1)];
   for d in [2047,2048,2049]{words.push(coin.last.saturating_add(d));words.push(coin.last.saturating_sub(d));}
   for _ in 0..12{words.push(rng.gen::<u64>());}
   for w in words{let sample=((w>>11)as f64)*(1.0/9007199254740992.0);assert_eq!(coin.test_word(w),sample<p,"p={p:?},word={w}");}
   let mut a=rng.clone();let mut b=rng.clone();assert_eq!(coin.draw(&mut a),b.gen::<f64>()<p);assert_eq!(a.gen::<u64>(),b.gen::<u64>());
  }
 }
}
}
mod raw_roulette {
// rand 0.8 Standard<f64> uses the upper 53 bits of one next_u64 word.
// Prefix-hit predicates are monotone over that exact grid. Store the inclusive
// last accepted RAW WORD, including all 11 ignored low bits. GRID maps safely
// to u64::MAX through u128 arithmetic; even zero weights accept grid point zero.
const D:usize=8;
const GRID:u64=1u64<<53;
const BUCKET_BITS:usize=5;
const BUCKETS:usize=1<<BUCKET_BITS;
pub(crate) struct RawRoulette {cuts:Vec<[u64;3]>,fast:Vec<u64>}
impl RawRoulette {
    pub(crate) fn new(weights:&[f64])->Self {
        assert!(!weights.is_empty());
        let mut cuts=vec![[0;3];D*D*D];let mut fast=vec![0u64;D*D*D];
        for a in 0..D {for b in 0..D {for c in 0..D {
            let w=[weights[a.min(weights.len()-1)],weights[b.min(weights.len()-1)],weights[c.min(weights.len()-1)]];
            let mut sum=0.0;for &v in &w {sum+=v;}
            let idx=(a*D+b)*D+c;
            for stage in 0..3 {
                let hit=|k:u64|{let mut r=(k as f64)*(1.0/(GRID as f64))*sum;for j in 0..=stage {r-=w[j];}r<=0.0};
                let(mut lo,mut hi)=(0u64,GRID);
                while lo<hi {let mid=lo+(hi-lo)/2;if hit(mid){lo=mid+1;}else{hi=mid;}}
                assert!(lo>0); // r at grid point zero is non-positive
                cuts[idx][stage]=(((lo as u128)<<11)-1)as u64;
            }
            let cut=cuts[idx];
            let mut codes=0u64;
            for bucket in 0..BUCKETS {
                let low=(bucket as u64)<<(64-BUCKET_BITS);
                let high=low|((1u64<<(64-BUCKET_BITS))-1);
                // Certify the whole interval, not just equal endpoint outputs:
                // no prefix boundary may lie inside this bucket.
                let crosses=cut.iter().any(|&last|last>=low&&last<high);
                let choice=if crosses{3}else{Self::raw_choice(cut,low)};
                codes|=(choice as u64)<<(2*bucket);
            }
            fast[idx]=codes;
        }}}
        Self{cuts,fast}
    }
    #[inline(always)]pub(crate) fn sample(word:u64)->f64 {((word>>11)as f64)*(1.0/(GRID as f64))}
    #[inline(always)]fn raw_choice(c:[u64;3],word:u64)->usize {
        if word<=c[0]{0}else if word<=c[1]{1}else if word<=c[2]{2}else{0}
    }
    #[inline(always)]pub(crate) fn choose(&self,a:usize,b:usize,c:usize,word:u64)->usize {
        unsafe {
            let idx=(a*D+b)*D+c;
            let packed=*self.fast.get_unchecked(idx);
            let choice=((packed>>(2*(word>>(64-BUCKET_BITS))))&3)as usize;
            if choice!=3{return choice;}
            Self::raw_choice(*self.cuts.get_unchecked(idx),word)
        }
    }
}
#[cfg(test)]mod tests {
    use super::*;
    use rand::{Rng,RngCore,SeedableRng,rngs::SmallRng};
    fn reference(weights:&[f64],a:usize,b:usize,c:usize,word:u64)->usize {
        let w=[weights[a],weights[b],weights[c]];let mut sum=0.0;for &v in &w{sum+=v;}
        let mut r=RawRoulette::sample(word)*sum;
        for j in 0..3{r-=w[j];if r<=0.0{return j;}}0
    }
    #[test]fn raw_word_conversion_matches_rand_standard(){
        let mut a=SmallRng::seed_from_u64(18273645);let mut b=a.clone();
        for _ in 0..100000{assert_eq!(RawRoulette::sample(a.next_u64()).to_bits(),b.gen::<f64>().to_bits());}
    }
    #[test]fn certified_buckets_and_every_boundary_neighbor(){
        let weights:Vec<f64>=(0..D).map(|i|2.06f64.powf(-(i as f64))).collect();
        let table=RawRoulette::new(&weights);let mut rng=SmallRng::seed_from_u64(567123);
        for a in 0..D{for b in 0..D{for c in 0..D{
            let mut words=vec![0,1,u64::MAX,u64::MAX-1];
            for &cut in &table.cuts[(a*D+b)*D+c]{for delta in [0,1,2,2047,2048,2049]{words.push(cut.saturating_add(delta));words.push(cut.saturating_sub(delta));}}
            for bucket in 0..BUCKETS{let lo=(bucket as u64)<<(64-BUCKET_BITS);let hi=lo|((1u64<<(64-BUCKET_BITS))-1);words.push(lo);words.push(hi);}
            for _ in 0..96{words.push(rng.next_u64());}
            for word in words{assert_eq!(table.choose(a,b,c,word),reference(&weights,a,b,c,word),"scores={a},{b},{c}, word={word}");}
        }}}
    }
    #[test]fn underflow_zero_and_unit_weights(){
        let weights=[0.0,f64::from_bits(1),f64::MIN_POSITIVE,1e-250,1e-20,0.001,1.0,2.0];
        let table=RawRoulette::new(&weights);let mut rng=SmallRng::seed_from_u64(871236);
        for a in 0..D{for b in 0..D{for c in 0..D{
            let cuts=table.cuts[(a*D+b)*D+c];
            let mut words=vec![0,1,u64::MAX];
            for cut in cuts{words.push(cut);words.push(cut.saturating_add(1));words.push(cut.saturating_sub(1));}
            for _ in 0..32{words.push(rng.next_u64());}
            for word in words{assert_eq!(table.choose(a,b,c,word),reference(&weights,a,b,c,word));}
        }}}
    }
}
}
mod signed_order {
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
  super::clause_order::ClauseOrder::choose_zero(v,mask,r)
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
mod clause_order {
// Search reads only variable IDs after exact break caching. Literal signs stay
// in the immutable preprocessing arrays used for initialization/reinitialization.
// The mutable permutation of each clause is represented by three 17-bit IDs and
// a two-bit length. Every baseline swap is mirrored, including across restarts.
pub(crate) struct ClauseOrder { words:Vec<u64> }
impl ClauseOrder {
    pub(crate) fn new(cl:&[i32],co:&[u32])->Self {
        let mut words=Vec::with_capacity(co.len()-1);
        for c in 0..co.len()-1 {
            let lits=&cl[co[c] as usize..co[c+1] as usize];
            assert!(!lits.is_empty() && lits.len()<=3);
            let mut w=(lits.len() as u64)<<51;
            for (j,&l) in lits.iter().enumerate() {
                let v=(l.abs()-1) as u64;
                assert!(v<(1<<17));
                w |= v<<(17*j);
            }
            words.push(w);
        }
        Self{words}
    }
    #[inline(always)] pub(crate) unsafe fn variables(&self,c:usize)->[usize;3] {
        let w=*self.words.get_unchecked(c);
        [(w&131071)as usize,((w>>17)&131071)as usize,((w>>34)&131071)as usize]
    }
    // A two-member zero set needs only the original random parity. A
    // singleton is fixed. For all three zero candidates, the ordinal is exactly
    // the random%3 value ALREADY computed for this clause's first-literal swap.
    #[inline(always)] pub(crate) fn choose_zero_rank(v:[usize;3],mask:usize,random:usize,rank:usize)->usize{
        let shift=(mask<<2)|((random&1)<<1);
        let table=((160056576u32>>shift)&3)as usize;
        let k=if mask==7{rank}else{table};
        if k==0{v[0]}else if k==1{v[1]}else{v[2]}
    }
    #[inline(always)] pub(crate) unsafe fn choose_zero(v:[usize;3],mask:usize,random:usize)->usize {
        const COUNTS:[usize;8]=[0,1,1,2,1,2,2,3];
        const ORDER:[[usize;3];8]=[[0,0,0],[0,0,0],[1,0,0],[0,1,0],[2,0,0],[0,2,0],[1,2,0],[0,1,2]];
        let count=*COUNTS.get_unchecked(mask);
        let k=super::exact_div::clause_rem(random,count);
        *v.get_unchecked(*ORDER.get_unchecked(mask).get_unchecked(k))
    }
    #[inline(always)] pub(crate) unsafe fn len_bounded(&self,c:usize)->usize {
        ((*self.words.get_unchecked(c)>>51)&3) as usize
    }
    #[inline(always)] pub(crate) unsafe fn len(&self,c:usize)->usize {
        (*self.words.get_unchecked(c)>>51) as usize
    }
    // Synthetic positive literal: only abs(lit)-1 is observed in the hot search.
    #[inline(always)] pub(crate) unsafe fn lit(&self,c:usize,j:usize)->i32 {
        (((*self.words.get_unchecked(c)>>(17*j))&((1<<17)-1)) as i32)+1
    }
    #[inline(always)] pub(crate) unsafe fn swap(&mut self,c:usize,a:usize,b:usize) {
        let w=self.words.get_unchecked_mut(c);
        let d=((*w>>(17*a))^(*w>>(17*b)))&((1<<17)-1);
        *w ^= (d<<(17*a))|(d<<(17*b));
    }
}

#[cfg(test)]mod rank_tests{
 use super::*;
 #[test]fn exact_zero_choice_all_masks_lengths_and_random_words(){
  let mut r=0x91827364deadbeefusize;let vv=[197,731,19];
  for len in 1..=3{for mask in 1usize..1<<len{for _ in 0..100000{
   r=r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
   assert_eq!(ClauseOrder::choose_zero_rank(vv,mask,r,r%len),unsafe{ClauseOrder::choose_zero(vv,mask,r)});
  }}}
 }
}
}
mod exact_div {
// Let M=floor((2^64-1)/d), for ANY positive d, including d=1.
// 0 <= 2^64/d - M <= 1. Since x<2^64, floor(x*M/2^64) is at
// most ONE below floor(x/d). One correction of r=x-q*d therefore returns
// exactly x%d. Unlike floor(2^64/d), M is representable for d=1 too.
const N:usize=65536;
const fn reciprocals()->[u64;N]{
 let mut out=[0u64;N];let mut d=1;
 while d<N{out[d]=u64::MAX/(d as u64);d+=1;}out
}
static RECIPROCALS:[u64;N]=reciprocals();
#[inline(always)]pub(crate) fn rem(x:usize,d:usize)->usize{
 // One interval check admits precisely 1<=d<N, retaining divisor-zero panic.
 if d.wrapping_sub(1)<N-1{
  let m=unsafe{*RECIPROCALS.get_unchecked(d)};
  let q=(((x as u128)*(m as u128))>>64)as usize;
  let r=x-q*d;
  if r>=d{r-d}else{r}
 }else{x%d}
}
#[inline(always)]pub(crate) fn clause_rem(x:usize,d:usize)->usize{
 match d{3=>x%3,2=>x&1,1=>0,_=>x%d}
}
#[cfg(test)]mod total_reciprocal_tests{
 use super::*;
 #[test]fn every_table_divisor_boundary_and_random_word(){
  let mut x=0xa5a5123456784321usize;
  for d in 1usize..N+128{
   for v in [0,1,d-1,d,d+1,2*d-1,2*d,usize::MAX,usize::MAX-1,1usize<<63]{assert_eq!(rem(v,d),v%d);}
   for _ in 0..48{x^=x<<13;x^=x>>7;x^=x<<17;assert_eq!(rem(x,d),x%d);}
  }
 }
}

const fn domain32_reciprocals()->[u32;65536]{let mut a=[0u32;65536];let mut d=1;while d<65536{a[d]=u32::MAX/(d as u32);d+=1;}a}
static DOMAIN32_RECIPROCALS:[u32;65536]=domain32_reciprocals();
// Callers prove 1<=d<=retained_nc<=21335. Numerator is the ORIGINAL low32 bits.
#[inline(always)]pub(crate) unsafe fn rem32_nonzero_small(x:u32,d:usize)->usize{
 let m=*DOMAIN32_RECIPROCALS.get_unchecked(d);let q=(((x as u64)*(m as u64))>>32)as u32;
 let r=x-q*(d as u32);if r>=d as u32{(r-d as u32)as usize}else{r as usize}
}
#[cfg(test)]mod domain32_tests{
 use super::*;
 #[test]fn every_admissible_divisor_and_boundary_words(){let mut x=0xa341316cu32;for d in 1..65536{for r in [0,1,d as u32-1,d as u32,d as u32+1,u32::MAX-1,u32::MAX]{assert_eq!(unsafe{rem32_nonzero_small(r,d)},(r as usize)%d);}for _ in 0..32{x^=x<<13;x^=x>>17;x^=x<<5;assert_eq!(unsafe{rem32_nonzero_small(x,d)},(x as usize)%d);}}}
}
}
// sat_base_mix -- per-track best-of composite BASELINE.
//
// NOT an optimisation. Each track is delegated, unmodified, to whichever shipped
// algorithm measured best on that track (32 nonces, seed `base1`, one batch per
// track, 2026-09-09). The three source algorithms are vendored verbatim under
// `hybrid/`, `tw6/` and `ours/`; nothing inside them is edited, so each retains
// its own internal dispatch and its own baked hyperparameter defaults.
//
// Track assignment and the evidence behind it are in README.md. Two of the five
// picks (t1, t2) rest on a ONE-solve difference and are inside sampling noise --
// they are the first thing an optimiser should re-measure, not treat as settled.
//
// Delegation is by whole-algorithm `solve_challenge`, not by reaching into
// internals: each vendored crate re-derives the same track from the unchanged
// Challenge, so there is no risk of mis-wiring an engine to the wrong track.
#[path = "hybrid_mod.rs"]
mod hybrid;
#[path = "ours_mod.rs"]
mod ours;
#[path = "tw6_mod.rs"]
mod tw6;

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn help() {
    println!("sat_base_mix - per-track best-of baseline (hybrid t1/t5, tailwalk_v6 t2, giveup2 t3/t4)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();

    match (nv, nc) {
        // t1  n=5000  r4267 -- sat_hybrid    7/32 @ 14,024 core_s  (ours 6/32 @ 9,288)
        (5000, 21335) => hybrid::solve_challenge(challenge, save_solution, hyperparameters),
        // t2  n=7500  r4267 -- sat_tailwalk_v6 5/32 @ 20,524      (ours 4/32 @ 9,512)
        (7500, 32002) => tw6::solve_challenge(challenge, save_solution, hyperparameters),
        // t3  n=10000 r4267 -- sat_imp_giveup2 1/32 @ 19,511, fastest of four at parity
        (10000, 42670) => ours::solve_challenge(challenge, save_solution, hyperparameters),
        // t4  n=100000 r4150 -- sat_imp_giveup2 32/32 @ 906, fastest of four
        (100000, 415000) => ours::solve_challenge(challenge, save_solution, hyperparameters),
        // t5  n=100000 r4200 -- sat_hybrid 32/32 @ 15,686, fastest of four (ours 21,844)
        (100000, 420000) => hybrid::solve_challenge(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!(
            "unknown track config (num_variables={}, num_clauses={})",
            nv,
            nc
        )),
    }
}
