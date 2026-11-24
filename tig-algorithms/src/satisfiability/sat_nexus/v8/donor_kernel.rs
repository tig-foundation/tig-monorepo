// Adapted from c001_a125 / sat_exact by ChervovNikita; upstream helper attribution is retained in README.md.
// Exact union of R21, R22 and R23 shared helpers.
pub(crate) mod clause_order {
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











}
pub(crate) mod compact_ranges {
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











}
pub(crate) mod exact_coin {
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
 
}















}
pub(crate) mod exact_div {
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











const fn domain32_reciprocals()->[u32;65536]{let mut a=[0u32;65536];let mut d=1;while d<65536{a[d]=u32::MAX/(d as u32);d+=1;}a}
static DOMAIN32_RECIPROCALS:[u32;65536]=domain32_reciprocals();
// Callers prove 1<=d<=retained_nc<=21335. Numerator is the ORIGINAL low32 bits.
#[inline(always)]pub(crate) unsafe fn rem32_nonzero_small(x:u32,d:usize)->usize{
 let m=*DOMAIN32_RECIPROCALS.get_unchecked(d);let q=(((x as u64)*(m as u64))>>32)as u32;
 let r=x-q*(d as u32);if r>=d as u32{(r-d as u32)as usize}else{r as usize}
}




}
pub(crate) mod large_buffer {
// Plain Vec buffers, no standard-library module paths. Every element is written at construction,
// as the original explicit zeroing did. Contents equal an explicitly
// zeroed/copied buffer, so the solver path is unchanged.
pub(crate) type LargeBuf<T>=Vec<T>;
pub(crate) fn zeros<T:Copy+Default>(len:usize)->Vec<T>{let mut buf=Vec::with_capacity(len);buf.resize(len,T::default());buf}
pub(crate) fn from_slice<T:Copy>(values:&[T])->Vec<T>{let mut buf=Vec::with_capacity(values.len());buf.extend_from_slice(values);buf}




}
pub(crate) mod paged_cache {
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











































}
pub(crate) mod paged_flip_ranges {
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





}
pub(crate) mod paged_order {
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














}
pub(crate) mod paged_small_order {
// Small-track-only helper; variable IDs fit exactly in u16.
use super::large_buffer::LargeBuf;
// Search reads only variable IDs after exact break caching. Literal signs stay
// in the immutable preprocessing arrays used for initialization/reinitialization.
// The mutable permutation of each clause is represented by three 17-bit IDs and
// a two-bit length. Every baseline swap is mirrored, including across restarts.
#[derive(Clone)]
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











}
pub(crate) mod paged_small_signed {
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














}
pub(crate) mod phase_div {
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










}
pub(crate) mod raw_roulette {
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



































}
pub(crate) mod short_mask {
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












}

pub(crate) mod subset_cache {
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











































}
