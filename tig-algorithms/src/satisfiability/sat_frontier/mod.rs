mod paged_flip_ranges {
use super::large_buffer::LargeBuf;
pub(crate) struct FlipRanges {data:LargeBuf<[u64;2]>}
impl FlipRanges {
pub(crate) fn restore_variables(&self,off:&[u32],mid:&[u32],vars:&mut[bool]){
assert_eq!(vars.len(),self.data.len());
for v in 0..vars.len(){
if off[v+1]!=off[v]{
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
use super::large_buffer::LargeBuf;
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
use super::large_buffer::LargeBuf;
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
const D:usize=8;const GRID:u64=1u64<<53;
pub(crate) struct AdditiveRoulette{cuts:Vec<[u64;2]>}
impl AdditiveRoulette{
pub(crate) fn new(weights:&[f64])->Self{
assert!(weights.len()>=D);
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
const N:usize=65536;
const fn reciprocals()->[u64;N]{
let mut out=[0u64;N];let mut d=1;
while d<N{out[d]=u64::MAX/(d as u64);d+=1;}out
}
static RECIPROCALS:[u64;N]=reciprocals();
#[inline(always)]pub(crate) fn rem(x:usize,d:usize)->usize{
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
#[inline(always)]pub(crate) unsafe fn rem32_nonzero_small(x:u32,d:usize)->usize{
let m=*DOMAIN32_RECIPROCALS.get_unchecked(d);let q=(((x as u64)*(m as u64))>>32)as u32;
let r=x-q*(d as u32);if r>=d as u32{(r-d as u32)as usize}else{r as usize}
}
#[cfg(test)]mod domain32_tests{
use super::*;
#[test]fn every_admissible_divisor_and_boundary_words(){let mut x=0xa341316cu32;for d in 1..65536{for r in [0,1,d as u32-1,d as u32,d as u32+1,u32::MAX-1,u32::MAX]{assert_eq!(unsafe{rem32_nonzero_small(r,d)},(r as usize)%d);}for _ in 0..32{x^=x<<13;x^=x>>17;x^=x<<5;assert_eq!(unsafe{rem32_nonzero_small(x,d)},(x as usize)%d);}}}
}
}
mod hybrid {
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
mod engine_b {
mod track_t1 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess(challenge, save_solution);
let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
let max_fuel = hp.max_fuel_high.unwrap_or(default_fuel);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
for v in 0..nv {
let np = p_cnt[v] as usize;
let nn = n_cnt[v] as usize;
if np > nn {
vars[v] = true;
} else if np < nn {
vars[v] = false;
} else {
vars[v] = rng.gen_bool(0.5);
}
}
let mut num_good = vec![0u8; nc];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[i] += 1;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
let mut true_unsat = 0usize;
for i in 0..nc {
if num_good[i] == 0 {
residual.push(i as u32);
true_unsat += 1;
}
}
if true_unsat == 0 {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let mut best_unsat = true_unsat;
let mut best_vars = vars.clone();
let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let mut probsat_weights = vec![0.0f64; nc + 1];
if avg_clause_size <= 3.2 {
let cb: f64 = 2.06;
for i in 0..=nc {
probsat_weights[i] = cb.powf(-(i as f64));
}
} else {
let cb: f64 = if avg_clause_size <= 4.2 {
2.85
} else if avg_clause_size <= 5.2 {
3.7
} else if avg_clause_size <= 6.2 {
5.1
} else {
5.4
};
for i in 0..=nc {
probsat_weights[i] = (i as f64 + 1.0).powf(-cb);
}
}
let mut last_check_unsat = true_unsat;
let mut rounds = 0usize;
let mut stagnation = 0usize;
unsafe {
loop {
if rounds >= max_flips { break; }
if true_unsat == 0 { break; }
if rounds % check_interval == 0 && rounds > 0 {
let progress = last_check_unsat as i64 - true_unsat as i64;
let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
if progress <= 0 {
stagnation += 1;
if stagnation >= 4 {
if stagnation >= 15 {
vars.copy_from_slice(&best_vars);
let perturb_cnt = (nv / 20).max(1);
for _ in 0..perturb_cnt {
let v = rng.gen::<usize>() % nv;
*vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
}
residual.clear();
true_unsat = 0;
for i in 0..nc {
let s = *co.get_unchecked(i) as usize;
let e = *co.get_unchecked(i + 1) as usize;
let mut good = 0u8;
for j in s..e {
let l = *cl.get_unchecked(j);
let v = (l.abs() - 1) as usize;
if (l > 0 && *vars.get_unchecked(v)) || (l < 0 && !*vars.get_unchecked(v)) {
good = good.saturating_add(1);
}
}
*num_good.get_unchecked_mut(i) = good;
if good == 0 {
residual.push(i as u32);
true_unsat += 1;
}
}
stagnation = 0;
} else {
let kicks = if stagnation >= 8 { 6 } else { 3 };
for _ in 0..kicks {
if true_unsat == 0 { break; }
let rid = rng.gen::<usize>() % residual.len();
let pcid = *residual.get_unchecked(rid) as usize;
if *num_good.get_unchecked(pcid) > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
*ng = ng.saturating_add(1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
let new_val = ng.saturating_sub(1);
*ng = new_val;
if new_val == 0 {
residual.push(c as u32);
true_unsat += 1;
}
}
*vars.get_unchecked_mut(v) = !was_true;
}
stagnation = 0;
}
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
} else {
stagnation = 0;
}
last_check_unsat = true_unsat;
}
if true_unsat == 0 { break; }
let mut cid = usize::MAX;
let mut min_len = usize::MAX;
for _ in 0..3 {
while !residual.is_empty() {
let id = rng.gen::<usize>() % residual.len();
let cand = *residual.get_unchecked(id) as usize;
if *num_good.get_unchecked(cand) > 0 {
residual.swap_remove(id);
} else {
let c_s = *co.get_unchecked(cand) as usize;
let c_e = *co.get_unchecked(cand + 1) as usize;
let clen = c_e - c_s;
if clen < min_len {
min_len = clen;
cid = cand;
}
break;
}
}
if residual.is_empty() { break; }
}
if cid == usize::MAX { break; }
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
if clen > 1 {
let ri = rng.gen::<usize>() % clen;
cl.swap(cs, cs + ri);
}
let mut zero_found: Option<usize> = None;
'outer_h: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
continue 'outer_h;
}
}
zero_found = Some(abs_l);
break;
}
let v_idx = if let Some(v) = zero_found {
v
} else {
let mut total_weight = 0.0;
let mut weights = [0.0; 256];
let clen_actual = (ce - cs).min(256);
for idx in 0..clen_actual {
let j = cs + idx;
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
sad += 1;
}
}
let w = *probsat_weights.get_unchecked(sad.min(nc));
weights[idx] = w;
total_weight += w;
}
let mut r = rng.gen::<f64>() * total_weight;
let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
for idx in 0..clen_actual {
r -= weights[idx];
if r <= 0.0 {
v_min = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
break;
}
}
v_min
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
*ng = ng.saturating_add(1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
let new_val = ng.saturating_sub(1);
*ng = new_val;
if new_val == 0 {
residual.push(c as u32);
true_unsat += 1;
}
}
*vars.get_unchecked_mut(v_idx) = !was_true;
if true_unsat < best_unsat {
best_unsat = true_unsat;
best_vars.copy_from_slice(&vars);
}
rounds += 1;
}
}
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
}
mod track_t3 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess(challenge, save_solution);
let nvf = nv as f64;
let max_fuel = hp.max_fuel_low.unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
let appearances: Vec<u8> = (0..nv).map(|v| {
((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
}).collect();
drop(p_cnt);
drop(n_cnt);
let ng_len = (nc + 3) >> 2;
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let shift = (i & 3) << 1;
let byte_idx = i >> 2;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[byte_idx] += 1u8 << shift;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
for i in 0..nc {
if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
residual.push(i as u32);
}
}
if residual.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let base_prob = hp.base_prob
.unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
let mut current_prob = base_prob;
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let max_random_prob = hp.max_prob.unwrap_or(0.9);
let prob_adjustment_factor = 0.03;
let smoothing_factor = 0.8;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.perturbation_flips
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.stagnation_limit
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = residual.len();
let mut stagnation = 0usize;
let mut var_age = vec![0u8; nv];
let mut countdown = check_interval;
let mut rounds = 0usize;
let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
unsafe {
loop {
if residual.is_empty() || rounds >= max_flips { break; }
countdown -= 1;
if countdown == 0 {
countdown = check_interval;
let progress = last_check_residual as i64 - residual.len() as i64;
let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
if progress <= 0 {
stagnation += 1;
let prob_adjustment = prob_adjustment_factor
* (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
current_prob = (current_prob + prob_adjustment).min(max_random_prob);
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if residual.is_empty() { break; }
let rid = rng.gen::<usize>() % residual.len();
let pcid = *residual.get_unchecked(rid) as usize;
let ng_val = (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3;
if ng_val > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
*num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
residual.push(c as u32);
}
}
*vars.get_unchecked_mut(v) = !was_true;
*var_age.get_unchecked_mut(v) = 0;
}
stagnation = 0;
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
current_prob = base_prob;
} else {
stagnation = 0;
current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
}
last_check_residual = residual.len();
}
let rand_val = rng.gen::<usize>();
let mut cid = 0usize;
let mut found = false;
while !residual.is_empty() {
let id = rand_val % residual.len();
let candidate = *residual.get_unchecked(id) as usize;
let ng_val = (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3;
if ng_val > 0 {
residual.swap_remove(id);
} else {
cid = candidate;
found = true;
break;
}
}
if !found { break; }
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
if clen > 1 {
let ri = rand_val % clen;
cl.swap(cs, cs + ri);
}
let mut zero_buf: [usize; 3] = [0; 3];
let mut zero_cnt: usize = 0;
'outer: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if l > 0 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
continue 'outer;
}
}
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
let v_idx = if zero_cnt > 0 {
if zero_cnt == 1 {
*zero_buf.get_unchecked(0)
} else {
*zero_buf.get_unchecked(rand_val % zero_cnt)
}
} else if rng.gen::<f64>() < current_prob {
(cl.get_unchecked(cs).abs() - 1) as usize
} else {
let mut min_sad = usize::MAX;
let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
let mut min_weight = usize::MAX;
for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if l > 0 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
sad += 1;
}
if sad >= min_sad { break; }
}
if sad == 0 {
let app = *appearances.get_unchecked(abs_l) as usize;
let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
let adjusted_weight = app.saturating_sub(age_bonus);
if min_sad > 0 || adjusted_weight < min_weight {
min_sad = 0;
min_weight = adjusted_weight;
v_min = abs_l;
}
} else if min_sad > 0 {
let app = *appearances.get_unchecked(abs_l) as usize;
let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
if combined_weight < min_weight {
min_sad = sad;
min_weight = combined_weight;
v_min = abs_l;
}
if min_sad <= 1 { break; }
}
}
v_min
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
*num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
residual.push(c as u32);
}
}
*vars.get_unchecked_mut(v_idx) = !was_true;
*var_age.get_unchecked_mut(v_idx) = 0;
for j in cs..ce {
let l = *cl.get_unchecked(j);
let var = (l.abs() - 1) as usize;
let age = var_age.get_unchecked_mut(var);
*age = age.saturating_add(1);
}
rounds += 1;
}
}
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
}
mod track_t4 {
use super::super::super::explicit_roulette;
use super::super::super::signed_order::{SignedOrder as ClauseOrder,LiteralRanges};
use super::super::super::exact_div;
use super::super::super::subset_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
#[inline(always)]
unsafe fn choose_three<C: CacheOps>(
order: &mut ClauseOrder,
cid: usize,
cache: &C,
word: usize,
) -> usize {
order.swap(cid, 0, exact_div::clause_rem(word, 3));
let ll = order.codes(cid);
let b0 = cache.breaks_of(ll[0] >> 1);
let b1 = cache.breaks_of(ll[1] >> 1);
let b2 = cache.breaks_of(ll[2] >> 1);
let mask = ((b0 == 0) as usize)
| (((b1 == 0) as usize) << 1)
| (((b2 == 0) as usize) << 2);
if mask != 0 {
ClauseOrder::choose_zero(ll, mask, word)
} else {
ll[explicit_roulette::integer_three(b0, b1, b2, word as u32)]
}
}
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
let max_degree=all_off.windows(2).map(|w|(w[1]-w[0])as usize).max().unwrap_or(0);
let mut changed=Vec::<u32>::with_capacity(max_degree);
let changed_ptr=changed.as_mut_ptr();
let mut cache = C::new(nv, &cl, &co, &vars);
let mut next_check_interval = check_interval;
unsafe {
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
}
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v,changed_ptr);
let len=unsat_list.len();
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
let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32, unsat_list.len())) as usize;
let cs = 0usize;
let ce = order.len(cid);
let clen = ce - cs;
let chosen_code = if clen == 3 {
choose_three(&mut order, cid, &cache, rand_val)
} else {
if clen > 1 {
let ri = exact_div::clause_rem(rand_val, clen);
order.swap(cid, 0, ri);
}
let ll = order.codes(cid);
let b0 = cache.breaks_of(ll[0] >> 1);
let b1 = if clen > 1 { cache.breaks_of(ll[1] >> 1) } else { 1 };
let b2 = if clen > 2 { cache.breaks_of(ll[2] >> 1) } else { 1 };
let zero_mask = ((b0 == 0) as usize)
| (((b1 == 0) as usize) << 1)
| (((b2 == 0) as usize) << 2);
if zero_mask != 0 {
ClauseOrder::choose_zero(ll, zero_mask, rand_val)
} else {
let w0 = *probs_break.get_unchecked(b0.min(15));
let w1 = if clen > 1 { *probs_break.get_unchecked(b1.min(15)) } else { 0 };
let w2 = if clen > 2 { *probs_break.get_unchecked(b2.min(15)) } else { 0 };
let total = w0 + w1 + w2;
let r = (rand_val as u32) % total.max(1);
if r < w0 { ll[0] }
else if clen > 1 && r - w0 < w1 { ll[1] }
else if clen > 2 && r - w0 - w1 < w2 { ll[2] }
else { ll[0] }
}
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
}
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,changed_ptr);
let len=unsat_list.len();
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
}
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v,changed_ptr);
let len=unsat_list.len();
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
let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32, unsat_list.len())) as usize;
let cs = 0usize;
let ce = order.len(cid);
let clen = ce - cs;
let chosen_code = if clen == 3 {
choose_three(&mut order, cid, &cache, rand_val)
} else {
if clen > 1 {
let ri = exact_div::clause_rem(rand_val, clen);
order.swap(cid, 0, ri);
}
let ll = order.codes(cid);
let b0 = cache.breaks_of(ll[0] >> 1);
let b1 = if clen > 1 { cache.breaks_of(ll[1] >> 1) } else { 1 };
let b2 = if clen > 2 { cache.breaks_of(ll[2] >> 1) } else { 1 };
let zero_mask = ((b0 == 0) as usize)
| (((b1 == 0) as usize) << 1)
| (((b2 == 0) as usize) << 2);
if zero_mask != 0 {
ClauseOrder::choose_zero(ll, zero_mask, rand_val)
} else {
let w0 = *probs_break.get_unchecked(b0.min(15));
let w1 = if clen > 1 { *probs_break.get_unchecked(b1.min(15)) } else { 0 };
let w2 = if clen > 2 { *probs_break.get_unchecked(b2.min(15)) } else { 0 };
let total = w0 + w1 + w2;
let r = (rand_val as u32) % total.max(1);
if r < w0 { ll[0] }
else if clen > 1 && r - w0 < w1 { ll[1] }
else if clen > 2 && r - w0 - w1 < w2 { ll[2] }
else { ll[0] }
}
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
}
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,changed_ptr);
let len=unsat_list.len();
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
} else {
}
}
}
let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
let _ = save_solution(&Solution { variables: final_vars });
Ok(())
}
}
mod track_t5 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
let super::Prepared {
mut rng,
nv,
nc,
density: _density,
p_cnt,
n_cnt,
all_off,
p_bound,
all_data,
mut cl,
co,
} = super::preprocess(challenge, save_solution);
let max_fuel = hp.max_fuel_high.unwrap_or(155_000_000_000.0);
let base_fuel = 50_000_000.0;
let flip_fuel = 200.0;
let max_flips = if max_fuel > base_fuel {
((max_fuel - base_fuel) / flip_fuel) as u64
} else {
10_000_000
};
const WALK_P: f64 = 0.52;
const RESTART_PERIOD: u64 = 80_000_000;
const REINIT_MIN_UNSAT: usize = 30;
let mut assignment = vec![false; nv];
let mut best_assignment = vec![false; nv];
let mut true_lit_count = vec![0u8; nc];
let mut unsat_clauses: Vec<usize> = Vec::with_capacity(nc);
let mut clause_pos_in_unsat = vec![usize::MAX; nc];
for v in 0..nv {
let pc = p_cnt[v] as f64;
let nc_v = n_cnt[v] as f64;
let total = pc + nc_v;
if total == 0.0 {
assignment[v] = rng.gen_bool(0.5);
} else {
assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
}
}
for c in 0..nc {
let off = co[c] as usize;
let end = co[c + 1] as usize;
let mut trues = 0u8;
for i in off..end {
let lit = cl[i];
let v = (lit.abs() - 1) as usize;
if assignment[v] == (lit > 0) { trues += 1; }
}
true_lit_count[c] = trues;
if trues == 0 {
clause_pos_in_unsat[c] = unsat_clauses.len();
unsat_clauses.push(c);
}
}
let mut best_unsat = unsat_clauses.len();
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
let mut period_best_unsat = best_unsat;
let mut var_age = vec![0u8; nv];
let mut global_flips: u64 = 0;
unsafe {
loop {
if global_flips >= max_flips { break; }
if unsat_clauses.is_empty() { break; }
if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
if period_best_unsat >= REINIT_MIN_UNSAT {
for v in 0..nv {
*assignment.get_unchecked_mut(v) = rng.gen_bool(0.5);
}
true_lit_count.fill(0);
unsat_clauses.clear();
clause_pos_in_unsat.fill(usize::MAX);
for c in 0..nc {
let off = *co.get_unchecked(c) as usize;
let end = *co.get_unchecked(c + 1) as usize;
let mut trues = 0u8;
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
}
*true_lit_count.get_unchecked_mut(c) = trues;
if trues == 0 {
*clause_pos_in_unsat.get_unchecked_mut(c) = unsat_clauses.len();
unsat_clauses.push(c);
}
}
var_age.fill(0);
let cur = unsat_clauses.len();
if cur < best_unsat {
best_unsat = cur;
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
}
}
period_best_unsat = unsat_clauses.len();
}
let cur_unsat = unsat_clauses.len();
if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
global_flips += 1;
let r_idx = rng.gen_range(0..unsat_clauses.len());
let c = *unsat_clauses.get_unchecked(r_idx);
let off = *co.get_unchecked(c) as usize;
let end = *co.get_unchecked(c + 1) as usize;
let len = end - off;
if len > 1 {
let ri = (global_flips as usize) % len;
cl.swap(off, off + ri);
}
let mut picked_v = usize::MAX;
let mut vars = [0usize; 3];
let mut breaks = [0u8; 3];
let mut nvars = 0;
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
*vars.get_unchecked_mut(nvars) = v;
let val = *assignment.get_unchecked(v);
let (start, stop) = if val {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
let mut b = 0u8;
for k in start..stop {
if *true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1 { b += 1; }
}
*breaks.get_unchecked_mut(nvars) = b;
if b == 0 {
picked_v = v;
break;
}
nvars += 1;
}
if picked_v == usize::MAX {
if nvars == 0 {
if let Some(&lit) = cl.get(off) {
picked_v = (lit.abs() - 1) as usize;
}
} else if rng.gen_bool(WALK_P) {
picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
} else {
let mut min_b = u8::MAX;
let mut best_idx = 0;
for i in 0..nvars {
let b = *breaks.get_unchecked(i);
let vi = *vars.get_unchecked(i);
let vb = *vars.get_unchecked(best_idx);
if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
min_b = b;
best_idx = i;
}
}
picked_v = *vars.get_unchecked(best_idx);
}
}
if picked_v == usize::MAX { continue; }
let new_val = !*assignment.get_unchecked(picked_v);
*assignment.get_unchecked_mut(picked_v) = new_val;
let p_start = *all_off.get_unchecked(picked_v) as usize;
let p_end = *p_bound.get_unchecked(picked_v) as usize;
for k in p_start..p_end {
let c_idx = *all_data.get_unchecked(k) as usize;
if new_val {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
if trues == 0 {
let last_c = unsat_clauses.pop().unwrap_unchecked();
let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
if last_c != c_idx {
*unsat_clauses.get_unchecked_mut(pos) = last_c;
*clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
}
}
} else {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
if trues == 1 {
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
unsat_clauses.push(c_idx);
}
}
}
let n_start = *p_bound.get_unchecked(picked_v) as usize;
let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
for k in n_start..n_end {
let c_idx = *all_data.get_unchecked(k) as usize;
if !new_val {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
if trues == 0 {
let last_c = unsat_clauses.pop().unwrap_unchecked();
let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
if last_c != c_idx {
*unsat_clauses.get_unchecked_mut(pos) = last_c;
*clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
}
}
} else {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
if trues == 1 {
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
unsat_clauses.push(c_idx);
}
}
}
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
let a = *var_age.get_unchecked(v);
*var_age.get_unchecked_mut(v) = a.saturating_add(1);
}
let cur_unsat = unsat_clauses.len();
if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
if cur_unsat < best_unsat {
best_unsat = cur_unsat;
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
if cur_unsat == 0 { break; }
}
}
}
let _ = save_solution(&Solution { variables: best_assignment });
Ok(())
}
}
mod track_t38 {
use super::super::super::one_load_ranges::{Ranges,PackedRanges};
use super::super::super::phase_div;
use super::super::super::explicit_roulette;
use super::super::super::paged_order::SignedOrder as ClauseOrder;
use super::super::super::signed_order::LiteralRanges;
use super::super::super::exact_div;
use super::super::super::paged_cache::{ExactBreakCache, NarrowLargeBreakCache, CacheOps};
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
let prepared=super::preprocess(challenge, save_solution);
let narrow=prepared.p_cnt.iter().zip(&prepared.n_cnt).all(|(&p,&n)|p+n<=255);
let packed=prepared.all_data.len()<(1<<21);
if narrow && packed { solve_impl::<NarrowLargeBreakCache,PackedRanges>(prepared,save_solution,hp) }
else { solve_impl::<ExactBreakCache,LiteralRanges>(prepared,save_solution,hp) }
}
fn solve_impl<C:CacheOps,R:Ranges>(
prepared: super::Prepared,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
let nvf = nv as f64;
let max_fuel = hp.max_fuel_low.unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
drop(p_cnt);
drop(n_cnt);
let ng_len = nc; // one byte per clause; count is in 0..=3
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let byte_idx = i;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[byte_idx] += 1u8;
}
}
}
let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
let mut unsat_pos = super::super::super::large_buffer::zeros::<u32>(nc);
for i in 0..nc {
if num_good[i] == 0 {
unsat_pos[i] = unsat_list.len() as u32;
unsat_list.push(i as u32);
}
}
if unsat_list.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.perturbation_flips
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.stagnation_limit
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = unsat_list.len();
let mut stagnation = 0usize;
let mut countdown = check_interval;
let mut rounds = 0usize;
let cb = if avg_clause_size > 4.5 {
3.5f64
} else if avg_clause_size > 3.5 {
2.85f64
} else {
2.06f64
};
let mut probs_break = [0.0f64; 256];
for i in 0..256 {
probs_break[i] = cb.powf(-(i as f64));
}
let additive=explicit_roulette::AdditiveRoulette::new(&probs_break);
let mut order = ClauseOrder::new(&cl, &co);
let ranges=R::new(&all_off,&p_bound);
let all_data=super::super::super::large_buffer::from_slice(&all_data);
let max_degree=all_off.windows(2).map(|w|(w[1]-w[0])as usize).max().unwrap_or(0);
let mut changed=Vec::<u32>::with_capacity(max_degree);
let changed_ptr=changed.as_mut_ptr();
let mut cache = C::new(nv, &cl, &co, &vars);
unsafe {
let mut end=if check_interval==0{max_flips}else{(check_interval-1).min(max_flips)};
'search: loop{
let steps=(end-rounds).min(64);let batch_end=rounds+steps;
let small=steps.checked_mul(max_degree).and_then(|n|unsat_list.len().checked_add(n)).map_or(false,|n|n<phase_div::LIMIT);
let done=if small{run_explicit_phase::<C,R,true>(&mut cache,&mut order,&ranges,&mut unsat_list,&mut unsat_pos,&all_data,changed_ptr,&mut rng,&probs_break,&additive,&mut rounds,batch_end)}else{run_explicit_phase::<C,R,false>(&mut cache,&mut order,&ranges,&mut unsat_list,&mut unsat_pos,&all_data,changed_ptr,&mut rng,&probs_break,&additive,&mut rounds,batch_end)};
if done{break 'search;}
if rounds<end{continue;}
if unsat_list.is_empty()||rounds>=max_flips{break;}
let progress = last_check_residual as i64 - unsat_list.len() as i64;
if progress <= 0 {
stagnation += 1;
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if unsat_list.is_empty() { break; }
let rid = exact_div::rem(rng.gen::<usize>(), unsat_list.len());
let pcid = *unsat_list.get_unchecked(rid) as usize;
let pcs = 0usize;
let pce = order.len_bounded(pcid);
if pcs == pce { continue; }
let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true=lit<0;
let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v,changed_ptr);
super::super::super::list_events::remove_events(&mut unsat_list,&mut unsat_pos,changed_ptr,gains);
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v,changed_ptr);
super::super::super::list_events::insert_events(&mut unsat_list,&mut unsat_pos,changed_ptr,losses);
cache.set_break(v,gains as u32);
}
stagnation = 0;
}
} else {
stagnation = 0;
}
last_check_residual = unsat_list.len();
if unsat_list.is_empty(){let _:usize=rng.gen();break;}
end=rounds.saturating_add(check_interval).min(max_flips);
}
}
cache.restore_variables(&cl,&co,&mut vars);
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
#[inline(always)]
unsafe fn run_explicit_phase<C:CacheOps,R:Ranges,const SMALL:bool>(
cache:&mut C,order:&mut ClauseOrder,ranges:&R,
unsat_list:&mut Vec<u32>,unsat_pos:&mut[u32],all_data:&[u32],changed_ptr:*mut u32,
rng:&mut rand::rngs::SmallRng,probs_break:&[f64;256],additive:&explicit_roulette::AdditiveRoulette,
rounds:&mut usize,end:usize)->bool{
while *rounds<end{
if unsat_list.is_empty(){return true;}
let rand_val = rng.gen::<usize>();
if unsat_list.is_empty() {return true;}
let cid = *unsat_list.get_unchecked(if SMALL{phase_div::rem_nonzero_small(rand_val,unsat_list.len())}else{exact_div::rem(rand_val,unsat_list.len())}) as usize;
let cs = 0usize;
let ce = order.len_bounded(cid);
let clen = ce - cs;
let permutation_rank=if clen>1{let rank=exact_div::clause_rem(rand_val,clen);order.swap(cid,0,rank);rank}else{0};
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0{super::super::super::clause_order::ClauseOrder::choose_zero_rank(ll,zero_mask,rand_val,permutation_rank)}else{
let word=rng.gen::<u64>();
if clen==3 && (b0|b1|b2)<8{ll[additive.choose(b0,b1,b2,word)]}else{
let w0=*probs_break.get_unchecked(b0.min(255));
let w1=if clen>1{*probs_break.get_unchecked(b1.min(255))}else{0.0};
let w2=if clen>2{*probs_break.get_unchecked(b2.min(255))}else{0.0};
let mut total=0.0;total+=w0;if clen>1{total+=w1;}if clen>2{total+=w2;}
let threshold=explicit_roulette::AdditiveRoulette::sample(word)*total;
let mut accum=0.0;accum+=w0;
if accum>=threshold{ll[0]}else if clen>1{
accum+=w1;
if accum>=threshold{ll[1]}else if clen>2{accum+=w2;if accum>=threshold{ll[2]}else{ll[0]}}else{ll[0]}
}else{ll[0]}
}
};
let v_idx=chosen_code>>1;
let was_true=chosen_code&1!=0;
let (is,ie,ds,de)=ranges.of(chosen_code);
let gains=cache.gain_collect(all_data.get_unchecked(is..ie),v_idx,changed_ptr);
super::super::super::list_events::remove_events(unsat_list,unsat_pos,changed_ptr,gains);
let losses=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,changed_ptr);
super::super::super::list_events::insert_events(unsat_list,unsat_pos,changed_ptr,losses);
cache.set_break(v_idx,gains as u32);
*rounds += 1;
}
false
}
}
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
pub base_prob: Option<f64>,
pub max_prob: Option<f64>,
pub check_interval: Option<usize>,
pub stagnation_limit: Option<usize>,
pub perturbation_flips: Option<usize>,
pub max_fuel_high: Option<f64>,
pub max_fuel_low: Option<f64>,
pub max_reinits: Option<usize>,
}
impl Hparams {
pub fn for_t1() -> Self {
let mut h = Self::default();
h.max_fuel_high = Some(180_000_000_000.0);
h
}
pub fn for_t3() -> Self { Self::default() }
pub fn for_t4() -> Self {
let mut h = Self::default();
h.max_reinits = Some(15);
h
}
pub fn for_t5() -> Self { Self::default() }
pub fn for_t38() -> Self {
let mut h = Self::default();
h.stagnation_limit = Some(3);
h
}
fn merge_user(mut self, user: Option<&Map<String, Value>>) -> Self {
if let Some(m) = user {
if let Ok(u) = serde_json::from_value::<Hparams>(Value::Object(m.clone())) {
if u.base_prob.is_some() { self.base_prob = u.base_prob; }
if u.max_prob.is_some() { self.max_prob = u.max_prob; }
if u.check_interval.is_some() { self.check_interval = u.check_interval; }
if u.stagnation_limit.is_some() { self.stagnation_limit = u.stagnation_limit; }
if u.perturbation_flips.is_some() { self.perturbation_flips = u.perturbation_flips; }
if u.max_fuel_high.is_some() { self.max_fuel_high = u.max_fuel_high; }
if u.max_fuel_low.is_some() { self.max_fuel_low = u.max_fuel_low; }
if u.max_reinits.is_some() { self.max_reinits = u.max_reinits; }
}
}
self
}
}
pub(crate) struct Prepared {
pub rng: SmallRng,
pub nv: usize,
pub nc: usize,
pub density: f64,
pub p_cnt: Vec<u32>,
pub n_cnt: Vec<u32>,
pub all_off: Vec<u32>,
pub p_bound: Vec<u32>,
pub all_data: Vec<u32>,
pub cl: Vec<i32>,
pub co: Vec<u32>,
}
#[inline(always)]
pub(crate) fn preprocess(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Prepared {
let nv = challenge.num_variables;
let _ = save_solution(&Solution { variables: vec![false; nv] });
let rng = SmallRng::seed_from_u64(u64::from_le_bytes(
challenge.seed[..8].try_into().unwrap(),
));
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
let density = nc as f64 / nv as f64;
Prepared { rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co }
}
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let nv = challenge.num_variables;
let nc_total = challenge.clauses.len();
let user = hyperparameters.as_ref();
match (nv, nc_total) {
(10000, 42670) => {
let hp = Hparams::for_t1().merge_user(user);
track_t1::solve(challenge, save_solution, &hp)
}
(100000, 415000) => {
let hp = Hparams::for_t3().merge_user(user);
track_t3::solve(challenge, save_solution, &hp)
}
(5000, 21335) => {
let hp = Hparams::for_t4().merge_user(user);
track_t4::solve(challenge, save_solution, &hp)
}
(7500, 32002) => {
let hp = Hparams::for_t5().merge_user(user);
track_t5::solve(challenge, save_solution, &hp)
}
(100000, 420000) => {
let hp = Hparams::for_t38().merge_user(user);
track_t38::solve(challenge, save_solution, &hp)
}
_ => Err(anyhow::anyhow!(
"engine: unknown track config (num_variables={}, num_clauses={})",
nv, nc_total
)),
}
}
}
pub fn solve_challenge(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let nv = challenge.num_variables;
let nc = challenge.clauses.len();
match (nv, nc) {
(100000, 415000) => engine_b::solve(challenge, save_solution, hyperparameters),
(5000, 21335) => engine_b::solve(challenge, save_solution, hyperparameters),
(100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
_ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
}
}
#[allow(dead_code)]
pub fn help() {
println!("sat_hybrid - per-track SAT solver");
}
}
mod ours {
mod track3 {
use super::super::short_mask::ShortMask;
use super::super::phase_div;
use super::super::raw_roulette::RawRoulette;
use super::super::paged_small_signed::{SignedOrder as ClauseOrder,LiteralRanges};
use super::super::exact_div;
use super::super::paged_cache::{SmallBreakCache,Byte14BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;
const GIVEUP_LEVEL: usize = 6;
const GIVEUP_DEADLINE_FLIPS: usize = 563_000_000;
const GIVEUP_ENABLE_NV_MAX: usize = 25000;
pub fn solve(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let narrow=nv<=16384 && p_cnt.iter().zip(&n_cnt).all(|(&p,&n)|p+n<=255);
if narrow {solve_impl::<Byte14BreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)}else{solve_impl::<SmallBreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)}
}
fn solve_impl<C:CacheOps>(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let max_loss_per_flip=p_cnt.iter().chain(n_cnt.iter()).copied().max().unwrap_or(0) as usize;
let capacity_batch_limit=64usize.min((4096/max_loss_per_flip.max(1)).max(1));
let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(default_fuel);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let mut max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let base_max_flips = max_flips;
let mut vars = vec![false; nv];
let mut max_len = 0usize;
let mut lengths = vec![0usize; nc];
for i in 0..nc {
let len = (co[i + 1] - co[i]) as usize;
lengths[i] = len;
if len > max_len {
max_len = len;
}
}
let mut buckets: Vec<Vec<usize>> = (0..=max_len).map(|_| Vec::new()).collect();
for i in 0..nc {
buckets[lengths[i]].push(i);
}
for l in 1..=max_len {
for &cid in buckets[l].iter() {
let s = co[cid] as usize;
let e = co[cid + 1] as usize;
let mut already = false;
for j in s..e {
let lit = cl[j];
let v = (lit.abs() - 1) as usize;
if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
already = true;
break;
}
}
if already {
continue;
}
let mut best_score: u32 = 0;
let mut best_v = 0usize;
let mut best_target = false;
let mut count = 0;
for j in s..e {
let lit = cl[j];
let v = (lit.abs() - 1) as usize;
let target_val = lit > 0;
if vars[v] == target_val {
continue;
}
let score = if target_val { p_cnt[v] } else { n_cnt[v] };
if score > best_score {
best_score = score;
best_v = v;
best_target = target_val;
count = 1;
} else if score == best_score {
count += 1;
if rng.gen::<usize>() % count == 0 {
best_v = v;
best_target = target_val;
}
}
}
if best_score == 0 {
let idx = rng.gen::<usize>() % (e - s);
let lit = cl[s + idx];
best_v = (lit.abs() - 1) as usize;
best_target = lit > 0;
}
vars[best_v] = best_target;
}
}
let mut num_good = vec![0u8; nc];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let mut good = 0u8;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
good = good + 1;
}
}
num_good[i] = good;
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
let mut true_unsat = 0usize;
for i in 0..nc {
if num_good[i] == 0 {
residual.push(i as u32);
true_unsat += 1;
}
}
if true_unsat == 0 {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let giveup_deadline: usize = if nv <= GIVEUP_ENABLE_NV_MAX {
GIVEUP_DEADLINE_FLIPS
} else {
usize::MAX
};
let mut giveup_latched: bool = true_unsat <= GIVEUP_LEVEL;
let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
let check_interval = hp.as_ref().and_then(|h| h.check_interval)
.unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let mut probsat_weights = vec![0.0f64; nc + 1];
if avg_clause_size <= 3.2 {
let cb: f64 = 2.06;
for i in 0..=nc {
probsat_weights[i] = cb.powf(-(i as f64));
}
} else {
let cb: f64 = if avg_clause_size <= 4.2 {
2.85
} else if avg_clause_size <= 5.2 {
3.7
} else if avg_clause_size <= 6.2 {
5.1
} else {
5.4
};
for i in 0..=nc {
probsat_weights[i] = (i as f64 + 1.0).powf(-cb);
}
}
let roulette=RawRoulette::new(&probsat_weights);
let mut last_check_unsat = true_unsat;
let mut rounds = 0usize;
let mut stagnation = 0usize;
let paged_data=super::super::large_buffer::from_slice(all_data);
let all_data=&*paged_data;
let mut order = ClauseOrder::new(&cl, &co);
let ranges=LiteralRanges::new(all_off,p_bound);
let mut cache = C::new(nv, &cl, &co, &vars);
let mut short_mask=ShortMask::new(nv,&cl,&co,&vars);
let mut next_check_interval = check_interval;
unsafe {
if check_interval==0 && max_flips>0{let _=rounds%check_interval;}
let mut end=check_interval.min(max_flips);
'search: loop{
let capacity_steps=(end-rounds).min(capacity_batch_limit);
residual.reserve(capacity_steps*max_loss_per_flip);
let capacity_end=rounds+capacity_steps;
let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
let done=if small{if giveup_latched{run_sample_phase::<C,true,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}else{run_sample_phase::<C,true,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}}else{if giveup_latched{run_sample_phase::<C,false,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}else{run_sample_phase::<C,false,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}};
if done{break 'search;}
if rounds<end {continue;}
if rounds>=max_flips||true_unsat==0{break;}
if !giveup_latched && rounds >= giveup_deadline {break 'search;}
let progress = last_check_unsat as i64 - true_unsat as i64;
let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
if progress <= 0 {
stagnation += 1;
if stagnation >= 4 {
let kicks = 3; // stagnation reaches at most 4 before this reset
for _ in 0..kicks {
if true_unsat == 0 { break; }
let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
let pcid = *residual.get_unchecked(rid) as usize;
if cache.good(pcid) > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = 0usize;
let pce = order.len(pcid);
if pcs == pce { continue; }
let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
let v = (lit.abs() - 1) as usize;
let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v);
residual.reserve(de-ds);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v,gains);
short_mask.flip(v);
true_unsat=true_unsat+appended-gains as usize;
}
stagnation = 0;
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
} else {
stagnation = 0;
}
last_check_unsat = true_unsat;
if progress_ratio > 0.2 {
let increase = (base_max_flips / 100).max(1);
max_flips = (max_flips + increase).min(2 * base_max_flips);
}
end=rounds.saturating_add(check_interval).min(max_flips);
}
}
cache.restore_variables(&cl,&co,&mut vars);
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
#[inline(always)]
unsafe fn choose_shape<const LENGTH:usize,C:CacheOps>(ll:[usize;3],length:usize,cache:&C,roulette:&RawRoulette,rng:&mut SmallRng,probsat_weights:&[f64],nc:usize)->usize{
let clen=if LENGTH==0{length}else{LENGTH};let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0 {
if zero_mask&1!=0{ll[0]}else if zero_mask&2!=0{ll[1]}else{ll[2]}
}else{
let word=rng.gen::<u64>();
if clen==3 && (b0|b1|b2)<8 {
ll[roulette.choose(b0,b1,b2,word)]
}else{
let u=RawRoulette::sample(word);
let w0=*probsat_weights.get_unchecked(b0.min(nc));
let w1=if clen>1{*probsat_weights.get_unchecked(b1.min(nc))}else{0.0};
let w2=if clen>2{*probsat_weights.get_unchecked(b2.min(nc))}else{0.0};
let mut total=0.0;total+=w0;if clen>1{total+=w1;}if clen>2{total+=w2;}
let mut r=u*total;
r-=w0;
if r<=0.0{ll[0]}else if clen>1{
r-=w1;
if r<=0.0{ll[1]}else if clen>2{r-=w2;if r<=0.0{ll[2]}else{ll[0]}}else{ll[0]}
}else{ll[0]}
}
};
chosen_code
}
#[cold]
#[inline(never)]
unsafe fn choose_short<C:CacheOps>(ll:[usize;3],length:usize,cache:&C,roulette:&RawRoulette,rng:&mut SmallRng,probsat_weights:&[f64],nc:usize)->usize{
choose_shape::<0,C>(ll,length,cache,roulette,rng,probsat_weights,nc)
}
#[inline(always)]
unsafe fn run_sample_phase<C:CacheOps,const SMALL:bool,const LATCHED:bool>(cache:&mut C,order:&mut ClauseOrder,ranges:&LiteralRanges,residual:&mut Vec<u32>,rng:&mut SmallRng,roulette:&RawRoulette,probsat_weights:&[f64],nc:usize,all_data:&[u32],rounds:&mut usize,end:usize,true_unsat:&mut usize,giveup_latched:&mut bool,short_mask:&mut ShortMask)->bool{
while *rounds<end{
if *true_unsat==0{return true;}
let cid=if short_mask.all_satisfied(){
let mut cid=0usize;
for sample in 0..3{
let candidate=loop{
let word=rng.gen::<usize>();let len=residual.len();
let id=if SMALL{phase_div::rem_nonzero_small(word,len)}else{exact_div::rem(word,len)};
let c=*residual.get_unchecked(id)as usize;
if cache.good(c)>0{
let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
}else{break c;}
};
if sample==0{cid=candidate;}
}
cid
}else{
let mut cid=0usize;let mut min_len=usize::MAX;
for _ in 0..3{
let candidate=loop{
let word=rng.gen::<usize>();let len=residual.len();
let id=if SMALL{phase_div::rem_nonzero_small(word,len)}else{exact_div::rem(word,len)};
let c=*residual.get_unchecked(id)as usize;
if cache.good(c)>0{
let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
}else{break c;}
};
let len=order.len(candidate);if len<min_len{min_len=len;cid=candidate;}
}
cid
};
let cs = 0usize;
let ce = order.len(cid);
let clen = ce - cs;
if clen > 1 {
let ri = exact_div::clause_rem(rng.gen::<usize>(), clen);
order.swap(cid, 0, ri);
}
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let chosen_code=if clen==3{choose_shape::<3,C>(ll,clen,cache,roulette,rng,probsat_weights,nc)}else{choose_short(ll,clen,cache,roulette,rng,probsat_weights,nc)};
let v_idx=chosen_code>>1;
let (is,ie,ds,de)=ranges.of(chosen_code);
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v_idx);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v_idx,gains);
short_mask.flip(v_idx);
*true_unsat=*true_unsat+appended-gains as usize;
if !LATCHED{*giveup_latched |= *true_unsat <= GIVEUP_LEVEL;}
*rounds += 1;
}
false
}
}
mod track4 {
use super::super::phase_div;
use super::super::paged_order::SignedOrder;
use super::super::compact_ranges::CompactRanges;
use super::super::exact_coin::ExactCoin;
use super::super::clause_order::ClauseOrder;
use super::super::exact_div;
use super::super::paged_cache::{ExactBreakCache, NarrowLargeBreakCache, CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;
pub fn solve(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let narrow = p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n <= 255);
if narrow { solve_impl::<NarrowLargeBreakCache,true>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
else { solve_impl::<ExactBreakCache,false>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
}
fn solve_impl<C:CacheOps,const PACKED:bool>(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let page_data=super::super::large_buffer::from_slice(all_data);
let all_data:&[u32]=&page_data;
let max_loss_per_flip=p_cnt.iter().chain(n_cnt.iter()).copied().max().unwrap_or(0)as usize;
let capacity_batch_limit=64usize.min((4096/max_loss_per_flip.max(1)).max(1));
let nvf = nv as f64;
let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_low).unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
let appearances: Vec<u8> = (0..nv).map(|v| {
((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
}).collect();
drop(p_cnt);
drop(n_cnt);
let ng_len = nc; // one byte per clause; count is in 0..=3
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let byte_idx = i;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[byte_idx] += 1u8;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
for i in 0..nc {
if num_good[i] == 0 {
residual.push(i as u32);
}
}
if residual.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let base_prob = hp.as_ref().and_then(|h| h.base_prob)
.unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
let mut current_prob = base_prob;
let mut coin = ExactCoin::new(current_prob);
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.as_ref().and_then(|h| h.check_interval)
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let max_random_prob = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
let prob_adjustment_factor = 0.03;
let smoothing_factor = 0.8;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = residual.len();
let mut stagnation = 0usize;
let mut var_age = vec![0u8; nv];
let mut countdown = check_interval;
let mut rounds = 0usize;
let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
let mut order=SignedOrder::new(&cl,&co);
let ranges=CompactRanges::<PACKED>::new(all_off,p_bound);
let mut cache = C::new(nv, &cl, &co, &vars);
unsafe {
let mut end=if check_interval==0{max_flips}else{(check_interval-1).min(max_flips)};
'search: loop{
if residual.is_empty() {break 'search;}
let capacity_steps=(end-rounds).min(capacity_batch_limit);
residual.reserve(capacity_steps*max_loss_per_flip);
let capacity_end=rounds+capacity_steps;
let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
let done=if small{run_hot_phase::<C,PACKED,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,&appearances,&mut var_age,all_data,&mut rounds,capacity_end)}else{run_hot_phase::<C,PACKED,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,&appearances,&mut var_age,all_data,&mut rounds,capacity_end)};
if done{break 'search;}
if rounds<end{continue;}
if residual.is_empty()||rounds>=max_flips{break;}
let progress = last_check_residual as i64 - residual.len() as i64;
let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
if progress <= 0 {
stagnation += 1;
let prob_adjustment = prob_adjustment_factor
* (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
current_prob = (current_prob + prob_adjustment).min(max_random_prob);
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if residual.is_empty() { break; }
let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
let pcid = *residual.get_unchecked(rid) as usize;
let ng_val = cache.good(pcid);
if ng_val > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = 0usize;
let pce = order.len_bounded(pcid);
if pcs == pce { continue; }
let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
let v = (lit.abs() - 1) as usize;
let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v);
residual.reserve(de-ds);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v,gains);
*var_age.get_unchecked_mut(v) = 0;
}
stagnation = 0;
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
current_prob = base_prob;
} else {
stagnation = 0;
current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
}
coin=ExactCoin::new(current_prob);
last_check_residual = residual.len();
end=rounds.saturating_add(check_interval).min(max_flips);
}
}
cache.restore_variables(&cl,&co,&mut vars);
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
#[inline(always)]unsafe fn run_hot_phase<C:CacheOps,const PACKED:bool,const SMALL:bool>(
cache:&mut C,order:&mut SignedOrder,ranges:&CompactRanges<PACKED>,
residual:&mut Vec<u32>,rng:&mut SmallRng,coin:&super::super::exact_coin::ExactCoin,
appearances:&[u8],var_age:&mut[u8],all_data:&[u32],
rounds_ref:&mut usize,capacity_end:usize
)->bool{
let mut rounds=*rounds_ref;
while rounds<capacity_end{
let rand_val = rng.gen::<usize>();
let cid=loop {
let len=residual.len();
let id=if SMALL{phase_div::rem_nonzero_small(rand_val,len)}else{exact_div::rem(rand_val,len)};
let c=*residual.get_unchecked(id) as usize;
if cache.good(c)==0 {break c;}
let p=residual.as_mut_ptr();
p.add(id).write(*p.add(len-1));
residual.set_len(len-1);
if len==1 {*rounds_ref=rounds;return true;}
};
let cs = 0usize;
let ce = order.len_bounded(cid);
let clen = ce - cs;
let permutation_rank=if clen>1{
let rank=exact_div::clause_rem(rand_val,clen);order.swap(cid,0,rank);rank
}else{0};
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0 {
ClauseOrder::choose_zero_rank(ll,zero_mask,rand_val,permutation_rank)
}else if coin.draw(rng){ll[0]}else{
let mut selected=ll[0];let mut min_sad=b0;
let mut penalty=(*appearances.get_unchecked(vv[0])as i32)-(((*var_age.get_unchecked(vv[0])as usize)/2).min(50)as i32);
if clen>1 && min_sad>1 {
let p=(*appearances.get_unchecked(vv[1])as i32)-(((*var_age.get_unchecked(vv[1])as usize)/2).min(50)as i32);
if b1<min_sad || p<penalty {selected=ll[1];min_sad=b1.min(min_sad);penalty=p;}
}
if clen>2 && min_sad>1 {
let p=(*appearances.get_unchecked(vv[2])as i32)-(((*var_age.get_unchecked(vv[2])as usize)/2).min(50)as i32);
if b2<min_sad || p<penalty {selected=ll[2];}
}
selected
};
let v_idx=chosen_code>>1;
let (is,ie,ds,de)=ranges.of(chosen_code);
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v_idx);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v_idx,gains);
*var_age.get_unchecked_mut(v_idx)=1;
if clen==3 {
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};
let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
let x=var_age.get_unchecked_mut(a);*x=x.saturating_add(1);
let x=var_age.get_unchecked_mut(b);*x=x.saturating_add(1);
}else if clen==2 {
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};
let x=var_age.get_unchecked_mut(a);*x=x.saturating_add(1);
}
rounds += 1;
}
*rounds_ref=rounds;
false
}
}
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
pub base_prob: Option<f64>,
pub max_prob: Option<f64>,
pub check_interval: Option<usize>,
pub stagnation_limit: Option<usize>,
pub perturbation_flips: Option<usize>,
pub max_fuel_high: Option<f64>,
pub max_fuel_low: Option<f64>,
}
#[allow(dead_code)]
pub fn help() {
println!("SAT imp v3 - per-track files (track1/track2/track3/track4/track5)");
}
pub fn solve_challenge(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
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
let density = nc as f64 / nv as f64;
if density >= 4.25 {
if nv <= 5000 {
return Err(anyhow!("sat_exact: ours route track1 is not used by this composite"));
}
if nv <= 7500 {
return Err(anyhow!("sat_exact: ours route track2 is not used by this composite"));
}
return track3::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
}
if density < 4.18 {
return track4::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
}
Err(anyhow!("sat_exact: ours route track5 is not used by this composite"))
}
}
mod tw6 {
mod track2 {
use super::super::phase_div;
use super::super::exact_coin::ExactCoin;
use super::super::paged_flip_ranges::FlipRanges;
use super::super::paged_small_order::ClauseOrder;
use super::super::exact_div;
use super::super::paged_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;
#[inline(always)]
unsafe fn apply_flip<C:CacheOps>(
v_idx: usize,
all_data:&[u32],
ranges:&mut FlipRanges,
cache: &mut C,
residual: &mut Vec<u32>,
unsat_count: &mut usize,
var_age: &mut [u8],
) {
let (is,ie,ds,de)=ranges.flip(v_idx);
let ia=all_data;let da=all_data;
let gains=cache.gain_many(ia.get_unchecked(is..ie),v_idx);
residual.reserve(de-ds);
let old_len=residual.len();
let appended=cache.lose_many(da.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v_idx,gains);
*unsat_count=*unsat_count+appended-gains as usize;
*var_age.get_unchecked_mut(v_idx) = 0;
}
pub fn solve(
challenge: &Challenge,
hp: &Option<Hyperparameters>,
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let nv=challenge.num_variables;
let mut degree=vec![0u32;nv];
for orig in &challenge.clauses {for &l in orig {degree[(l.abs()-1)as usize]+=1;}}
if nv<=8192 && degree.iter().all(|&d|d<=255) {
solve_impl::<Byte13BreakCache,true>(challenge,hp,save_solution)
} else {solve_impl::<SmallBreakCache,false>(challenge,hp,save_solution)}
}
fn solve_impl<C:CacheOps,const COMPACT:bool>(
challenge: &Challenge,
hp: &Option<Hyperparameters>,
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let nv = challenge.num_variables;
let _ = save_solution(&Solution { variables: vec![false; nv] });
let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
let mut p_cnt = vec![0u32; nv];
let mut n_cnt = vec![0u32; nv];
let mut compact_clauses = Vec::with_capacity(challenge.clauses.len());
let mut compact_len = Vec::with_capacity(challenge.clauses.len());
let mut total_lits = 0usize;
for orig in &challenge.clauses {
let (a, b, c) = (orig[0], orig[1], orig[2]);
if a == -b || a == -c || b == -c { continue; }
let mut lits = [0i32; 3];
let mut len = 1usize;
lits[0] = a;
let va = (a.abs() - 1) as usize;
if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
if b != a {
lits[len] = b;
len += 1;
let vb = (b.abs() - 1) as usize;
if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
}
if c != a && c != b {
lits[len] = c;
len += 1;
let vc = (c.abs() - 1) as usize;
if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
}
total_lits += len;
compact_clauses.push(lits);
compact_len.push(len as u8);
}
let nc = compact_clauses.len();
let mut p_off = vec![0u32; nv + 1];
let mut n_off = vec![0u32; nv + 1];
for v in 0..nv {
p_off[v + 1] = p_off[v] + p_cnt[v];
n_off[v + 1] = n_off[v] + n_cnt[v];
}
let mut p_data = vec![0u32; p_off[nv] as usize];
let mut n_data = vec![0u32; n_off[nv] as usize];
let mut p_pos = p_off[..nv].to_vec();
let mut n_pos = n_off[..nv].to_vec();
let mut cl = Vec::with_capacity(total_lits);
let mut co = Vec::with_capacity(nc + 1);
co.push(0u32);
for (ci, lits) in compact_clauses.iter().enumerate() {
let ci_u32 = ci as u32;
let len = compact_len[ci] as usize;
for j in 0..len {
let lit = lits[j];
let v = (lit.abs() - 1) as usize;
if lit > 0 {
p_data[p_pos[v] as usize] = ci_u32;
p_pos[v] += 1;
} else {
n_data[n_pos[v] as usize] = ci_u32;
n_pos[v] += 1;
}
}
cl.extend_from_slice(&lits[..len]);
co.push(cl.len() as u32);
}
let density = nc as f64 / nv as f64;
let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(250_000_000_000.0);
let var_appearances: Vec<u16> = (0..nv)
.map(|v| (p_cnt[v] + n_cnt[v]) as u16)
.collect();
let max_app = *var_appearances.iter().max().unwrap_or(&1) as f64;
let base_keep_prob = if nv <= 20000 { 0.15 } else { 0.09 };
let phase_save_threshold: Vec<u32> = (0..nv).map(|v| {
let app = var_appearances[v] as f64;
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
let skew = if np + nn > 0.0 { (np - nn).abs() / (np + nn) } else { 0.0 };
let prob = (base_keep_prob + 0.35 * (app / max_app) + 0.25 * skew).clamp(0.0, 0.90);
(prob * 4294967295.0) as u32
}).collect();
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let nad = 1.0;
let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
let base_prob: f64 = 0.52;
let max_random_prob: f64 = 0.9;
let smoothing_factor: f64 = 0.8;
let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
let check_interval = (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;
let variance_interval = 1000usize;
let raw_restarts = if nv <= 12000 { 4usize } else if nv <= 40000 { 3usize } else { 2usize };
let restart_attempts = raw_restarts
.saturating_sub(if density > 5.0 { 1 } else { 0 })
.max(1)
.min(max_flips.max(1));
let mut best_unsat = nc + 1;
let mut record_limit=best_unsat.min(13);
let mut best_vars = vec![false; nv];
let mut vars = vec![false; nv];
let mut num_good = vec![0u8; nc];
let mut residual: Vec<u32> = Vec::with_capacity(nc);
let mut var_age = vec![0u8; nv];
let mut order = ClauseOrder::new(&cl, &co);
let mut all_off=Vec::with_capacity(nv+1);let mut p_bound=Vec::with_capacity(nv);
let mut all_data=Vec::with_capacity(p_data.len()+n_data.len());all_off.push(0u32);
for v in 0..nv {
all_data.extend_from_slice(&p_data[p_off[v]as usize..p_off[v+1]as usize]);
p_bound.push(all_data.len()as u32);
all_data.extend_from_slice(&n_data[n_off[v]as usize..n_off[v+1]as usize]);
all_off.push(all_data.len()as u32);
}
let all_data=super::super::large_buffer::from_slice(&all_data);
for attempt in 0..restart_attempts {
let attempt_budget = if restart_attempts == 1 {
max_flips
} else {
max_flips / restart_attempts
+ if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
};
let attempt_random_threshold = if attempt == 0 {
random_threshold
} else {
(random_threshold * (1.0 + 0.45 * attempt as f64)).min(0.08)
};
let attempt_relax = if attempt == 0 {
0.0
} else {
(0.06 * attempt as f64).min(0.18)
};
vars.fill(false);
for v in 0..nv {
let np = p_cnt[v] as usize;
let nn = n_cnt[v] as usize;
if nn == 0 && np > 0 { vars[v] = true; continue; }
if np == 0 && nn > 0 { continue; }
let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
if vad <= nad {
vars[v] = rng.gen_bool(attempt_random_threshold);
} else {
let bias_prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
let prob = (bias_prob * (1.0 - attempt_relax) + 0.5 * attempt_relax).clamp(0.0, 1.0);
vars[v] = rng.gen_bool(prob);
}
}
if attempt > 0 && best_unsat < nc {
for v in 0..nv {
if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<u32>() < phase_save_threshold[v] {
vars[v] = best_vars[v];
}
}
}
num_good.fill(0);
residual.clear();
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let mut ng = 0u8;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
ng += 1;
}
}
num_good[i] = ng;
if ng == 0 {
residual.push(i as u32);
}
}
let mut unsat_count = residual.len();
if unsat_count < best_unsat {
best_unsat = unsat_count;
record_limit=best_unsat.min(13);
best_vars.copy_from_slice(&vars);
}
if unsat_count == 0 {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let mut current_prob = base_prob;
var_age.fill(0);
let mut rounds = 0usize;
let mut stagnation = 0usize;
let mut max_unsat_window = unsat_count;
let mut min_unsat_window = unsat_count;
let mut coin=ExactCoin::new(current_prob);
let mut cache = C::new(nv, &cl, &co, &vars);
let mut ranges=FlipRanges::new(&all_off,&p_bound,&vars);
let mut next_check_interval = check_interval;
let mut next_variance_interval = variance_interval;
unsafe {
if rounds<attempt_budget && unsat_count>0{
if check_interval==0{let _=rounds%check_interval;}
if variance_interval==0{let _=rounds%variance_interval;}
}
let max_loss_per_flip=(0..nv).map(|v|(p_bound[v]-all_off[v]).max(all_off[v+1]-p_bound[v])as usize).max().unwrap_or(0);
let capacity_batch_limit=(4096usize/max_loss_per_flip.max(1)).max(1).min(64);
'attempt_search: loop{
let event_end=next_check_interval.min(next_variance_interval).min(attempt_budget);
let steps=(event_end-rounds).min(capacity_batch_limit);
residual.reserve(steps*max_loss_per_flip);
let batch_end=rounds+steps;
let small=residual.len().checked_add(steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
if small{
while (rounds<batch_end) & (unsat_count!=0){
if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }
let rand_val = rng.gen::<usize>();
let cid=loop{
let id=phase_div::rem_nonzero_small(rand_val,residual.len());
let c=*residual.get_unchecked(id)as usize;
if cache.good(c)>0{
let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
}else{break c;}
};
let cs = 0usize;
let ce = order.len(cid);
let clen = ce - cs;
if clen>1{order.swap(cid,0,exact_div::clause_rem(rand_val,clen));}
let vv=order.variables(cid);
let v_idx=if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)};
apply_flip_main(
v_idx,
&all_data,&mut ranges,
&mut cache,
&mut residual,
&mut unsat_count,
&mut var_age,
);
*var_age.get_unchecked_mut(v_idx)=1;
if clen==3{
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
let age=var_age.get_unchecked_mut(b);*age=age.saturating_add(1);
}else if clen==2{
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
}
rounds += 1;
if unsat_count < record_limit {
best_unsat = unsat_count;
record_limit=best_unsat.min(13);
ranges.restore_variables(&all_off,&p_bound,&mut vars);
best_vars.copy_from_slice(&vars);
}
}
}else{
while (rounds<batch_end) & (unsat_count!=0){
if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }
let rand_val = rng.gen::<usize>();
let cid=loop{
let id=exact_div::rem(rand_val,residual.len());
let c=*residual.get_unchecked(id)as usize;
if cache.good(c)>0{
let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
}else{break c;}
};
let cs = 0usize;
let ce = order.len(cid);
let clen = ce - cs;
if clen>1{order.swap(cid,0,exact_div::clause_rem(rand_val,clen));}
let vv=order.variables(cid);
let v_idx=if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)};
apply_flip_main(
v_idx,
&all_data,&mut ranges,
&mut cache,
&mut residual,
&mut unsat_count,
&mut var_age,
);
*var_age.get_unchecked_mut(v_idx)=1;
if clen==3{
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
let age=var_age.get_unchecked_mut(b);*age=age.saturating_add(1);
}else if clen==2{
let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
}
rounds += 1;
if unsat_count < record_limit {
best_unsat = unsat_count;
record_limit=best_unsat.min(13);
ranges.restore_variables(&all_off,&p_bound,&mut vars);
best_vars.copy_from_slice(&vars);
}
}
}
if unsat_count==0{break 'attempt_search;}
if rounds<event_end{continue;}
if rounds>=attempt_budget||unsat_count==0{break;}
if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }
if rounds==next_check_interval{
next_check_interval=next_check_interval.checked_add(check_interval).unwrap_or(usize::MAX);
if unsat_count < best_unsat {
best_unsat = unsat_count;
record_limit=best_unsat.min(13);
ranges.restore_variables(&all_off,&p_bound,&mut vars);
best_vars.copy_from_slice(&vars);
}
}
if rounds==next_variance_interval{
next_variance_interval=next_variance_interval.checked_add(variance_interval).unwrap_or(usize::MAX);
let variance = max_unsat_window.saturating_sub(min_unsat_window);
if variance <= 2 {
stagnation += 1;
current_prob = (current_prob + 0.15).min(max_random_prob);
} else if variance <= 6 {
stagnation += 1;
current_prob = (current_prob + 0.05).min(max_random_prob);
} else if variance >= 20 {
stagnation = 0;
current_prob = base_prob;
} else {
stagnation = 0;
current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
}
if stagnation >= 3 {
let kicks = if stagnation >= 6 { 8 } else { 4 };
for _ in 0..kicks {
if residual.is_empty() || unsat_count == 0 { break; }
let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
let pcid = *residual.get_unchecked(rid) as usize;
if cache.good(pcid) > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = 0usize;
let pce = order.len(pcid);
if pcs == pce { continue; }
let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
let v = (lit.abs() - 1) as usize;
apply_flip(
v,
&all_data,&mut ranges,
&mut cache,
&mut residual,
&mut unsat_count,
&mut var_age,
);
}
stagnation = 0;
}
coin=ExactCoin::new(current_prob);
max_unsat_window = unsat_count;
min_unsat_window = unsat_count;
}
if unsat_count==0{let _:usize=rng.gen();break;}
}
}
if unsat_count < best_unsat {
best_unsat = unsat_count;
record_limit=best_unsat.min(13);
ranges.restore_variables(&all_off,&p_bound,&mut vars);
best_vars.copy_from_slice(&vars);
}
if unsat_count == 0 {
ranges.restore_variables(&all_off,&p_bound,&mut vars);
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
}
let _ = save_solution(&Solution { variables: best_vars });
Ok(())
}
#[inline(always)]
unsafe fn apply_flip_main<C:CacheOps>(
v_idx: usize,
all_data:&[u32],
ranges:&mut FlipRanges,
cache: &mut C,
residual: &mut Vec<u32>,
unsat_count: &mut usize,
var_age: &mut [u8],
) {
let (is,ie,ds,de)=ranges.flip(v_idx);
let ia=all_data;let da=all_data;
let gains=cache.gain_many(ia.get_unchecked(is..ie),v_idx);
let old_len=residual.len();
let appended=cache.lose_many(da.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v_idx,gains);
*unsat_count=*unsat_count+appended-gains as usize;
}
#[inline(always)]
unsafe fn choose_shape<C:CacheOps,const COMPACT:bool,const LENGTH:usize>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
let clen=if LENGTH==0{length}else{LENGTH};
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_result=if zero_mask!=0 {
if zero_mask&1!=0{vv[0]}else if zero_mask&2!=0{vv[1]}else{vv[2]}
}else if b0==1{let _:u64=rng.gen();vv[0]}else if coin.draw(rng){vv[0]}
else if COMPACT && clen>1 && b1==1{vv[1]}
else if COMPACT && clen>2 && b2==1{vv[2]}else{
if COMPACT {
let mut selected=vv[0];
let mut min_sad=b0;
let mut penalty=(*var_appearances.get_unchecked(vv[0])as i32)-(((*var_age.get_unchecked(vv[0])as usize)/2).min(50)as i32);
if clen>1 && min_sad>1 {
let p=(*var_appearances.get_unchecked(vv[1])as i32)-(((*var_age.get_unchecked(vv[1])as usize)/2).min(50)as i32);
if b1<min_sad || p<penalty {selected=vv[1];min_sad=b1.min(min_sad);penalty=p;}
}
if clen>2 && min_sad>1 {
let p=(*var_appearances.get_unchecked(vv[2])as i32)-(((*var_age.get_unchecked(vv[2])as usize)/2).min(50)as i32);
if b2<min_sad || p<penalty {selected=vv[2];min_sad=b2.min(min_sad);penalty=p;}
}
selected
}else{
let mut selected=vv[0];
let mut min_sad=b0;
let mut min_weight=(b0)*1000+(*var_appearances.get_unchecked(vv[0]) as usize)-(((*var_age.get_unchecked(vv[0]) as usize)/2).min(50));
if clen>1 && min_sad>1 {
let sad=b1.min(min_sad);
let weight=(sad)*1000+(*var_appearances.get_unchecked(vv[1]) as usize)-(((*var_age.get_unchecked(vv[1]) as usize)/2).min(50));
if weight<min_weight {selected=vv[1];min_sad=sad;min_weight=weight;}
}
if clen>2 && min_sad>1 {
let sad=b2.min(min_sad);
let weight=(sad)*1000+(*var_appearances.get_unchecked(vv[2]) as usize)-(((*var_age.get_unchecked(vv[2]) as usize)/2).min(50));
if weight<min_weight {selected=vv[2];min_sad=sad;min_weight=weight;}
}
selected
}
};
chosen_result
}
#[cold]#[inline(never)]
unsafe fn choose_short<C:CacheOps,const COMPACT:bool>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
choose_shape::<C,COMPACT,0>(vv,length,cache,coin,rng,var_appearances,var_age)
}
#[cfg(test)]mod one_tests{
fn reference(b:[usize;3],p:[i32;3],len:usize)->usize{let mut chosen=0;let mut low=b[0];let mut w=1000*(low as i32)+p[0];for i in 1..len{if low<=1{break;}let n=b[i].min(low);let nw=1000*(n as i32)+p[i];if nw<w{chosen=i;low=n;w=nw;}}chosen}
#[test]fn positive_score_one_shortcut_all_penalty_extrema(){for a in 1..8{for b in 1..8{for c in 1..8{for p in [-50,-1,0,1,127,255]{for q in [-50,0,255]{for r in [-50,0,255]{for len in 1..=3{let scores=[a,b,c];if let Some(i)=(0..len).find(|&i|scores[i]==1){assert_eq!(reference(scores,[p,q,r],len),i);}}}}}}}}
}
}
}
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
#[derive(Serialize, Deserialize)]
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
#[allow(dead_code)]
pub fn help() {
println!(
"sat_tailwalk_v6: use {{\"max_fuel_high\":175000000000}} for n_vars=10000, ratio=4267; use null for other tracks"
);
}
pub fn solve_challenge(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
serde_json::from_value(Value::Object(m.clone())).ok()
});
let nv = challenge.num_variables;
let _ = save_solution(&Solution { variables: vec![false; nv] });
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
Err(anyhow!(
"sat_exact: tw6 route {:?} is not used by this composite (only Track2)",
route
))
}
}
mod sp128 {
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
mod engine_b {
mod track_t1 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess(challenge, save_solution);
let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
let max_fuel = hp.max_fuel_high.unwrap_or(default_fuel);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
for v in 0..nv {
let np = p_cnt[v] as usize;
let nn = n_cnt[v] as usize;
if np > nn {
vars[v] = true;
} else if np < nn {
vars[v] = false;
} else {
vars[v] = rng.gen_bool(0.5);
}
}
let mut num_good = vec![0u8; nc];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[i] += 1;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
let mut true_unsat = 0usize;
for i in 0..nc {
if num_good[i] == 0 {
residual.push(i as u32);
true_unsat += 1;
}
}
if true_unsat == 0 {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let mut best_unsat = true_unsat;
let mut best_vars = vars.clone();
let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let mut probsat_weights = vec![0.0f64; nc + 1];
if avg_clause_size <= 3.2 {
let cb: f64 = 2.06;
for i in 0..=nc {
probsat_weights[i] = cb.powf(-(i as f64));
}
} else {
let cb: f64 = if avg_clause_size <= 4.2 {
2.85
} else if avg_clause_size <= 5.2 {
3.7
} else if avg_clause_size <= 6.2 {
5.1
} else {
5.4
};
for i in 0..=nc {
probsat_weights[i] = (i as f64 + 1.0).powf(-cb);
}
}
let mut last_check_unsat = true_unsat;
let mut rounds = 0usize;
let mut stagnation = 0usize;
unsafe {
loop {
if rounds >= max_flips { break; }
if true_unsat == 0 { break; }
if rounds % check_interval == 0 && rounds > 0 {
let progress = last_check_unsat as i64 - true_unsat as i64;
let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
if progress <= 0 {
stagnation += 1;
if stagnation >= 4 {
if stagnation >= 15 {
vars.copy_from_slice(&best_vars);
let perturb_cnt = (nv / 20).max(1);
for _ in 0..perturb_cnt {
let v = rng.gen::<usize>() % nv;
*vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
}
residual.clear();
true_unsat = 0;
for i in 0..nc {
let s = *co.get_unchecked(i) as usize;
let e = *co.get_unchecked(i + 1) as usize;
let mut good = 0u8;
for j in s..e {
let l = *cl.get_unchecked(j);
let v = (l.abs() - 1) as usize;
if (l > 0 && *vars.get_unchecked(v)) || (l < 0 && !*vars.get_unchecked(v)) {
good = good.saturating_add(1);
}
}
*num_good.get_unchecked_mut(i) = good;
if good == 0 {
residual.push(i as u32);
true_unsat += 1;
}
}
stagnation = 0;
} else {
let kicks = if stagnation >= 8 { 6 } else { 3 };
for _ in 0..kicks {
if true_unsat == 0 { break; }
let rid = rng.gen::<usize>() % residual.len();
let pcid = *residual.get_unchecked(rid) as usize;
if *num_good.get_unchecked(pcid) > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
*ng = ng.saturating_add(1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
let new_val = ng.saturating_sub(1);
*ng = new_val;
if new_val == 0 {
residual.push(c as u32);
true_unsat += 1;
}
}
*vars.get_unchecked_mut(v) = !was_true;
}
stagnation = 0;
}
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
} else {
stagnation = 0;
}
last_check_unsat = true_unsat;
}
if true_unsat == 0 { break; }
let mut cid = usize::MAX;
let mut min_len = usize::MAX;
for _ in 0..3 {
while !residual.is_empty() {
let id = rng.gen::<usize>() % residual.len();
let cand = *residual.get_unchecked(id) as usize;
if *num_good.get_unchecked(cand) > 0 {
residual.swap_remove(id);
} else {
let c_s = *co.get_unchecked(cand) as usize;
let c_e = *co.get_unchecked(cand + 1) as usize;
let clen = c_e - c_s;
if clen < min_len {
min_len = clen;
cid = cand;
}
break;
}
}
if residual.is_empty() { break; }
}
if cid == usize::MAX { break; }
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
if clen > 1 {
let ri = rng.gen::<usize>() % clen;
cl.swap(cs, cs + ri);
}
let mut zero_found: Option<usize> = None;
'outer_h: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
continue 'outer_h;
}
}
zero_found = Some(abs_l);
break;
}
let v_idx = if let Some(v) = zero_found {
v
} else {
let mut total_weight = 0.0;
let mut weights = [0.0; 256];
let clen_actual = (ce - cs).min(256);
for idx in 0..clen_actual {
let j = cs + idx;
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
sad += 1;
}
}
let w = *probsat_weights.get_unchecked(sad.min(nc));
weights[idx] = w;
total_weight += w;
}
let mut r = rng.gen::<f64>() * total_weight;
let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
for idx in 0..clen_actual {
r -= weights[idx];
if r <= 0.0 {
v_min = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
break;
}
}
v_min
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
*ng = ng.saturating_add(1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = num_good.get_unchecked_mut(c);
let new_val = ng.saturating_sub(1);
*ng = new_val;
if new_val == 0 {
residual.push(c as u32);
true_unsat += 1;
}
}
*vars.get_unchecked_mut(v_idx) = !was_true;
if true_unsat < best_unsat {
best_unsat = true_unsat;
best_vars.copy_from_slice(&vars);
}
rounds += 1;
}
}
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
}
mod track_t3 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess(challenge, save_solution);
let nvf = nv as f64;
let max_fuel = hp.max_fuel_low.unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
let appearances: Vec<u8> = (0..nv).map(|v| {
((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
}).collect();
drop(p_cnt);
drop(n_cnt);
let ng_len = (nc + 3) >> 2;
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let shift = (i & 3) << 1;
let byte_idx = i >> 2;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[byte_idx] += 1u8 << shift;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
for i in 0..nc {
if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
residual.push(i as u32);
}
}
if residual.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let base_prob = hp.base_prob
.unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
let mut current_prob = base_prob;
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let max_random_prob = hp.max_prob.unwrap_or(0.9);
let prob_adjustment_factor = 0.03;
let smoothing_factor = 0.8;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.perturbation_flips
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.stagnation_limit
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = residual.len();
let mut stagnation = 0usize;
let mut var_age = vec![0u8; nv];
let mut countdown = check_interval;
let mut rounds = 0usize;
let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
unsafe {
't3: loop {
if residual.is_empty() || rounds >= max_flips { break; }
countdown -= 1;
if countdown == 0 {
countdown = check_interval;
let progress = last_check_residual as i64 - residual.len() as i64;
let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
if progress <= 0 {
stagnation += 1;
let prob_adjustment = prob_adjustment_factor
* (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
current_prob = (current_prob + prob_adjustment).min(max_random_prob);
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if residual.is_empty() { break; }
let rid = rng.gen::<usize>() % residual.len();
let pcid = *residual.get_unchecked(rid) as usize;
let ng_val = (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3;
if ng_val > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
*num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
let rl = residual.len();
*residual.get_unchecked_mut(rl) = c as u32;
residual.set_len(rl + 1);
}
}
*vars.get_unchecked_mut(v) = !was_true;
*var_age.get_unchecked_mut(v) = 0;
}
stagnation = 0;
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
current_prob = base_prob;
} else {
stagnation = 0;
current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
}
last_check_residual = residual.len();
}
let r0 = rounds;
let nblk = (max_flips - rounds).min(countdown);
for _ in 0..nblk {
let rand_val = rng.gen::<usize>();
let mut cid = 0usize;
let mut found = false;
while !residual.is_empty() {
let id = rand_val % residual.len();
let candidate = *residual.get_unchecked(id) as usize;
let ng_val = (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3;
if ng_val > 0 {
residual.swap_remove(id);
} else {
cid = candidate;
found = true;
break;
}
}
if !found { break 't3; }   // i1: sortie DEFINITIVE, pas seulement du bloc
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
let ri = rand_val % clen;
cl.swap(cs, cs + ri);
let mut zero_buf: [usize; 3] = [0; 3];
let mut zero_cnt: usize = 0;
if clen == 3 {
macro_rules! probe3 { ($j:expr) => {{
let l = *cl.get_unchecked($j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if l > 0 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
let mut crit = false;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
crit = true;
break;
}
}
if !crit {
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
}}}
probe3!(cs);
probe3!(cs + 1);
probe3!(cs + 2);
} else {
'outer: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if l > 0 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
continue 'outer;
}
}
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
}
let v_idx = if zero_cnt > 0 {
if zero_cnt == 1 {
*zero_buf.get_unchecked(0)
} else {
*zero_buf.get_unchecked(rand_val % zero_cnt)
}
} else if rng.gen::<f64>() < current_prob {
(cl.get_unchecked(cs).abs() - 1) as usize
} else {
let mut min_sad = usize::MAX;
let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
let mut min_weight = usize::MAX;
for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if l > 0 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
sad += 1;
}
if sad >= min_sad { break; }
}
if sad == 0 {
let app = *appearances.get_unchecked(abs_l) as usize;
let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
let adjusted_weight = app.saturating_sub(age_bonus);
if min_sad > 0 || adjusted_weight < min_weight {
min_sad = 0;
min_weight = adjusted_weight;
v_min = abs_l;
}
} else if min_sad > 0 {
let app = *appearances.get_unchecked(abs_l) as usize;
let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
if combined_weight < min_weight {
min_sad = sad;
min_weight = combined_weight;
v_min = abs_l;
}
if min_sad <= 1 { break; }
}
}
v_min
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
*num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
let rl = residual.len();
*residual.get_unchecked_mut(rl) = c as u32;
residual.set_len(rl + 1);
}
}
*vars.get_unchecked_mut(v_idx) = !was_true;
*var_age.get_unchecked_mut(v_idx) = 0;
for j in cs..ce {
let l = *cl.get_unchecked(j);
let var = (l.abs() - 1) as usize;
let age = var_age.get_unchecked_mut(var);
*age = age.saturating_add(1);
}
rounds += 1;
if residual.is_empty() { break; }   // (B) non hoistable, re-testee par tour
}   // i1 HOIST: fin du bloc
countdown -= (rounds - r0).saturating_sub(1);
}
}
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
}
mod track_t4 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess(challenge, save_solution);
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
let w = vec![1u8; nc];
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
unsafe {
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
best_unsat = unsat_list.len();
best_vars.copy_from_slice(&vars);
stagnation_count = 0;
}
if rounds % check_interval == 0 && rounds > 0 {
let progress = last_check_residual as i64 - unsat_list.len() as i64;
if progress <= 0 {
stagnation += 1;
if stagnation >= stagnation_limit_t4 {
let kicks = if stagnation >= 8 { 6 } else { 3 };
for _ in 0..kicks {
if unsat_list.is_empty() { break; }
let rid = rng.gen::<usize>() % unsat_list.len();
let pcid = *unsat_list.get_unchecked(rid) as usize;
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit.abs() - 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = *num_good.get_unchecked(c);
if ng == 0 {
let pos = *unsat_pos.get_unchecked(c) as usize;
let last_idx = unsat_list.len() - 1;
let last_c = *unsat_list.get_unchecked(last_idx) as usize;
*unsat_list.get_unchecked_mut(pos) = last_c as u32;
*unsat_pos.get_unchecked_mut(last_c) = pos as u32;
*unsat_pos.get_unchecked_mut(c) = u32::MAX;
unsat_list.pop();
}
*num_good.get_unchecked_mut(c) = ng + 1;
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = *num_good.get_unchecked(c);
*num_good.get_unchecked_mut(c) = ng - 1;
if ng == 1 {
*unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
unsat_list.push(c as u32);
}
}
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
let cid = {
let uc = unsat_list.len();
let i1 = (rand_val as u32 as usize) % uc;
let i2 = (rand_val >> 32) % uc;
let c1 = *unsat_list.get_unchecked(i1) as usize;
let c2 = *unsat_list.get_unchecked(i2) as usize;
if *w.get_unchecked(c1) >= *w.get_unchecked(c2) { c1 } else { c2 }
};
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
if clen > 1 {
let ri = rand_val % clen;
cl.swap(cs, cs + ri);
}
let mut zero_buf: [usize; 3] = [0; 3];
let mut zero_cnt: usize = 0;
'outer_h: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
continue 'outer_h;
}
}
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
let v_idx = if zero_cnt > 0 {
if zero_cnt == 1 {
*zero_buf.get_unchecked(0)
} else {
*zero_buf.get_unchecked(rand_val % zero_cnt)
}
} else {
let mut pw_weights: [u32; 3] = [0; 3];
let mut pw_vars: [usize; 3] = [0; 3];
let mut pw_cnt: usize = 0;
let mut total_pw: u32 = 0;
for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l.abs() - 1) as usize;
let (os, oe) = if *vars.get_unchecked(abs_l) {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
} else {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if *num_good.get_unchecked(c) == 1 {
sad += 1;
}
}
let b_idx = sad.min(15);
let pw = *probs_break.get_unchecked(b_idx);
*pw_weights.get_unchecked_mut(pw_cnt) = pw;
*pw_vars.get_unchecked_mut(pw_cnt) = abs_l;
total_pw += pw;
pw_cnt += 1;
}
let mut r = (rand_val as u32) % total_pw.max(1);
let mut chosen = *pw_vars.get_unchecked(0);
for i in 0..pw_cnt {
let pw = *pw_weights.get_unchecked(i);
if r < pw {
chosen = *pw_vars.get_unchecked(i);
break;
}
r -= pw;
}
chosen
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let ng = *num_good.get_unchecked(c);
if ng == 0 {
let pos = *unsat_pos.get_unchecked(c) as usize;
let last_idx = unsat_list.len() - 1;
let last_c = *unsat_list.get_unchecked(last_idx) as usize;
*unsat_list.get_unchecked_mut(pos) = last_c as u32;
*unsat_pos.get_unchecked_mut(last_c) = pos as u32;
*unsat_pos.get_unchecked_mut(c) = u32::MAX;
unsat_list.pop();
}
*num_good.get_unchecked_mut(c) = ng + 1;
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let ng = *num_good.get_unchecked(c);
*num_good.get_unchecked_mut(c) = ng - 1;
if ng == 1 {
*unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
unsat_list.push(c as u32);
}
}
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
}
}
let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
let _ = save_solution(&Solution { variables: final_vars });
Ok(())
}
}
mod track_t5 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
let super::Prepared {
mut rng,
nv,
nc,
density: _density,
p_cnt,
n_cnt,
all_off,
p_bound,
all_data,
mut cl,
co,
} = super::preprocess(challenge, save_solution);
let max_fuel = hp.max_fuel_high.unwrap_or(155_000_000_000.0);
let base_fuel = 50_000_000.0;
let flip_fuel = 200.0;
let max_flips = if max_fuel > base_fuel {
((max_fuel - base_fuel) / flip_fuel) as u64
} else {
10_000_000
};
const WALK_P: f64 = 0.52;
const RESTART_PERIOD: u64 = 80_000_000;
const REINIT_MIN_UNSAT: usize = 30;
let mut assignment = vec![false; nv];
let mut best_assignment = vec![false; nv];
let mut true_lit_count = vec![0u8; nc];
let mut unsat_clauses: Vec<usize> = Vec::with_capacity(nc);
let mut clause_pos_in_unsat = vec![usize::MAX; nc];
for v in 0..nv {
let pc = p_cnt[v] as f64;
let nc_v = n_cnt[v] as f64;
let total = pc + nc_v;
if total == 0.0 {
assignment[v] = rng.gen_bool(0.5);
} else {
assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
}
}
for c in 0..nc {
let off = co[c] as usize;
let end = co[c + 1] as usize;
let mut trues = 0u8;
for i in off..end {
let lit = cl[i];
let v = (lit.abs() - 1) as usize;
if assignment[v] == (lit > 0) { trues += 1; }
}
true_lit_count[c] = trues;
if trues == 0 {
clause_pos_in_unsat[c] = unsat_clauses.len();
unsat_clauses.push(c);
}
}
let mut best_unsat = unsat_clauses.len();
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
let mut period_best_unsat = best_unsat;
let mut var_age = vec![0u8; nv];
let mut global_flips: u64 = 0;
unsafe {
loop {
if global_flips >= max_flips { break; }
if unsat_clauses.is_empty() { break; }
if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
if period_best_unsat >= REINIT_MIN_UNSAT {
for v in 0..nv {
*assignment.get_unchecked_mut(v) = rng.gen_bool(0.5);
}
true_lit_count.fill(0);
unsat_clauses.clear();
clause_pos_in_unsat.fill(usize::MAX);
for c in 0..nc {
let off = *co.get_unchecked(c) as usize;
let end = *co.get_unchecked(c + 1) as usize;
let mut trues = 0u8;
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
}
*true_lit_count.get_unchecked_mut(c) = trues;
if trues == 0 {
*clause_pos_in_unsat.get_unchecked_mut(c) = unsat_clauses.len();
unsat_clauses.push(c);
}
}
var_age.fill(0);
let cur = unsat_clauses.len();
if cur < best_unsat {
best_unsat = cur;
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
}
}
period_best_unsat = unsat_clauses.len();
}
let cur_unsat = unsat_clauses.len();
if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
global_flips += 1;
let r_idx = rng.gen_range(0..unsat_clauses.len());
let c = *unsat_clauses.get_unchecked(r_idx);
let off = *co.get_unchecked(c) as usize;
let end = *co.get_unchecked(c + 1) as usize;
let len = end - off;
if len > 1 {
let ri = (global_flips as usize) % len;
cl.swap(off, off + ri);
}
let mut picked_v = usize::MAX;
let mut vars = [0usize; 3];
let mut breaks = [0u8; 3];
let mut nvars = 0;
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
*vars.get_unchecked_mut(nvars) = v;
let val = *assignment.get_unchecked(v);
let (start, stop) = if val {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
let mut b = 0u8;
for k in start..stop {
if *true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1 { b += 1; }
}
*breaks.get_unchecked_mut(nvars) = b;
if b == 0 {
picked_v = v;
break;
}
nvars += 1;
}
if picked_v == usize::MAX {
if nvars == 0 {
if let Some(&lit) = cl.get(off) {
picked_v = (lit.abs() - 1) as usize;
}
} else if rng.gen_bool(WALK_P) {
picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
} else {
let mut min_b = u8::MAX;
let mut best_idx = 0;
for i in 0..nvars {
let b = *breaks.get_unchecked(i);
let vi = *vars.get_unchecked(i);
let vb = *vars.get_unchecked(best_idx);
if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
min_b = b;
best_idx = i;
}
}
picked_v = *vars.get_unchecked(best_idx);
}
}
if picked_v == usize::MAX { continue; }
let new_val = !*assignment.get_unchecked(picked_v);
*assignment.get_unchecked_mut(picked_v) = new_val;
let p_start = *all_off.get_unchecked(picked_v) as usize;
let p_end = *p_bound.get_unchecked(picked_v) as usize;
for k in p_start..p_end {
let c_idx = *all_data.get_unchecked(k) as usize;
if new_val {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
if trues == 0 {
let last_c = unsat_clauses.pop().unwrap_unchecked();
let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
if last_c != c_idx {
*unsat_clauses.get_unchecked_mut(pos) = last_c;
*clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
}
}
} else {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
if trues == 1 {
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
unsat_clauses.push(c_idx);
}
}
}
let n_start = *p_bound.get_unchecked(picked_v) as usize;
let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
for k in n_start..n_end {
let c_idx = *all_data.get_unchecked(k) as usize;
if !new_val {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
if trues == 0 {
let last_c = unsat_clauses.pop().unwrap_unchecked();
let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
if last_c != c_idx {
*unsat_clauses.get_unchecked_mut(pos) = last_c;
*clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
}
}
} else {
let trues = *true_lit_count.get_unchecked(c_idx);
*true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
if trues == 1 {
*clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
unsat_clauses.push(c_idx);
}
}
}
for i in off..end {
let lit = *cl.get_unchecked(i);
let v = (lit.abs() - 1) as usize;
let a = *var_age.get_unchecked(v);
*var_age.get_unchecked_mut(v) = a.saturating_add(1);
}
let cur_unsat = unsat_clauses.len();
if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
if cur_unsat < best_unsat {
best_unsat = cur_unsat;
best_assignment.copy_from_slice(&assignment);
let _ = save_solution(&Solution { variables: best_assignment.clone() });
if cur_unsat == 0 { break; }
}
}
}
let _ = save_solution(&Solution { variables: best_assignment });
Ok(())
}
}
#[inline(always)]
fn enc(l: i32) -> i32 { (((l.abs() - 1) << 1) | ((l > 0) as i32)) }
mod track_t38 {
use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;
use super::Hparams;
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hp: &Hparams,
) -> Result<()> {
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
} = super::preprocess_enc(challenge, save_solution);
let nvf = nv as f64;
let max_fuel = hp.max_fuel_low.unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
drop(p_cnt);
drop(n_cnt);
let vars_prod = vars.clone();
{
let (seed, u_seed, it_best, it_tot) =
super::super::engine_spr::seed_enc(
nv, nc, &cl, &co, &mut rng,
challenge.num_variables == 100000 && challenge.clauses.len() == 420000,
);
let mut u_base = 0usize;
for i in 0..nc {
let mut ok = false;
for j in co[i] as usize..co[i + 1] as usize {
let l = cl[j];
if ((l & 1) == 1) == vars[(l >> 1) as usize] { ok = true; break; }
}
if !ok { u_base += 1; }
}
println!("MI_PROBE SPR nv={} nc={} unsat_base={} unsat_seed={} it_best={} it_tot={}",
nv, nc, u_base, u_seed, it_best, it_tot);
vars.copy_from_slice(&seed);
}
for phase in 0..2usize {
if phase == 1 {
vars.copy_from_slice(&vars_prod);
println!("MI_PROBE T38FALLBACK phase=1");
}
let ng_len = (nc + 3) >> 2;
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let shift = (i & 3) << 1;
let byte_idx = i >> 2;
for j in s..e {
let l = cl[j];
let v = (l >> 1) as usize;
if ((l & 1) == 1) == vars[v] {
num_good[byte_idx] += 1u8 << shift;
}
}
}
let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
let mut unsat_pos = vec![u32::MAX; nc];
for i in 0..nc {
if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
unsat_pos[i] = unsat_list.len() as u32;
unsat_list.push(i as u32);
}
}
if unsat_list.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.check_interval
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.perturbation_flips
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.stagnation_limit
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = unsat_list.len();
let mut stagnation = 0usize;
let mut countdown = check_interval;
let mut rounds = 0usize;
let cb = if avg_clause_size > 4.5 {
3.5f64
} else if avg_clause_size > 3.5 {
2.85f64
} else {
2.06f64
};
let mut probs_break = [0.0f64; 256];
for i in 0..256 {
probs_break[i] = (0.9f64 + i as f64).powf(-cb);
}
unsafe {
loop {
if unsat_list.is_empty() || rounds >= max_flips { break; }
countdown -= 1;
if countdown == 0 {
countdown = check_interval;
let progress = last_check_residual as i64 - unsat_list.len() as i64;
if progress <= 0 && phase == 1 {
stagnation += 1;
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if unsat_list.is_empty() { break; }
let rid = rng.gen::<usize>() % unsat_list.len();
let pcid = *unsat_list.get_unchecked(rid) as usize;
let pcs = *co.get_unchecked(pcid) as usize;
let pce = *co.get_unchecked(pcid + 1) as usize;
if pcs == pce { continue; }
let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
let v = (lit >> 1) as usize;
let was_true = *vars.get_unchecked(v);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
} else {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
} else {
(*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
if old == 0 {
let pos = *unsat_pos.get_unchecked(c) as usize;
let last_idx = unsat_list.len() - 1;
let last_c = *unsat_list.get_unchecked(last_idx) as usize;
*unsat_list.get_unchecked_mut(pos) = last_c as u32;
*unsat_pos.get_unchecked_mut(last_c) = pos as u32;
*unsat_pos.get_unchecked_mut(c) = u32::MAX;
unsat_list.set_len(last_idx);
}
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
let ul = unsat_list.len();
*unsat_pos.get_unchecked_mut(c) = ul as u32;
*unsat_list.as_mut_ptr().add(ul) = c as u32;
unsat_list.set_len(ul + 1);
}
}
*vars.get_unchecked_mut(v) = !was_true;
}
stagnation = 0;
}
} else {
stagnation = 0;
}
last_check_residual = unsat_list.len();
}
let r0 = rounds;
let nblk = (max_flips - rounds).min(countdown);
for _ in 0..nblk {
let rand_val = rng.gen::<usize>();
if unsat_list.is_empty() { break; }
let cid = *unsat_list.get_unchecked(rand_val % unsat_list.len()) as usize;
let cs = *co.get_unchecked(cid) as usize;
let ce = *co.get_unchecked(cid + 1) as usize;
let clen = ce - cs;
let ri = rand_val % clen;
cl.swap(cs, cs + ri);
let mut zero_buf: [usize; 3] = [0; 3];
let mut zero_cnt: usize = 0;
const SBUF: usize = 4;
let mut scores_f = [0.0f64; SBUF];
let mut sum_scores_f = 0.0f64;
let mut fused = false;
if clen == 3 {
macro_rules! probe3 { ($j:expr) => {{
let l = *cl.get_unchecked($j);
let abs_l = (l >> 1) as usize;
let (os, oe) = if (l & 1) == 1 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
sad += ((*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1) as usize;
}
let score = *probs_break.get_unchecked(sad.min(255));
sum_scores_f += score;
*scores_f.get_unchecked_mut($j - cs) = score;
if sad == 0 {
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
}}}
probe3!(cs);
probe3!(cs + 1);
probe3!(cs + 2);
fused = true;
} else {
'outer: for j in cs..ce {
let l = *cl.get_unchecked(j);
let abs_l = (l >> 1) as usize;
let (os, oe) = if (l & 1) == 1 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
continue 'outer;
}
}
*zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
zero_cnt += 1;
}
}
let v_idx = if zero_cnt > 0 {
if zero_cnt == 1 {
*zero_buf.get_unchecked(0)
} else {
*zero_buf.get_unchecked(rand_val % zero_cnt)
}
} else {
let mut sum_scores = sum_scores_f;
let mut scores = scores_f;
let limit = (ce - cs).min(SBUF);
for idx in (if fused { limit } else { 0 })..limit {
let j = cs + idx;
let l = *cl.get_unchecked(j);
let abs_l = (l >> 1) as usize;
let (os, oe) = if (l & 1) == 1 {
(*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
} else {
(*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
};
let mut sad = 0usize;
for k in os..oe {
let c = *all_data.get_unchecked(k) as usize;
if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
sad += 1;
}
}
let score = *probs_break.get_unchecked(sad.min(255));
sum_scores += score;
*scores.get_unchecked_mut(idx) = score;
}
let threshold = rng.gen::<f64>() * sum_scores;
let mut accum = 0.0;
let mut v_sel = (*cl.get_unchecked(cs) >> 1) as usize;
for idx in 0..limit {
accum += *scores.get_unchecked(idx);
if accum >= threshold {
v_sel = (*cl.get_unchecked(cs + idx) >> 1) as usize;
break;
}
}
v_sel
};
let was_true = *vars.get_unchecked(v_idx);
let (is, ie) = if was_true {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
} else {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
};
let (ds, de) = if was_true {
(*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
} else {
(*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
};
for k in is..ie {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
if old == 0 {
let pos = *unsat_pos.get_unchecked(c) as usize;
let last_idx = unsat_list.len() - 1;
let last_c = *unsat_list.get_unchecked(last_idx) as usize;
*unsat_list.get_unchecked_mut(pos) = last_c as u32;
*unsat_pos.get_unchecked_mut(last_c) = pos as u32;
*unsat_pos.get_unchecked_mut(c) = u32::MAX;
unsat_list.set_len(last_idx);
}
}
for k in ds..de {
let c = *all_data.get_unchecked(k) as usize;
let shift = (c & 3) << 1;
let byte_idx = c >> 2;
let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
*num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
if ng_before == 1 {
let ul = unsat_list.len();
*unsat_pos.get_unchecked_mut(c) = ul as u32;
*unsat_list.get_unchecked_mut(ul) = c as u32;
unsat_list.set_len(ul + 1);
}
}
*vars.get_unchecked_mut(v_idx) = !was_true;
rounds += 1;
}   // i7 HOIST: fin du bloc
countdown -= (rounds - r0).saturating_sub(1);
}
}
if unsat_list.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
}   // fin des phases
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
}
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
pub base_prob: Option<f64>,
pub max_prob: Option<f64>,
pub check_interval: Option<usize>,
pub stagnation_limit: Option<usize>,
pub perturbation_flips: Option<usize>,
pub max_fuel_high: Option<f64>,
pub max_fuel_low: Option<f64>,
pub max_reinits: Option<usize>,
}
impl Hparams {
pub fn for_t1() -> Self {
let mut h = Self::default();
h.max_fuel_high = Some(180_000_000_000.0);
h
}
pub fn for_t3() -> Self { Self::default() }
pub fn for_t4() -> Self {
let mut h = Self::default();
h.max_reinits = Some(15);
h
}
pub fn for_t5() -> Self { Self::default() }
pub fn for_t38() -> Self {
let mut h = Self::default();
h.stagnation_limit = Some(3);
h
}
fn merge_user(mut self, user: Option<&Map<String, Value>>) -> Self {
if let Some(m) = user {
if let Ok(u) = serde_json::from_value::<Hparams>(Value::Object(m.clone())) {
if u.base_prob.is_some() { self.base_prob = u.base_prob; }
if u.max_prob.is_some() { self.max_prob = u.max_prob; }
if u.check_interval.is_some() { self.check_interval = u.check_interval; }
if u.stagnation_limit.is_some() { self.stagnation_limit = u.stagnation_limit; }
if u.perturbation_flips.is_some() { self.perturbation_flips = u.perturbation_flips; }
if u.max_fuel_high.is_some() { self.max_fuel_high = u.max_fuel_high; }
if u.max_fuel_low.is_some() { self.max_fuel_low = u.max_fuel_low; }
if u.max_reinits.is_some() { self.max_reinits = u.max_reinits; }
}
}
self
}
}
pub(crate) struct Prepared {
pub rng: SmallRng,
pub nv: usize,
pub nc: usize,
pub density: f64,
pub p_cnt: Vec<u32>,
pub n_cnt: Vec<u32>,
pub all_off: Vec<u32>,
pub p_bound: Vec<u32>,
pub all_data: Vec<u32>,
pub cl: Vec<i32>,
pub co: Vec<u32>,
}
#[inline(always)]
pub(crate) fn preprocess(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Prepared {
let nv = challenge.num_variables;
let _ = save_solution(&Solution { variables: vec![false; nv] });
let rng = SmallRng::seed_from_u64(u64::from_le_bytes(
challenge.seed[..8].try_into().unwrap(),
));
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
let density = nc as f64 / nv as f64;
Prepared { rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co }
}
pub(crate) fn preprocess_enc(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Prepared {
let nv = challenge.num_variables;
let _ = save_solution(&Solution { variables: vec![false; nv] });
let rng = SmallRng::seed_from_u64(u64::from_le_bytes(
challenge.seed[..8].try_into().unwrap(),
));
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
cl.push(enc(a));
if b != a { cl.push(enc(b)); }
if c != a && c != b { cl.push(enc(c)); }
co.push(cl.len() as u32);
ci += 1;
}
}
let density = nc as f64 / nv as f64;
Prepared { rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co }
}
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let nv = challenge.num_variables;
let nc_total = challenge.clauses.len();
let user = hyperparameters.as_ref();
match (nv, nc_total) {
(10000, 42670) => {
let hp = Hparams::for_t1().merge_user(user);
track_t1::solve(challenge, save_solution, &hp)
}
(100000, 415000) => {
let hp = Hparams::for_t3().merge_user(user);
track_t3::solve(challenge, save_solution, &hp)
}
(5000, 21335) => {
let hp = Hparams::for_t4().merge_user(user);
track_t4::solve(challenge, save_solution, &hp)
}
(7500, 32002) => {
let hp = Hparams::for_t5().merge_user(user);
track_t5::solve(challenge, save_solution, &hp)
}
(100000, 420000) => {
let hp = Hparams::for_t38().merge_user(user);
track_t38::solve(challenge, save_solution, &hp)
}
_ => Err(anyhow::anyhow!(
"engine: unknown track config (num_variables={}, num_clauses={})",
nv, nc_total
)),
}
}
}
mod engine_spr {
const MAXIT: usize = 1500;
const RATE: f64 = 0.001;
const GMAX: f64 = 0.5;
const LAM: f64 = 0.1;
const DAMP: f64 = 0.3;
const CHECK_EVERY: usize = 10;
const PATIENCE: usize = 150;
pub fn seed_enc(nv: usize, nc: usize, cl: &[i32], co: &[u32], rng: &mut impl rand::Rng, stop_when_seed_good: bool)
-> (Vec<bool>, usize, usize, usize)
{
let ne = co[nc] as usize;
let mut ev = vec![0u32; ne];
let mut es = vec![0u8; ne];
for a in 0..nc {
for j in co[a] as usize..co[a + 1] as usize {
let l = cl[j];
ev[j] = (l >> 1) as u32; es[j] = (l & 1) as u8;
}
}
let mut eta: Vec<f32> = (0..ne).map(|_| rng.gen::<f32>()).collect();
let mut p = vec![1f32; 2 * nv];
let mut hp = vec![0f32; nv];
let mut hm = vec![0f32; nv];
let mut wp = vec![0f32; nv];
let mut wm = vec![0f32; nv];
let mut best_x = vec![false; nv];
let mut best_u = usize::MAX;
let mut best_it = 0usize;
let mut it = 0usize;
let damp = DAMP as f32;
while it < MAXIT {
it += 1;
let gamma = ((RATE * it as f64).min(GMAX)) as f32;
for v in 0..nv {
p[2 * v] = 1.0 - hm[v];
p[2 * v + 1] = 1.0 - hp[v];
}
for e in 0..ne {
let o = (1.0 - eta[e]).max(1e-6);
p[(ev[e] as usize) * 2 + es[e] as usize] *= o;
}
let mut rr = [0f32; 3];
for a in 0..nc {
let s0 = co[a] as usize; let e0 = co[a + 1] as usize;
let len = e0 - s0;
if len == 3 {
for q in 0..3 {
let e = s0 + q;
let v = ev[e] as usize; let sg = es[e] as usize;
let o = (1.0 - eta[e]).max(1e-6);
let av = p[v * 2 + sg] / o;
let bv = p[v * 2 + (1 - sg)];
let pu = (1.0 - bv) * av; let ps = (1.0 - av) * bv; let p0 = av * bv;
let den = pu + ps + p0;
rr[q] = if den > 0.0 { pu / den } else { 0.0 };
}
let prod0 = (1f32 * rr[1]) * rr[2];
let prod1 = (1f32 * rr[0]) * rr[2];
let prod2 = (1f32 * rr[0]) * rr[1];
eta[s0] = damp * eta[s0] + (1.0 - damp) * prod0.min(1.0);
eta[s0 + 1] = damp * eta[s0 + 1] + (1.0 - damp) * prod1.min(1.0);
eta[s0 + 2] = damp * eta[s0 + 2] + (1.0 - damp) * prod2.min(1.0);
continue;
}
for q in 0..len {
let e = s0 + q;
let v = ev[e] as usize; let sg = es[e] as usize;
let o = (1.0 - eta[e]).max(1e-6);
let av = p[v * 2 + sg] / o;
let bv = p[v * 2 + (1 - sg)];
let pu = (1.0 - bv) * av; let ps = (1.0 - av) * bv; let p0 = av * bv;
let den = pu + ps + p0;
rr[q] = if den > 0.0 { pu / den } else { 0.0 };
}
for q in 0..len {
let mut prod = 1f32;
for k in 0..len { if k != q { prod *= rr[k]; } }
let e = s0 + q;
eta[e] = damp * eta[e] + (1.0 - damp) * prod.min(1.0);
}
}
for v in 0..nv {
let pm = p[2 * v]; let pp = p[2 * v + 1];
let pip = (1.0 - pp) * pm; let pim = (1.0 - pm) * pp; let pi0 = pp * pm;
let den = pip + pim + pi0;
if den > 0.0 { wp[v] = pip / den; wm[v] = pim / den; } else { wp[v] = 0.0; wm[v] = 0.0; }
hp[v] = (1.0 - LAM as f32) * hp[v] + (LAM as f32) * gamma * wp[v];
hm[v] = (1.0 - LAM as f32) * hm[v] + (LAM as f32) * gamma * wm[v];
}
if it % CHECK_EVERY == 0 {
let mut u = 0usize;
for a in 0..nc {
let mut ok = false;
for j in co[a] as usize..co[a + 1] as usize {
let v = ev[j] as usize;
if (es[j] == 1) == (wp[v] > wm[v]) { ok = true; break; }
}
if !ok { u += 1; }
}
if u < best_u {
best_u = u; best_it = it;
for v in 0..nv { best_x[v] = wp[v] > wm[v]; }
if u == 0 { break; }
} else if it - best_it >= PATIENCE {
break;
}
if stop_when_seed_good && it >= 100 && best_u <= 128 {
break;
}
}
}
(best_x, best_u, best_it, it)
}
}
pub fn solve_challenge(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let nv = challenge.num_variables;
let nc = challenge.clauses.len();
match (nv, nc) {
(100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
_ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
}
}
}
mod tw7 {
mod donor_kernel {
pub(crate) mod clause_order {
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
pub(crate) mod compact_ranges {
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
pub(crate) mod exact_coin {
use rand::Rng;
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
pub(crate) mod exact_div {
const N:usize=65536;
const fn reciprocals()->[u64;N]{
let mut out=[0u64;N];let mut d=1;
while d<N{out[d]=u64::MAX/(d as u64);d+=1;}out
}
static RECIPROCALS:[u64;N]=reciprocals();
#[inline(always)]pub(crate) fn rem(x:usize,d:usize)->usize{
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
#[inline(always)]pub(crate) unsafe fn rem32_nonzero_small(x:u32,d:usize)->usize{
let m=*DOMAIN32_RECIPROCALS.get_unchecked(d);let q=(((x as u64)*(m as u64))>>32)as u32;
let r=x-q*(d as u32);if r>=d as u32{(r-d as u32)as usize}else{r as usize}
}
#[cfg(test)]mod domain32_tests{
use super::*;
#[test]fn every_admissible_divisor_and_boundary_words(){let mut x=0xa341316cu32;for d in 1..65536{for r in [0,1,d as u32-1,d as u32,d as u32+1,u32::MAX-1,u32::MAX]{assert_eq!(unsafe{rem32_nonzero_small(r,d)},(r as usize)%d);}for _ in 0..32{x^=x<<13;x^=x>>17;x^=x<<5;assert_eq!(unsafe{rem32_nonzero_small(x,d)},(x as usize)%d);}}}
}
}
pub(crate) mod large_buffer {
pub(crate) type LargeBuf<T>=Vec<T>;
pub(crate) fn zeros<T:Copy+Default>(len:usize)->Vec<T>{let mut buf=Vec::with_capacity(len);buf.resize(len,T::default());buf}
pub(crate) fn from_slice<T:Copy>(values:&[T])->Vec<T>{let mut buf=Vec::with_capacity(values.len());buf.extend_from_slice(values);buf}
#[cfg(test)]mod tests{
use super::*;
#[test]fn initialized_elements_and_copy(){for n in [0usize,1,513,32768,131072,524289]{let mut a:LargeBuf<u32>=zeros(n);assert_eq!(a.len(),n);assert!(a.iter().all(|&x|x==0));if n>0{for i in 0..n{a[i]=(i as u32).wrapping_mul(17);}let b=from_slice(&a);assert_eq!(a,b);}}}
}
}
pub(crate) mod paged_cache {
use super::large_buffer::LargeBuf;
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
pub(crate) mod paged_flip_ranges {
use super::large_buffer::LargeBuf;
pub(crate) struct FlipRanges {data:LargeBuf<[u64;2]>}
impl FlipRanges {
pub(crate) fn restore_variables(&self,off:&[u32],mid:&[u32],vars:&mut[bool]){
assert_eq!(vars.len(),self.data.len());
for v in 0..vars.len(){
if off[v+1]!=off[v]{
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
pub(crate) mod paged_order {
use super::large_buffer::LargeBuf;
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
pub(crate) mod paged_small_order {
use super::large_buffer::LargeBuf;
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
pub(crate) mod paged_small_signed {
use super::large_buffer::LargeBuf;
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
pub(crate) mod phase_div {
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
pub(crate) mod raw_roulette {
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
pub(crate) mod short_mask {
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
pub(crate) mod subset_cache {
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
}
mod track1_cached {
pub(super) mod explicit_roulette {
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
}
mod engine_iv {
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
pub base_prob: Option<f64>,
pub max_prob: Option<f64>,
pub check_interval: Option<usize>,
pub stagnation_limit: Option<usize>,
pub perturbation_flips: Option<usize>,
pub max_fuel_high: Option<f64>,
pub max_fuel_low: Option<f64>,
}
pub fn solve(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
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
let density = nc as f64 / nv as f64;
track1::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution)
}
mod track1 {
use super::super::donor_kernel::exact_div;
use super::super::track1_cached::explicit_roulette;
use super::super::track1_cached::signed_order::{SignedOrder as ClauseOrder,LiteralRanges};
use super::super::donor_kernel::subset_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;
pub fn solve(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let narrow = nv <= 8192 && p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n <= 255);
if narrow {
solve_impl::<Byte13BreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
} else {
solve_impl::<SmallBreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
}
}
fn solve_impl<C:CacheOps>(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(160_000_000_000.0);
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
let check_interval = hp.as_ref().and_then(|h| h.check_interval)
.unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let mut last_check_residual = unsat_list.len();
let mut rounds = 0usize;
let mut stagnation = 0usize;
let stagnation_limit_t4 = hp.as_ref().and_then(|h| h.stagnation_limit).unwrap_or(3);
let probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
let mut current_reinit_stagnation: usize = ((nv * nc / 1000) as usize).clamp(1000, 500000);
const REINIT_MIN_UNSAT: usize = 10;
const MAX_REINITS: usize = 50;
let mut best_unsat = unsat_list.len();
let mut best_vars = vars.clone();
let mut stagnation_count: usize = 0;
let mut reinit_count: usize = 0;
let max_degree = all_off.windows(2).map(|w| (w[1]-w[0]) as usize).max().unwrap_or(0).max(1);
let mut changed = Vec::<u32>::with_capacity(max_degree);
let scratch = changed.as_mut_ptr();
let mut order = ClauseOrder::new(cl,co);
let ranges = LiteralRanges::new(all_off,p_bound);
let mut next_check_interval = check_interval;
let mut cache = C::new(nv, &cl, co, &vars);
unsafe {
if best_unsat >= REINIT_MIN_UNSAT {
loop {
if rounds >= max_flips { break; }
if unsat_list.is_empty() { break; }
if reinit_count >= MAX_REINITS && stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT {
break;
}
if stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT && reinit_count < MAX_REINITS {
reinit_count += 1;
let reinit_factor = if density > 4.0 { 1.5 } else { 1.3 };
current_reinit_stagnation = ((current_reinit_stagnation as f64 * reinit_factor) as usize).clamp(1000, 500000);
for v in 0..nv { vars[v] = false; }
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
unsat_list.clear();
unsat_pos.fill(u32::MAX);
for c in 0..nc {
let s = co[c] as usize;
let e = co[c + 1] as usize;
let mut g = 0u8;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
}
if g == 0 {
unsat_pos[c] = unsat_list.len() as u32;
unsat_list.push(c as u32);
}
}
cache = C::new(nv, &cl, co, &vars);
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
let rid = exact_div::rem(rng.gen::<usize>(),unsat_list.len());
let pcid = *unsat_list.get_unchecked(rid) as usize;
let len = order.len(pcid);
if len == 0 { continue; }
let lit = order.lit(pcid,exact_div::clause_rem(rng.gen::<usize>(),len));
let v = (lit.abs()-1) as usize;
let was_true = lit < 0;
let (is,ie,ds,de) = ranges.of((v<<1)|(was_true as usize));
flip_cached(&mut cache,all_data,is,ie,ds,de,v,&mut unsat_list,&mut unsat_pos,scratch);
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
let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32,unsat_list.len())) as usize;
let clen = order.len(cid);
let chosen_code=if clen==3 {
super::super::track1_cached::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
} else {
if clen > 1 { order.swap(cid,0,exact_div::clause_rem(rand_val,clen)); }
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}
else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}
else{
let w0=*probs_break.get_unchecked(b0.min(15));
let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
let total=w0+w1;
let r=(rand_val as u32)%total.max(1);
if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else{ll[0]}
};
chosen_code
};
let v_idx=chosen_code>>1;
let was_true=chosen_code&1!=0;
let (is,ie,ds,de)=ranges.of(chosen_code);
flip_cached(&mut cache,all_data,is,ie,ds,de,v_idx,&mut unsat_list,&mut unsat_pos,scratch);
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
if best_unsat < REINIT_MIN_UNSAT { break; }
}
}
if best_unsat < REINIT_MIN_UNSAT {
loop {
if rounds >= max_flips || unsat_list.is_empty() { break; }
if (rounds == next_check_interval || check_interval == 0) && rounds % check_interval == 0 && rounds > 0 {
next_check_interval = next_check_interval.wrapping_add(check_interval);
let progress = last_check_residual as i64 - unsat_list.len() as i64;
if progress <= 0 {
stagnation += 1;
if stagnation >= stagnation_limit_t4 {
let kicks = if stagnation >= 8 { 6 } else { 3 };
for _ in 0..kicks {
if unsat_list.is_empty() { break; }
let rid = exact_div::rem(rng.gen::<usize>(),unsat_list.len());
let pcid = *unsat_list.get_unchecked(rid) as usize;
let len = order.len(pcid);
if len == 0 { continue; }
let lit = order.lit(pcid,exact_div::clause_rem(rng.gen::<usize>(),len));
let v = (lit.abs()-1) as usize;
let was_true = lit < 0;
let (is,ie,ds,de) = ranges.of((v<<1)|(was_true as usize));
flip_cached(&mut cache,all_data,is,ie,ds,de,v,&mut unsat_list,&mut unsat_pos,scratch);
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
let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32,unsat_list.len())) as usize;
let clen = order.len(cid);
let chosen_code=if clen==3 {
super::super::track1_cached::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
} else {
if clen > 1 { order.swap(cid,0,exact_div::clause_rem(rand_val,clen)); }
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}
else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}
else{
let w0=*probs_break.get_unchecked(b0.min(15));
let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
let total=w0+w1;
let r=(rand_val as u32)%total.max(1);
if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else{ll[0]}
};
chosen_code
};
let v_idx=chosen_code>>1;
let was_true=chosen_code&1!=0;
let (is,ie,ds,de)=ranges.of(chosen_code);
flip_cached(&mut cache,all_data,is,ie,ds,de,v_idx,&mut unsat_list,&mut unsat_pos,scratch);
rounds += 1;
let cur = unsat_list.len();
if cur < best_unsat {
best_unsat = cur;
cache.restore_variables(cl,co,&mut vars);
best_vars.copy_from_slice(&vars);
}
}
}
}
if unsat_list.is_empty() {cache.restore_variables(cl,co,&mut vars);}
let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
let _ = save_solution(&Solution { variables: final_vars });
Ok(())
}
#[inline(always)]
unsafe fn flip_cached<C:CacheOps>(
cache:&mut C, all_data:&[u32], is:usize,ie:usize,ds:usize,de:usize,
variable:usize, unsat_list:&mut Vec<u32>,unsat_pos:&mut [u32],scratch:*mut u32,
) {
let gains=cache.gain_collect(all_data.get_unchecked(is..ie),variable,scratch);
for i in 0..gains {
let c=*scratch.add(i) as usize;
let pos=*unsat_pos.get_unchecked(c) as usize;
let last_index=unsat_list.len()-1;
let last_clause=*unsat_list.get_unchecked(last_index) as usize;
*unsat_list.get_unchecked_mut(pos)=last_clause as u32;
*unsat_pos.get_unchecked_mut(last_clause)=pos as u32;
unsat_list.set_len(last_index);
}
let losses=cache.lose_many(all_data.get_unchecked(ds..de),variable,scratch);
let len=unsat_list.len();
for i in 0..losses {
let c=*scratch.add(i);
*unsat_pos.get_unchecked_mut(c as usize)=(len+i) as u32;
unsat_list.as_mut_ptr().add(len+i).write(c);
}
unsat_list.set_len(len+losses);
cache.set_break(variable,gains as u32);
}
}
}
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::{Challenge, Solution};
pub(super) fn solve_complement(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
budget: u64,
) -> Result<()> {
let mut hp = Map::new();
hp.insert("max_fuel_high".into(), Value::from(budget));
engine_iv::solve(challenge, save_solution, &Some(hp))
}
pub(super) fn complement_budget(hyperparameters: &Option<Map<String, Value>>) -> u64 {
hyperparameters.as_ref()
.and_then(|m| m.get("t4_fuel_b"))
.and_then(Value::as_u64)
.unwrap_or(0)
}
mod track4 {
use super::donor_kernel::phase_div;
use super::donor_kernel::paged_order::SignedOrder;
use super::donor_kernel::compact_ranges::CompactRanges;
use super::donor_kernel::exact_coin::ExactCoin;
use super::donor_kernel::clause_order::ClauseOrder;
use super::donor_kernel::exact_div;
use super::donor_kernel::paged_cache::{ExactBreakCache, NarrowLargeBreakCache, CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;
pub fn solve(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let narrow = p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n <= 255);
if narrow { solve_impl::<NarrowLargeBreakCache,true>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
else { solve_impl::<ExactBreakCache,false>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
}
fn solve_impl<C:CacheOps,const PACKED:bool>(
hp: &Option<Hyperparameters>,
rng: &mut SmallRng,
nv: usize, nc: usize, density: f64,
p_cnt: Vec<u32>, n_cnt: Vec<u32>,
all_off: &[u32], p_bound: &[u32],
all_data: &[u32],
cl: &mut Vec<i32>, co: &[u32],
save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
let page_data=super::donor_kernel::large_buffer::from_slice(all_data);
let all_data:&[u32]=&page_data;
let max_loss_per_flip=p_cnt.iter().chain(n_cnt.iter()).copied().max().unwrap_or(0)as usize;
let capacity_batch_limit=64usize.min((4096/max_loss_per_flip.max(1)).max(1));
let nvf = nv as f64;
let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_low).unwrap_or(150_000_000_000.0);
let avg_clause_size = cl.len() as f64 / nc as f64;
let difficulty_factor = density * avg_clause_size.sqrt();
let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
let remaining = (max_fuel - base_fuel).max(0.0);
let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
let mut vars = vec![false; nv];
let nad = 1.0;
let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
for v in 0..nv {
let np = p_cnt[v] as f64;
let nn = n_cnt[v] as f64;
if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
if np == 0.0 { continue; }
let vad = np / nn;
let bias_prob = (np + 0.25) / (np + nn + 1.2);
let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
vars[v] = rng.gen_bool(prob);
}
drop(p_cnt);
drop(n_cnt);
let ng_len = nc; // one byte per clause; count is in 0..=3
let mut num_good = vec![0u8; ng_len];
for i in 0..nc {
let s = co[i] as usize;
let e = co[i + 1] as usize;
let byte_idx = i;
for j in s..e {
let l = cl[j];
let v = (l.abs() - 1) as usize;
if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
num_good[byte_idx] += 1u8;
}
}
}
let mut residual: Vec<u32> = Vec::with_capacity(nc);
for i in 0..nc {
if num_good[i] == 0 {
residual.push(i as u32);
}
}
if residual.is_empty() {
let _ = save_solution(&Solution { variables: vars });
return Ok(());
}
let base_prob = hp.as_ref().and_then(|h| h.base_prob)
.unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
let mut current_prob = base_prob;
let mut coin = ExactCoin::new(current_prob);
let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
let base_interval = 60.0 - 30.0 * large_problem_scale;
let min_interval = 25.0 - 10.0 * large_problem_scale;
let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
let density_factor = 1.0 + 0.2 * density_s;
let check_interval = hp.as_ref().and_then(|h| h.check_interval)
.unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
let max_random_prob = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
let prob_adjustment_factor = 0.03;
let smoothing_factor = 0.8;
let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
.unwrap_or(1 + (2.0 * size_scale) as usize);
let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
.unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);
let mut last_check_residual = residual.len();
let mut stagnation = 0usize;
let mut countdown = check_interval;
let mut rounds = 0usize;
let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
let mut order=SignedOrder::new(&cl,&co);
let ranges=CompactRanges::<PACKED>::new(all_off,p_bound);
let mut cache = C::new(nv, &cl, &co, &vars);
unsafe {
let mut end=if check_interval==0{max_flips}else{(check_interval-1).min(max_flips)};
'search: loop{
if residual.is_empty() {break 'search;}
let capacity_steps=(end-rounds).min(capacity_batch_limit);
residual.reserve(capacity_steps*max_loss_per_flip);
let capacity_end=rounds+capacity_steps;
let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
let done=if small{run_hot_phase::<C,PACKED,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,all_data,&mut rounds,capacity_end)}else{run_hot_phase::<C,PACKED,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,all_data,&mut rounds,capacity_end)};
if done{break 'search;}
if rounds<end{continue;}
if residual.is_empty()||rounds>=max_flips{break;}
let progress = last_check_residual as i64 - residual.len() as i64;
let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
if progress <= 0 {
stagnation += 1;
let prob_adjustment = prob_adjustment_factor
* (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
current_prob = (current_prob + prob_adjustment).min(max_random_prob);
if stagnation >= stagnation_limit {
let kicks = if stagnation >= 5 {
(perturbation_flips * 12).min(100)
} else if stagnation >= 4 {
(perturbation_flips * 6).min(50)
} else if stagnation >= 3 {
(perturbation_flips * 3).min(20)
} else {
(perturbation_flips + 2).min(10)
};
for _ in 0..kicks {
if residual.is_empty() { break; }
let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
let pcid = *residual.get_unchecked(rid) as usize;
let ng_val = cache.good(pcid);
if ng_val > 0 {
residual.swap_remove(rid);
continue;
}
let pcs = 0usize;
let pce = order.len_bounded(pcid);
if pcs == pce { continue; }
let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
let v = (lit.abs() - 1) as usize;
let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v);
residual.reserve(de-ds);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v,gains);
}
stagnation = 0;
}
} else if progress_ratio > progress_threshold {
stagnation = 0;
current_prob = base_prob;
} else {
stagnation = 0;
current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
}
coin=ExactCoin::new(current_prob);
last_check_residual = residual.len();
end=rounds.saturating_add(check_interval).min(max_flips);
}
}
cache.restore_variables(&cl,&co,&mut vars);
let _ = save_solution(&Solution { variables: vars });
Ok(())
}
#[inline(always)]unsafe fn run_hot_phase<C:CacheOps,const PACKED:bool,const SMALL:bool>(
cache:&mut C,order:&mut SignedOrder,ranges:&CompactRanges<PACKED>,
residual:&mut Vec<u32>,rng:&mut SmallRng,coin:&super::donor_kernel::exact_coin::ExactCoin,
all_data:&[u32],
rounds_ref:&mut usize,capacity_end:usize
)->bool{
let mut rounds=*rounds_ref;
while rounds<capacity_end{
let rand_val = rng.gen::<usize>();
let cid=loop {
let len=residual.len();
let id=if SMALL{phase_div::rem_nonzero_small(rand_val,len)}else{exact_div::rem(rand_val,len)};
let c=*residual.get_unchecked(id) as usize;
if cache.good(c)==0 {break c;}
let p=residual.as_mut_ptr();
p.add(id).write(*p.add(len-1));
residual.set_len(len-1);
if len==1 {*rounds_ref=rounds;return true;}
};
let cs = 0usize;
let ce = order.len_bounded(cid);
let clen = ce - cs;
let permutation_rank=if clen>1{
let rank=exact_div::clause_rem(rand_val,clen);order.swap(cid,0,rank);rank
}else{0};
let ll=order.codes(cid);
let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
let b0=cache.breaks_of(vv[0]);
let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
let chosen_code=if zero_mask!=0 {
ClauseOrder::choose_zero_rank(ll,zero_mask,rand_val,permutation_rank)
}else if coin.draw(rng){ll[0]}else{
let mut selected=ll[0];let mut min_sad=b0;
if clen>1 && min_sad>1 && b1<min_sad {
selected=ll[1];min_sad=b1;
}
if clen>2 && min_sad>1 && b2<min_sad {
selected=ll[2];
}
selected
};
let v_idx=chosen_code>>1;
let (is,ie,ds,de)=ranges.of(chosen_code);
let gains=cache.gain_many(all_data.get_unchecked(is..ie),v_idx);
let old_len=residual.len();
let appended=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
residual.set_len(old_len+appended);
cache.set_break(v_idx,gains);
rounds += 1;
}
*rounds_ref=rounds;
false
}
}
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
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
pub(super) fn solve_track4(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
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
track4::solve(
&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
&p_bound, &all_data, &mut cl, &co, save_solution,
)
}
}
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
pub fn help() {
println!("sat_frontier: use null or {{}} on four tracks; for n_vars=10000,ratio=4267 use {{\"max_fuel_high\":180000000000}}. Optional 5000-variable rescue: {{\"t4_fuel_b\":50000000000}}.");
}
pub fn solve_challenge(
challenge: &Challenge,
save_solution: &dyn Fn(&Solution) -> Result<()>,
hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
let nv = challenge.num_variables;
let nc = challenge.clauses.len();
match (nv, nc) {
(5000, 21335) => {
let rescue_budget = tw7::complement_budget(hyperparameters);
if rescue_budget == 0 {
return hybrid::solve_challenge(challenge, save_solution, hyperparameters);
}
let solved = std::cell::Cell::new(false);
let is_sat = |s: &Solution| s.variables.len() == nv
&& challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
s.variables[(lit.abs() - 1) as usize] == (lit > 0)
}));
let primary_output = |s: &Solution| {
let sat = is_sat(s);
if !solved.get() || sat {
save_solution(s)?;
}
if sat { solved.set(true); }
Ok(())
};
hybrid::solve_challenge(challenge, &primary_output, hyperparameters)?;
if solved.get() { return Ok(()); }
let rescue_output = |s: &Solution| {
if is_sat(s) { save_solution(s)?; }
Ok(())
};
tw7::solve_complement(challenge, &rescue_output, rescue_budget)
}
(7500, 32002) => tw6::solve_challenge(challenge, save_solution, hyperparameters),
(10000, 42670) => ours::solve_challenge(challenge, save_solution, hyperparameters),
(100000, 415000) => tw7::solve_track4(challenge, save_solution, hyperparameters),
(100000, 420000) => sp128::solve_challenge(challenge, save_solution, hyperparameters),
_ => Err(anyhow!(
"unknown track config (num_variables={}, num_clauses={})",
nv,
nc
)),
}
}
