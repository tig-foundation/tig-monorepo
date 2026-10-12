//! The input core is already in the ORIGINAL density order. A fractional
//! upper bound and feasible greedy lower bound fix membership common to EVERY
//! optimal core solution. Solve only the unresolved items, still in original
//! order, and encode precisely the final path observed by existing callers.
pub fn fill(
    core:&[usize],weights:&[u32],values:&[i32],capacity:usize,
    choices:&mut Vec<u8>,scratch:&mut Vec<i32>,
)->Option<usize> {
    if let Some(best)=reduced(core,weights,values,capacity,choices,scratch) {
        return Some(best);
    }
    dense(core,weights,values,capacity,choices,scratch)
}

fn reduced(
    core:&[usize],weights:&[u32],values:&[i32],capacity:usize,
    choices:&mut Vec<u8>,scratch:&mut Vec<i32>,
)->Option<usize> {
    if core.len()<16 || capacity<16 {return None;}
    const SCALE:[i64;11]=[0,2520,1260,840,630,504,420,360,315,280,252];
    let mut prior=i64::MAX;let mut magnitude=0i64;
    let mut fractional_remaining=capacity;let mut fractional_value=0i64;
    let mut multiplier=None;let mut greedy_remaining=capacity;let mut greedy_value=0i64;
    for &i in core {
        let w=weights[i] as usize;let v=values[i] as i64;
        if w==0 || w>10 || v<=0 {return None;}
        let key=v*SCALE[w];
        if key>prior {return None;}
        prior=key;magnitude+=v;
        if multiplier.is_none() {
            if w<=fractional_remaining {fractional_remaining-=w;fractional_value+=v;}
            else {multiplier=Some(key);}
        }
        if w<=greedy_remaining {greedy_remaining-=w;greedy_value+=v;}
    }
    // Preserve the original32-bit sentinel proof and i64 fallback boundary.
    if magnitude>=(1i64<<29) {return None;}
    let lambda=multiplier?;
    let upper=fractional_value*2520+lambda*(fractional_remaining as i64);
    let gap=upper-greedy_value*2520;
    if gap<0 {return None;}
    let mut flags=vec![0u8;core.len()];let mut uncertain=Vec::new();let mut fixed_weight=0usize;
    for (t,&i) in core.iter().enumerate() {
        let reduced=values[i] as i64*2520-lambda*(weights[i] as i64);
        if reduced>gap {flags[t]=1;fixed_weight+=weights[i] as usize;}
        else if reduced>=-gap {uncertain.push(t);}
    }
    let residual_capacity=capacity.checked_sub(fixed_weight)?;
    let original_work=core.len().checked_mul(capacity.checked_add(1)?)?;
    let reduced_work=uncertain.len().checked_mul(residual_capacity.checked_add(1)?)?;
    // Do not spend a certificate merely to reproduce almost the same DP.
    if reduced_work>original_work/2 {return None;}
    let remaining_weight=if uncertain.is_empty() || residual_capacity==0 {0} else {
        let residual_core:Vec<usize>=uncertain.iter().map(|&t|core[t]).collect();
        let mut local_choices=Vec::new();
        let best=dense(&residual_core,weights,values,residual_capacity,&mut local_choices,scratch)?;
        let width=residual_capacity+1;let mut cur=best;
        for (r,&t) in uncertain.iter().enumerate().rev() {
            let w=weights[core[t]] as usize;
            if w<=cur && local_choices[r*width+cur]!=0 {flags[t]=1;cur-=w;}
        }
        debug_assert_eq!(cur,0);best
    };
    let best=fixed_weight+remaining_weight;
    let width=capacity+1;choices.resize(core.len()*width,0);
    // Callers observe exactly this descending path, not unvisited table cells
    // or scratch rows. A later dense fallback fully reinitializes its table.
    let mut cur=best;
    for t in (0..core.len()).rev() {
        choices[t*width+cur]=flags[t];
        if flags[t]!=0 {cur-=weights[core[t]] as usize;}
    }
    debug_assert_eq!(cur,0);Some(best)
}

fn dense(
    core: &[usize],
    weights: &[u32],
    values: &[i32],
    capacity: usize,
    choices: &mut Vec<u8>,
    scratch: &mut Vec<i32>,
) -> Option<usize> {
    let magnitude: i64 = core.iter().map(|&i| (values[i] as i64).abs()).sum();
    // Every real path lies in [-magnitude, magnitude]. Every unreachable
    // pseudo-path lies below INIT + magnitude < -magnitude. Additions cannot
    // overflow, so changing the sentinel cannot change the winning real path.
    if magnitude >= (1i64 << 29) {
        return None;
    }
    const INIT: i32 = -(1 << 30);
    let width = capacity + 1;
    scratch.resize(2 * width, INIT);
    scratch[..2 * width].fill(INIT);
    choices.resize(core.len() * width, 0);
    choices[..core.len() * width].fill(0);
    scratch[0] = 0;
    let mut row = 0;
    let mut hi = 0usize;
    for (t, &item) in core.iter().enumerate() {
        let weight = weights[item] as usize;
        if weight > capacity {
            continue;
        }
        let value = values[item];
        let next_hi = (hi + weight).min(capacity);
        let (first, second) = scratch[..2 * width].split_at_mut(width);
        let (src, dst) = if row == 0 {
            (first, second)
        } else {
            (second, first)
        };
        dst[..weight].copy_from_slice(&src[..weight]);
        let source = &src[..=next_hi - weight];
        let prior = &src[weight..=next_hi];
        let output = &mut dst[weight..=next_hi];
        let bits = &mut choices[t * width + weight..=t * width + next_hi];
        for (((out, bit), &old), &from) in output.iter_mut().zip(bits).zip(prior).zip(source) {
            let candidate = from + value;
            *bit = (candidate > old) as u8;
            *out = candidate.max(old);
        }
        hi = next_hi;
        row ^= 1;
    }
    let last = &scratch[row * width..(row + 1) * width];
    Some((0..=capacity).max_by_key(|&w| last[w]).unwrap_or(0))
}

/// Certify the current membership as the UNIQUE solution of the linear
/// knapsack that DP would solve. Every selected item must have positive
/// density strictly above every unselected item, and selected weight must
/// fill capacity. A positive separating multiplier then makes every change
/// strictly worse, including changes at equal total weight. Thus DP cannot
/// change membership, regardless of its internal tie order.
///
/// Caller supplies keys in original item order, with exact density in the
/// signed upper 48 bits, and positive weights. No hash equivalence is used.
pub(super) fn strict_density_solution(
    packed: &[usize], selected: &[bool], weights: &[u32], capacity: u32,
) -> bool {
    assert_eq!(packed.len(), selected.len());
    assert_eq!(packed.len(), weights.len());
    let mut lowest_selected = i64::MAX;
    let mut highest_unused = i64::MIN;
    let mut weight = 0u64;
    for ((&key, &chosen), &w) in packed.iter().zip(selected).zip(weights) {
        let density = (key as i64) >> 16;
        lowest_selected = lowest_selected.min(if chosen { density } else { i64::MAX });
        highest_unused = highest_unused.max(if chosen { i64::MIN } else { density });
        weight += w as u64 * chosen as u64;
    }
    weight == capacity as u64 && lowest_selected > 0 && lowest_selected > highest_unused
}

/// Preserve the caller's EXACT sentinel, every table decision byte and its
/// largest-final-weight tie. Only storage changes: immutable previous row
/// removes the in-place reverse-loop dependence and permits vectorization.
pub(super) fn preserving_sentinel<F:Fn(usize)->(usize,i32)>(
    count:usize,capacity:usize,sentinel:i32,choices:&mut [u8],item:F,
)->(usize,i64) {
    let width=capacity+1;
    assert_eq!(choices.len(),count*width);
    let mut rows=vec![sentinel;2*width];rows[0]=0;
    let mut current=0usize;let mut hi=0usize;
    for t in 0..count {
        let (weight,value)=item(t);
        if weight>capacity {continue;}
        let next_hi=(hi+weight).min(capacity);
        let (first,second)=rows.split_at_mut(width);
        let (src,dst)=if current==0 {(first,second)}else{(second,first)};
        dst[..weight].copy_from_slice(&src[..weight]);
        let bits=&mut choices[t*width+weight..=t*width+next_hi];
        for (((out,flag),&old),&from) in dst[weight..=next_hi].iter_mut()
            .zip(bits).zip(&src[weight..=next_hi]).zip(&src[..=next_hi-weight]) {
            let candidate=from+value;
            *flag=(candidate>old) as u8;
            *out=candidate.max(old);
        }
        current^=1;hi=next_hi;
    }
    let last=&rows[current*width..(current+1)*width];
    let best=(0..=capacity).max_by_key(|&i|last[i]).unwrap_or(0);
    (best,last[best] as i64)
}
