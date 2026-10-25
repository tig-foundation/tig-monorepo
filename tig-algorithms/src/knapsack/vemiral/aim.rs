use serde_json::{Map, Value};
use tig_challenges::knapsack::Challenge;

const SCALE: f64 = 1.25;

fn fraction(t: f64) -> f64 {
    if t > 1.0 { t / 1e6 } else { t }
}

struct Track {
    n_items: usize,
    budget_max: u32,
    target_ppm: i64,
    below: f64,
    above: f64,
    gain_ppm: &'static [i64],
}

#[rustfmt::skip]
const TRACKS: [Track; 3] = [
    Track { n_items: 1000, budget_max: 100, target_ppm: 53400, below: 0.001, above: 0.002,
        gain_ppm: &[16323, 2739, 1504, 1342, 1232, 1232, 1232, 0] },
    Track { n_items: 5000, budget_max: 17, target_ppm: 165000, below: 0.004, above: 0.006,
        gain_ppm: &[4963, 3949, 3556, 3556, 3556, 3488, 3480, 3480, 3480, 3480, 3480, 3480, 3480, 3480, 2229, 1521, 1521, 1390, 1390, 1390, 1390, 1390, 1390, 1390, 1390, 1390, 1002, 1002, 1002, 1002, 135, 30] },
    Track { n_items: 5000, budget_max: 100, target_ppm: 51000, below: 0.001, above: 0.002,
        gain_ppm: &[2856, 2263, 1835, 1763, 1590, 1067, 1067, 973, 958, 958, 958, 958, 958, 958, 958, 587, 587, 587, 587, 587, 587, 585, 585, 585, 263, 261, 261, 261, 261, 261, 25, 0] },
];

pub struct Aim {
    baseline: f64,
    target: f64,
    below: f64,
    above: f64,
    gain: &'static [i64],
}

impl Aim {
    pub fn new(ch: &Challenge, hp: &Option<Map<String, Value>>, row_sums: Option<&[i64]>) -> Option<Self> {
        let get = |key: &str| hp.as_ref().and_then(|m| m.get(key)).and_then(Value::as_f64);
        if get("aim") == Some(0.0) {
            return None;
        }
        let sum_w: u64 = ch.weights.iter().map(|&w| w as u64).sum();
        let budget = (ch.max_weight as u64 * 100 / sum_w.max(1)) as u32;
        let t = TRACKS.iter().find(|t| t.n_items == ch.num_items && budget <= t.budget_max)?;
        let owned;
        let rs = match row_sums {
            Some(r) => r,
            None => {
                owned = row_sums_of(ch);
                &owned
            }
        };
        Some(Self {
            baseline: baseline_value(ch, rs) as f64,
            target: get("aim_target").map(fraction).unwrap_or(t.target_ppm as f64 / 1e6),
            below: t.below,
            above: t.above,
            gain: t.gain_ppm,
        })
    }

    pub fn settled(&self, value: i64, done: usize, total: usize) -> bool {
        if done >= total {
            return false;
        }
        let q = value as f64 / self.baseline - 1.0;
        let len = self.gain.len();
        let gain = self.gain[((done - 1) * len / total).min(len - 1)] as f64 / 1e6;
        q >= self.target + self.above || q + SCALE * gain + self.below < self.target
    }
}

struct Lane {
    budget_max: u32,
    target_ppm: i64,
    band_ppm: i64,
    heavy: &'static [(&'static str, u64)],
}

const LANES: [Lane; 2] = [
    Lane { budget_max: 7, target_ppm: 467900, band_ppm: 15000,
        heavy: &[("core_ktgt", 200), ("core_work", 3000000)] },
    Lane { budget_max: 17, target_ppm: 235700, band_ppm: 15000,
        heavy: &[("core_kmax", 400), ("core_work", 27000000)] },
];

pub struct Boost {
    baseline: f64,
    target: f64,
    band: f64,
    hp: Map<String, Value>,
}

impl Boost {
    pub fn new(ch: &Challenge, hp: &Option<Map<String, Value>>, budget: u32) -> Option<Self> {
        let get = |key: &str| hp.as_ref().and_then(|m| m.get(key)).and_then(Value::as_f64);
        if get("aim") == Some(0.0) {
            return None;
        }
        let lane = LANES.iter().find(|l| budget <= l.budget_max)?;
        let mut heavy = hp.clone().unwrap_or_default();
        for &(k, v) in lane.heavy {
            heavy.insert(k.to_string(), Value::from(v));
        }
        Some(Self {
            baseline: baseline_value(ch, &row_sums_of(ch)) as f64,
            target: get("aim_target").map(fraction).unwrap_or(lane.target_ppm as f64 / 1e6),
            band: lane.band_ppm as f64 / 1e6,
            hp: heavy,
        })
    }

    pub fn heavy_hp(&self, value: i64) -> Option<Map<String, Value>> {
        let q = value as f64 / self.baseline - 1.0;
        if q < self.target && q >= self.target - self.band {
            Some(self.hp.clone())
        } else {
            None
        }
    }
}

pub fn value(ch: &Challenge, items: &[usize]) -> i64 {
    let mut v = 0i64;
    for (a, &i) in items.iter().enumerate() {
        v += ch.values[i] as i64;
        let row = &ch.interaction_values[i];
        for &j in &items[a + 1..] {
            v += row[j] as i64;
        }
    }
    v
}

pub fn feasible(ch: &Challenge, items: &[usize]) -> bool {
    let mut seen = vec![false; ch.num_items];
    let mut w = 0u64;
    for &i in items {
        if i >= ch.num_items || seen[i] {
            return false;
        }
        seen[i] = true;
        w += ch.weights[i] as u64;
    }
    w <= ch.max_weight as u64
}

pub fn small_target(ch:&Challenge,hp:&Option<Map<String,Value>>)->Option<(i64,i64)> {
    let get=|key:&str|hp.as_ref().and_then(|m|m.get(key)).and_then(Value::as_f64);
    if ch.num_items!=1000 || get("aim")==Some(0.0) {return None;}
    let sum:u64=ch.weights.iter().map(|&w|w as u64).sum();
    let budget=(ch.max_weight as u64*100/sum.max(1)) as u32;
    let lane=LANES.iter().find(|lane|budget<=lane.budget_max)?;
    let baseline=baseline_value(ch,&row_sums_of(ch));
    let target=get("aim_target").map(fraction).unwrap_or(lane.target_ppm as f64/1e6);
    Some((baseline,(baseline as f64*(1.0+target)).ceil() as i64))
}

pub fn finish_before_search(ch:&Challenge,hp:&Option<Map<String,Value>>,value:i64,upper:i64)->bool {
    if let Some((baseline,target))=small_target(ch,hp) {
        let margin=(baseline+999)/1000;
        value>=target.saturating_add(margin) || upper<target.saturating_sub(margin)
    }else{false}
}

fn row_sums_of(ch: &Challenge) -> Vec<i64> {
    ch.interaction_values
        .iter()
        .map(|row| {
            let mut acc = [0i32; 8];
            let mut chunks = row.chunks_exact(8);
            for c in &mut chunks {
                for k in 0..8 {
                    acc[k] += c[k];
                }
            }
            let tail: i32 = chunks.remainder().iter().sum();
            (acc.iter().sum::<i32>() + tail) as i64
        })
        .collect()
}

fn baseline_value(ch: &Challenge, row_sums: &[i64]) -> i64 {
    let n = ch.num_items;
    let w = &ch.weights;
    let iv = &ch.interaction_values;
    let cap = ch.max_weight;
    let mut order: Vec<(usize, f32)> = (0..n)
        .map(|i| (i, (ch.values[i] as i32 + row_sums[i] as i32) as f32 / w[i] as f32))
        .collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut sel: Vec<usize> = Vec::with_capacity(n);
    let mut uns: Vec<usize> = Vec::with_capacity(n);
    let mut tw = 0u32;
    for &(i, _) in &order {
        if tw + w[i] <= cap {
            tw += w[i];
            sel.push(i);
        } else {
            uns.push(i);
        }
    }
    let mut isl: Vec<i32> = (0..n).map(|x| ch.values[x] as i32).collect();
    for &s in &sel {
        for (a, &b) in isl.iter_mut().zip(iv[s].iter()) {
            *a += b;
        }
    }
    let mut tabu = vec![0u8; n];
    let mut small: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..100 {
        let mut fl = [i32::MAX; 12];
        for &s in &sel {
            let c = w[s] as usize;
            if tabu[s] == 0 && isl[s] < fl[c] {
                fl[c] = isl[s];
            }
        }
        for c in (1..11).rev() {
            fl[c] = fl[c].min(fl[c + 1]);
        }
        small.clear();
        for (p, &s) in sel.iter().enumerate() {
            if tabu[s] == 0 && isl[s] <= fl[w[s] as usize].saturating_add(1000) {
                small.push(p as u32);
            }
        }
        let slack = cap as i32 - tw as i32;
        let class = |u: usize| ((w[u] as i32 - slack).max(1) as usize).min(11);
        let best_rem = |u: usize| -> (i32, u32) {
            let row = &iv[u];
            let c0 = class(u) as u32;
            let lim = fl[c0 as usize].saturating_add(1000);
            let mut m = i32::MAX;
            let mut mp = u32::MAX;
            for &p in &small {
                let s = sel[p as usize];
                if w[s] < c0 || isl[s] > lim {
                    continue;
                }
                let c = isl[s] + row[s];
                if c < m || (c == m && p < mp) {
                    m = c;
                    mp = p;
                }
            }
            (m, mp)
        };
        let bound = |u: usize| isl[u].saturating_sub(fl[class(u)]);
        let mut best = 0i32;
        let mut bi = u32::MAX;
        let mut bj = u32::MAX;
        let mut top = u32::MAX;
        let mut top_b = i32::MIN;
        for (p, &u) in uns.iter().enumerate() {
            if tabu[u] == 0 && bound(u) > top_b {
                top_b = bound(u);
                top = p as u32;
            }
        }
        if top != u32::MAX && top_b > 0 {
            let (m, mp) = best_rem(uns[top as usize]);
            let d = isl[uns[top as usize]] - m;
            if mp != u32::MAX && d > 0 {
                best = d;
                bi = top;
                bj = mp;
            }
        }
        for (p, &u) in uns.iter().enumerate() {
            if tabu[u] != 0 || bound(u) < best || bound(u) <= 0 || (p as u32) == top {
                continue;
            }
            let (m, mp) = best_rem(u);
            if mp == u32::MAX {
                continue;
            }
            let d = isl[u] - m;
            if d > best || (d == best && d > 0 && (p as u32) < bi) {
                best = d;
                bi = p as u32;
                bj = mp;
            }
        }
        if bi == u32::MAX {
            break;
        }
        let (ui, si) = (bi as usize, bj as usize);
        let nu = uns[ui];
        let rs = sel[si];
        sel.swap_remove(si);
        uns.swap_remove(ui);
        sel.push(nu);
        uns.push(rs);
        tw = tw + w[nu] - w[rs];
        for ((a, &x), &y) in isl.iter_mut().zip(iv[nu].iter()).zip(iv[rs].iter()) {
            *a += x - y;
        }
        tabu[nu] = 3;
        tabu[rs] = 3;
        for t in tabu.iter_mut() {
            *t = t.saturating_sub(1);
        }
    }
    let v: i64 = sel.iter().map(|&s| isl[s] as i64 + ch.values[s] as i64).sum();
    (v / 2).max(0)
}

pub fn near_target(ch:&Challenge,hp:&Option<Map<String,Value>>,value:i64,width:f64)->bool{
 let get=|key:&str|hp.as_ref().and_then(|m|m.get(key)).and_then(Value::as_f64);
 if get("aim")==Some(0.0){return false;}
 let sum_w:u64=ch.weights.iter().map(|&w|w as u64).sum();
 let budget=(ch.max_weight as u64*100/sum_w.max(1)) as u32;
 let t=match TRACKS.iter().find(|t|t.n_items==ch.num_items && budget<=t.budget_max){Some(t)=>t,None=>return false};
 let target=get("aim_target").map(fraction).unwrap_or(t.target_ppm as f64/1e6);
 let baseline=baseline_value(ch,&row_sums_of(ch)) as f64;
 let q=value as f64/baseline-1.0;
 q<target && q>=target-width
}
