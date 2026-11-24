use tig_challenges::knapsack::{Challenge, Solution};
#[derive(Clone)]
struct Block {
    ids: Vec<usize>,
    weight: i64,
    value: i64,
}
fn groups(ch: &Challenge, bits: &[bool], c: &[i64], take: bool, width: usize) -> Vec<Block> {
    let ids: Vec<usize> = (0..ch.num_items).filter(|&i| bits[i] == take).collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for &anchor in &ids {
        let mut near: Vec<(i32, usize)> = ids
            .iter()
            .filter(|&&i| i != anchor && ch.interaction_values[anchor][i] > 0)
            .map(|&i| (ch.interaction_values[anchor][i], i))
            .collect();
        let k = width.min(near.len());
        if k < 1 {
            continue;
        }
        if k < near.len() {
            near.select_nth_unstable_by(k, |a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            near.truncate(k);
        }
        let mut a = vec![anchor];
        a.extend(near.iter().map(|&(_, i)| i));
        a.sort_unstable();
        loop {
            let weight: i64 = a.iter().map(|&i| ch.weights[i] as i64).sum();
            let mut v: i64 = a.iter().map(|&i| c[i]).sum();
            for x in 0..a.len() {
                for y in x + 1..a.len() {
                    let q = ch.interaction_values[a[x]][a[y]] as i64;
                    if take {
                        v -= q;
                    } else {
                        v += q;
                    }
                }
            }
            let mut rem = a.len();
            let mut worst = 0i64;
            for (x, &i) in a.iter().enumerate() {
                let internal: i64 = a
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| ch.interaction_values[i][j] as i64)
                    .sum();
                let marginal = if take {
                    c[i] - internal
                } else {
                    c[i] + internal
                };
                let excess = if take {
                    marginal * weight - v * ch.weights[i] as i64
                } else {
                    v * ch.weights[i] as i64 - marginal * weight
                };
                if excess > worst {
                    worst = excess;
                    rem = x;
                }
            }
            if rem < a.len() && a.len() > 2 {
                a.remove(rem);
                continue;
            }
            if seen.insert(a.clone()) {
                out.push(Block {
                    ids: a,
                    weight,
                    value: v,
                });
            }
            break;
        }
    }
    if take {
        out.sort_by(|a, b| {
            (a.value * b.weight)
                .cmp(&(b.value * a.weight))
                .then(a.ids.cmp(&b.ids))
        });
    } else {
        out.sort_by(|a, b| {
            (b.value * a.weight)
                .cmp(&(a.value * b.weight))
                .then(a.ids.cmp(&b.ids))
        });
    }
    out.truncate(128);
    out
}
fn combine(ch: &Challenge, groups: &mut Vec<Block>, take: bool) {
    let mut more = Vec::new();
    let mut seen: std::collections::BTreeSet<Vec<usize>> =
        groups.iter().map(|b| b.ids.clone()).collect();
    for a in 0..groups.len().min(24) {
        for b in a + 1..groups.len().min(24) {
            if groups[a]
                .ids
                .iter()
                .any(|i| groups[b].ids.binary_search(i).is_ok())
            {
                continue;
            }
            let cross: i64 = groups[a]
                .ids
                .iter()
                .flat_map(|&i| {
                    groups[b]
                        .ids
                        .iter()
                        .map(move |&j| ch.interaction_values[i][j] as i64)
                })
                .sum();
            let mut ids = groups[a].ids.clone();
            ids.extend_from_slice(&groups[b].ids);
            ids.sort_unstable();
            if seen.insert(ids.clone()) {
                more.push(Block {
                    ids,
                    weight: groups[a].weight + groups[b].weight,
                    value: groups[a].value + groups[b].value + if take { -cross } else { cross },
                });
            }
        }
    }
    groups.extend(more);
    if take {
        groups.sort_by(|a, b| {
            (a.value * b.weight)
                .cmp(&(b.value * a.weight))
                .then(a.ids.cmp(&b.ids))
        });
    } else {
        groups.sort_by(|a, b| {
            (b.value * a.weight)
                .cmp(&(a.value * b.weight))
                .then(a.ids.cmp(&b.ids))
        });
    }
    groups.truncate(256);
}
fn candidates(
    ch: &Challenge,
    sol: &Solution,
    width: usize,
    count: usize,
    compound: bool,
) -> Vec<Solution> {
    let n = ch.num_items;
    let mut bits = vec![false; n];
    let mut c: Vec<i64> = ch.values.iter().map(|&v| v as i64).collect();
    let mut weight = 0i64;
    for &i in &sol.items {
        bits[i] = true;
        weight += ch.weights[i] as i64;
        for j in 0..n {
            c[j] += ch.interaction_values[i][j] as i64;
        }
    }
    let mut incoming = groups(ch, &bits, &c, false, width);
    let mut outgoing = groups(ch, &bits, &c, true, width);
    if compound {
        combine(ch, &mut incoming, false);
        combine(ch, &mut outgoing, true);
    }
    let lam = (0..n)
        .filter(|&i| bits[i])
        .map(|i| c[i] as f64 / ch.weights[i] as f64)
        .fold(f64::INFINITY, f64::min);
    let mut ranked = Vec::new();
    for (a, add) in incoming.iter().enumerate() {
        for (b, rem) in outgoing.iter().enumerate() {
            let cross: i64 = add
                .ids
                .iter()
                .flat_map(|&i| {
                    rem.ids
                        .iter()
                        .map(move |&j| ch.interaction_values[i][j] as i64)
                })
                .sum();
            let d = add.value - rem.value - cross;
            let dw = add.weight - rem.weight;
            if dw > if compound { 80 } else { 40 } {
                continue;
            }
            ranked.push((d as f64 - lam * dw as f64, a, b));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut result = Vec::new();
    for &(_, a, b) in ranked.iter().take(count) {
        let mut x = bits.clone();
        let mut d = c.clone();
        let mut wt = weight;
        for &i in &outgoing[b].ids {
            x[i] = false;
            wt -= ch.weights[i] as i64;
            for j in 0..n {
                d[j] -= ch.interaction_values[i][j] as i64;
            }
        }
        for &i in &incoming[a].ids {
            x[i] = true;
            wt += ch.weights[i] as i64;
            for j in 0..n {
                d[j] += ch.interaction_values[i][j] as i64;
            }
        }
        while wt > ch.max_weight as i64 {
            let mut worst = n;
            for i in 0..n {
                if x[i]
                    && !incoming[a].ids.contains(&i)
                    && (worst == n
                        || d[i] * (ch.weights[worst] as i64) < d[worst] * (ch.weights[i] as i64))
                {
                    worst = i;
                }
            }
            if worst == n {
                break;
            }
            x[worst] = false;
            wt -= ch.weights[worst] as i64;
            for j in 0..n {
                d[j] -= ch.interaction_values[worst][j] as i64;
            }
        }
        while wt < (ch.max_weight as i64) {
            let mut best = n;
            for i in 0..n {
                if !x[i]
                    && wt + ch.weights[i] as i64 <= ch.max_weight as i64
                    && !outgoing[b].ids.contains(&i)
                    && (best == n
                        || d[i] * (ch.weights[best] as i64) > d[best] * (ch.weights[i] as i64))
                {
                    best = i;
                }
            }
            if best == n {
                break;
            }
            x[best] = true;
            wt += ch.weights[best] as i64;
            for j in 0..n {
                d[j] += ch.interaction_values[best][j] as i64;
            }
        }
        let s = Solution {
            items: (0..n).filter(|&i| x[i]).collect(),
        };
        if super::aim::feasible(ch, &s.items) {
            result.push(s);
        }
    }
    result
}

pub fn improve(ch: &Challenge, initial: &Solution) -> Solution {
    let mut best = initial.clone();
    let mut best_value = super::aim::value(ch, &best.items);
    for candidate in candidates(ch, initial, 24, 12, true) {
        let out = super::track4::refine(ch, &candidate);
        if super::aim::feasible(ch, &out.items) {
            let value = super::aim::value(ch, &out.items);
            if value > best_value {
                best = out;
                best_value = value;
            }
        }
    }
    best
}
