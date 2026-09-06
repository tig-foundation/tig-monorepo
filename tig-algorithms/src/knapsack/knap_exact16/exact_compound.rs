//! Exact bounded exchanges. Every candidate retains its original ordinal.
//! The quadratic interactions must be nonnegative, as in the TIG generator.

#[derive(Clone, Copy, Debug)]
pub(super) struct Group {
    pub ids: [usize; 3],
    pub len: usize,
    pub weight: i64,
    pub value: i64,
    pub ordinal: usize,
}

pub(super) fn groups<W, C, Q>(
    items: &[usize],
    count: usize,
    weight: W,
    value: C,
    q: Q,
    add: bool,
) -> Vec<Group>
where
    W: Fn(usize) -> i64,
    C: Fn(usize) -> i64,
    Q: Fn(usize, usize) -> i64,
{
    let mut out = Vec::new();
    for (i, &a) in items.iter().enumerate() {
        if count == 1 {
            out.push(Group {
                ids: [a, 0, 0],
                len: 1,
                weight: weight(a),
                value: value(a),
                ordinal: out.len(),
            });
            continue;
        }
        for (j, &b) in items.iter().enumerate().skip(i + 1) {
            let pair = value(a) + value(b) + if add { q(a, b) } else { -q(a, b) };
            let pair_weight = weight(a) + weight(b);
            if count == 2 {
                out.push(Group {
                    ids: [a, b, 0],
                    len: 2,
                    weight: pair_weight,
                    value: pair,
                    ordinal: out.len(),
                });
            } else {
                for &c in &items[j + 1..] {
                    let extra = q(a, c) + q(b, c);
                    out.push(Group {
                        ids: [a, b, c],
                        len: 3,
                        weight: pair_weight + weight(c),
                        value: pair + value(c) + if add { extra } else { -extra },
                        ordinal: out.len(),
                    });
                }
            }
        }
    }
    out
}

pub(super) fn best<Q>(
    adds: &[Group],
    removes: &[Group],
    slack: i64,
    add_outer: bool,
    stop_nonpositive: bool,
    q: Q,
) -> Option<(Group, Group)>
where
    Q: Fn(usize, usize) -> i64,
{
    if adds.is_empty() || removes.is_empty() {
        return None;
    }
    let mut best = 0i64;
    let mut order = (usize::MAX, usize::MAX);
    let mut result = None;
    let consider = |r: Group, a: Group| {
        let mut delta = a.value - r.value;
        for &i in &a.ids[..a.len] {
            for &j in &r.ids[..r.len] {
                delta -= q(i, j);
            }
        }
        let key = if add_outer {
            (a.ordinal, r.ordinal)
        } else {
            (r.ordinal, a.ordinal)
        };
        (delta, key)
    };
    // Small integral weights are a generator property. The fallback retains
    // candidate order and the same cross penalties for other weights.
    if adds
        .iter()
        .chain(removes)
        .any(|g| !(0..=30).contains(&g.weight))
    {
        if add_outer {
            for &a in adds {
                if stop_nonpositive && a.value <= 0 && best > 0 {
                    break;
                }
                for &r in removes {
                    if a.weight > slack + r.weight {
                        continue;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        } else {
            for &r in removes {
                for &a in adds {
                    if a.weight > slack + r.weight {
                        continue;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
        return result;
    }
    // A cross penalty cannot improve a move. First eliminate groups that
    // cannot beat even the cheapest weight-feasible opposite group.
    let mut buckets: [Vec<usize>; 31] = Default::default();
    if add_outer {
        let mut maximum = [i64::MIN; 31];
        for a in adds {
            maximum[a.weight as usize] = maximum[a.weight as usize].max(a.value);
        }
        for w in 1..31 {
            maximum[w] = maximum[w].max(maximum[w - 1]);
        }
        for (i, r) in removes.iter().enumerate() {
            let budget = slack + r.weight;
            if budget >= 0 && maximum[budget.min(30) as usize] > r.value {
                buckets[r.weight as usize].push(i);
            }
        }
        for b in &mut buckets {
            b.sort_unstable_by_key(|&i| (removes[i].value, removes[i].ordinal));
        }
        let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
        weights.sort_unstable_by_key(|&w| removes[buckets[w][0]].value);
        for &a in adds {
            if stop_nonpositive && a.value <= 0 && best > 0 {
                break;
            }
            for &w in &weights {
                if a.value - removes[buckets[w][0]].value < best {
                    break;
                }
                if a.weight > slack + w as i64 {
                    continue;
                }
                for &i in &buckets[w] {
                    let r = removes[i];
                    if a.value - r.value < best {
                        break;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
    } else {
        let mut minimum = [i64::MAX; 31];
        for r in removes {
            minimum[r.weight as usize] = minimum[r.weight as usize].min(r.value);
        }
        for w in (0..30).rev() {
            minimum[w] = minimum[w].min(minimum[w + 1]);
        }
        for (i, a) in adds.iter().enumerate() {
            let need = (a.weight - slack).max(0);
            if need <= 30 && a.value > minimum[need as usize] {
                buckets[a.weight as usize].push(i);
            }
        }
        for b in &mut buckets {
            b.sort_unstable_by_key(|&i| (std::cmp::Reverse(adds[i].value), adds[i].ordinal));
        }
        let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
        weights.sort_unstable_by_key(|&w| std::cmp::Reverse(adds[buckets[w][0]].value));
        for &r in removes {
            for &w in &weights {
                if adds[buckets[w][0]].value - r.value < best {
                    break;
                }
                if w as i64 > slack + r.weight {
                    continue;
                }
                for &i in &buckets[w] {
                    let a = adds[i];
                    if a.value - r.value < best {
                        break;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
    }
    result
}
