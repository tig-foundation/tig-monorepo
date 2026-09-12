//! Exact maximization of a two-for-two exchange with the caller's tie order.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub a: usize,
    pub b: usize,
    pub weight: i64,
    pub value: i64,
    pub ordinal: usize,
}

pub fn best_exchange<F: Fn(usize, usize) -> i64>(
    additions: &mut [Pair],
    removals: &mut [Pair],
    slack: i64,
    cross: F,
) -> Option<(Pair, Pair)> {
    if additions.is_empty() || removals.is_empty() {
        return None;
    }
    if additions.iter().all(|p| (0..=20).contains(&p.weight)) {
        return best_by_weight(additions, removals, slack, cross);
    }
    additions.sort_unstable_by(|a, b| b.value.cmp(&a.value).then(a.ordinal.cmp(&b.ordinal)));
    removals.sort_unstable_by(|a, b| a.value.cmp(&b.value).then(a.ordinal.cmp(&b.ordinal)));
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for &r in removals.iter() {
        if additions[0].value - r.value < best {
            break;
        }
        let budget = slack + r.weight;
        for &a in additions.iter() {
            let bound = a.value - r.value;
            if bound < best {
                break;
            }
            let order = (r.ordinal, a.ordinal);
            if bound <= 0 || (bound == best && order >= best_order) || a.weight > budget {
                continue;
            }
            let delta =
                bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
            if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                best = delta;
                best_order = order;
                movement = Some((r, a));
            }
        }
    }
    movement
}

fn best_by_weight<F: Fn(usize, usize) -> i64>(
    additions: &mut [Pair],
    removals: &mut [Pair],
    slack: i64,
    cross: F,
) -> Option<(Pair, Pair)> {
    // Filter using feasible opposite groups before sorting. Sorting 8-byte
    // indices keeps the 40-byte pair records stationary. Original ordinals,
    // rather than the work-buffer permutation, resolve every equal delta.
    let mut minimum = [i64::MAX; 21];
    for r in removals.iter() {
        let budget = slack + r.weight;
        if budget >= 0 {
            let b = budget.min(20) as usize;
            minimum[b] = minimum[b].min(r.value);
        }
    }
    for w in (0..20).rev() {
        minimum[w] = minimum[w].min(minimum[w + 1]);
    }
    let mut ai: Vec<usize> = additions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.value > minimum[a.weight as usize])
        .map(|(i, _)| i)
        .collect();
    if ai.is_empty() {
        return None;
    }
    ai.sort_unstable_by(|&i, &j| {
        let a = &additions[i];
        let b = &additions[j];
        a.weight
            .cmp(&b.weight)
            .then(b.value.cmp(&a.value))
            .then(a.ordinal.cmp(&b.ordinal))
    });
    let mut starts = [0usize; 22];
    for &i in &ai {
        starts[additions[i].weight as usize + 1] += 1;
    }
    for w in 1..22 {
        starts[w] += starts[w - 1];
    }
    let mut maxima = [i64::MIN; 21];
    let mut groups = [0usize; 21];
    let mut ng = 0;
    for w in 0..21 {
        if starts[w] != starts[w + 1] {
            maxima[w] = additions[ai[starts[w]]].value;
            groups[ng] = w;
            ng += 1;
        }
    }
    groups[..ng].sort_unstable_by_key(|&w| std::cmp::Reverse(maxima[w]));
    for w in 1..21 {
        maxima[w] = maxima[w].max(maxima[w - 1]);
    }
    let mut ri: Vec<usize> = removals
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            let budget = slack + r.weight;
            budget >= 0 && maxima[budget.min(20) as usize] > r.value
        })
        .map(|(i, _)| i)
        .collect();
    ri.sort_unstable_by(|&i, &j| {
        removals[i]
            .value
            .cmp(&removals[j].value)
            .then(removals[i].ordinal.cmp(&removals[j].ordinal))
    });
    let mut best = 0i64;
    let mut order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for rindex in ri {
        let r = removals[rindex];
        if maxima[20] - r.value < best {
            break;
        }
        let limit = (slack + r.weight).min(20) as usize;
        if maxima[limit] - r.value < best {
            continue;
        }
        for &w in &groups[..ng] {
            let indices = &ai[starts[w]..starts[w + 1]];
            if additions[indices[0]].value - r.value < best {
                break;
            }
            if w > limit {
                continue;
            }
            for &aindex in indices {
                let a = additions[aindex];
                let bound = a.value - r.value;
                if bound < best {
                    break;
                }
                let at = (r.ordinal, a.ordinal);
                if bound <= 0 || (bound == best && at >= order) {
                    continue;
                }
                let delta =
                    bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
                if delta > 0 && (delta > best || (delta == best && at < order)) {
                    best = delta;
                    order = at;
                    movement = Some((r, a));
                }
            }
        }
    }
    movement
}
