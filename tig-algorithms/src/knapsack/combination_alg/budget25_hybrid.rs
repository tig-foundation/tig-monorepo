use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::knapsack::{Challenge, Solution};

use super::{superfast_t1000, superfast_t5000, v11_t40, v11_t43};

#[derive(Clone, Copy)]
struct PairMove {
    first: usize,
    second: usize,
    weight: u32,
    score: i64,
}

struct State<'a> {
    challenge: &'a Challenge,
    selected: Vec<bool>,
    contribution: Vec<i64>,
    value: i64,
    weight: u32,
}

impl<'a> State<'a> {
    fn from_solution(challenge: &'a Challenge, solution: &Solution) -> Self {
        let n = challenge.num_items;
        let mut state = Self {
            challenge,
            selected: vec![false; n],
            contribution: challenge.values.iter().map(|&value| value as i64).collect(),
            value: 0,
            weight: 0,
        };
        for &item in &solution.items {
            if item < n
                && !state.selected[item]
                && state.weight + challenge.weights[item] <= challenge.max_weight
            {
                state.add(item);
            }
        }
        state
    }

    #[inline(always)]
    fn slack(&self) -> u32 {
        self.challenge.max_weight - self.weight
    }

    #[inline(always)]
    fn add(&mut self, item: usize) {
        self.value += self.contribution[item];
        self.weight += self.challenge.weights[item];
        let row = unsafe { self.challenge.interaction_values.get_unchecked(item) };
        for (contribution, &interaction) in self.contribution.iter_mut().zip(row.iter()) {
            *contribution += interaction as i64;
        }
        self.selected[item] = true;
    }

    #[inline(always)]
    fn remove(&mut self, item: usize) {
        self.value -= self.contribution[item];
        self.weight -= self.challenge.weights[item];
        let row = unsafe { self.challenge.interaction_values.get_unchecked(item) };
        for (contribution, &interaction) in self.contribution.iter_mut().zip(row.iter()) {
            *contribution -= interaction as i64;
        }
        self.selected[item] = false;
    }

    fn solution(&self) -> Solution {
        Solution {
            items: (0..self.challenge.num_items)
                .filter(|&item| self.selected[item])
                .collect(),
        }
    }

    fn best_add_or_swap(&mut self) -> bool {
        let n = self.challenge.num_items;
        let slack = self.slack();
        let mut best_delta = 0i64;
        let mut best_move: Option<(Option<usize>, usize)> = None;

        for add in 0..n {
            if self.selected[add] {
                continue;
            }
            let add_weight = self.challenge.weights[add];
            if add_weight <= slack && self.contribution[add] > best_delta {
                best_delta = self.contribution[add];
                best_move = Some((None, add));
            }
        }

        for remove in 0..n {
            if !self.selected[remove] {
                continue;
            }
            let available = slack + self.challenge.weights[remove];
            for add in 0..n {
                if self.selected[add] || self.challenge.weights[add] > available {
                    continue;
                }
                let delta = self.contribution[add]
                    - self.contribution[remove]
                    - self.challenge.interaction_values[add][remove] as i64;
                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some((Some(remove), add));
                }
            }
        }

        match best_move {
            Some((None, add)) => self.add(add),
            Some((Some(remove), add)) => {
                self.remove(remove);
                self.add(add);
            }
            None => return false,
        }
        true
    }
}

fn retain_pair(bucket: &mut Vec<PairMove>, pair: PairMove, limit: usize, keep_largest: bool) {
    bucket.push(pair);
    if bucket.len() <= limit {
        return;
    }
    let mut worst = 0usize;
    for index in 1..bucket.len() {
        let is_worse = if keep_largest {
            bucket[index].score < bucket[worst].score
        } else {
            bucket[index].score > bucket[worst].score
        };
        if is_worse {
            worst = index;
        }
    }
    bucket.swap_remove(worst);
}

fn build_pair_buckets(state: &State, limit: usize) -> (Vec<Vec<PairMove>>, Vec<Vec<PairMove>>) {
    let n = state.challenge.num_items;
    let mut remove_pairs = vec![Vec::new(); 21];
    let mut add_pairs = vec![Vec::new(); 21];

    for first in 0..n {
        for second in (first + 1)..n {
            if state.selected[first] != state.selected[second] {
                continue;
            }
            let weight = state.challenge.weights[first] + state.challenge.weights[second];
            let interaction = state.challenge.interaction_values[first][second] as i64;
            if state.selected[first] {
                let loss = state.contribution[first] + state.contribution[second] - interaction;
                retain_pair(
                    &mut remove_pairs[weight as usize],
                    PairMove {
                        first,
                        second,
                        weight,
                        score: loss,
                    },
                    limit,
                    false,
                );
            } else {
                let gain = state.contribution[first] + state.contribution[second] + interaction;
                retain_pair(
                    &mut add_pairs[weight as usize],
                    PairMove {
                        first,
                        second,
                        weight,
                        score: gain,
                    },
                    limit,
                    true,
                );
            }
        }
    }
    (remove_pairs, add_pairs)
}

enum CompositeMove {
    OneToTwo(usize, usize, usize),
    TwoToOne(usize, usize, usize),
    TwoToTwo(usize, usize, usize, usize),
}

fn apply_best_composite(state: &mut State, pair_limit: usize) -> bool {
    let n = state.challenge.num_items;
    let slack = state.slack();
    let (remove_pairs, add_pairs) = build_pair_buckets(state, pair_limit);
    let mut best_delta = 0i64;
    let mut best_move = None;

    // One selected item to two unselected items. Candidate pairs are ranked by
    // their exact current gain, then corrected for the removed cross edges.
    for remove in 0..n {
        if !state.selected[remove] {
            continue;
        }
        let available = slack + state.challenge.weights[remove];
        for add_weight in 2..=20usize {
            if add_weight as u32 > available {
                continue;
            }
            for pair in &add_pairs[add_weight] {
                let cross = state.challenge.interaction_values[pair.first][remove] as i64
                    + state.challenge.interaction_values[pair.second][remove] as i64;
                let delta = pair.score - cross - state.contribution[remove];
                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some(CompositeMove::OneToTwo(remove, pair.first, pair.second));
                }
            }
        }
    }

    // Two selected items to one unselected item.
    for remove_weight in 2..=20usize {
        for pair in &remove_pairs[remove_weight] {
            let available = slack + pair.weight;
            for add in 0..n {
                if state.selected[add] || state.challenge.weights[add] > available {
                    continue;
                }
                let cross = state.challenge.interaction_values[add][pair.first] as i64
                    + state.challenge.interaction_values[add][pair.second] as i64;
                let delta = state.contribution[add] - cross - pair.score;
                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some(CompositeMove::TwoToOne(pair.first, pair.second, add));
                }
            }
        }
    }

    // Edge-aware 2-for-2: retain strong incoming pairs and weak outgoing pairs
    // for each of the only 19 possible pair weights, then evaluate cross terms
    // exactly. This reaches useful pairs that contribution-only top-k VND misses.
    for remove_weight in 2..=20usize {
        for removed in &remove_pairs[remove_weight] {
            let available = slack + removed.weight;
            for add_weight in 2..=20usize {
                if add_weight as u32 > available {
                    continue;
                }
                for added in &add_pairs[add_weight] {
                    let cross = state.challenge.interaction_values[added.first][removed.first]
                        as i64
                        + state.challenge.interaction_values[added.first][removed.second] as i64
                        + state.challenge.interaction_values[added.second][removed.first] as i64
                        + state.challenge.interaction_values[added.second][removed.second] as i64;
                    let delta = added.score - cross - removed.score;
                    if delta > best_delta {
                        best_delta = delta;
                        best_move = Some(CompositeMove::TwoToTwo(
                            removed.first,
                            removed.second,
                            added.first,
                            added.second,
                        ));
                    }
                }
            }
        }
    }

    match best_move {
        Some(CompositeMove::OneToTwo(remove, first, second)) => {
            state.remove(remove);
            state.add(first);
            state.add(second);
        }
        Some(CompositeMove::TwoToOne(first, second, add)) => {
            state.remove(first);
            state.remove(second);
            state.add(add);
        }
        Some(CompositeMove::TwoToTwo(remove_first, remove_second, add_first, add_second)) => {
            state.remove(remove_first);
            state.remove(remove_second);
            state.add(add_first);
            state.add(add_second);
        }
        None => return false,
    }
    true
}

fn polish(challenge: &Challenge, solution: &Solution) -> Solution {
    let mut state = State::from_solution(challenge, solution);
    let (round_limit, pair_limit) = if challenge.num_items <= 1200 {
        (24usize, 128usize)
    } else {
        (12usize, 48usize)
    };

    for _ in 0..round_limit {
        while state.best_add_or_swap() {}
        if !apply_best_composite(&mut state, pair_limit) {
            break;
        }
    }
    while state.best_add_or_swap() {}
    state.solution()
}

fn exact_core_move(state: &mut State, core: &[usize]) -> bool {
    let m = core.len();
    if m == 0 || m >= usize::BITS as usize {
        return false;
    }

    let mut in_core = vec![false; state.challenge.num_items];
    let mut outside_weight = state.weight;
    for &item in core {
        in_core[item] = true;
        if state.selected[item] {
            outside_weight -= state.challenge.weights[item];
        }
    }
    let capacity = state.challenge.max_weight - outside_weight;

    let mut linear = vec![0i64; m];
    for (position, &item) in core.iter().enumerate() {
        let mut gain = state.challenge.values[item] as i64;
        for other in 0..state.challenge.num_items {
            if state.selected[other] && !in_core[other] {
                gain += state.challenge.interaction_values[item][other] as i64;
            }
        }
        linear[position] = gain;
    }

    let mut current_mask = 0usize;
    let mut current_core_value = 0i64;
    for (position, &item) in core.iter().enumerate() {
        if state.selected[item] {
            current_mask |= 1usize << position;
            current_core_value += linear[position];
            for previous in 0..position {
                if current_mask & (1usize << previous) != 0 {
                    current_core_value +=
                        state.challenge.interaction_values[item][core[previous]] as i64;
                }
            }
        }
    }

    let mut best_mask = current_mask;
    let mut best_value = current_core_value;
    let mut gray_mask = 0usize;
    let mut gray_weight = 0u32;
    let mut gray_value = 0i64;
    for step in 1usize..(1usize << m) {
        let next_mask = step ^ (step >> 1);
        let changed = gray_mask ^ next_mask;
        let position = changed.trailing_zeros() as usize;
        let item = core[position];
        let adding = next_mask & changed != 0;
        let mut interaction = 0i64;
        let other_mask = if adding { gray_mask } else { next_mask };
        let mut remaining = other_mask;
        while remaining != 0 {
            let other_position = remaining.trailing_zeros() as usize;
            interaction += state.challenge.interaction_values[item][core[other_position]] as i64;
            remaining &= remaining - 1;
        }
        if adding {
            gray_weight += state.challenge.weights[item];
            gray_value += linear[position] + interaction;
        } else {
            gray_weight -= state.challenge.weights[item];
            gray_value -= linear[position] + interaction;
        }
        gray_mask = next_mask;
        if gray_weight <= capacity && gray_value > best_value {
            best_value = gray_value;
            best_mask = gray_mask;
        }
    }

    if best_value <= current_core_value {
        return false;
    }
    for &item in core.iter().rev() {
        if state.selected[item] {
            state.remove(item);
        }
    }
    for (position, &item) in core.iter().enumerate() {
        if best_mask & (1usize << position) != 0 {
            state.add(item);
        }
    }
    true
}

fn block_delta(state: &State, adds: &[usize], removes: &[usize]) -> i64 {
    let mut delta = 0i64;
    for (position, &item) in adds.iter().enumerate() {
        delta += state.contribution[item];
        for &previous in &adds[..position] {
            delta += state.challenge.interaction_values[item][previous] as i64;
        }
        for &removed in removes {
            delta -= state.challenge.interaction_values[item][removed] as i64;
        }
    }
    for (position, &item) in removes.iter().enumerate() {
        delta -= state.contribution[item];
        for &previous in &removes[..position] {
            delta += state.challenge.interaction_values[item][previous] as i64;
        }
    }
    delta
}

fn block_polish(challenge: &Challenge, solution: &Solution) -> Solution {
    let mut state = State::from_solution(challenge, solution);
    let n = challenge.num_items;
    let (round_limit, anchor_limit, bundle_limit) = if n <= 1200 {
        (8usize, 256usize, 10usize)
    } else {
        (4usize, 192usize, 8usize)
    };
    let global_interaction: Vec<i64> = (0..n)
        .map(|item| {
            challenge.interaction_values[item]
                .iter()
                .map(|&v| v as i64)
                .sum()
        })
        .collect();

    for _ in 0..round_limit {
        while state.best_add_or_swap() {}
        let mut anchors: Vec<usize> = (0..n).filter(|&item| !state.selected[item]).collect();
        anchors.sort_unstable_by_key(|&item| {
            std::cmp::Reverse(
                (global_interaction[item] + 4 * state.contribution[item]) * 1024
                    / challenge.weights[item] as i64,
            )
        });
        anchors.truncate(anchor_limit);

        let selected: Vec<usize> = (0..n).filter(|&item| state.selected[item]).collect();
        let mut best_delta = 0i64;
        let mut best_adds = Vec::new();
        let mut best_removes = Vec::new();

        for anchor in anchors {
            let mut in_bundle = vec![false; n];
            let mut adds = vec![anchor];
            in_bundle[anchor] = true;
            while adds.len() < bundle_limit {
                let mut next = None;
                let mut next_score = i64::MIN;
                for item in 0..n {
                    if state.selected[item] || in_bundle[item] {
                        continue;
                    }
                    let mutual: i64 = adds
                        .iter()
                        .map(|&member| challenge.interaction_values[item][member] as i64)
                        .sum();
                    let score = (state.contribution[item] + 8 * mutual) * 1024
                        / challenge.weights[item] as i64;
                    if score > next_score {
                        next_score = score;
                        next = Some(item);
                    }
                }
                let Some(next_item) = next else { break };
                adds.push(next_item);
                in_bundle[next_item] = true;
                if adds.len() < 2 {
                    continue;
                }

                let add_weight: u32 = adds.iter().map(|&item| challenge.weights[item]).sum();
                let required = add_weight.saturating_sub(state.slack());
                let mut ranked_removals: Vec<(usize, i64)> = selected
                    .iter()
                    .map(|&item| {
                        let cross: i64 = adds
                            .iter()
                            .map(|&added| challenge.interaction_values[added][item] as i64)
                            .sum();
                        let cost = (state.contribution[item] + cross) * 1024
                            / challenge.weights[item] as i64;
                        (item, cost)
                    })
                    .collect();
                ranked_removals.sort_unstable_by_key(|&(_, cost)| cost);
                let mut removes = Vec::new();
                let mut removed_weight = 0u32;
                for (item, _) in ranked_removals {
                    removes.push(item);
                    removed_weight += challenge.weights[item];
                    if removed_weight >= required {
                        break;
                    }
                }
                if removed_weight < required {
                    continue;
                }
                let delta = block_delta(&state, &adds, &removes);
                if delta > best_delta {
                    best_delta = delta;
                    best_adds = adds.clone();
                    best_removes = removes;
                }
            }
        }

        if best_delta <= 0 {
            break;
        }
        for &item in best_removes.iter().rev() {
            state.remove(item);
        }
        for &item in &best_adds {
            state.add(item);
        }
    }
    while state.best_add_or_swap() {}
    state.solution()
}

fn exact_core_polish(challenge: &Challenge, solution: &Solution) -> Solution {
    let mut state = State::from_solution(challenge, solution);
    let n = challenge.num_items;
    let (rounds, side) = if n <= 1200 {
        (12usize, 10usize)
    } else {
        (8usize, 10usize)
    };
    let global_interaction: Vec<i64> = (0..n)
        .map(|item| {
            challenge.interaction_values[item]
                .iter()
                .map(|&v| v as i64)
                .sum()
        })
        .collect();

    for round in 0..rounds {
        while state.best_add_or_swap() {}

        let mut selected: Vec<usize> = (0..n).filter(|&item| state.selected[item]).collect();
        selected.sort_unstable_by_key(|&item| {
            let weight = challenge.weights[item] as i64;
            if round % 2 == 0 {
                state.contribution[item] * 1024 / weight
            } else {
                state.contribution[item]
            }
        });
        let mut core: Vec<usize> = selected.into_iter().take(side).collect();

        let mut unselected: Vec<usize> = (0..n).filter(|&item| !state.selected[item]).collect();
        if round < 2 {
            unselected.sort_unstable_by_key(|&item| {
                let weight = challenge.weights[item] as i64;
                let score = if round == 0 {
                    state.contribution[item] * 1024 / weight
                } else {
                    state.contribution[item]
                };
                std::cmp::Reverse(score)
            });
            core.extend(unselected.into_iter().take(side));
        } else {
            let mut anchors = unselected.clone();
            anchors.sort_unstable_by_key(|&item| {
                std::cmp::Reverse(global_interaction[item] * 1024 / challenge.weights[item] as i64)
            });
            let anchor = anchors[(round - 2) % anchors.len().min(64)];
            unselected.sort_unstable_by_key(|&item| {
                let cluster_score = state.contribution[item]
                    + 6 * challenge.interaction_values[anchor][item] as i64;
                std::cmp::Reverse(cluster_score * 1024 / challenge.weights[item] as i64)
            });
            core.push(anchor);
            core.extend(
                unselected
                    .into_iter()
                    .filter(|&item| item != anchor)
                    .take(side - 1),
            );
        }
        core.sort_unstable();
        core.dedup();

        if exact_core_move(&mut state, &core) {
            while state.best_add_or_swap() {}
        }
    }
    state.solution()
}

fn objective(challenge: &Challenge, solution: &Solution) -> i64 {
    let mut value = 0i64;
    for (position, &first) in solution.items.iter().enumerate() {
        value += challenge.values[first] as i64;
        for &second in &solution.items[(position + 1)..] {
            value += challenge.interaction_values[first][second] as i64;
        }
    }
    value
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if challenge.num_items != 1000 && challenge.num_items != 5000 {
        return Err(anyhow!("budget25_hybrid supports only 1000 or 5000 items"));
    }

    let candidates = RefCell::new(Vec::<Solution>::new());
    let capture = |solution: &Solution| -> Result<()> {
        candidates.borrow_mut().push(solution.clone());
        save(solution)
    };

    match challenge.num_items {
        1000 => {
            superfast_t1000::solve(challenge, &capture, hyperparameters)?;
            v11_t40::solve(challenge, &capture, hyperparameters)?;
        }
        5000 => {
            superfast_t5000::solve(challenge, &capture, hyperparameters)?;
            v11_t43::solve(challenge, &capture, hyperparameters)?;
        }
        _ => unreachable!(),
    }

    let captured = candidates.into_inner();
    let mut best = captured
        .iter()
        .max_by_key(|solution| objective(challenge, solution))
        .cloned()
        .ok_or_else(|| anyhow!("reference solvers returned no solution"))?;

    // Polish both portfolio members independently. Their different seeds and
    // neighborhoods occasionally expose different edge-aware exchanges.
    for candidate in &captured {
        let refined = polish(challenge, candidate);
        if objective(challenge, &refined) > objective(challenge, &best) {
            best = refined;
        }
    }

    if challenge.num_items <= 1200 {
        let block_refined = block_polish(challenge, &best);
        if objective(challenge, &block_refined) > objective(challenge, &best) {
            best = block_refined;
        }
    }

    let exact_refined = exact_core_polish(challenge, &best);
    if objective(challenge, &exact_refined) > objective(challenge, &best) {
        best = exact_refined;
    }

    save(&best)
}

pub fn help() {
    println!("budget25_hybrid: specialized portfolio for n_items=1000/5000, budget=25");
}
