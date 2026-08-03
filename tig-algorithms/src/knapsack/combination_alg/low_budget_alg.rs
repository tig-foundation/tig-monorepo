use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::cmp::Reverse;
use tig_challenges::knapsack::{Challenge, Solution};

/// Mutable exact objective state. `contribution[i]` is the marginal value of
/// adding `i` to the current selection (or its current removal contribution
/// when selected).
#[derive(Clone)]
struct State<'a> {
    challenge: &'a Challenge,
    selected: Vec<bool>,
    contribution: Vec<i64>,
    value: i64,
    weight: u32,
}

impl<'a> State<'a> {
    fn from_solution(challenge: &'a Challenge, solution: &Solution) -> Self {
        let mut state = Self {
            challenge,
            selected: vec![false; challenge.num_items],
            contribution: challenge.values.iter().map(|&value| value as i64).collect(),
            value: 0,
            weight: 0,
        };
        for &item in &solution.items {
            if item < challenge.num_items
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

    /// Steepest feasible add or 1-for-1 exchange. The large-neighborhood move
    /// deliberately leaves a rough boundary, so this closes that boundary
    /// exactly before candidates are compared.
    fn best_add_or_swap(&mut self) -> bool {
        let n = self.challenge.num_items;
        let slack = self.slack();
        let mut best_delta = 0i64;
        let mut best_move: Option<(Option<usize>, usize)> = None;

        for add in 0..n {
            if self.selected[add] {
                continue;
            }
            if self.challenge.weights[add] <= slack && self.contribution[add] > best_delta {
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

#[derive(Clone)]
struct BeamNode {
    weight: u32,
    gain: i64,
    items: Vec<usize>,
}

#[derive(Clone, Copy)]
struct TripleMove {
    items: [usize; 3],
    weight: u32,
    score: i64,
}

fn retain_triple(
    bucket: &mut Vec<TripleMove>,
    candidate: TripleMove,
    limit: usize,
    keep_largest: bool,
) {
    bucket.push(candidate);
    if bucket.len() <= limit {
        return;
    }
    let mut worst = 0usize;
    for index in 1..bucket.len() {
        let candidate_is_worse = if keep_largest {
            bucket[index].score < bucket[worst].score
        } else {
            bucket[index].score > bucket[worst].score
        };
        if candidate_is_worse {
            worst = index;
        }
    }
    bucket.swap_remove(worst);
}

fn insert_weight_bucket(bucket: &mut Vec<BeamNode>, node: BeamNode, limit: usize) {
    bucket.push(node);
    if bucket.len() <= limit {
        return;
    }
    let mut worst = 0usize;
    for index in 1..bucket.len() {
        if bucket[index].gain < bucket[worst].gain {
            worst = index;
        }
    }
    bucket.swap_remove(worst);
}

/// Systematic 3-for-3 exchange.
///
/// Existing reference neighborhoods stop at two coordinated removals/additions
/// or solve a small hand-picked core. Here, promising triples are generated
/// independently on both sides, retained by their exact total weight, and then
/// cross-corrected exactly. Weight bucketing makes the six-item neighborhood
/// practical while still considering triples drawn from a much wider pool.
fn best_triple_exchange(
    state: &mut State,
    selected_limit: usize,
    unselected_limit: usize,
    triples_per_weight: usize,
) -> bool {
    let n = state.challenge.num_items;
    let mut selected_ranked: Vec<(i64, usize)> = (0..n)
        .filter(|&item| state.selected[item])
        .map(|item| {
            (
                state.contribution[item] * 1024 / state.challenge.weights[item] as i64,
                item,
            )
        })
        .collect();
    selected_ranked.sort_unstable_by_key(|&(density, item)| (density, item));
    let selected: Vec<usize> = selected_ranked
        .into_iter()
        .take(selected_limit)
        .map(|(_, item)| item)
        .collect();

    // Include latent within-complement synergy in the incoming ranking. A
    // marginal-only pool cannot see a profitable triple whose members have
    // little value until they enter together.
    let mut unselected_ranked = Vec::new();
    for item in 0..n {
        if state.selected[item] {
            continue;
        }
        let mut top_first = 0i64;
        let mut top_second = 0i64;
        for other in 0..n {
            if other == item || state.selected[other] {
                continue;
            }
            let interaction = state.challenge.interaction_values[item][other] as i64;
            if interaction > top_first {
                top_second = top_first;
                top_first = interaction;
            } else if interaction > top_second {
                top_second = interaction;
            }
        }
        let score = state.contribution[item] + 3 * (top_first + top_second);
        let density = score * 1024 / state.challenge.weights[item] as i64;
        unselected_ranked.push((density, item));
    }
    unselected_ranked.sort_unstable_by_key(|&(density, item)| (Reverse(density), item));
    let unselected: Vec<usize> = unselected_ranked
        .into_iter()
        .take(unselected_limit)
        .map(|(_, item)| item)
        .collect();

    if selected.len() < 3 || unselected.len() < 3 {
        return false;
    }

    let mut remove_buckets = vec![Vec::<TripleMove>::new(); 31];
    for first_pos in 0..selected.len() {
        let first = selected[first_pos];
        for second_pos in (first_pos + 1)..selected.len() {
            let second = selected[second_pos];
            let first_second = state.challenge.interaction_values[first][second] as i64;
            for third_pos in (second_pos + 1)..selected.len() {
                let third = selected[third_pos];
                let weight = state.challenge.weights[first]
                    + state.challenge.weights[second]
                    + state.challenge.weights[third];
                let loss = state.contribution[first]
                    + state.contribution[second]
                    + state.contribution[third]
                    - first_second
                    - state.challenge.interaction_values[first][third] as i64
                    - state.challenge.interaction_values[second][third] as i64;
                retain_triple(
                    &mut remove_buckets[weight as usize],
                    TripleMove {
                        items: [first, second, third],
                        weight,
                        score: loss,
                    },
                    triples_per_weight,
                    false,
                );
            }
        }
    }

    let mut add_buckets = vec![Vec::<TripleMove>::new(); 31];
    for first_pos in 0..unselected.len() {
        let first = unselected[first_pos];
        for second_pos in (first_pos + 1)..unselected.len() {
            let second = unselected[second_pos];
            let first_second = state.challenge.interaction_values[first][second] as i64;
            for third_pos in (second_pos + 1)..unselected.len() {
                let third = unselected[third_pos];
                let weight = state.challenge.weights[first]
                    + state.challenge.weights[second]
                    + state.challenge.weights[third];
                let gain = state.contribution[first]
                    + state.contribution[second]
                    + state.contribution[third]
                    + first_second
                    + state.challenge.interaction_values[first][third] as i64
                    + state.challenge.interaction_values[second][third] as i64;
                retain_triple(
                    &mut add_buckets[weight as usize],
                    TripleMove {
                        items: [first, second, third],
                        weight,
                        score: gain,
                    },
                    triples_per_weight,
                    true,
                );
            }
        }
    }

    let slack = state.slack();
    let mut best_delta = 0i64;
    let mut best_move: Option<(TripleMove, TripleMove)> = None;
    for remove_weight in 3..=30usize {
        for removed in &remove_buckets[remove_weight] {
            let capacity = slack + removed.weight;
            for add_weight in 3..=30usize {
                if add_weight as u32 > capacity {
                    continue;
                }
                for added in &add_buckets[add_weight] {
                    let mut cross = 0i64;
                    for &incoming in &added.items {
                        for &outgoing in &removed.items {
                            cross += state.challenge.interaction_values[incoming][outgoing] as i64;
                        }
                    }
                    let delta = added.score - removed.score - cross;
                    if delta > best_delta {
                        best_delta = delta;
                        best_move = Some((*removed, *added));
                    }
                }
            }
        }
    }

    let Some((removed, added)) = best_move else {
        return false;
    };
    for &item in removed.items.iter().rev() {
        state.remove(item);
    }
    for &item in &added.items {
        state.add(item);
    }
    true
}

/// Finds nodes that are weakly represented by the incumbent but have a strong
/// latent team among other unselected nodes. This differs from marginal-gain
/// ranking: a whole community can be invisible to a one-item neighborhood.
fn latent_anchors(state: &State, limit: usize) -> Vec<usize> {
    let n = state.challenge.num_items;
    let mut ranked = Vec::new();
    for item in 0..n {
        if state.selected[item] {
            continue;
        }
        let mut top = [0i64; 4];
        for other in 0..n {
            if other == item || state.selected[other] {
                continue;
            }
            let interaction = state.challenge.interaction_values[item][other] as i64;
            if interaction <= top[3] {
                continue;
            }
            top[3] = interaction;
            top.sort_unstable_by(|left, right| right.cmp(left));
        }
        let latent: i64 = top.iter().sum();
        let score =
            (state.contribution[item] + 5 * latent) * 1024 / state.challenge.weights[item] as i64;
        ranked.push((score, latent, item));
    }
    ranked.sort_unstable_by_key(|&(score, latent, item)| (Reverse(score), Reverse(latent), item));

    // Prefer distinct latent communities. A second highly similar anchor is
    // admitted only after the diversified half of the list has been filled.
    let mut anchors = Vec::new();
    for &(_, _, item) in &ranked {
        let redundant = anchors
            .iter()
            .any(|&anchor| state.challenge.interaction_values[item][anchor] >= 250);
        if !redundant || anchors.len() >= limit / 2 {
            anchors.push(item);
            if anchors.len() == limit {
                break;
            }
        }
    }
    anchors
}

fn guided_anchors(state: &State, guide: Option<&Solution>, limit: usize) -> Vec<usize> {
    let mut anchors = Vec::new();
    if let Some(guide) = guide {
        let mut disagreement: Vec<(i64, usize)> = guide
            .items
            .iter()
            .copied()
            .filter(|&item| item < state.challenge.num_items && !state.selected[item])
            .map(|item| {
                let mut latent = 0i64;
                for &other in &guide.items {
                    if other != item && other < state.challenge.num_items {
                        latent += state.challenge.interaction_values[item][other] as i64;
                    }
                }
                let score = (state.contribution[item] + 3 * latent) * 1024
                    / state.challenge.weights[item] as i64;
                (score, item)
            })
            .collect();
        disagreement.sort_unstable_by_key(|&(score, item)| (Reverse(score), item));
        anchors.extend(
            disagreement
                .into_iter()
                .take(limit / 2)
                .map(|(_, item)| item),
        );
    }
    for anchor in latent_anchors(state, limit) {
        if !anchors.contains(&anchor) {
            anchors.push(anchor);
            if anchors.len() == limit {
                break;
            }
        }
    }
    anchors
}

fn weakest_selected(state: &State, count: usize, anchor: usize) -> Vec<usize> {
    let mut selected: Vec<(i64, usize)> = (0..state.challenge.num_items)
        .filter(|&item| state.selected[item])
        .map(|item| {
            // Keeping an incumbent that already connects to the entering
            // anchor is valuable. Penalize its removal accordingly.
            let keep_value = state.contribution[item]
                + 2 * state.challenge.interaction_values[item][anchor] as i64;
            let density = keep_value * 1024 / state.challenge.weights[item] as i64;
            (density, item)
        })
        .collect();
    selected.sort_unstable_by_key(|&(density, item)| (density, item));
    selected
        .into_iter()
        .take(count)
        .map(|(_, item)| item)
        .collect()
}

fn candidate_pool(
    survivor: &State,
    incumbent: &State,
    anchor: usize,
    removed: &[usize],
    pool_limit: usize,
) -> Vec<usize> {
    let n = survivor.challenge.num_items;
    let mut ranked = Vec::new();
    for item in 0..n {
        if survivor.selected[item] || item == anchor {
            continue;
        }
        let anchor_edge = survivor.challenge.interaction_values[anchor][item] as i64;

        // Reward both compatibility with the forced anchor and value against
        // the surviving incumbent. The extra strongest edge encourages a
        // second latent subcluster rather than a pure anchor star.
        let mut strongest = 0i64;
        for other in 0..n {
            if other != item && !incumbent.selected[other] {
                strongest =
                    strongest.max(survivor.challenge.interaction_values[item][other] as i64);
            }
        }
        let score = (survivor.contribution[item] + 7 * anchor_edge + 2 * strongest) * 1024
            / survivor.challenge.weights[item] as i64;
        ranked.push((score, anchor_edge, item));
    }
    ranked.sort_unstable_by_key(|&(score, edge, item)| (Reverse(score), Reverse(edge), item));

    let mut in_pool = vec![false; n];
    in_pool[anchor] = true;
    let mut pool = Vec::new();
    for &item in removed {
        if !in_pool[item] {
            pool.push(item);
            in_pool[item] = true;
        }
    }
    for &(_, _, item) in &ranked {
        if !in_pool[item] {
            pool.push(item);
            in_pool[item] = true;
            if pool.len() >= pool_limit {
                break;
            }
        }
    }

    // Process high-potential items early so that their interaction signatures
    // survive the bounded per-weight beam.
    pool.sort_unstable_by_key(|&item| {
        let edge = survivor.challenge.interaction_values[anchor][item] as i64;
        Reverse(
            (survivor.contribution[item] + 7 * edge) * 1024
                / survivor.challenge.weights[item] as i64,
        )
    });
    pool
}

/// Anchor-conditioned beam large-neighborhood search.
///
/// A latent anchor is forced into the team, several weak incumbents are
/// ejected, and a bounded beam solves the resulting quadratic refill problem.
/// Keeping several states for every exact weight is important: two partial
/// teams with equal current score can have very different future synergies.
fn anchored_beam_move<'a>(
    incumbent: &State<'a>,
    anchor: usize,
    remove_count: usize,
    pool_limit: usize,
    states_per_weight: usize,
) -> State<'a> {
    let removed = weakest_selected(incumbent, remove_count, anchor);
    let mut survivor = incumbent.clone();
    for &item in removed.iter().rev() {
        survivor.remove(item);
    }
    let capacity = survivor.challenge.max_weight - survivor.weight;
    let anchor_weight = survivor.challenge.weights[anchor];
    if anchor_weight > capacity {
        return incumbent.clone();
    }

    let pool = candidate_pool(&survivor, incumbent, anchor, &removed, pool_limit);
    let mut beam = vec![BeamNode {
        weight: anchor_weight,
        gain: survivor.contribution[anchor],
        items: vec![anchor],
    }];

    for item in pool {
        let item_weight = survivor.challenge.weights[item];
        let previous = beam.clone();
        let mut buckets = vec![Vec::<BeamNode>::new(); capacity as usize + 1];
        for node in previous.iter().cloned() {
            insert_weight_bucket(&mut buckets[node.weight as usize], node, states_per_weight);
        }
        for node in previous {
            let next_weight = node.weight + item_weight;
            if next_weight > capacity {
                continue;
            }
            let mutual: i64 = node
                .items
                .iter()
                .map(|&chosen| survivor.challenge.interaction_values[item][chosen] as i64)
                .sum();
            let mut items = node.items;
            items.push(item);
            insert_weight_bucket(
                &mut buckets[next_weight as usize],
                BeamNode {
                    weight: next_weight,
                    gain: node.gain + survivor.contribution[item] + mutual,
                    items,
                },
                states_per_weight,
            );
        }
        beam = buckets.into_iter().flatten().collect();
    }

    let Some(best_refill) = beam.into_iter().max_by_key(|node| node.gain) else {
        return incumbent.clone();
    };
    for item in best_refill.items {
        survivor.add(item);
    }

    let ascent_limit = if survivor.challenge.num_items <= 1200 {
        16
    } else {
        8
    };
    for _ in 0..ascent_limit {
        if !survivor.best_add_or_swap() {
            break;
        }
    }
    survivor
}

fn low_budget_search(
    challenge: &Challenge,
    solution: &Solution,
    guide: Option<&Solution>,
) -> Solution {
    let mut best = State::from_solution(challenge, solution);
    let (passes, anchor_limit, removal_sizes, pool_limit, states_per_weight) =
        if challenge.num_items <= 1200 {
            (2usize, 24usize, [6usize, 10usize], 38usize, 10usize)
        } else {
            (2usize, 14usize, [10usize, 18usize], 44usize, 8usize)
        };

    for _ in 0..passes {
        let anchors = guided_anchors(&best, guide, anchor_limit);
        let mut pass_best = best.clone();
        for anchor in anchors {
            for remove_count in removal_sizes {
                let candidate =
                    anchored_beam_move(&best, anchor, remove_count, pool_limit, states_per_weight);
                if candidate.value > pass_best.value {
                    pass_best = candidate;
                }
            }
        }
        if pass_best.value <= best.value {
            break;
        }
        best = pass_best;
    }

    let (triple_rounds, selected_limit, unselected_limit, triples_per_weight) =
        if challenge.num_items <= 1200 {
            (6usize, 52usize, 76usize, 64usize)
        } else {
            (4usize, 64usize, 88usize, 48usize)
        };
    for _ in 0..triple_rounds {
        if !best_triple_exchange(
            &mut best,
            selected_limit,
            unselected_limit,
            triples_per_weight,
        ) {
            break;
        }
        while best.best_add_or_swap() {}
    }
    best.solution()
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

fn tuned_v11_hyperparameters(
    num_items: usize,
    observed_budget_pct: u64,
) -> Option<Map<String, Value>> {
    if observed_budget_pct <= 7 {
        let Value::Object(parameters) = serde_json::json!({
            "n_crossover_gen": 13,
            "n_random_starts": 3
        }) else {
            unreachable!()
        };
        return Some(parameters);
    }
    if num_items <= 1200 {
        return Some(Map::new());
    }
    let Value::Object(parameters) = serde_json::json!({
        "bounded_2_2_k": 20,
        "ils_restart_interval": 15,
        "ils_rounds": 115,
        "ils_vnd_level": 3,
        "n_crossover_gen": 16,
        "n_random_starts": 2,
        "n_sa_members": 0,
        "perturb_base_frac": 7,
        "perturb_max_frac": 7,
        "sa_iter": 0,
        "sa_rounds": 1
    }) else {
        unreachable!()
    };
    Some(parameters)
}

fn effective_v11_hyperparameters(
    num_items: usize,
    observed_budget_pct: u64,
    supplied: &Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let mut parameters =
        tuned_v11_hyperparameters(num_items, observed_budget_pct).unwrap_or_default();
    if let Some(supplied) = supplied {
        for (name, value) in supplied {
            let is_default = value.is_null()
                || value
                    .as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case("default"));
            if name != "track" && !is_default {
                parameters.insert(name.clone(), value.clone());
            }
        }
    }
    Some(parameters)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if challenge.num_items != 1000 && challenge.num_items != 5000 {
        return Err(anyhow!("low_budget_alg supports only 1000 or 5000 items"));
    }
    let sum_weight: u64 = challenge.weights.iter().map(|&weight| weight as u64).sum();
    let budget_pct = if sum_weight == 0 {
        0
    } else {
        challenge.max_weight as u64 * 100 / sum_weight
    };
    if !(3..=17).contains(&budget_pct) || (challenge.num_items == 5000 && budget_pct <= 7) {
        return Err(anyhow!(
            "low_budget_alg supports the 1000-item budget=5 track and the budget=10 tracks (observed n_items={}, budget_pct={})",
            challenge.num_items,
            budget_pct
        ));
    }

    // The existing portfolio supplies strong deterministic incumbents. The
    // selected track preset, including any concrete overrides, is shared by
    // every child; each child ignores keys it does not support.
    let captured = RefCell::new(Vec::<Solution>::new());
    let capture = |solution: &Solution| -> Result<()> {
        captured.borrow_mut().push(solution.clone());
        save(solution)
    };
    super::budget25_hybrid::solve_challenge(challenge, &capture, hyperparameters)?;
    let v11_hyperparameters =
        effective_v11_hyperparameters(challenge.num_items, budget_pct, hyperparameters);
    if budget_pct <= 7 {
        super::v11_t41::solve(challenge, &capture, &v11_hyperparameters)?;
    } else if challenge.num_items <= 1200 {
        super::v11_t39::solve(challenge, &capture, &v11_hyperparameters)?;
    } else {
        super::v11_t42::solve(challenge, &capture, &v11_hyperparameters)?;
    }
    let seeds = captured.into_inner();
    if seeds.is_empty() {
        return Err(anyhow!("incumbent solvers returned no solution"));
    }
    let mut best = seeds
        .iter()
        .max_by_key(|solution| objective(challenge, solution))
        .cloned()
        .unwrap();
    for (index, seed) in seeds.iter().enumerate() {
        let guide = seeds
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != index)
            .max_by_key(|(_, solution)| objective(challenge, solution))
            .map(|(_, solution)| solution);
        let improved = low_budget_search(challenge, seed, guide);
        if objective(challenge, &improved) > objective(challenge, &best) {
            best = improved;
        }
    }
    save(&best)
}

pub fn help() {
    println!(
        "low_budget_alg: budget-5/10 solver with latent-community anchored beam neighborhoods"
    );
}
