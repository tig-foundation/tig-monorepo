use rand::rngs::SmallRng;
use rand::Rng;
use super::types::Rule;

#[inline]
pub fn rule_idx(r: Rule) -> usize {
    match r {
        Rule::Adaptive => 0,
        Rule::BnHeavy => 1,
        Rule::EndTight => 2,
        Rule::CriticalPath => 3,
        Rule::MostWork => 4,
        Rule::LeastFlex => 5,
        Rule::Regret => 6,
        Rule::ShortestProc => 7,
        Rule::FlexBalance => 8,
    }
}

#[inline]
pub fn sample_roulette(rng: &mut SmallRng, weights: &[f64]) -> usize {
    let mut sum = 0.0;
    for &w in weights {
        sum += w.max(0.0);
    }
    if !(sum > 0.0) {
        return rng.gen_range(0..weights.len());
    }
    let mut r = rng.gen::<f64>() * sum;
    for (i, &w) in weights.iter().enumerate() {
        r -= w.max(0.0);
        if r <= 0.0 {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

pub fn choose_rule_bandit(
    rng: &mut SmallRng,
    rules: &[Rule],
    rule_best: &[u32],
    rule_tries: &[u32],
    global_best: u32,
    margin: u32,
    stuck: usize,
    chaos_like: bool,
    late_phase: bool,
) -> Rule {
    if rules.is_empty() {
        return Rule::Adaptive;
    }

    let mut best_seen = global_best;
    for &mk in rule_best {
        if mk < best_seen {
            best_seen = mk;
        }
    }

    let scale = (margin as f64).max(1.0);
    let s = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
    let explore_mix = (0.10 + 0.55 * s).clamp(0.10, 0.65);

    let mut w = vec![0.0f64; rules.len()];
    for (i, &r) in rules.iter().enumerate() {
        let mk = rule_best[rule_idx(r)];
        let t = rule_tries[rule_idx(r)].max(1) as f64;

        let delta = mk.saturating_sub(best_seen) as f64;
        let exploit = (-delta / scale).exp();

        let explore = (1.0 / t).sqrt();

        let mut ww = (1.0 - explore_mix) * exploit + explore_mix * explore;
        ww = ww.max(1e-6);

        if chaos_like {
            ww = ww.powf(0.70);
        } else if late_phase {
            ww = ww.powf(1.18);
        }

        w[i] = ww;
    }

    let idx = sample_roulette(rng, &w);
    rules[idx]
}
