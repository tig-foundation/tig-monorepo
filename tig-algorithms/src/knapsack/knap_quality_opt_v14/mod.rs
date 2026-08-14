use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::knapsack::*;

mod track_t39;
mod track_t40;
mod track_t41;
mod track_t42;
mod track_t43;

fn merge_hp(user: &Option<Map<String, Value>>, defs: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user.clone().unwrap_or_default();
    for (k, v) in defs { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn n(v: i64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    let n_items = challenge.num_items;
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
    match (n_items, budget_pct) {
        (1000, b) if b <= 7  => { let hp = merge_hp(hp, vec![("core_final_mode", n(2))]); track_t41::solve(challenge, save, &hp) }
        (1000, b) if b <= 17 => { let hp = merge_hp(hp, vec![("n_perturbation_rounds", n(16))]); track_t39::solve(challenge, save, &hp) }
        // PROMU 2026-08-11 : seed HPF/QKBP injecte dans la population de l'ILS (t40_hpf_pop).
        // Mesure : 54 064 @ 6,5 s contre 54 062 @ 37,5 s pour l'ancien defaut (n_full_restarts=60).
        // Meme Q a k=8 et a k=60 (courbe plate) => les 52 restarts supplementaires ne payaient plus.
        // Detail : challenges/c003_knapsack/knowledge/03_indev_analysis/t40_i4_seed_dans_la_population_GAIN.md
        (1000, _)            => { let hp = merge_hp(hp, vec![("n_full_restarts", n(8)), ("t40_hpf_pop", n(1))]); track_t40::solve(challenge, save, &hp) }
        // PROMU 2026-08-14 (revise) : ils_rounds 110 -> 40 sur t42.
        // Choisi sur Σ PAR NONCE, pas sur la mediane : la mediane est identique de 25 a 60 et
        // ne departage pas. Σ cumule sur 5 jeux = -4075 (14 nonces degrades) contre -7292 (25)
        // pour ils=30, et le temps est MEILLEUR (79,52 s contre 80,52, medianes de 4 et 2 reps
        // sur local ; champion 91,03 s). ils=40 domine donc strictement ils=30 sur les deux axes.
        // ils=60 fait mieux en Σ (-1466) mais coute 4,25 s de plus : laisse au front.
        // Detail : knowledge/t42_LE_POINT_LIVRE_EST_DOMINE_choisir_sur_SIGMA.md
        (5000, b) if b <= 17 => { let hp = merge_hp(hp, vec![("lagr_mode", n(3)), ("ils_rounds", n(40))]); track_t42::solve(challenge, save, &hp) }
        (5000, _)            => {
            let hp = merge_hp(hp, vec![
                ("rc_mode", n(5)), ("rc_fraction", f(0.45)), ("refine_mode", n(5)), ("use_flat_iv", n(1)),
                ("base_profile", n(1)), ("use_rflip_screen", n(0)), ("use_screen_break", n(1)), ("use_light_fast_replace", n(1)),
            ]);
            track_t43::solve(challenge, save, &hp)
        }
        _ => Err(anyhow::anyhow!("unknown track config (n_items={}, budget_pct={})", n_items, budget_pct)),
    }
}
pub fn help() { println!("knap_quality_opt_v13 - per-track knapsack solver"); }
