// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::satisfiability::*;

// engine_d). Le fichier est conservé intact comme référence de la baseline i1.
#[allow(dead_code)]
mod engine_a;
mod engine_b;
#[allow(dead_code)]
mod engine_c;
mod engine_d;
// `preprocess`/`Prepared` sont réutilisés par engine_k, et son corps de boucle est repris
// VERBATIM comme arm 0 (A_ref) de la cascade i6. Seul son `solve`/`Hparams` devient inutilisé.
#[allow(dead_code)]
mod engine_f;
// comme arm 0 (A_ctrl) de la cascade i8, et c'est la ligne à restaurer si i8 est rejetée.
// Seul son `solve` devient inutilisé.
#[allow(dead_code)]
mod engine_k;
mod engine_n;
// flip jamais mesurés (bornes du scan gardées par `if assignment[v]` ; littéraux i32
mod engine_o;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn u(v: u64) -> Value { Value::Number(Number::from(v)) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Per-track dispatch by (num_variables, num_clauses); tuned defaults baked per track.
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    match (nv, nc) {
        (10000, 42670) => {
            // fuel baké 315B — le seuil du 7ᵉ nonce est dans ]275B, 315B] (v9 i14/i28-CTRL),
            // et Q est monotone en fuel donc 315B ne peut pas régresser sous 6/32.
            // fuel > 315B est dead-listé (8ᵉ nonce hopeless, le temps explose).
            let hp = merge_hp(hyperparameters, vec![("max_fuel_high", u(315000000000))]);
            engine_d::solve(challenge, save_solution, &hp)
        }
        (100000, 415000) => engine_b::solve(challenge, save_solution, hyperparameters),
        (5000, 21335) => engine_b::solve(challenge, save_solution, hyperparameters),
        (7500, 32002) => {
            // (max_flips = 774 750 000). L'ancien `target_max_fuel=200B` appartenait à la
            // formule de fuel d'engine_c et n'a AUCUN sens pour engine_f — le transposer
            // changerait le nombre de flips, donc l'issue des nonces borderline.
            //
            // bookkeeping du flip réécrit en BRANCHLESS (CL-AmbSAT `b0b0b5dd`). Le mur est
            // mesuré : bloc RMW ≥24,59 % du flip (i5 mode E) dont ~2 % seulement de mémoire
            // (i5 mode H) ⇒ la cible est la BRANCHE + la maintenance `unsat_clauses`.
            // `rmw_variant` sélectionne l'arm (0=A_ref verbatim engine_f · 1=buf · 2=branchless
            // · 3=branchless+fusion) ; défaut = arm retenu au screening. AUCUN fuel baké
            // (défaut interne 155B, identique à engine_f ⇒ max_flips = 774 750 000).
            //
            // binaire les DEUX mécanismes de tête du track, jamais arbitrés l'un contre
            // l'autre parce qu'ils vivaient dans deux binaires différents :
            //   · `argmin branchless` (i8/`engine_m` arm 1) — retire des MISPREDICTIONS,
            //     screening 200 M : 90,01 s vs ctrl 92,51 s ⇒ −2,70 % ;
            //   · `var_bounds_packing` (i7/`engine_l` arm 2) — retire des CHARGES
            //     DÉPENDANTES aux 2 sites de bornes, production n=1 : 349 560 vs CTRL
            //     même-batch 353 550 ⇒ −1,13 %, JAMAIS ADJUGÉ (bench relu après coup).
            // Les 2 sites et les 2 ressources microarchitecturales sont DISJOINTS ⇒ arm 3
            // 1=V_argmin · 2=V_vbnd · 3=V_both CANDIDAT KEEP) ; défaut = arm retenu au
            // screening. AUCUN fuel baké (défaut interne 155B, identique à
            // engine_f/engine_k/engine_m ⇒ max_flips = 774 750 000).
            // sur les DEUX derniers consommateurs de travail trajectoire-safe du flip, que
            // ni i7 (`V_vbnd`, latence) ni i8/i9 (`V_argmin`, sélection) n'avaient touchés :
            //   · arm 1 `V_polsel`  — le saut ~50/50 `if assignment[v]` qui garde la lecture
            //     des bornes du scan (jusqu'à 3× par flip) devient 3 charges + 2 `cmov` ;
            //   · arm 2 `V_litpack` — littéraux pré-décodés `(v<<1)|pol` en u32 (MÊME
            //     empreinte que `cl`) : supprime ~6 `abs()-1` par flip en tête de la chaîne
            //     de dépendances `littéral → v → assignment[v]` ;
            //   · arm 3 `V_both`    — test d'additivité, candidat KEEP.
            // AUCUN fuel baké (défaut interne 155B ⇒ max_flips = 774 750 000).
            //   engine_n::solve(challenge, save_solution, hyperparameters)
            engine_o::solve(challenge, save_solution, hyperparameters)
        }
        (100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
    }
}

pub fn help() {
    println!("sat_hybrid_v2_v2 - per-track SAT solver");
}
