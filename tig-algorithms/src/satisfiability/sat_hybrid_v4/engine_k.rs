// SAT solver engine K — T5 (nv=7500, nc=32002, α=4.267 = seuil SAT-UNSAT exact)
// Directive `directives/2026-07-28T042652Z_t5.md`.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// CE QUE LA MESURE A ÉTABLI (sondes i4/i5, à travail fixe, ZÉRO confond)
// ══════════════════════════════════════════════════════════════════════════════════════
//   i4 (SCAN)  : retirer la mémoire du break-count RALENTIT (B_l1 +5,26 %, C_nomem +3,51 %)
//                ⇒ le scan n'est le mur dans AUCUNE direction (dead 169-172).
//   i5 (RMW)   : E_rmw_strip (ablation)   = **−24,59 %** ⇒ le bloc RMW vaut ≥24,59 % du flip.
//                H_rmw_dup  (duplication, trajectoire byte-identique) = **+1,72 %**
//                ⇒ **sa MÉMOIRE ne vaut que ~2 %** (dead 173 `rmw_counter_memory`).
//   ⇒ Par élimination, les ~130 ns/flip restants du RMW sont la **BRANCHE data-dépendante
//     `trues == 0` / `trues == 1`** + le **push/pop de `unsat_clauses`** + le **scatter
//     `clause_pos_in_unsat`** (~35 cycles/occurrence = signature de MISPRÉDICTION).
//   ⛔ La famille LAYOUT est close PARTOUT sur t5 : aucun coup mémoire n'est licite ici.
//
// ── HYPOTHÈSE (Hindsight) ─────────────────────────────────────────────────────────────
// `b0b0b5dd` / `d2141025` / `67b03a44` (CL-AmbSAT, Takeuchi-Aono-Hara-Azumi-Ayala 2018) :
//   « CL-AmbSAT uses **simple combinational logic for variable updates**, unlike AmbSAT
//     which uses **complex conditional branches** » ; `4cc376f5` : les branches
//   conditionnelles complexes de l'AmbSAT gâchent le parallélisme physique.
// `253afd81` / `072c3cec` : « the flip operation **only updates clauses' sets and true
//   literal numbers**, leaving break calculation to pickVar » ⇒ le bookkeeping du flip est de
//   l'**ÉTAT PUR** : toute réécriture menant au MÊME état final est **byte-identique PAR
//   CONSTRUCTION** (RNG intouché, même nombre de flips, même trajectoire) — c'est la SEULE
//   famille légitime sur un track dont la trajectoire est intouchable (`d843a3ff`).
// `bd53cbdf` / `4713f991` : notre propre lignée a déjà validé le branchless break-count
//   dead-listé 171 — c'est le PRINCIPE branchless qui est armé, pas le court-circuit 0-break.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// LA TRANSFORMATION — ÉTAT-ÉQUIVALENTE, PROUVÉE LIGNE À LIGNE
// ══════════════════════════════════════════════════════════════════════════════════════
// 1) **Conteneur `unsat_clauses` : `Vec` push/pop → tampon plat 1-BASED + `ulen` scalaire.**
//    L'indice **0 est un EMPLACEMENT POUBELLE** (les vrais éléments vivent en `1..=ulen`) et
//    `cpos` reçoit une **entrée poubelle en `nc`**. Ces deux poubelles sont ce qui rend le
//    branchless possible : quand la condition est FAUSSE, on écrit quand même — dans la
//    poubelle. Sans elles, une écriture inconditionnelle corromprait la liste.
//    Le tampon est PRÉ-INITIALISÉ (`vec![0; nc+2]`) ⇒ toute lecture d'indice ≤ nc est définie
//    (pas d'UB), y compris la lecture spéculative de `ubuf[ulen]` quand `ulen == 0`.
// 2) **Suppression des branches `if trues == 0` / `if trues == 1`** → masque plein
//    (`0usize.wrapping_sub(cond)`) sur l'INDICE cible + écriture INCONDITIONNELLE. C'est la
//    « combinational logic » de CL-AmbSAT transposée en scalaire x86 (`sete`+`neg`+`and`,
//    aucun saut conditionnel ⇒ aucune misprédiction possible).
// 3) **Fusion des 4 boucles RMW en 1** (arm 3) : `p_end == n_start` par construction
//    (`p_bound[v] = all_off[v] + p_cnt[v]`), donc `p_start..n_end` visite les occurrences
//    dans EXACTEMENT le même ordre que « boucle p puis boucle n ». ⚠️ L'ORDRE DE VISITE EST
//    SACRÉ : il pilote l'ordre des push/pop, donc le contenu de `unsat_clauses`, donc le
//    clause tirée par `gen_range` ⇒ la trajectoire. Toute fusion qui inverserait p et n
//    (p.ex. « boucle des +1 » puis « boucle des −1 ») CASSERAIT Q. Celle-ci ne le fait pas.
//
// ── POURQUOI L'ÉTAT RESTE IDENTIQUE (les 3 points subtils) ────────────────────────────
//  (a) `clause_pos_in_unsat[c] = usize::MAX` lors d'un retrait est **DÉCORATIF** : cette table
//      n'est LUE que dans le chemin de retrait, c.-à-d. quand `c` EST dans la liste (donc
//      après un push qui a réécrit sa position). Une valeur périmée n'est jamais lue. Vérifié
//      par inspection exhaustive des accès (`engine_f` L198/224/261/273/377-381/391/406-410/420).
//  (b) Le cas `last == c_idx` (on retire le DERNIER élément) : `engine_f` saute les écritures
//      via `if last_c != c_idx`. Ici on écrit `ubuf[pos] = last` avec `pos == ulen` ⇒ écriture
//      idempotente au-delà du nouveau sommet, et `cpos[c] = pos` ⇒ périmé-non-lu, cf (a).
//  (c) Les occurrences positives et négatives d'une variable portent sur des clauses
//      DISJOINTES (`preprocess` L120 rejette toute clause contenant `x` et `¬x`), mais on ne
//      s'appuie PAS là-dessus : l'ordre de visite est préservé de toute façon (point 3).
//
// ── LES 4 ARMS DE LA CASCADE R7 (`rmw_variant`, const generic ⇒ ZÉRO surcoût dispatch) ──
//                         ANCRE : doit re-rendre ~28,5 s en mode sonde (= i4 et i5).
//   1 = **V1_buf**      : tampon plat 1-based + `ulen` — **branches CONSERVÉES**.
//                         Isole la famille « conteneur » (coût `Vec` len/cap/`pop().unwrap`).
//   2 = **V2_branchless**: V1 + **suppression des 2 branches** (masque + poubelles). Famille
//                         « branchless » = le cœur CL-AmbSAT, la cible de la directive.
//   3 = **V3_fused**    : V2 + **fusion des 4 boucles en 1** (`p_start..n_end`, sens ±1 calculé
//                         par `(k < p_end) == new_val`). Famille « loop fusion / I-cache ».
//   Les 4 arms sont **state-équivalents** ⇒ Q=93750 byte-identique attendue pour TOUS.
//   La cascade attribue le gain à CHAQUE mécanisme (conteneur / branchless / fusion) au lieu
//   de mesurer un paquet de 3 changements.
//
// ── PROTOCOLE DE SCREENING (`probe_max_flips` renseigné) ──────────────────────────────
// Même harnais qu'`engine_i` (i4/i5) : **flips-count FIXE** + **early-SAT-exit OFF** + pas de
// memcpy `best_assignment` dans la boucle ⇒ les 4 arms exécutent EXACTEMENT le même travail,
// Δt = pur ns/flip. 32 nonces / 32 workers (⛔ jamais de bench < 32 nonces sur CPU).
// Le survivant passe SEUL le bench de production (early-exit ON, fuel réel, Q gatée 93750).
// ⛔ Wall-clock INTERDIT : `probe_max_flips` est un compteur de TRAVAIL, pas un budget temps.
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use super::engine_f::{preprocess, Prepared};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub max_fuel_high: Option<f64>,
    /// 0=A_ref (engine_f verbatim) · 1=V1_buf · 2=V2_branchless · 3=V3_fused.
    pub rmw_variant: Option<u64>,
    /// Renseigné ⇒ MODE SONDE (travail fixe + early-exit OFF). Absent ⇒ PRODUCTION.
    pub probe_max_flips: Option<u64>,
}

/// Arm retenu en PRODUCTION (défaut). Mis à jour après le screening de la cascade.
const DEFAULT_VARIANT: u64 = 2;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hparams = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value::<Hparams>(Value::Object(m.clone())).ok())
        .unwrap_or_default();

    // Fuel VERBATIM engine_f : 155B par défaut ⇒ max_flips = 774 750 000. Toute déviation
    // change le nombre de flips donc l'issue des nonces borderline (dead: trim max_fuel_high).
    let max_fuel = hp.max_fuel_high.unwrap_or(155_000_000_000.0);
    let base_fuel = 50_000_000.0;
    let flip_fuel = 200.0;
    let prod_flips = if max_fuel > base_fuel {
        ((max_fuel - base_fuel) / flip_fuel) as u64
    } else {
        10_000_000
    };

    let variant = hp.rmw_variant.unwrap_or(DEFAULT_VARIANT);
    match hp.probe_max_flips {
        None => match variant {
            0 => run_vec::<false>(challenge, save_solution, prod_flips),
            1 => run_buf::<1, false>(challenge, save_solution, prod_flips),
            3 => run_buf::<3, false>(challenge, save_solution, prod_flips),
            _ => run_buf::<2, false>(challenge, save_solution, prod_flips),
        },
        Some(n) => match variant {
            0 => run_vec::<true>(challenge, save_solution, n),
            1 => run_buf::<1, true>(challenge, save_solution, n),
            3 => run_buf::<3, true>(challenge, save_solution, n),
            _ => run_buf::<2, true>(challenge, save_solution, n),
        },
    }
}

const WALK_P: f64 = 0.52;
const RESTART_PERIOD: u64 = 80_000_000;
const REINIT_MIN_UNSAT: usize = 30;
const PERTURB_K: usize = 100;

// ══════════════════════════════════════════════════════════════════════════════════════
// Seul ajout : le const generic PROBE (compilé away) pour le harnais de mesure.
// ══════════════════════════════════════════════════════════════════════════════════════
fn run_vec<const PROBE: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    max_flips: u64,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density: _density, p_cnt, n_cnt, all_off, p_bound, all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    let mut assignment = vec![false; nv];
    let mut best_assignment = vec![false; nv];
    let mut true_lit_count = vec![0u8; nc];
    let mut unsat_clauses: Vec<usize> = Vec::with_capacity(nc);
    let mut clause_pos_in_unsat = vec![usize::MAX; nc];
    let mut perturb_indices: Vec<usize> = (0..nv).collect();

    for v in 0..nv {
        let pc = p_cnt[v] as f64;
        let nc_v = n_cnt[v] as f64;
        let total = pc + nc_v;
        if total == 0.0 {
            assignment[v] = rng.gen_bool(0.5);
        } else {
            assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
        }
    }

    for c in 0..nc {
        let off = co[c] as usize;
        let end = co[c + 1] as usize;
        let mut trues = 0u8;
        for i in off..end {
            let lit = cl[i];
            let v = (lit.abs() - 1) as usize;
            if assignment[v] == (lit > 0) { trues += 1; }
        }
        true_lit_count[c] = trues;
        if trues == 0 {
            clause_pos_in_unsat[c] = unsat_clauses.len();
            unsat_clauses.push(c);
        }
    }

    let mut best_unsat = unsat_clauses.len();
    best_assignment.copy_from_slice(&assignment);
    let _ = save_solution(&Solution { variables: best_assignment.clone() });

    let mut period_best_unsat = best_unsat;
    let mut var_age = vec![0u8; nv];
    let mut global_flips: u64 = 0;
    let mut solved_marked = false;

    unsafe {
        loop {
            if global_flips >= max_flips { break; }

            if unsat_clauses.is_empty() {
                if !PROBE { break; }
                // SONDE : on consomme le budget de flips pour que tous les arms exécutent le
                // même travail. Le `save_solution` unique fait ressortir le nonce à 1 000 000
                // dans PER_NONCE_QUALITIES = détecteur de contamination.
                if !solved_marked {
                    solved_marked = true;
                    let _ = save_solution(&Solution { variables: assignment.clone() });
                }
                global_flips += 1;
                continue;
            }

            if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
                if period_best_unsat >= REINIT_MIN_UNSAT {
                    assignment.copy_from_slice(&best_assignment);
                    for i in 0..PERTURB_K {
                        let j = rng.gen_range(i..nv);
                        perturb_indices.swap(i, j);
                        let v = *perturb_indices.get_unchecked(i);
                        let cur = *assignment.get_unchecked(v);
                        *assignment.get_unchecked_mut(v) = !cur;
                    }
                    true_lit_count.fill(0);
                    unsat_clauses.clear();
                    clause_pos_in_unsat.fill(usize::MAX);
                    for c in 0..nc {
                        let off = *co.get_unchecked(c) as usize;
                        let end = *co.get_unchecked(c + 1) as usize;
                        let mut trues = 0u8;
                        for i in off..end {
                            let lit = *cl.get_unchecked(i);
                            let v = (lit.abs() - 1) as usize;
                            if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
                        }
                        *true_lit_count.get_unchecked_mut(c) = trues;
                        if trues == 0 {
                            *clause_pos_in_unsat.get_unchecked_mut(c) = unsat_clauses.len();
                            unsat_clauses.push(c);
                        }
                    }
                    var_age.fill(0);
                    let cur = unsat_clauses.len();
                    if cur < best_unsat {
                        best_unsat = cur;
                        if !PROBE {
                            best_assignment.copy_from_slice(&assignment);
                            let _ = save_solution(&Solution { variables: best_assignment.clone() });
                        }
                    }
                }
                period_best_unsat = unsat_clauses.len();
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }

            global_flips += 1;

            let r_idx = rng.gen_range(0..unsat_clauses.len());
            let c = *unsat_clauses.get_unchecked(r_idx);
            let off = *co.get_unchecked(c) as usize;
            let end = *co.get_unchecked(c + 1) as usize;
            let len = end - off;

            if len > 1 {
                let ri = (global_flips as usize) % len;
                cl.swap(off, off + ri);
            }

            let mut picked_v = usize::MAX;
            let mut vars = [0usize; 3];
            let mut breaks = [0u8; 3];
            let mut nvars = 0;

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                *vars.get_unchecked_mut(nvars) = v;
                let val = *assignment.get_unchecked(v);
                let (start, stop) = if val {
                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                } else {
                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                };
                let mut b = 0u8;
                for k in start..stop {
                    b += (*true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1) as u8;
                }
                *breaks.get_unchecked_mut(nvars) = b;
                if b == 0 { picked_v = v; break; }
                nvars += 1;
            }

            if picked_v == usize::MAX {
                if nvars == 0 {
                    if let Some(&lit) = cl.get(off) { picked_v = (lit.abs() - 1) as usize; }
                } else if rng.gen_bool(WALK_P) {
                    picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
                } else {
                    let mut min_b = u8::MAX;
                    let mut best_idx = 0;
                    for i in 0..nvars {
                        let b = *breaks.get_unchecked(i);
                        let vi = *vars.get_unchecked(i);
                        let vb = *vars.get_unchecked(best_idx);
                        if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
                            min_b = b;
                            best_idx = i;
                        }
                    }
                    picked_v = *vars.get_unchecked(best_idx);
                }
            }

            if picked_v == usize::MAX { continue; }

            let new_val = !*assignment.get_unchecked(picked_v);
            *assignment.get_unchecked_mut(picked_v) = new_val;

            let p_start = *all_off.get_unchecked(picked_v) as usize;
            let p_end = *p_bound.get_unchecked(picked_v) as usize;
            if new_val {
                for k in p_start..p_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                }
            } else {
                for k in p_start..p_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            let n_start = *p_bound.get_unchecked(picked_v) as usize;
            let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
            if !new_val {
                for k in n_start..n_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                }
            } else {
                for k in n_start..n_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                let a = *var_age.get_unchecked(v);
                *var_age.get_unchecked_mut(v) = a.saturating_add(1);
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                if !PROBE {
                    best_assignment.copy_from_slice(&assignment);
                    let _ = save_solution(&Solution { variables: best_assignment.clone() });
                    if cur_unsat == 0 { break; }
                }
            }
        }
    }

    if PROBE {
        let _ = save_solution(&Solution { variables: assignment });
    } else {
        let _ = save_solution(&Solution { variables: best_assignment });
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════════════
// ARMS 1-3 — tampon plat 1-based (`ubuf[0]` = POUBELLE, éléments en `1..=ulen`) +
// `cpos[nc]` = POUBELLE. Identique à `run_vec` HORS du bloc RMW et de la représentation
// de la liste ; toutes les séquences RNG et l'ordre de visite sont préservés.
// ══════════════════════════════════════════════════════════════════════════════════════
fn run_buf<const RMW: u8, const PROBE: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    max_flips: u64,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density: _density, p_cnt, n_cnt, all_off, p_bound, all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    let mut assignment = vec![false; nv];
    let mut best_assignment = vec![false; nv];
    let mut true_lit_count = vec![0u8; nc];
    // +2 : indice 0 = poubelle, indices 1..=nc = éléments. Tout est INITIALISÉ ⇒ la lecture
    // spéculative `ubuf[ulen]` (y compris `ulen == 0`) est définie, jamais de l'UB.
    let mut ubuf: Vec<usize> = vec![0usize; nc + 2];
    let mut ulen: usize = 0;
    // +1 : l'entrée `nc` est le puits des écritures annulées par le masque.
    let mut cpos = vec![usize::MAX; nc + 1];
    let mut perturb_indices: Vec<usize> = (0..nv).collect();

    for v in 0..nv {
        let pc = p_cnt[v] as f64;
        let nc_v = n_cnt[v] as f64;
        let total = pc + nc_v;
        if total == 0.0 {
            assignment[v] = rng.gen_bool(0.5);
        } else {
            assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
        }
    }

    for c in 0..nc {
        let off = co[c] as usize;
        let end = co[c + 1] as usize;
        let mut trues = 0u8;
        for i in off..end {
            let lit = cl[i];
            let v = (lit.abs() - 1) as usize;
            if assignment[v] == (lit > 0) { trues += 1; }
        }
        true_lit_count[c] = trues;
        if trues == 0 {
            ulen += 1;
            ubuf[ulen] = c;
            cpos[c] = ulen;
        }
    }

    let mut best_unsat = ulen;
    best_assignment.copy_from_slice(&assignment);
    let _ = save_solution(&Solution { variables: best_assignment.clone() });

    let mut period_best_unsat = best_unsat;
    let mut var_age = vec![0u8; nv];
    let mut global_flips: u64 = 0;
    let mut solved_marked = false;

    unsafe {
        loop {
            if global_flips >= max_flips { break; }

            if ulen == 0 {
                if !PROBE { break; }
                if !solved_marked {
                    solved_marked = true;
                    let _ = save_solution(&Solution { variables: assignment.clone() });
                }
                global_flips += 1;
                continue;
            }

            if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
                if period_best_unsat >= REINIT_MIN_UNSAT {
                    assignment.copy_from_slice(&best_assignment);
                    for i in 0..PERTURB_K {
                        let j = rng.gen_range(i..nv);
                        perturb_indices.swap(i, j);
                        let v = *perturb_indices.get_unchecked(i);
                        let cur = *assignment.get_unchecked(v);
                        *assignment.get_unchecked_mut(v) = !cur;
                    }
                    true_lit_count.fill(0);
                    ulen = 0;
                    cpos.fill(usize::MAX);
                    for c in 0..nc {
                        let off = *co.get_unchecked(c) as usize;
                        let end = *co.get_unchecked(c + 1) as usize;
                        let mut trues = 0u8;
                        for i in off..end {
                            let lit = *cl.get_unchecked(i);
                            let v = (lit.abs() - 1) as usize;
                            if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
                        }
                        *true_lit_count.get_unchecked_mut(c) = trues;
                        if trues == 0 {
                            ulen += 1;
                            *ubuf.get_unchecked_mut(ulen) = c;
                            *cpos.get_unchecked_mut(c) = ulen;
                        }
                    }
                    var_age.fill(0);
                    let cur = ulen;
                    if cur < best_unsat {
                        best_unsat = cur;
                        if !PROBE {
                            best_assignment.copy_from_slice(&assignment);
                            let _ = save_solution(&Solution { variables: best_assignment.clone() });
                        }
                    }
                }
                period_best_unsat = ulen;
            }

            let cur_unsat = ulen;
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }

            global_flips += 1;

            // de la lecture change ⇒ MÊME clause tirée.
            let r_idx = rng.gen_range(0..ulen);
            let c = *ubuf.get_unchecked(r_idx + 1);
            let off = *co.get_unchecked(c) as usize;
            let end = *co.get_unchecked(c + 1) as usize;
            let len = end - off;

            if len > 1 {
                let ri = (global_flips as usize) % len;
                cl.swap(off, off + ri);
            }

            let mut picked_v = usize::MAX;
            let mut vars = [0usize; 3];
            let mut breaks = [0u8; 3];
            let mut nvars = 0;

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                *vars.get_unchecked_mut(nvars) = v;
                let val = *assignment.get_unchecked(v);
                let (start, stop) = if val {
                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                } else {
                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                };
                let mut b = 0u8;
                for k in start..stop {
                    b += (*true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1) as u8;
                }
                *breaks.get_unchecked_mut(nvars) = b;
                if b == 0 { picked_v = v; break; }
                nvars += 1;
            }

            if picked_v == usize::MAX {
                if nvars == 0 {
                    if let Some(&lit) = cl.get(off) { picked_v = (lit.abs() - 1) as usize; }
                } else if rng.gen_bool(WALK_P) {
                    picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
                } else {
                    let mut min_b = u8::MAX;
                    let mut best_idx = 0;
                    for i in 0..nvars {
                        let b = *breaks.get_unchecked(i);
                        let vi = *vars.get_unchecked(i);
                        let vb = *vars.get_unchecked(best_idx);
                        if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
                            min_b = b;
                            best_idx = i;
                        }
                    }
                    picked_v = *vars.get_unchecked(best_idx);
                }
            }

            if picked_v == usize::MAX { continue; }

            let new_val = !*assignment.get_unchecked(picked_v);
            *assignment.get_unchecked_mut(picked_v) = new_val;

            let p_start = *all_off.get_unchecked(picked_v) as usize;
            let p_end = *p_bound.get_unchecked(picked_v) as usize;
            let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
            // `p_bound[v] == all_off[v] + p_cnt[v]` (preprocess L110-116) ⇒ la plage négative
            // commence EXACTEMENT à `p_end` : `p_start..p_end..n_end` est contigu et ordonné.

            if RMW == 3 {
                // ── ARM 3 : FUSION — une seule boucle sur `p_start..n_end`. L'ordre de visite
                let up_is_p = new_val as usize; // les occurrences p montent ssi new_val==true
                for k in p_start..n_end {
                    let is_p = (k < p_end) as usize;
                    let inc = (is_p == up_is_p) as usize; // 1 ⇒ +1, 0 ⇒ −1
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    let new_trues = trues.wrapping_add((inc as u8) << 1).wrapping_sub(1);
                    *true_lit_count.get_unchecked_mut(c_idx) = new_trues;

                    // RETRAIT (clause devenue satisfaite) : inc ET trues == 0.
                    let rc = inc & ((trues == 0) as usize);
                    let rm = 0usize.wrapping_sub(rc);
                    let last = *ubuf.get_unchecked(ulen);
                    let pos = *cpos.get_unchecked(c_idx) & rm;
                    ulen -= rc;
                    *ubuf.get_unchecked_mut(pos) = last;
                    *cpos.get_unchecked_mut((last & rm) | (nc & !rm)) = pos;

                    // AJOUT (clause devenue insatisfaite) : new_trues == 0 (impossible en inc).
                    let pc2 = (new_trues == 0) as usize;
                    let pm = 0usize.wrapping_sub(pc2);
                    ulen += pc2;
                    let wpos = ulen & pm;
                    *ubuf.get_unchecked_mut(wpos) = c_idx;
                    *cpos.get_unchecked_mut((c_idx & pm) | (nc & !pm)) = wpos;
                }
            } else {
                // ── ARMS 1 & 2 : `new_val` hoisté hors des boucles (acquis i21), 4 corps.
                let n_start = p_end;
                if new_val {
                    rmw_inc::<RMW>(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                    rmw_dec::<RMW>(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
                } else {
                    rmw_dec::<RMW>(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                    rmw_inc::<RMW>(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
                }
            }

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                let a = *var_age.get_unchecked(v);
                *var_age.get_unchecked_mut(v) = a.saturating_add(1);
            }

            let cur_unsat = ulen;
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                if !PROBE {
                    best_assignment.copy_from_slice(&assignment);
                    let _ = save_solution(&Solution { variables: best_assignment.clone() });
                    if cur_unsat == 0 { break; }
                }
            }
        }
    }

    if PROBE {
        let _ = save_solution(&Solution { variables: assignment });
    } else {
        let _ = save_solution(&Solution { variables: best_assignment });
    }
    Ok(())
}

/// Occurrences dont le littéral DEVIENT vrai : `trues += 1`, et si `trues` valait 0 la clause
/// sort de la liste (swap-pop). RMW==1 ⇒ branche ; RMW==2 ⇒ masque + poubelles.
#[inline(always)]
unsafe fn rmw_inc<const RMW: u8>(
    all_data: &[u32],
    tlc: &mut [u8],
    ubuf: &mut [usize],
    ulen: &mut usize,
    cpos: &mut [usize],
    nc: usize,
    start: usize,
    stop: usize,
) {
    for k in start..stop {
        let c_idx = *all_data.get_unchecked(k) as usize;
        let trues = *tlc.get_unchecked(c_idx);
        *tlc.get_unchecked_mut(c_idx) = trues + 1;
        if RMW == 1 {
            if trues == 0 {
                let last = *ubuf.get_unchecked(*ulen);
                let pos = *cpos.get_unchecked(c_idx);
                *ulen -= 1;
                if last != c_idx {
                    *ubuf.get_unchecked_mut(pos) = last;
                    *cpos.get_unchecked_mut(last) = pos;
                }
            }
        } else {
            let cond = (trues == 0) as usize;
            let m = 0usize.wrapping_sub(cond);
            let last = *ubuf.get_unchecked(*ulen);
            let pos = *cpos.get_unchecked(c_idx) & m; // 0 = poubelle si !cond
            *ulen -= cond;
            *ubuf.get_unchecked_mut(pos) = last;
            *cpos.get_unchecked_mut((last & m) | (nc & !m)) = pos; // nc = poubelle si !cond
        }
    }
}

/// Occurrences dont le littéral DEVIENT faux : `trues -= 1`, et si `trues` valait 1 la clause
/// entre dans la liste (push). RMW==1 ⇒ branche ; RMW==2 ⇒ masque + poubelles.
#[inline(always)]
unsafe fn rmw_dec<const RMW: u8>(
    all_data: &[u32],
    tlc: &mut [u8],
    ubuf: &mut [usize],
    ulen: &mut usize,
    cpos: &mut [usize],
    nc: usize,
    start: usize,
    stop: usize,
) {
    for k in start..stop {
        let c_idx = *all_data.get_unchecked(k) as usize;
        let trues = *tlc.get_unchecked(c_idx);
        *tlc.get_unchecked_mut(c_idx) = trues - 1;
        if RMW == 1 {
            if trues == 1 {
                *ulen += 1;
                *ubuf.get_unchecked_mut(*ulen) = c_idx;
                *cpos.get_unchecked_mut(c_idx) = *ulen;
            }
        } else {
            let cond = (trues == 1) as usize;
            let m = 0usize.wrapping_sub(cond);
            *ulen += cond;
            let wpos = *ulen & m; // 0 = poubelle si !cond
            *ubuf.get_unchecked_mut(wpos) = c_idx;
            *cpos.get_unchecked_mut((c_idx & m) | (nc & !m)) = wpos;
        }
    }
}
