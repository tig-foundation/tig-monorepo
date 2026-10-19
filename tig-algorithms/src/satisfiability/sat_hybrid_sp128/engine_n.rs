// SAT solver engine N — T5 (nv=7500, nc=32002, α=4.267 = seuil SAT-UNSAT exact)
//
// ══════════════════════════════════════════════════════════════════════════════════════
// POURQUOI CET ITER EXISTE — TROIS CANDIDATS DE TÊTE, JAMAIS MIS DANS LE MÊME BATCH
// ══════════════════════════════════════════════════════════════════════════════════════
// Au 2026-07-28T07:0xZ le track porte TROIS temps de production distincts, tous à
// Q=93 750 byte-identique, mais mesurés dans TROIS BINAIRES DIFFÉRENTS :
//
//   · i7  `engine_l` arm 2 `var_bounds_packing` 349 560 ms vs son CTRL même-batch 353 550
//                                              ⇒ **−1,13 %**, n=1, **JAMAIS ADJUGÉ**
//                                              (bench 25109/25110 atterri 06:20Z, jamais relu)
//   · i8  `engine_m` arm 1 `argmin branchless`  screening 200 M : 90,01 s vs ctrl 92,51 s
//                                              ⇒ **−2,70 %**, production PAS ENCORE LANCÉE
//
// Aucun de ces deux gains n'a été arbitré contre l'autre, et **personne n'a testé leur
// SOMME**. Les deux mécanismes sont ORTHOGONAUX par construction :
//   – `argmin branchless` retire des MISPREDICTIONS (site : sélection de variable) ;
//   – `var_bounds_packing` retire des CHARGES DÉPENDANTES (site : bornes du scan + du RMW).
// Ils ne touchent ni le même site ni la même ressource microarchitecturale ⇒ additivité
// PLAUSIBLE, mais non garantie (pression de registres, ports d'émission partagés).
//
// ⇒ **i9 = `engine_n` met les QUATRE états dans UN SEUL binaire**, screené dans UN SEUL
//    batch, sur UNE SEULE machine, avec le MÊME seed. C'est la seule façon de trancher un
//    3-way sans confond de binaire — et c'est exactement le pré-requis que pose
//
// ══════════════════════════════════════════════════════════════════════════════════════
// LE DISCRIMINANT — `P_oversubscribed_latency_is_free` (né d'i7), VERSION CORRIGÉE
// ══════════════════════════════════════════════════════════════════════════════════════
// La production tourne à 32 nonces × 32 workers sur une machine ~2,8× plus étroite
// (mono-thread ≈ 165 ns/flip vs ≈ 457 ns/flip en batch). Un cœur qui cale sur une charge
// est immédiatement occupé par un autre thread ⇒ la LATENCE est fortement AMORTIE.
//
// ⚠️ La formulation d'origine (« toute latence rend 0 ») est SUR-ÉNONCÉE et a été
// **réfutée par la mesure de production d'i7-arm2** : −2,88 % en mono-thread → −1,13 % en
// batch, soit un facteur d'amortissement ≈2,5, **pas une annulation**. La version
// soutenable est :
//
//     ⇒ les MISPREDICTIONS paient PLEIN TARIF   (i6 : −17,84 %)
//     ⇒ la LATENCE paie AU TIERS environ        (i7-arm2 : −2,88 % mono → −1,13 % batch)
//     ⇒ le discriminant est un CLASSEUR DE PRIORITÉ, pas un théorème d'annulation.
//
// C'est précisément ce qui rend arm 3 intéressant : on empile un gain « plein tarif »
// (mispredict) et un gain « au tiers » (latence) qui ne se disputent pas la même ressource.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// LES 4 ARMS (`sel_variant`, const generic ⇒ ZÉRO surcoût de dispatch dans la boucle)
// ══════════════════════════════════════════════════════════════════════════════════════
//                     3-way (**5ᵉ ré-ancrage** ; les 4 précédents ont reproduit à 0,23-0,28 %
//                     près ⇒ dérive machine nulle, le CTRL est fiable).
//  1 = **V_argmin** : argmin min-break + tie-break `var_age` rendu BRANCHLESS (port
//                     VERBATIM d'`engine_m` arm 1, i8). Retire des MISPREDICTIONS.
//  2 = **V_vbnd**   : `vbnd[v] = all_off[v] | p_bound[v]<<20 | all_off[v+1]<<40` — 1 charge
//                     packée au lieu de 3 dépendantes, aux DEUX sites (scan + RMW) (port
//                     VERBATIM d'`engine_l` arm 2, i7). Retire des CHARGES DÉPENDANTES.
//  3 = **V_both**   : arms 1 + 2 SIMULTANÉS = **CANDIDAT KEEP**.
//
//  Les 4 arms sont state-équivalents PAR CONSTRUCTION : aucun ne touche au RNG, à l'ordre
//  de visite, ni à une valeur d'état ⇒ **Q=93 750 byte-identique attendue pour les 4**.
//  Une Q ≠ 93 750 sur un arm = un BUG (trajectoire divergée), pas un résultat.
//
// ⚠️ CONTRE-EXEMPLE À RESPECTER (Hindsight `59c7da0e`, v9 t3 2026-06-24, +11,8 % REVERTED) :
// le branchless a PERDU là où la branche était BIEN PRÉDITE. Le gate est la MESURE, jamais
// l'idée ⇒ screening par cascade OBLIGATOIRE, arm de contrôle dans le MÊME binaire.
// ⚠️ Hindsight `4a8daf24`/`68b553e3` (« l'axe pick_variable est mort sur t1 : tout changement
// de sélection détruit la solution fragile ») ne s'applique PAS ici : les arms 1 et 3 ne
// changent pas la SÉLECTION, ils changent le CODEGEN qui la calcule — `best_idx` est
// byte-identique par construction (preuve en ligne au site du code).
//
// ── PROTOCOLE DE SCREENING (`probe_max_flips` renseigné) ──────────────────────────────
// Harnais i4-i8 : **flips-count FIXE** + **early-SAT-exit OFF** + pas de memcpy
// `best_assignment` dans la boucle ⇒ les 4 arms exécutent EXACTEMENT le même travail,
// Δt = pur ns/flip. 32 nonces / 32 workers (⛔ jamais de bench < 32 nonces sur CPU).
// `probe_max_flips = 200 M` (⇒ arm 0 ≈ 92 s) : la granularité d'horloge du client est de
// 0,5 s DURE (toutes les valeurs en base sont des multiples de 0,5 s) ⇒ résolution ±0,53 %,
// nécessaire pour résoudre un effet attendu à −1 % (à 50 M la résolution est ±2,1 % et
// **c'est ce qui a rendu i7 aveugle sur son propre arm 2**).
// Le survivant passe le bench de PRODUCTION (early-exit ON, fuel réel, Q gatée 93 750) avec
// un CTRL `sel_variant=0` dans le MÊME batch et le MÊME binaire.
//
// ⛔ Wall-clock INTERDIT : `probe_max_flips` est un compteur de TRAVAIL, pas un temps.
// ⛔ `engine_k` / `engine_f` / `engine_m` ne sont JAMAIS édités en place (isolation de track

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use super::engine_f::{preprocess, Prepared};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub max_fuel_high: Option<f64>,
    /// 0=A_ctrl (engine_k arm 2 VERBATIM) · 1=V_argmin · 2=V_vbnd · 3=V_both (candidat KEEP).
    pub sel_variant: Option<u64>,
    /// Renseigné ⇒ MODE SONDE (travail fixe + early-exit OFF). Absent ⇒ PRODUCTION.
    pub probe_max_flips: Option<u64>,
}

/// Arm retenu en PRODUCTION (défaut). Mis à jour après le screening de la cascade.
///
/// **i10 (2026-07-28) : 0 → 1 = `V_argmin` BAKÉ.** Seul et unique delta d'i10 vs i9.
/// Le screening à travail fixe a tranché sur DEUX builds indépendants (`probe_max_flips`=200 M,
/// early-exit OFF, 32 nonces / 32 workers, Q=31 250 IDENTIQUE sur tous les arms ⇒ identité de
/// trajectoire vérifiée empiriquement EN PLUS de la preuve en ligne `engine_n:372-410`) :
///   · i8 `engine_m` ver 4367 : arm1 90,2 s (n=3) vs ctrl 92,5 s (n=2) ⇒ **−2,5 %**
///   · i9 `engine_n` ver 4368 : arm1 90,5 s (n=2) vs ctrl ~94,0 s (n=2) ⇒ **−3,7 %**
///   · arm2 `V_vbnd` neutre, arm3 `V_both` == arm1 seul ⇒ **vbnd rend 0 en plus, NON EMBARQUÉ**
///     (latence absorbée par l'oversubscription 32×32, `P_oversubscribed_latency_is_free`).
/// `hp={}` : c'est la MÊME monomorphisation `run::<1,false>`, seul le bras du `match` de dispatch
/// change — hors boucle chaude ⇒ zéro effet de codegen sur le corps mesuré.
const DEFAULT_VARIANT: u64 = 1;

// Champs du mot packé `vbnd` (arms 2 et 3). 20 bits par champ ⇒ valide tant que
// `all_data.len() < 2^20`. t5 : nc=32002, entrées ≈ 96 006 ⇒ très large marge. Un garde
// runtime (`solve`, AVANT le dispatch) retombe sur l'arm 0 si la borne était dépassée, pour
// que `SV` reste la SEULE condition dans le corps (const generic ⇒ zéro test au runtime).
const FSH: u32 = 20;
const FMASK: usize = (1usize << FSH) - 1;
const PACK_LIMIT: usize = 1usize << FSH;

#[inline(always)]
fn pack3(a: usize, b: usize, c: usize) -> usize {
    a | (b << FSH) | (c << (2 * FSH))
}

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

    let mut variant = hp.sel_variant.unwrap_or(DEFAULT_VARIANT);
    // Garde de sûreté du packing, évaluée AVANT le dispatch (cf `PACK_LIMIT`). `all_data`
    // contient au plus 3 entrées par clause ⇒ `clauses.len()*3` majore sa longueur.
    if (variant == 2 || variant == 3) && challenge.clauses.len().saturating_mul(3) >= PACK_LIMIT {
        variant = if variant == 3 { 1 } else { 0 };
    }
    match hp.probe_max_flips {
        None => match variant {
            1 => run::<1, false>(challenge, save_solution, prod_flips),
            2 => run::<2, false>(challenge, save_solution, prod_flips),
            3 => run::<3, false>(challenge, save_solution, prod_flips),
            _ => run::<0, false>(challenge, save_solution, prod_flips),
        },
        Some(n) => match variant {
            1 => run::<1, true>(challenge, save_solution, n),
            2 => run::<2, true>(challenge, save_solution, n),
            3 => run::<3, true>(challenge, save_solution, n),
            _ => run::<0, true>(challenge, save_solution, n),
        },
    }
}

const WALK_P: f64 = 0.52;
const RESTART_PERIOD: u64 = 80_000_000;
const REINIT_MIN_UNSAT: usize = 30;
const PERTURB_K: usize = 100;


// ══════════════════════════════════════════════════════════════════════════════════════
// KEPT d'i6 : tampon plat 1-based `ubuf[0]`=POUBELLE + `cpos[nc]`=POUBELLE + RMW masqué).
// Les arms RMW 1 et 3 d'engine_k (dead 174) sont retirés : ici RMW est FIGÉ sur le masque.
// `SV` ne touche QUE (a) le bloc de sélection de variable (arms 1 et 3 ⇒ argmin branchless)
// et (b) la LECTURE DES BORNES `(start, stop)` / `(p_start, p_end, n_end)` (arms 2 et 3 ⇒
// 1 charge packée au lieu de 3). Aucun arm ne consomme de RNG, ne change l'ordre de visite,
// ni ne change une valeur d'état ⇒ Q=93 750 byte-identique attendue pour les 4.
// ══════════════════════════════════════════════════════════════════════════════════════
fn run<const SV: u8, const PROBE: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    max_flips: u64,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density: _density, p_cnt, n_cnt, all_off, p_bound, all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    // La borne 20 bits a été vérifiée dans `solve` AVANT le dispatch (sinon l'arm retombe),
    // donc ici `SV == 2` / `SV == 3` sont des conditions PUREMENT const-generic.
    debug_assert!(!(SV == 2 || SV == 3) || all_data.len() < PACK_LIMIT);
    // ARMS 2 & 3 — `vbnd[v] = all_off[v] | p_bound[v]<<20 | all_off[v+1]<<40` : remplace 3
    // charges dépendantes par 1, aux DEUX sites (bornes du scan ET bornes du RMW).
    // Port VERBATIM d'`engine_l` arm 2 (i7). Table de `nv` usize = 60 KB sur t5 ; c'est un
    // COÛT MÉMOIRE ajouté, mais la famille mémoire est absorbée ici (dead 169-173) alors que
    // la chaîne de charges, elle, valait −2,88 % en mono-thread ⇒ le pari est net.
    // ⚠️ Construit hors de la boucle chaude, une fois par nonce (preprocess est déjà O(nc)).
    let vbnd: Vec<usize> = if SV == 2 || SV == 3 {
        (0..nv)
            .map(|v| pack3(all_off[v] as usize, p_bound[v] as usize, all_off[v + 1] as usize))
            .collect()
    } else {
        Vec::new()
    };

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
                // ── SITE 1 DES BORNES (scan / break-count) ────────────────────────────
                // Arms 2 & 3 : 1 charge packée `vbnd[v]` au lieu des 3 charges dépendantes
                // `all_off[v]`, `p_bound[v]`, `all_off[v+1]`. Les valeurs extraites sont
                // ARITHMÉTIQUEMENT IDENTIQUES (pack3 est injectif sous `PACK_LIMIT`, garde
                // vérifiée dans `solve`) ⇒ `start`/`stop` byte-identiques, donc `b`, donc
                // la trajectoire. Le delta est purement microarchitectural.
                let (start, stop) = if SV == 2 || SV == 3 {
                    let w = *vbnd.get_unchecked(v);
                    if val {
                        (w & FMASK, (w >> FSH) & FMASK)
                    } else {
                        ((w >> FSH) & FMASK, w >> (2 * FSH))
                    }
                } else if val {
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
                } else if SV == 1 || SV == 3 {
                    // ══════════════════════════════════════════════════════════════════
                    // ARMS 1 & 3 `V_argmin` — argmin min-break + tie-break
                    // `var_age`, rendu BRANCHLESS. C'est le DERNIER branchement
                    // data-dépendant ET trajectoire-safe du flip (le RMW est tombé en i6,
                    // l'accumulation du scan est déjà branchless, le coin-flip `WALK_P` est
                    // sous VETO car le rendre branchless consommerait du RNG).
                    //
                    // ── ÉQUIVALENCE BYTE-À-BYTE (par construction, PAS par mesure) ────
                    //     `b0 < u8::MAX || (b0 == u8::MAX && age[v0] < age[v0])`
                    //     = `b0 < 255` (le tie-break se compare à LUI-MÊME ⇒ faux).
                    //     Les DEUX issues laissent `best_idx = 0` et `min_b = b0` (dans la
                    //     branche non prise, `min_b` valait déjà 255 == b0). L'état après
                    //     i=0 est donc INCONDITIONNELLEMENT (min_b=breaks[0], best_idx=0).
                    //     Aucune hypothèse sur la valeur de `b` n'est requise.
                    // (2) i>=1 : `best_age` remplace la relecture `var_age[vars[best_idx]]`.
                    //     `var_age` n'est écrit qu'APRÈS le flip (boucle L « saturating_add »),
                    //     jamais dans cette boucle ⇒ la valeur cachée est exacte, et elle est
                    //     mise à jour EXACTEMENT quand `best_idx` l'est. Même prédicat, même
                    //     ordre, même premier-gagnant sur égalité stricte.
                    // ⇒ `best_idx` (donc `picked_v`, donc la trajectoire, donc Q=93 750) est
                    //   IDENTIQUE. Le seul delta est le codegen : `|`/`&` non court-circuitants
                    //   + `select` ⇒ `cmov`, zéro saut conditionnel data-dépendant.
                    // Bonus non recherché : 2 charges de moins par itération (`vars[best_idx]`
                    // et sa relecture `var_age[..]`) — de la LATENCE, donc gratuite ici
                    // (`P_oversubscribed_latency_is_free`) ; le pari porte sur le mispredict.
                    // ══════════════════════════════════════════════════════════════════
                    let mut best_idx = 0usize;
                    let mut min_b = *breaks.get_unchecked(0);
                    let mut best_age = *var_age.get_unchecked(*vars.get_unchecked(0));
                    for i in 1..nvars {
                        let b = *breaks.get_unchecked(i);
                        let age = *var_age.get_unchecked(*vars.get_unchecked(i));
                        // `|` et `&` sont NON court-circuitants ⇒ pas de saut caché.
                        let take = (b < min_b) | ((b == min_b) & (age < best_age));
                        min_b = if take { b } else { min_b };
                        best_age = if take { age } else { best_age };
                        best_idx = if take { i } else { best_idx };
                    }
                    picked_v = *vars.get_unchecked(best_idx);
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

            // ── SITE 2 DES BORNES (RMW / bookkeeping du flip) ─────────────────────────
            // Même substitution qu'au site 1, mêmes valeurs par injectivité de `pack3`.
            let (p_start, p_end, n_end) = if SV == 2 || SV == 3 {
                let w = *vbnd.get_unchecked(picked_v);
                (w & FMASK, (w >> FSH) & FMASK, w >> (2 * FSH))
            } else {
                (
                    *all_off.get_unchecked(picked_v) as usize,
                    *p_bound.get_unchecked(picked_v) as usize,
                    *all_off.get_unchecked(picked_v + 1) as usize,
                )
            };
            // `p_bound[v] == all_off[v] + p_cnt[v]` (preprocess L110-116) ⇒ la plage négative
            // commence EXACTEMENT à `p_end` : `p_start..p_end..n_end` est contigu et ordonné.

            // `new_val` hoisté hors des boucles (acquis i21), 4 corps. RMW = masque (arm 2 i6).
            let n_start = p_end;
            if new_val {
                rmw_inc(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                rmw_dec(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
            } else {
                rmw_dec(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                rmw_inc(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
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
/// sort de la liste (swap-pop). VERSION MASQUÉE FIGÉE (= engine_k arm 2, KEPT i6).
#[inline(always)]
unsafe fn rmw_inc(
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
        let cond = (trues == 0) as usize;
        let m = 0usize.wrapping_sub(cond);
        let last = *ubuf.get_unchecked(*ulen);
        let pos = *cpos.get_unchecked(c_idx) & m; // 0 = poubelle si !cond
        *ulen -= cond;
        *ubuf.get_unchecked_mut(pos) = last;
        *cpos.get_unchecked_mut((last & m) | (nc & !m)) = pos; // nc = poubelle si !cond
    }
}

/// Occurrences dont le littéral DEVIENT faux : `trues -= 1`, et si `trues` valait 1 la clause
/// entre dans la liste (push). VERSION MASQUÉE FIGÉE (= engine_k arm 2, KEPT i6).
#[inline(always)]
unsafe fn rmw_dec(
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
        let cond = (trues == 1) as usize;
        let m = 0usize.wrapping_sub(cond);
        *ulen += cond;
        let wpos = *ulen & m; // 0 = poubelle si !cond
        *ubuf.get_unchecked_mut(wpos) = c_idx;
        *cpos.get_unchecked_mut((c_idx & m) | (nc & !m)) = wpos;
    }
}
