// SAT solver engine O — T5 (nv=7500, nc=32002, α=4.267 = seuil SAT-UNSAT exact)
//
// ══════════════════════════════════════════════════════════════════════════════════════
// POURQUOI CET ITER EXISTE — LA ROADMAP DISAIT « l'argmin était le DERNIER branchement
// data-dépendant trajectoire-safe ». C'EST FAUX, PAR INSPECTION DU CODE.
// ══════════════════════════════════════════════════════════════════════════════════════
// M13, job 25145) contient ENCORE, dans le corps du flip, deux consommateurs de travail
// jamais mesurés isolément et tous deux TRAJECTOIRE-SAFE :
//
//  (A) `engine_n:363-367` — la lecture des bornes du scan est gardée par
//        `} else if val { (all_off[v], p_bound[v]) } else { (p_bound[v], all_off[v+1]) }`
//      où `val = assignment[v]` est l'assignation COURANTE d'une variable tirée au hasard
//      dans une clause tirée au hasard ⇒ **~50/50, structurellement IMPRÉDICTIBLE**, et
//      exécuté **jusqu'à 3 fois par flip** (une par candidate scannée). C'est exactement la
//      classe qui paie PLEIN TARIF sur ce track (`P_oversubscribed_latency_is_free`) — la
//      même classe que le RMW d'i6 (−17,84 %) et l'argmin d'i10 (−2,7 %).
//      ⚠️ L'arm `V_vbnd` d'i7/i9 N'A PAS testé ça : il remplaçait les 3 charges par 1 charge
//      packée (famille LATENCE, rendue 0) **en gardant le `if val`** (`engine_n:356-362`).
//      Le branchement, lui, n'a jamais été retiré. → arm 1 `V_polsel`.
//
//  (B) `engine_n:346-347`, `:379`, `:471-472` — chaque littéral est stocké en `i32` signé et
//      redécodé à CHAQUE lecture par `(lit.abs() - 1) as usize` (cdq/xor/sub + sub) puis,
//      dans la reconstruction post-restart, par `lit > 0`. Soit ~6 décodages par flip
//      (3 dans le scan + 3 dans le bump `var_age`), chacun en TÊTE d'une chaîne de
//      dépendances `charge → abs → sub → charge aléatoire assignment[v]`.
//      Pré-décoder une fois pour toutes en `lp[i] = (v << 1) | pol` (u32, **même empreinte
//      mémoire que `cl`** ⇒ la famille layout, morte ici, n'est PAS convoquée) réduit le
//      décodage à un `shr` et raccourcit la chaîne critique. → arm 2 `V_litpack`.
//
// Aucun des deux ne touche au RNG, ni à l'ordre de visite, ni à une valeur d'état : ce sont
// des transformations de CODEGEN et d'ENCODAGE, byte-identiques par construction (preuves en
// ligne aux deux sites). Q = 93 750 attendue à l'unité sur les 4 arms — toute autre valeur
// est un BUG, pas un résultat.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// CE QUI A ÉTÉ ÉCARTÉ AVANT D'ÊTRE CODÉ (et pourquoi — ne pas re-proposer)
// ══════════════════════════════════════════════════════════════════════════════════════
//  ⛔ `if new_val { inc(p); dec(n) } else { dec(p); inc(n) }` (engine_n:462-468) — c'est
//     pourtant un branchement ~50/50 par flip. Le rendre branchless en sélectionnant les
//     PLAGES (cmov) impose d'exécuter toujours `inc` puis `dec`, donc **d'inverser l'ordre**
//     des deux passes dans un cas sur deux. Or `inc` fait du swap-pop et `dec` du push sur
//     `ubuf` : les deux passes NE COMMUTENT PAS sur la PERMUTATION de `ubuf` (elles commutent
//     seulement sur `true_lit_count`, dont les plages sont disjointes). Comme la clause du
//     flip suivant est `ubuf[gen_range(0..ulen)+1]`, toute permutation différente = TRAJECTOIRE
//     différente ⇒ VETO. (L'unique forme ordre-préservante est la FUSION des deux boucles =
//     `brick:rmw_loop_fusion`, **dead-listée, mesurée 0 gain** en i6 arm 3.)
//  ⛔ `global_flips % RESTART_PERIOD` → compteur : famille idiv, **dead** (t5 id 177
//     `alu:flip_modulo_len3` + t4 id 167) — « l'idiv n'est pas le mur, y compris en position
//     de latence ».
//  ⛔ Tout ce qui touche `WALK_P` (consomme du RNG), la sélection, le fuel, ou un layout
//     mémoire (dead 169-173) : hors périmètre.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// LES 4 ARMS (`sel_variant`, const generic ⇒ ZÉRO surcoût de dispatch dans la boucle chaude)
// ══════════════════════════════════════════════════════════════════════════════════════
//                      + littéraux i32). ANCRE commune, 6ᵉ ré-ancrage du track.
//  1 = **V_polsel**  : (A) bornes du scan sélectionnées par `select`/cmov — 3 charges
//                      inconditionnelles + 2 cmov au lieu d'un saut ~50/50.
//  2 = **V_litpack** : (B) littéraux pré-décodés `u32 = (v<<1)|pol`, `lp` remplaçant `cl`
//                      dans TOUT le corps du flip (scan, swap, bump `var_age`, rebuild).
//  3 = **V_both**    : arms 1 + 2 simultanés = test d'ADDITIVITÉ (candidat KEEP si additif).
//
//
// ⚠️ CONTRE-EXEMPLE À RESPECTER (Hindsight `59c7da0e` / `0bf9d46e` / `44faa88b`, v9 t3
// 2026-06-24 : +2,0 % puis +11,8 %, REVERTED) : le branchless PERD quand la branche visée est
// BIEN PRÉDITE. Ici la branche visée est `assignment[v]` d'une variable tirée au hasard —
// classe OPPOSÉE (imprédictible par construction). Le gate reste la MESURE : cascade avec arm
// de contrôle DANS LE MÊME BINAIRE, jamais un pari.
//
// ── PROTOCOLE DE SCREENING (`probe_max_flips` renseigné) ──────────────────────────────
// Harnais i4-i10 : **flips-count FIXE** + **early-SAT-exit OFF** + pas de memcpy
// `best_assignment` dans la boucle ⇒ les 4 arms exécutent EXACTEMENT le même travail,
// Δt = pur ns/flip. 32 nonces / 32 workers (⛔ jamais de bench < 32 nonces sur CPU).
// `probe_max_flips = 200 M` ⇒ arm 0 ≈ 90 s : la granularité d'horloge du client est de 0,5 s
// DURE ⇒ résolution ±0,55 %, nécessaire pour résoudre un effet attendu à −1 à −3 %.
// Le survivant passe le bench de PRODUCTION (early-exit ON, fuel réel 155 B, Q gatée 93 750)
// avec un CTRL `sel_variant=0` dans le MÊME batch et le MÊME binaire.
//
// ⛔ Wall-clock INTERDIT : `probe_max_flips` est un compteur de TRAVAIL, pas un temps.
// ⛔ `engine_f`/`engine_k`/`engine_n` ne sont JAMAIS édités en place (isolation de track + on

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use super::engine_f::{preprocess, Prepared};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub max_fuel_high: Option<f64>,
    pub sel_variant: Option<u64>,
    /// Renseigné ⇒ MODE SONDE (travail fixe + early-exit OFF). Absent ⇒ PRODUCTION.
    pub probe_max_flips: Option<u64>,
}

/// Le screening à travail fixe d'i11 a tranché sur DEUX répétitions interleavées
/// (`probe_max_flips`=200 M, ver 4372, jobs 25153-25160, M13, Q=31 250 IDENTIQUE sur les 4 arms
/// ⇒ contrôle d'identité passé) :
///   arm 0 `A_ctrl`  90,51 / 90,52 s  — réf
///   arm 1 `V_polsel` 90,52 / 90,52 s — **+0,006 % = NULL** ⇒ dead-list `branch:polsel_bound_cmov`
///   arm 2 `V_litpack` 88,51 / 88,52 s — **−2,21 %** ⇒ SURVIVANT
///   arm 3 `V_both`   88,52 / 88,52 s — ≡ arm 2 seul ⇒ additivité NULLE (contrôle croisé)
/// PROD du survivant (job 25167, `hp={"sel_variant":2}`) : Q=93 750 32/32 0-inval @**336 050 ms**
///
/// ⚠️ POURQUOI CE BAKE EST OBLIGATOIRE (et pas cosmétique) : tant que le défaut reste 0, un build
/// ne serait alors une win QUE si l'override `sel_variant=2` accompagne le binaire — ce qui n'est
/// i10 était déjà le bake du survivant du screening d'i9 (`engine_n` `DEFAULT_VARIANT=1`).
/// Transformation hors boucle chaude (ligne de dispatch), trajectoire INCHANGÉE ⇒ Q=93 750 attendue
/// À L'UNITÉ ; toute autre valeur est un BUG, pas un résultat.
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

    let variant = hp.sel_variant.unwrap_or(DEFAULT_VARIANT);
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

/// Décodage d'un littéral du corps du flip, const-générique sur l'arm.
///
/// ── ÉQUIVALENCE (arms 2/3 vs 0/1) ────────────────────────────────────────────────────
/// `lp[i] = ((v as u32) << 1) | (pol as u32)` avec `v = cl[i].abs()-1`, `pol = cl[i] > 0`
/// est une BIJECTION sur le couple `(v, pol)` (v < nv ≤ 7500 < 2^31 ⇒ aucun débordement).
/// `lp` est construit dans le MÊME ordre que `cl` et subit EXACTEMENT les mêmes `swap`
/// ⇒ à tout instant `decode_v(lp, i) == cl[i].abs()-1` et `decode_pol(lp, i) == (cl[i] > 0)`.
/// ⇒ mêmes `v`, même ordre de visite, même `picked_v`, même trajectoire, même Q.
#[inline(always)]
unsafe fn decode_v<const SV: u8>(cl: &[i32], lp: &[u32], i: usize) -> usize {
    if SV >= 2 {
        (*lp.get_unchecked(i) >> 1) as usize
    } else {
        ((*cl.get_unchecked(i)).abs() - 1) as usize
    }
}

#[inline(always)]
unsafe fn decode_pol<const SV: u8>(cl: &[i32], lp: &[u32], i: usize) -> bool {
    if SV >= 2 {
        (*lp.get_unchecked(i) & 1) == 1
    } else {
        *cl.get_unchecked(i) > 0
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════
// d'i10 : tampon plat 1-based `ubuf[0]`=POUBELLE + `cpos[nc]`=POUBELLE + RMW masqué +
// argmin branchless). Les arms `V_vbnd` d'i9 (latence, rend 0) sont retirés.
// `SV` ne touche QUE (a) la sélection des bornes du scan (arms 1 et 3) et (b) l'encodage des
// littéraux (arms 2 et 3). Aucun arm ne consomme de RNG, ne change l'ordre de visite, ni ne
// change une valeur d'état ⇒ Q=93 750 byte-identique attendue pour les 4.
// ══════════════════════════════════════════════════════════════════════════════════════
fn run<const SV: u8, const PROBE: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    max_flips: u64,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density: _density, p_cnt, n_cnt, all_off, p_bound, all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    // ARMS 2 & 3 — littéraux pré-décodés. MÊME longueur, MÊME largeur (u32 vs i32) que `cl`
    // ⇒ aucune variation d'empreinte mémoire (la famille layout est morte sur ce track, on ne
    // la convoque pas). Construit une fois par nonce, hors de la boucle chaude.
    let mut lp: Vec<u32> = if SV >= 2 {
        cl.iter()
            .map(|&lit| (((lit.abs() - 1) as u32) << 1) | ((lit > 0) as u32))
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

    // Construction initiale : AVANT tout `swap`, `lp` et `cl` sont dans le même ordre ⇒ on
    // lit `cl` (identique pour les 4 arms, hors boucle chaude).
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
                            // `cl` a pu être permuté par `swap` (arms 0/1) tout comme `lp`
                            // (arms 2/3) : on lit TOUJOURS le tableau que l'arm permute.
                            let v = decode_v::<SV>(&cl, &lp, i);
                            let pol = decode_pol::<SV>(&cl, &lp, i);
                            if *assignment.get_unchecked(v) == pol { trues += 1; }
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
                // MÊME permutation appliquée au MÊME rang : `lp` est l'image bijective de `cl`.
                if SV >= 2 { lp.swap(off, off + ri); } else { cl.swap(off, off + ri); }
            }

            let mut picked_v = usize::MAX;
            let mut vars = [0usize; 3];
            let mut breaks = [0u8; 3];
            let mut nvars = 0;

            for i in off..end {
                let v = decode_v::<SV>(&cl, &lp, i);
                *vars.get_unchecked_mut(nvars) = v;
                let val = *assignment.get_unchecked(v);
                // ══════════════════════════════════════════════════════════════════════
                // SITE (A) — BORNES DU SCAN. ARMS 1 & 3 `V_polsel` : le saut conditionnel
                // sur `val` est remplacé par 3 charges INCONDITIONNELLES + 2 `select`.
                //
                // ── ÉQUIVALENCE BYTE-À-BYTE (par construction) ────────────────────────
                // Les trois valeurs `base = all_off[v]`, `mid = p_bound[v]`,
                // `top = all_off[v+1]` sont chargées quel que soit `val` ; le couple rendu
                // est `(base, mid)` si `val`, `(mid, top)` sinon — EXACTEMENT les deux
                // couples de l'arm 0. `all_off` a `nv+1` entrées (preprocess) ⇒ `v+1` est
                // TOUJOURS en borne, la charge supplémentaire est donc légale dans les deux
                // cas (l'arm 0 la faisait déjà dans sa branche `else`).
                // ⇒ `start`/`stop` identiques ⇒ `b` identique ⇒ trajectoire identique.
                // Le seul delta est le codegen : `select` ⇒ `cmov`, ZÉRO saut sur une
                // condition ~50/50 structurellement imprédictible (assignation courante
                // d'une variable tirée au hasard), exécutée jusqu'à 3× par flip.
                // Coût payé : 1 charge de plus par candidate, mais `all_off[v]` et
                // `all_off[v+1]` sont ADJACENTS (même ligne de cache dans ~15 cas sur 16)
                // ⇒ de la LATENCE, amortie ~×3 par l'oversubscription 32×32 (i7-arm2),
                // contre une misprédiction qui, elle, paie PLEIN TARIF (i6, i10).
                // ══════════════════════════════════════════════════════════════════════
                let (start, stop) = if SV == 1 || SV == 3 {
                    let base = *all_off.get_unchecked(v) as usize;
                    let mid = *p_bound.get_unchecked(v) as usize;
                    let top = *all_off.get_unchecked(v + 1) as usize;
                    (if val { base } else { mid }, if val { mid } else { top })
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
                    if SV >= 2 {
                        if let Some(&e) = lp.get(off) { picked_v = (e >> 1) as usize; }
                    } else if let Some(&lit) = cl.get(off) {
                        picked_v = (lit.abs() - 1) as usize;
                    }
                } else if rng.gen_bool(WALK_P) {
                    picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
                } else {
                    // (preuve d'équivalence : `engine_n:390-405`, i=0 déroulé + cache `best_age`)
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
                }
            }

            if picked_v == usize::MAX { continue; }

            let new_val = !*assignment.get_unchecked(picked_v);
            *assignment.get_unchecked_mut(picked_v) = new_val;

            // SITE 2 DES BORNES (RMW) — 3 charges inconditionnelles, AUCUN branchement ici
            let (p_start, p_end, n_end) = (
                *all_off.get_unchecked(picked_v) as usize,
                *p_bound.get_unchecked(picked_v) as usize,
                *all_off.get_unchecked(picked_v + 1) as usize,
            );

            // ⛔ L'ordre `inc` puis `dec` (resp. `dec` puis `inc`) N'EST PAS commutatif sur la
            // PERMUTATION de `ubuf` ⇒ le `if new_val` ne peut PAS devenir un select de plages
            // sans changer la trajectoire (cf en-tête). Laissé VERBATIM.
            let n_start = p_end;
            if new_val {
                rmw_inc(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                rmw_dec(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
            } else {
                rmw_dec(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, p_start, p_end);
                rmw_inc(&all_data, &mut true_lit_count, &mut ubuf, &mut ulen, &mut cpos, nc, n_start, n_end);
            }

            for i in off..end {
                let v = decode_v::<SV>(&cl, &lp, i);
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
