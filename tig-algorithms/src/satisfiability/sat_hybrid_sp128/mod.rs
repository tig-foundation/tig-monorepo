// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::satisfiability::*;

// engine_d). Le fichier est conservé intact comme référence de la baseline i1.
#[allow(dead_code)]
mod engine_a;
mod engine_b;
// t38/i1 (cellule, 20/08) : phase 1 SP-renforcement pour t38. Cf knowledge/t38_i1_*.md
mod engine_spr;
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
// t4/i14 (CONSOLIDATION) : `engine_e` fournit `preprocess`/`Prepared` a `engine_ag`. Copie
// BYTE-IDENTIQUE de `iters/t4/i7/code/engine_e.rs`. Son propre `solve`/`Hparams` n'est PAS cable.
#[allow(dead_code)]
mod engine_e;
// t4/i14 (CONSOLIDATION) : `engine_ag` = champion TEMPS de t4, terminus de la campagne i6->i28 de
// l'algo 138 `sat_hybrid` (296 040 -> 141 020 ms, -52 %, Q strictement constante 218 750 / 32-32).
// Copie BYTE-IDENTIQUE de `iters/t4/i7/code/engine_ag.rs`. `DEFAULT_VARIANT = 2` => `hp={}` route
// sur `V_lmfree`.
mod engine_ag;
// t4/i37 : `engine_iv` = port VERBATIM de sat_imp_v4 (qualifier mainnet), chemin track1 seul.
// Run B du portfolio t4 — preuve bench 34816 : @400e9 il resout {3,7,12,17,27,28} dont le
// nonce 28, HORS de l'union des 585 benchs a map de la base (t4 etait 7/32 sans lui).
mod engine_iv;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn u(v: u64) -> Value { Value::Number(Number::from(v)) }
// Budgets de portfolio surchargeables par HP (defaut = valeur prouvee du run).
// hp={} => valeurs par defaut => Q/temps bit-identiques. Baisser un budget reduit
// le temps de ce run au risque de perdre son/ses nonce(s) cible(s) (Q).
fn get_u64(hp: &Option<Map<String, Value>>, key: &str, default: u64) -> u64 {
    hp.as_ref().and_then(|m| m.get(key)).and_then(|v| v.as_u64()).unwrap_or(default)
}
fn with_fuel(hp: &Option<Map<String, Value>>, key: &str, default: u64) -> Option<Map<String, Value>> {
    let f = get_u64(hp, key, default);
    let mut m = hp.clone().unwrap_or_default();
    m.insert("max_fuel_high".to_string(), u(f));
    Some(m)
}

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
            // === t1/i36 : PORTFOLIO DE TRAJECTOIRES — cible = le plafond PROUVE 9/32 ===
            // 130 benchs relus : AUCUNE trajectoire n'attrape l'union {4,7,10,15,19,22,24,28,31}.
            // La prod (A) attrape 7 ; l'init perturbee d'i26 (B) attrape {4,7,10,15,22,24,31} —
            // fuels reels mesures 58-681 G (jobs 34618/34622, identiques a l'unite). Le temps n'a
            // AUCUNE valeur economique (tig-protocol) et le harness alloue 5 000 G/nonce : le
            // portfolio sequentiel est legal. A (315 G) puis, si non resolu, B (700 G >= 681).
            // « Resolu » = une solution qui satisfait TOUTES les clauses (le save trivial du
            // preprocess et le save best-effort final ne comptent pas).
            let solved = std::cell::Cell::new(false);
            let cb = |s: &Solution| {
                if !s.variables.is_empty()
                    && challenge.clauses.iter().all(|cl| {
                        cl.iter().any(|&l| {
                            let v = (l.abs() - 1) as usize;
                            if l > 0 { s.variables[v] } else { !s.variables[v] }
                        })
                    })
                {
                    solved.set(true);
                }
                save_solution(s)
            };
            let hp_a = with_fuel(hyperparameters, "t1_fuel_a", 315000000000);
            engine_d::solve(challenge, &cb, &hp_a)?;
            if !solved.get() {
                // i37 : 700e9 etait exprime en unites REELLES — or `max_fuel_high` est en unites MODELE
                // (facteur ~2,4 : la prod hp 315e9 consomme ~756 G reels/nonce bloque). 700e9 modele
                // = ~1 680 G reels ⇒ pire nonce 2 436 G ≈ 1 925 s > mur 1 800 s (timeout d'i36).
                // 300e9 modele ≈ 720 G reels ≥ 681 (le nonce le plus cher de B) ⇒ pire nonce
                // ~1 476 G ≈ 1 200-1 350 s ✓.
                let hp_b = with_fuel(hyperparameters, "t1_fuel_b", 300000000000);
                engine_d::solve_perturbed(challenge, &cb, &hp_b)?;
            }
            Ok(())
        }
        (100000, 415000) => engine_b::solve(challenge, save_solution, hyperparameters),
        (5000, 21335) => {
            // t4/i14 — CONSOLIDATION. Seul bras modifie vs `code/` : t4 passe de `engine_b`
            // (481,56 s, job 30353) a `engine_ag` (141,51 s, job 30364) a Q BIT-IDENTIQUE
            // (218 750 = 7/32, 32/32 finis, 0 invalide), soit -70,6 %. n=3+3 sur 2 dates.
            // Ce champion n'avait jamais ete cable : le finalizer ne juge que la Q, or Q est
            // binaire et SATUREE sur t4 (218 750 = plafond de tous les algos, 141 benchs)
            // => tout gain de TEMPS y est classe `no-win` et l'iteration `rejected`
            // (cf iters/t4/i7/code/lesson.md, et avant lui 138/iters/t4/i28 laissee stranded).
            // Isolation de track : `engine_b` reste cable pour t1/t3/t5/t38, non touches.
            // === t4/i37 : PORTFOLIO — cible le nonce #28 (prouve SAT par 34809/34815/34816,
            // jamais resolu par nos moteurs). Meme mecanisme que t1/i37 et t5/i19-20 (promus).
            // A = engine_ag AUX MEMES hp (bit-identique : la closure n'altere pas son RNG).
            // B (echec verifie seulement) = engine_iv avec UNE map propre {max_fuel_high:400e9}
            // = replique EXACTE des conditions du bench 34816 ({3,7,12,17,27,28}, 630 s) —
            // PAS merge_hp : un bake t4 heriterait des cles aux semantiques d'engine_ag
            // (stagnation_limit...) que imp_v4 parse autrement.
            // Pire cas temps : A ~450 s + B ~630 s = ~1 080 s < mur 1 800 s.
            //   Q = 250 000 (8/32) => plafond prouve ATTEINT (0,21 par A ; 28 par B) ;
            //   Q = 218 750 => B n'a pas reproduit 28 en contexte portfolio (A preserve).
            let solved = std::cell::Cell::new(false);
            let cb = |s: &Solution| {
                if !s.variables.is_empty()
                    && challenge.clauses.iter().all(|cl| {
                        cl.iter().any(|&l| {
                            let v = (l.abs() - 1) as usize;
                            if l > 0 { s.variables[v] } else { !s.variables[v] }
                        })
                    })
                {
                    solved.set(true);
                }
                save_solution(s)
            };
            // i38 : ORDRE INVERSE apres l'echec d'i37 (B jamais execute apres A : duree
            // strictement egale a A-seul sur 3 benchs 34839/34841/34846, fuel refute par
            // doublement sans effet, sondes println muettes sur ce chemin). B D'ABORD a
            // budget entier (replique du bench 34816 par construction), puis A sur echec
            // verifie. Chaque moteur seede son PROPRE RNG depuis challenge.seed => les
            // trajectoires par nonce sont independantes de l'ordre ; l'union A∪B est
            // garantie par construction. Pire cas temps ~630+325=955 s < mur.
            let mut hp_b = Map::new();
            hp_b.insert("max_fuel_high".to_string(), u(get_u64(hyperparameters, "t4_fuel_b", 400000000000)));
            engine_iv::solve(challenge, &cb, &Some(hp_b))?;
            if !solved.get() {
                engine_ag::solve(challenge, &cb, hyperparameters)?;
            }
            Ok(())
        }
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
            // === t5/i14 PROMU (17/08) — GAIN DE QUALITE : 93 750 -> 125 000 (+33 %) ===
            // Plan factoriel 2x2 complet sur t5 (n=7 500) :
            //           155 G          315 G
            //   engine_o  3/32  294 s   3/32   761 s   (sature : le budget ne lui rend RIEN)
            //   engine_d  1/32  412 s   4/32   830 s   (exploite le budget : +3 nonces)
            // et 630 G ne rend rien de plus (4/32, 1 637 s) => 315 G est l'OPTIMUM, pas un point
            // sur une pente. Le gain est une INTERACTION (moteur x budget) : ni `engine_d` seul
            // (1/32 a 155 G), ni le budget seul (`engine_o` @315 G = 3/32) ne l'expliquent.
            // Rejets de controle : `engine_ag` sur t5 = 1/32 (i13) ; `engine_d` @315 G sur t4
            // = 5/32 contre 7/32 (i34) => la fenetre est propre au couple, elle ne se transporte pas.
            // ⚠️ COUT : 829,6 s contre 294,0 s = x2,8. Assume : la DIRECTIVE cherche la Q.
            // Isolation de track : seul le bras (7500, 32002) change ; t1/t3/t4/t38 intacts.
            // === t5/i19 : PORTFOLIO sur t5 — cible le nonce #18 (prouve SAT, jamais resolu ici) ===
            // Meme mecanisme que t1/i37 (promu, plafond atteint). La trajectoire B (init perturbee
            // i26) n'a JAMAIS ete mesuree sur t5 : sa MAP est inconnue. Budget B prudent 150e9
            // modele (~360 G reels) pour rester loin du mur harness (pire nonce ~1 410 G ≈ 1 115 s).
            //   Q = 156 250 ⇒ #18 attrape = 2e plafond prouve atteint ;
            //   Q = 125 000 ⇒ B-t5 n'attrape pas #18 a ce budget (A preserve par construction).
            let solved = std::cell::Cell::new(false);
            let cb = |s: &Solution| {
                if !s.variables.is_empty()
                    && challenge.clauses.iter().all(|cl| {
                        cl.iter().any(|&l| {
                            let v = (l.abs() - 1) as usize;
                            if l > 0 { s.variables[v] } else { !s.variables[v] }
                        })
                    })
                {
                    solved.set(true);
                }
                save_solution(s)
            };
            let hp_a = with_fuel(hyperparameters, "t5_fuel_a", 315000000000);
            engine_d::solve(challenge, &cb, &hp_a)?;
            if !solved.get() {
                // i20 : le run B perturbe (i19) n'attrapait pas #18. La MAP de MON PROPRE bench
                // i18 (job 34608, v5462) le revelait : **engine_ag @ 315e9 sur t5 resout {18, 22}**.
                // La trajectoire vers #18 etait deja dans le binaire — run C = engine_ag.
                // Pire nonce ~A(1 050 G)+C(~900 G) ≈ 1 950 G ≈ 1 540 s < mur 1 800 s.
                let hp_c = with_fuel(hyperparameters, "t5_fuel_c", 315000000000);
                engine_ag::solve(challenge, &cb, &hp_c)?;
            }
            Ok(())
        }
        (100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
    }
}

pub fn help() {
    println!("sat_hybrid_v2_v2 - per-track SAT solver");
}
