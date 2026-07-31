use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod t51_engine;
pub mod t52_engine;
pub mod t53_engine;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

/// P-V7-7 — tag de build inerte (0 coût, jamais lu par le solveur). Répond à
/// « quel `mod.rs` est réellement dans ce `.so` ? » AVANT de dépenser un bench :
/// STALE avait été lié à un engine FRAIS — le diff de bake ne changeant ni la taille
/// ni le nombre de lignes du fichier, tout cache arbitrant par `(taille, mtime)` y
/// était aveugle. ⛔ À bumper à CHAQUE bake (un tag figé ne prouve plus rien).
///
/// 🔴 **i11/4133 = LA PREUVE VÉCUE DE CET AVERTISSEMENT.** i11 a baké le bon couple
/// (`esm=27` + `prune=2000`) mais **n'a PAS bumpé ce tag** ⇒ son `.so` (v4419) portait
/// `t53_i10_...`, indistinguable de v4413, et personne n'a pu voir avant le bench que
/// le binaire était faux. Symptôme mesuré : v4419 rend un Q **BYTE-IDENTIQUE**
/// `3 671 472 @ ~89 s` pour **4 `hp_json` différents** (`{}`, `{esm:27,prune:2000}`,
/// `{prune:1400}`, `{prune:3000}` — jobs 25546-25549), alors que le MÊME
/// `{esm:27,prune:2000}` donne `3 695 007 @ 153,03 s` sur v4413 (job 25524). Un binaire
/// insensible à ses propres HP = binaire figé ⇒ **mesure VOID, pas une réfutation**
/// (cf `feedback_identical_q_across_hp_means_dead_path`).
/// ⇒ **PROTOCOLE OBLIGATOIRE avant tout bench de bake** :
///   `strings -a <so> | grep TITANV7_BUILD=` DOIT rendre le tag de CETTE iter.
///   Si le tag est celui de l'iter précédente ⇒ NE PAS BENCHER, rebuild.
#[used]
#[no_mangle]
pub static TITANV7_BUILD_TAG: [u8; 40] = *b"TITANV7_BUILD=t53_i12_bake_esm27_prune2k";

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }
fn b(v: bool) -> Value { Value::Bool(v) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Baked HP = the iter's own baked defaults merged with the winning bench override,
    // so hp_json={} reproduces the winning per-track Q. User HP always win (merge_hp).
    match challenge.num_batteries {
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("emp_tail_nodes", self::n(1)),
                ("soc_levels", self::n(101)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(25)),
                ("ternary_iters", self::n(25)),
                ("convergence_tol", f(1e-4)),
                ("k_clusters", self::n(80)),
                ("deflator_iters", self::n(15)),
                ("lp_refine_sweeps", self::n(3)),
                ("cg_iters", self::n(20)),
                ("use_lp", b(true)),
                ("use_sdp", b(true)),
                ("use_cg", b(true)),
                ("network_derating", f(1.00)),
                ("use_analytical_pricing", b(true)),
                ("use_pce_affine_recourse", b(true)),
                ("use_emp_scenarios", b(true)),
                ("emp_tail_scale", f(2.0)),
                ("use_regret_weights", b(true)),
                ("regret_weight_mode", self::n(2)),
                ("emp_tail_nodes", self::n(0)),
            ]);
            track_t49::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("ct_ref_kappa", f(0.2)),
                ("soc_levels", self::n(101)),
                ("action_grid", self::n(20)),
                ("asca_iters", self::n(30)),
                ("ternary_iters", self::n(25)),
                ("deflator_iters", self::n(50)),
                ("network_derating", f(0.35)),
                ("max_admm_iters", self::n(10)),
                ("lp_total_pivots", self::n(15000)),
                ("dw_total_pivot_budget", self::n(8000)),
                ("lns_lp_pivots_total", self::n(6000)),
                ("use_lp", b(true)),
                ("use_dw", b(true)),
                ("use_lns", b(true)),
                ("use_kkt", b(true)),
                ("ct_step_eta", f(1.0)),
                ("dp_rho_jump", f(0.015)),
                ("ct_gdd_alpha", f(1.0)),
                ("lmp_threshold", f(0.60)),
                ("anticipate_lmp", b(true)),
                ("lmp_premium_scale", f(1.2)),
                ("use_primal_refine", b(true)),
                ("dw_mu_damping_alpha", f(0.9)),
                ("dw_boxstep_delta", f(0.25)),
                ("premium_shape_gamma", f(4.5)),
                ("use_cos_weights", b(true)),
                ("cos_line_weight_scale", f(2.0)),
                ("line_weight_w_min", f(0.5)),
                ("line_weight_w_max", f(2.0)),
                ("use_cos_cs_weights", b(false)),
                ("cos_alpha_under", f(1.0)),
                ("use_lmp_premiums_kkt", b(true)),
                ("use_ct_adaptive_per_line", b(true)),
                ("use_ptdf_constraint_tracking", b(true)),
                ("lp_obj_mode", self::n(2)),
                // `3` = geometric, finest segment AT the power rating; `r` measured, not
                // assumed: 8 points 32/32 on v4397 give a broad interior max on [2.6, 4.2]
                // (1.0=uniform 583020 / 2.0 585286 / 2.6 586108 / 3.0 585710 / 3.5 586194 /
                //  4.2 586151 / 6.0 583431). Time-neutral: the LP dimensions depend on the
                // segment COUNT only, so every point above measured 5.0 s.
                ("lp_pwl_spacing", self::n(3)),
                ("lp_pwl_geo_ratio", f(3.5)),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(hyperparameters, vec![
                ("lahc_init_alpha_span", f(0.0)),
                ("dp_soc_levels", self::n(97)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(80)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(2)),
                ("lookahead_horizon", self::n(24)),
                ("rh_stride", self::n(3)),
                ("pga_beta_end", f(0.6)),
                ("use_momentum", b(true)),
                ("use_bb_clamps", b(true)),
                ("soc_ref_lambda", f(0.05)),
                ("use_admm_solver", b(false)),
                ("use_cosine_beta", b(true)),
                ("soc_ref_dyn_stride", self::n(6)),
                ("joint_triplet_top_k", self::n(15)),
                ("use_rolling_horizon", b(true)),
                ("joint_triplet_budget", self::n(300)),
                ("use_joint_pair_polish", b(true)),
                ("use_joint_triplet_polish", b(true)),
                ("use_arb_seed", b(true)),
                ("arb_pct", self::n(75)),
                ("arb_inverse", b(true)),
                ("use_std_arb_third", b(true)),
                ("arb_pct_third", self::n(0)),
                ("use_da_arb_third", b(true)),
                ("use_rollout_additive", b(true)),
                ("rollout_window", self::n(12)),
                ("use_rollout_additive_2", b(true)),
                ("rollout_window_2", self::n(4)),
                ("use_rollout_additive_3", b(true)),
                ("rollout_window_3", self::n(2)),
                ("use_rollout_additive_4", b(true)),
                ("rollout_window_4", self::n(1)),
                ("use_basin_hop", b(true)),
                ("basin_hop_scale", f(0.05)),
                ("basin_hop_k", self::n(4)),
                ("pair_lahc_lh", self::n(5)),
                ("lahc_init_alpha_span", f(0.0)),
            ]);
            t51_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(hyperparameters, vec![
                ("arb_diversity_inverse", b(true)),
                ("use_arb_diversity_pair", b(true)),
                ("use_arb_diversity_seed", b(true)),
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(75)),
                ("grad_ls_iters", self::n(12)),
                ("bisect_iters", self::n(30)),
                // i30 KEPT: coord_polish 0→1 = +547Q prouvé hp_json.
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
                
                // hp_json on ver3797): 0.10=2387727(+530) 0.15=2386612 0.20=2385558 0.25=2387197(CTRL)
                // 0.40=2386806 0.55=2386632 0.70=2385293. t53 optimum 0.70 (P41) does NOT transfer
                // to t52 (n<=80, clusters=4): axis flat/rugged, deterministic peak at 0.10.
                ("cwv_lambda", f(0.10)),
                
                
                // (+5575, +0.234%). 0.25=2365859 0.35=+310 0.40=+4666 [0.50=+5575] 0.65=+3801
                // 0.85=-2239 1.00=CTRL 1.25=-4462 1.50=-6629. i80's cwv=0.10 (weak coupling) slid
                // the optimum down from 1.0; t53's 1.5 (P43, cwv=0.70) does NOT transfer.
                ("ct_step_eta", f(0.50)),
                ("use_dp_seed", b(false)),
                ("use_ptdf_ct", b(true)),
                ("ct_ref_kappa", f(0.0)),
                ("cwv_clusters", self::n(4)),
                ("use_momentum", b(true)),
                ("lmp_threshold", f(0.5)),
                ("lr_growth_cap", f(1.025)),
                ("use_bb_clamps", b(true)),
                ("use_zero_seed", b(false)),
                ("anticipate_lmp", b(true)),
                ("cwv_agg_levels", self::n(65)),
                ("use_cosine_beta", b(true)),
                ("use_composite_wv", b(true)),
                ("use_pwl_value_dp", b(false)),
                ("joint_pair_budget", self::n(1024)),
                ("lmp_premium_scale", f(2.0)),
                ("pwl_max_breakpoints", self::n(64)),
                ("congestion_grid_alpha", f(0.0)),
                ("use_joint_pair_polish", b(true)),
                // i31 combine P23: gram-incremental projection on top of P27 coord_polish=1.
                // iso-geometry (Q bit-exact vs i30), cuts wall-time ~36% (P23 measured on i25).
                ("use_gram_incremental_proj", b(true)),
                ("resync_period", self::n(16)),
                // i34 BB adaptive step-size sentinel: false=iso-i31 (CTRL). Sweep via hp_json.
                ("use_bb_step", b(false)),
                // i35 L7 disagg-order sentinel: 0=natural (CTRL iso-i31). Sweep {1,2,3} via hp_json.
                ("disagg_order_mode", self::n(0)),
                // i36 KEPT γ=4.0 (+28,436Q DOMINANT 2 AXES). i37 bake: 1.0→4.0.
                
                
                // unimodal (2.5..5.0 + fine 3.25/3.75), time-neutral 11.0s. Precedent t50/t53 gamma-peak shifts with stack.
                ("premium_shape_gamma", f(3.5)),
                // i45 KEPT P31 virtual-queue ct_vq_v=100.0 (+18,046Q +0.71%). i46 bake.
                ("ct_vq_v", f(100.0)),
            ]);
            t52_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                // i6/4102 BAKE — P21 rallumé. Le noyau de projection incrémental (Gram
                // `5abc8509`) est appelé PAR SEED : son économie est multipliée par la taille
                // du pool. Mesuré 32/32 0-inv m13 v4386 (0 build, pur override) :
                // {gram:true} à pool INCHANGÉ = 3,679,085 @76.52s vs CTRL baké i5
                // 3,677,479 @149.02s ⇒ **+1,606 Q ET −48,7 % de temps**.
                // ⚠️ Le « −720 Q du dedup » de la roadmap est RÉFUTÉ par cette mesure : il
                // venait d'une comparaison INTER-LIGNÉES (v6/i97 gram=true 1-seed vs v7
                // gram=false 1-seed) qui confondait le gram avec la taille du pool.
                ("use_gram_incremental_proj", b(true)),
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("premium_shape_gamma", f(5.0)),
                ("grad_outer_iters", self::n(150)),
                ("grad_ls_iters", self::n(12)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
                // i5/4093 BAKE multi-start: pool 2-seed {target, zero} (seed_order_mode=1).
                // i4/4089 ablation 32/32 0-inv m13: {target}=3,671,472@90.0s (CTRL) ;
                // {target,dp_seed}=3,672,375@156.5s ; {target,zero}=3,677,479@151.0s ;
                // {dp_seed,zero}=3,678,523@153.5s. Mode 1 retenu: +6,007 Q vs baseline,
                // mode 2 ne le bat que de +1,044 (0.028%, 17/13 en stats d'ordre) en droppant
                // `target` = non-monotone => marge non nette. Temps 151s << gate SOTA 184.03s.
                // i6/4102 BAKE — pool COMPLET 3-seed {target, zero, dp_seed}. Le +1,515 Q du
                // 3ᵉ seed était déjà mesuré (i3 job 25263 = 3,678,994) mais refusé à 216.53s
                // (> gate 184.03s). Avec le gram, le coût marginal d'un seed tombe de ~62.5s
                // à ~15s ⇒ le pool complet ENTRE dans le gate : 3,681,405 @106.52s
                // (job 25381, 32/32 0-inv m13) = +3,926 Q ET −42,5s (−28,5 %) vs i5.
                // On AJOUTE un seed sans retirer `target` ⇒ argmax total_step_value reste
                // mode ne fait plus que permuter un ensemble identique).
                ("num_seeds", self::n(3)),
                ("seed_order_mode", self::n(1)),
                // i7/4109 BAKE — seeds ADDITIFS 4e/5e portés de `t51_engine.rs`
                // (l. 2744-2799) : `arb` (seuil de prix binaire {lo,hi}, congestion-
                // agnostique ⇒ basin que ni `target` ni `dp_seed` n'atteignent,
                // d'action, sans information de prix). Le pool interne
                // `[target, dp_seed, zero]` était ÉPUISÉ à 3/3 ; le gram ayant fait
                // tomber le coût marginal d'un seed de ~62,5 s à ~15 s, les 77,5 s de
                // marge sous le gate 184,03 s s'achètent en BASINS, pas en grille
                // (les 2 axes du cube DP sont morts, dead-list SQL).
                // Bitmask : 0=aucun (bit-exact i6), 1=arb, 2=obl, 3=les deux.
                // On AJOUTE sans retirer ⇒ l'argmax `total_step_value` reste
                // monotone-sûr (`1d30845d`).
                //
                // i9/4121 BAKE — 6ᵉ seed `rollout` (bit 8, port de
                // `t51_engine.rs::use_rollout_additive` l. 2925-2946 : seuil P75 par
                // nœud sur la fenêtre day-ahead AVANT `[t, t+rollout_window)`). C'est
                // le seul constructeur que t51 ait promu en baseline PERMANENTE
                // (job 25454 = 3,691,147) mais à 206,53 s = HORS gate 184,03 s.
                // ⛔ `ramp` (bit 4) écarté : −988 Q mesurés (job 25453), dilution de
                // pool conforme à `b5b1b152`. Bitmask final : 11 = arb|obl|rollout.
                //
                // i11/4133 BAKE — 7ᵉ seed `arb_inverse` (bit 16, l. 3334 :
                // `discharge_when_high = !arb_inverse`, la POLARITÉ EXACTEMENT
                // INVERSÉE d'`arb`). Même argument de basin que `arb` (`813c100d`) :
                // un seuil de prix congestion-agnostique atteint un bassin que ni
                // `target` ni `dp_seed` ne visitent — et sa polarité opposée en
                // atteint un second, disjoint. Mesuré par i10 à **+3 297 Q**
                // (job 25518 `{esm:27}` = 3 694 678) mais à **214,53 s**, soit
                // 30,5 s HORS du gate 184,03 s. C'est le prune ci-dessous qui paie
                // ces 30,5 s (GOLDEN RULE `ed934a85`, 3ᵉ application consécutive).
                // ⛔ `arb_p75` (bit 32) écarté : `esm:43` = 3 681 387 (job 25517),
                // −8 000 Q — même polarité qu'`arb`, seuil redondant, dead-list SQL
                // `brick:extra_seed_arb_p75`. Bitmask final : 27 = arb|obl|rollout|arb_inverse.
                ("extra_seed_mode", self::n(27)),
                // i9/4121 BAKE — coupe du canal de SURPLUS du pool d'itérations
                // (`iter_pool`, écrit quand t53 était single-start). Donneurs (seeds
                // proches : ils sortent sur `g_norm < 1e-9`) et receveurs (seeds
                // lointains : ils ne convergent jamais) sont systématiquement les mêmes
                // deux groupes ⇒ le pool DOUBLE le budget des seeds exactement les plus
                // chers, pour un Q que la dead-list `hp:grad_outer_iters` dit déjà
                // nuisible au-delà de go=150. Le couper vaut ≈8,2 s PAR seed lointain :
                // −16,5 s à 2 (i8 job 25451), −26,5 s à 3 avec `rollout`.
                // C'est ce qui fait ENTRER le 6ᵉ seed sous le gate :
                // 206,53 s → 180,03 s à Q intact (3,691,381 ≥ 3,691,147).
                // ⚠️ Le budget de BASE des seeds cœur reste à `grad_outer_iters`=150
                // (son pic prouvé) — ce HP ne touche QUE le surplus, jamais la base.
                // GOLDEN RULE `ed934a85` : le levier temps FINANCE le levier Q.
                ("pga_pool_claim_pct", self::n(0)),
                // i11/4133 BAKE — DOMINANCE PRUNE du multi-start (`t53_engine.rs`
                // l. 2342 : on abandonne la trajectoire PGA d'un seed dès que la borne
                // OPTIMISTE `best_value + (p/100)·last_gain·(total_limit − iters_run)`
                // passe sous l'`incumbent` = la valeur déjà sécurisée par un seed
                // flow-faisable PRÉCÉDENT du MÊME timestep). C'est le levier TEMPS qui
                // finance le 7ᵉ seed : 214,53 s → **153,03 s**.
                //
                // ⚠️ **LA VALEUR EST NON MONOTONE — `2000` est un GENOU MESURÉ**, pas
                // « le plus grand possible ». Treillis i10, binaire UNIQUE v4413, m13,
                // 32/32 0-inv, mêmes nonces (`{esm:27, prune:X}`) :
                //   X=200  → 3 689 386 @114,52 s  (sous la baseline : prune trop AGRESSIF)
                //   X=800  → 3 692 973 @135,02 s  ✅ passe le gate
                //   X=2000 → **3 695 007 @153,03 s**  ✅ **RETENU** (Q max du treillis)
                //   X=6000 → 3 690 790 @162,03 s  (sous la baseline)
                //   X=0    → 3 694 678 @214,53 s  (pas de prune, HORS gate)
                // `p/100` = le facteur de SLACK sur la borne : p=2000 ⇒ ×20, donc on ne
                // coupe qu'un seed massivement sans espoir. Le prune SEUL (sans le 7ᵉ
                // seed) est Q-destructif : `{prune:200}` = 3 683 799, −7 582 Q
                // (job 25515) ⇒ **les deux valeurs ne se bakent QUE PAR PAIRE.**
                // La non-monotonie est attendue : un seed coupé renvoie quand même son
                // propre `best_action` (jamais un état vide), mais avec une valeur plus
                // basse ⇒ l'argmax `total_step_value` du timestep peut basculer et
                // dérouter la trajectoire gloutonne en aval. Le Q d'un `p` donné est donc
                // MESURÉ, pas déductible — ⛔ ne pas extrapoler, re-bencher tout nouveau `p`.
                //
                // ⚠️ Ce n'est PAS l'early-exit réfuté en juin (`0255dcb4`/`1e23d421`) :
                // cette réfutation portait sur le canal `iter_pool_donate/claim` qui
                // RÉINJECTAIT les itérations économisées ailleurs (gain annulé) ; i9 a
                // baké `pga_pool_claim_pct = 0`, donc ce canal est COUPÉ et l'itération
                // prunée est DÉTRUITE ⇒ l'économie est du vrai wall-clock.
                ("pga_dom_prune_pct", self::n(2000)),
                ("cwv_lambda", f(0.7)),
                ("ct_step_eta", f(1.5)),
                ("use_ptdf_ct", b(true)),
                ("ct_gdd_alpha", f(0.0)),
                ("ct_ref_kappa", f(0.0)),
                ("use_momentum", b(true)),
                ("anticipate_lmp", b(true)),
                ("use_cosine_beta", b(false)),
                ("use_composite_wv", b(true)),
                ("joint_pair_budget", self::n(5500)),
                ("use_joint_pair_polish", b(true)),
                ("proj_relax", f(1.0)),
                ("use_gram_incremental_proj", b(true)),
                ("use_basin_hop", b(true)),
                ("basin_hop_scale", f(0.05)),
                ("basin_hop_k", self::n(4)),
                ("pair_alpha_interval", b(true)),
                ("pair_alpha_max_passes", self::n(2)),
                ("joint_pair_early_exit_k", self::n(2000)),
                ("ct_vq_v", f(100.0)),
                ("use_dp_value_shift", b(true)),
                ("dp_value_curv_coef", f(0.65)),
                ("oco_full_rebuild", b(true)),
            ]);
            t53_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("the base solver: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("the base solver");
}
