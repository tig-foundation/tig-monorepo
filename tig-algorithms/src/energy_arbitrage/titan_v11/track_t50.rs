use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
    use std::sync::Arc;
    use tig_challenges::energy_arbitrage::*;

    #[derive(Clone, Debug)]
    pub struct TrackHp {
        pub soc_levels: usize,
        pub action_grid: usize,
        pub asca_iters: usize,
        pub ternary_iters: usize,
        pub convergence_tol: f64,
        pub anticipate_lmp: bool,
        pub lmp_threshold: f64,
        pub lmp_premium_scale: f64,
        pub jump_premium: f64,
        pub prune_ratio: f64,
        pub deflator_iters: usize,
        pub flow_margin: f64,
        /// Relative tolerance for the last-step LP-refinement feasibility acceptance.
        /// MUST stay <= framework EPS_FLOW (1e-6) to avoid the "schedule vide" bug
        /// (accepting actions the framework's verify_flows rejects). Default 1e-6.
        pub flow_feas_tol: f64,
        pub network_derating: f64,
        // i21 — BORNE D'ACTION DU DP, DÉRATAGE ÉTAT-DÉPENDANT.
        //
        // `network_derating` (0,22) borne la puissance que le DP croit délivrable :
        // `max_pwr = nameplate × 0,22`, un scalaire CONSTANT EN t ET EN b. Or la congestion
        // de t50 est ÉPISODIQUE et le reste du fichier calcule déjà un signal de congestion
        // riche par (t, b) (`expected_premiums`, `mu`) que cette borne n'utilise PAS.
        // Le DP croit donc que chaque batterie est bridée à 22 % de sa puissance MÊME aux
        // pas de temps totalement décongestionnés.
        //
        // Preuve pré-code que le scalaire est un PROXY d'une grandeur état-dépendante :
        // son optimum SE DÉPLACE quand on change les flux (0,25 → 0,35 sous GDD-EXP,
        // memory `c6a47d49`) — un vrai paramètre physique ne bougerait pas. Et la
        // distribution de f_exo/limit sur t50 est BIMODALE (memory `09d54049`), donc mal
        // résumée par une constante.
        //
        // ─── i23 — DEUX CORRECTIONS DURES À i21 (qui a mesuré le MAUVAIS SITE) ───────────
        //
        // (1) SITE. i21 a greffé les 3 modes sur `build_dp_stochastic_for_battery` (site 3)
        //     en affirmant en commentaire que c'était « le seul site vivant ». Les 3 modes y
        //     sont sortis BIT-EXACTS avec le CTRL, Y COMPRIS à `nd_free=0.05` (dérate ×20) :
        //     le site 3 n'influence PAS Q. La lecture d'i21 (« donc le levier vit dans
        //     `build_dp_with_mu*` ») est la bonne conclusion mais pour la mauvaise raison ;
        //     la vraie chaîne, lue dans le code (0 bench) :
        //       `skip_dead_prepass = use_sdp && dp_prepass_mode == 1` — les DEUX sont bakés
        //       (`use_sdp: true`, `dp_prepass_mode: 1`) ⇒ sur le chemin FINAL, `build_dp_with_mu*`
        //       n'est JAMAIS appelée. Elle n'est atteinte que par le PRESCREEN DW
        //       (`dw_prescreen_mode: 3` baké ⇒ `hp_ps.use_sdp = false` ⇒ `skip_dead_prepass`
        //       faux, L2852-2879), et là avec `mu = zero_mu`.
        //     ⇒ Les −133 528 Q de `{network_derating:0.05}` ne peuvent transiter QUE par les
        //     sites 1/2, donc par `prescreen_binding_lines` : `network_derating` n'est PAS
        //     une borne de dispatch, c'est un **réglage de fidélité du prescreen**, qui décide
        //     QUELLES lignes le maître DW contraint. C'est la prédiction falsifiable d'i23,
        //     pré-enregistrée et mesurée par les arms `nd_scalar_mask` (ci-dessous).
        //
        // (2) PORTE INATTEIGNABLE. `nd_gate` d'i21 était un seuil ABSOLU sur
        //     `max_l |f_exo(t,l)|/limit_l`, clampé à 10.0 — or la row dead
        //     `engine:action_aware_premium` dit que ce ratio reste SOUS 1,0 sur t50. L'arm
        //     `nd_gate:99.0` a donc dégénéré en constante (`d = nd_free` partout) au lieu de
        //     tester une porte. i23 rend le seuil **RELATIF** : `nd_gate` est un QUANTILE de
        //     la distribution empirique de `ratio[t]` ⇒ la porte est atteignable PAR
        //     CONSTRUCTION, quelle que soit l'échelle absolue des flux.
        //     Cf memory `unreachable-gate-is-a-rediscovery-magnet`.
        //
        // `nd_mode` : 0 = CTRL (bit-exact avec i15), 1 = porte bimodale (quantile),
        //   2 = rampe continue, 3 = dérate par EXPOSITION PTDF (spatial, pas temporel).
        // `nd_free` = dérate appliqué aux pas/batteries DÉCONGESTIONNÉS (1.0 = pleine puissance).
        // `nd_gate` = QUANTILE dans [0,1] de `ratio[t]` (0.5 = médiane).
        // `nd_scalar_mask` : attribution du SCALAIRE `network_derating` par site.
        //   0 = tous les sites (= i15 mot pour mot) · 1 = sites 1+2 seuls (site 3 → 1.0)
        //   · 2 = site 3 seul (sites 1+2 → 1.0). Sert de TÉMOIN DE VIVACITÉ PAR SITE.
        pub nd_mode: usize,
        pub nd_free: f64,
        pub nd_gate: f64,
        pub nd_probe: usize,
        pub nd_scalar_mask: usize,
        pub dual_iters: usize,
        pub da_step_size: f64,
        pub ldd_iters: usize,
        pub ldd_step_size: f64,
        pub use_kkt: bool,
        pub kkt_cong_threshold: f64,
        pub kkt_price_scale: f64,
        pub max_admm_iters: usize,
        pub admm_rho: f64,
        pub admm_primal_tol: f64,
        pub use_lp: bool,
        pub dantzig_in_dw: bool,
        pub dantzig_in_lns: bool,
        pub dantzig_in_kkt: bool,
        pub dantzig_bland_degen: bool,
        pub lp_soft_lambda: f64,
        pub lp_per_call_pivots: usize,  
        pub lp_total_pivots: usize,     
        pub use_policy: bool,
        pub use_warmstart: bool,
        pub use_mpc: bool,
        pub mpc_horizon: usize,
        pub mpc_pivot_budget: usize,
        pub use_dw: bool,
        pub dw_iters: usize,
        pub dw_max_lines: usize,
        pub dw_max_cols_per_batt: usize,
        pub dw_pivot_budget_per_solve: usize,
        pub dw_total_pivot_budget: usize,
        pub use_dw_prescreen: bool,
        pub use_lns: bool,
        pub lns_cg_iters: usize,
        pub lns_cg_column_limit: usize,
        pub lns_max_lines: usize,
        pub lns_lp_pivots_total: usize,
        pub use_pivot_reserve: bool,
        pub lp_max_lines: usize,
        pub use_parallel_dp: bool,
        pub use_sdp: bool,
        pub sdp_k: usize,
        /// i6 — élimination du prépasse DP déterministe mort.
        /// 0 = CTRL (comportement historique : le DP déterministe est construit PUIS
        ///     intégralement écrasé par le DP stochastique quand `use_sdp` est vrai) ;
        /// 1 = le prépasse déterministe est sauté quand `use_sdp` est vrai (défaut).
        /// Sans effet quand `use_sdp == false` : dans ce cas le prépasse EST le résultat.
        pub dp_prepass_mode: usize,
        /// i7 — FIDÉLITÉ DU DP DE PRESCREEN DW (site `dw_solve`, mu=0).
        /// Sa seule sortie est une liste de ≤ `dw_max_lines` INDICES DE LIGNES
        /// (`prescreen_binding_lines`) — c'est une SONDE, pas une valeur.
        /// 0 = CTRL (pleine résolution + stochastique = comportement historique) ;
        /// 1 = grille /2 (soc/2 plancher 31, action/2 plancher 15) — précédent en-code OCO ;
        /// 2 = sonde DÉTERMINISTE (`use_sdp=false`, grille pleine) — retire la quadrature ;
        /// 3 = grille 31/15 + déterministe — configuration EXACTE de la sonde ADMM du fichier.
        pub dw_prescreen_mode: usize,
        /// i9 — FORME DE L'ESPÉRANCE DE QUADRATURE dans `build_dp_stochastic_for_battery`.
        /// La boucle `for k in 0..num_q` est au CŒUR du nid le plus interne
        /// (num_t × soc_levels × (action_grid+1) × num_q). Or le seul terme qui dépend
        /// de `k` est `perturbation = scene_perturb[k]`, et il entre LINÉAIREMENT dans
        /// `revenue = u * (base + perturbation) * dt`. Tout le reste (`abs_u`, `tx`,
        /// `deg`, `v_next`, la branche `u > 0`) est INVARIANT en `k`.
        /// 0 = CTRL — code historique, invariants recalculés `num_q` fois ;
        /// 1 = HOIST — invariants sortis de la boucle `k`, ORDRE D'ACCUMULATION
        ///     INCHANGÉ ⇒ Q **bit-exact garanti par construction** ;
        /// 2 = COLLAPSE (défaut) — la boucle `k` est éliminée algébriquement :
        ///        Σ_k w_k·(u·dt·(base+p_k) − tx − deg + v_next)
        ///      = u·dt·(W·base + P̄) − W·(tx + deg) + W·v_next,
        ///      avec W = Σ_k w_k et P̄ = Σ_k w_k·p_k précalculés UNE FOIS par pas t.
        ///      Mathématiquement identique ; l'arrondi flottant peut différer d'1 ULP.
        pub sdp_expect_mode: usize,
        /// i10 — FORME D'ALLOCATION des nœuds de quadrature (même site que i9).
        /// Sous le collapse (`sdp_expect_mode == 2`) la boucle `k` a disparu, mais les
        /// nœuds étaient encore MATÉRIALISÉS en 3 `Vec<f64>` HEAP par (pas de temps ×
        /// batterie) : les 2 retours de `build_tail_mixture_quadrature` + `scene_perturb`.
        /// Or seuls deux SCALAIRES en sont extraits (`W = Σ w_k`, `P̄ = Σ w_k·p_k`).
        /// 0 = CTRL — état i9 exact : 3 `Vec` heap alloués/libérés à chaque pas ;
        /// 1 = INLINE (défaut) — nœuds/poids/perturbations en buffers PILE `[f64; 3]`,
        ///     zéro allocation heap dans la boucle `t`. Mêmes nœuds, mêmes poids,
        ///     MÊME ORDRE DE SOMMATION (gh2[0] → gh2[1] → jump) et mêmes branches de
        ///     garde (fallback `gh_quadrature(3)`) ⇒ Q **bit-exact par construction**.
        /// Le mode 0 reste le chemin des `sdp_expect_mode` 0/1, qui lisent les slices.
        pub sdp_alloc_mode: usize,
        /// i11 — RÈGLE D'ARRÊT DE LA COLUMN GENERATION dans `lns_dw_per_step`.
        /// Le bloc LNS pèse **1,5 s des 4,0 s du cycle** (étalon `use_lns=false`, job 26373)
        /// et les itérations CG 2-3 pèsent à elles seules **1,0 s** (étalon `lns_cg_iters=1`,
        /// job 26376 : 3,0 s / Q=586336 ≥ baseline) — 2× le quantum de métrologie.
        /// Le défaut structurel : la boucle CG ne teste PAS l'optimalité du pricing. Elle
        /// pousse une colonne dès qu'elle est NUMÉRIQUEMENT NEUVE (`|existing − u| ≥ 1e-4`)
        /// et reboucle (nouvel assemblage `a_mat` + nouveau solve simplexe) même quand la
        /// colonne ne peut PAS améliorer le maître. Le critère de Dantzig-Wolfe est le
        /// COÛT RÉDUIT : la colonne candidate de la batterie `b` vaut son Lagrangien
        /// `best_cand_val = max_u eval_profit_with_price(b, u, rt − cong_adj)` et n'améliore
        /// le maître que si `best_cand_val − σ_b > 0`, où `σ_b = duals[2*b]` est le dual de
        /// la contrainte de convexité `Σ_j λ_bj ≤ 1` (ligne `2*b`, cf assemblage :1508).
        /// 0 = CTRL — code historique (nouveauté numérique seule, pas de test d'optimalité) ;
        /// 1 = RC-GATE (défaut) — la colonne n'est poussée que si son coût réduit est
        ///     strictement positif ; sinon `added_any` reste faux et la boucle CG sort par
        ///     son `break` existant. C'est le test d'optimalité canonique de la CG ;
        /// 2 = SAT-SHORTCIRCUIT — si TOUTES les pools sont déjà au plafond
        ///     `lns_cg_column_limit`, la passe de pricing est un `continue` pour CHAQUE `b`
        ///     ⇒ `added_any == false` ⇒ `break`. On saute donc une passe dont la sortie est
        ///     PROUVÉE VIDE. Aucun état lu en aval n'est touché (`mu_prev` est assigné avant
        ///     et jamais relu après le break) ⇒ **Q bit-exact PAR CONSTRUCTION** ;
        /// 3 = les deux.
        pub lns_cg_stop_mode: usize,

        // ── i12 — ADMISSION GATE du bloc LNS (étage CONSOMMATEUR) ────────────────────
        /// Sonde (Q-neutre, `println!`) : 1 = journalise en fin de nonce le TAUX
        /// D'ACCEPTATION réel du bloc LNS **et** la précision CONTREFACTUELLE des 3
        /// prédicats candidats (« shadow ») — ils sont évalués mais JAMAIS appliqués.
        /// C'est l'étape MEASURE-FIRST : elle chiffre le gain de chaque famille et son
        /// coût en Q AVANT de laisser un prédicat décider quoi que ce soit.
        pub lns_probe: usize,
        /// 0 = CTRL (aucun gate, comportement historique bit-exact) ;
        /// 1 = F1 BORNE DE HEADROOM — `eval_profit` est SÉPARABLE par batterie et
        ///     `uncons_actions[b]` en est l'argmax non contraint ⇒
        ///     `Σ_b [f(uncons_b) − f(actions_b)]` MAJORE tout gain atteignable par le LNS.
        ///     Borne ≤ tol ⇒ `lns_p > base_p` est impossible ⇒ skip Q-exact ;
        /// 2 = F2 PERSISTANCE TEMPORELLE — après K rejets consécutifs, ne ré-essayer
        ///     qu'un pas sur `lns_gate_probe_period` (déterministe : compteur + index de
        ///     pas, aucun wall-clock) ;
        /// 3 = F3 REDONDANCE PRODUCTEUR — si le bloc LP a DÉJÀ déplacé l'incumbent à ce
        ///     pas, le couplage réseau a été exploité par un solveur exact et le LNS y
        ///     est redondant.
        /// Un skip rend `(actions, false)` = EXACTEMENT l'état aval d'un rejet
        /// (`used_lns=false` ⇒ `targeted_tight_line_polish` s'exécute comme au CTRL) :
        /// seuls les pas où le LNS aurait été ACCEPTÉ peuvent coûter du Q.
        pub lns_gate_mode: usize,
        pub lns_gate_headroom_tol: f64,
        pub lns_gate_streak_k: u64,
        pub lns_gate_probe_period: usize,

        // ── i13 — SEUIL DE MATÉRIALITÉ sur la boucle CG (étage MAGNITUDE) ─────────────
        /// Les deux étages voisins sont FERMÉS PAR MESURE :
        ///   * PRODUCTEUR (i11) : le RC-gate `rc > 1e-9` ne mord JAMAIS (Q bit-exact ×3)
        ///     ⇒ toutes les colonnes des iters CG 2-3 sont LÉGITIMES au sens de
        ///     Dantzig-Wolfe. `brick:lns_cg_pricing_optimality_test` = dead.
        ///   * CONSOMMATEUR (i12) : rétention MESURÉE = 81,25 % (2479/3051) ⇒ plafond
        ///     oracle `(1−0,8125)×1,5 s = 0,28 s` < quantum ⇒ 4 rows dead.
        /// Il ne reste qu'un étage : le consommateur ACCEPTE une amélioration de
        /// MAGNITUDE NULLE. i11 mesure `lns_cg_iters` 3→1 = **−1,0 s pour +142 Q (bruit)**
        /// ⇒ les colonnes des iters 2-3 sont légitimes, acceptées, et SANS VALEUR en Q.
        /// i11 testait l'EXISTENCE d'une colonne améliorante (toujours vraie) ; ici on
        /// teste sa MATÉRIALITÉ — l'axe exact que le RC-gate ne peut pas voir.
        ///
        /// Sonde SHADOW (Q-neutre, `println!` — `eprintln!` est MUET sur c008) : 1 =
        /// évalue les 3 mécanismes × 3 seuils SANS JAMAIS les appliquer, et journalise
        /// pour chacun (fire = itérations CG qui auraient été SAUTÉES = temps récupéré ;
        /// bad = appels où `best_obj` s'améliorait ENCORE après le point d'arrêt = Q
        /// menacé). 9 familles jugées pour 1 bench, à coût Q nul.
        pub lns_cg_probe: usize,
        /// Mécanisme APPLIQUÉ (0 = CTRL bit-exact, aucun calcul supplémentaire) :
        /// 1 = M1 GAP DE DUALITÉ — `Σ_b max(0, rc_b) ≤ δ·|best_obj|`. `rc_b` est le coût
        ///     réduit de la colonne candidate (`eval_action(u*) − σ_b`, cf i11). En DW la
        ///     somme des coûts réduits positifs MAJORE l'amélioration restante du maître
        ///     restreint ⇒ c'est une BORNE sur ce qui reste à gagner, pas une heuristique.
        ///     Seul mécanisme qui peut mordre dès la FIN de l'itération 0 ⇒ seul capable
        ///     de récupérer les 2 quanta complets mesurés par `lns_cg_iters=1`.
        /// 2 = M2 AMÉLIORATION MARGINALE RÉALISÉE — `(obj_k − obj_{k−1}) ≤ δ·|obj_{k−1}|`.
        ///     Formulation littérale de la roadmap. Ne peut mordre qu'à la fin de
        ///     l'itération 1 (il lui faut 2 points) ⇒ plafond ≈ 1 quantum, pas 2.
        /// 3 = M3 DÉPLACEMENT PRIMAL DES COLONNES — `max_b |u*_b − actions_t[b]| / span_b
        ///     ≤ δ`. Signal d'espace PRIMAL (déplacement) et non DUAL (gap) : une colonne
        ///     qui coïncide avec l'incumbent décodé ne peut pas déplacer le barycentre.
        pub lns_cg_mat_mode: usize,
        /// Seuil RELATIF δ du mécanisme appliqué. Sans dimension dans les 3 cas.
        pub lns_cg_mat_delta: f64,

        // ── i14 — OBJECTIF de la boucle CG : PONDÉRATION DU SURROGAT ──────────────────
        /// ⛔ CORRECTION DE PRÉMISSE (directive 2026-08-04T151008Z). La directive affirme
        /// que « la CG optimise un profit INSTANTANÉ pur » et prescrit d'AJOUTER
        /// `dp_lam[b]·Δsoc(u)` au coût des colonnes. **C'est FAUX par lecture** :
        /// `eval_profit_with_price` (:3970, appelée au coût des colonnes ET dans le
        /// pricing) rend déjà
        ///     `profit + V(t+1, soc'(u))`
        /// où `V = ca.dp[b][t_next][·]` est INTERPOLÉE linéairement en `soc'`. Le terme de
        /// continuation est donc DÉJÀ AU SITE, à l'ordre 0 (valeur exacte), et
        /// `dp_lambda` n'en est que la PENTE. L'ajouter reviendrait à le COMPTER DEUX FOIS
        /// (cf `feedback_measure_accepted_arity_not_offered_arity` : vérifier que le
        /// mécanisme du papier n'est pas déjà au site).
        ///
        /// Ce qui SURVIT du diagnostic d'i13, et qui est ici le vrai énoncé : la boucle CG
        /// garde `best_actions` au MAXIMUM de `eval_profit` (:1898 et :2015 — la vraie
        /// valeur au point décodé, pas la valeur LP) ⇒ **plus d'itérations CG améliorent
        /// STRICTEMENT `Σ_b (profit_b + V̂_b)` et pourtant Q BAISSE de 142**. C'est une
        /// preuve par dominance monotone que le désalignement n'est ni dans le décodage
        /// ni dans la sélection : il est dans **le surrogat `V̂` lui-même**, qui est un DP
        /// PAR BATTERIE construit SANS le réseau. Sur un track « congested », `V̂`
        /// sur-évalue les états de SoC que la congestion future rendra inexploitables ⇒
        /// sur-optimiser `profit + V̂` = OVERFIT du surrogat (memory `a7a7271e` : « DP can
        /// sometimes be MORE profitable due to the myopic nature of optimal control »).
        ///
        /// 3 familles DISTINCTES, toutes NEUTRES à leur défaut (⇒ CTRL bit-exact) :
        ///
        /// (A) `cg_cont_scale` — pondère la partie DIFFÉRENTIELLE du surrogat autour de
        ///     l'incumbent : `c_j ← profit_j + V̂_ref + w·(V̂_j − V̂_ref)`. Ancrer sur
        ///     `V̂_ref = V̂(u_base)` est essentiel : mettre `w·V̂_j` brut décalerait le
        ///     NIVEAU des colonnes et changerait l'arbitrage `Σλ ≤ 1` vs résidu-ancre,
        ///     ce qui n'est pas le mécanisme visé. `w = 1` ⇒ identité (chemin CTRL).
        ///     `w < 1` = « je fais moins confiance à V̂ » ; `w > 1` = l'inverse.
        ///
        /// (B) `cg_prox_rho` — TRUST-REGION / terme proximal sur le déplacement :
        ///     `c_j ← c_j − ρ·|price|·dt·(u_j − u_base)²/span`. Adimensionne ρ (le facteur
        ///     `|price|·dt/span` porte les unités de profit). Stabilisation canonique de
        ///     la CG (memories `704bbce5` régularisation du lagrangien, `8faa9e4b` RLSCG :
        ///     la stabilisation « substantially reduces iteration count »). ρ = 0 ⇒ CTRL.
        ///     ⛔ DISTINCT de `brick:lns_cg_primal_column_displacement` (dead, i13) : ce
        ///     dernier LISAIT le déplacement comme prédicat d'ARRÊT ; ici il entre dans
        ///     l'OBJECTIF et déplace l'optimum. Aucun compte d'itérations n'est touché.
        ///
        /// (C) `cg_cong_haircut` — DÉCOTE du surrogat PROPORTIONNELLE À LA CONGESTION vue
        ///     par la batterie : `c_j ← c_j − γ·util_b·(V̂_j − V̂_ref)`, avec
        ///     `util_b = max_{l ∈ lignes(b)} |flow_l|/limit_l` clampé à [0,1]. Correction
        ///     ÉTAT-DÉPENDANTE (A est un scalaire constant) : elle cible exactement le
        ///     biais structurel « le DP ignore le réseau », et s'annule là où le réseau
        ///     est libre. γ = 0 ⇒ CTRL.
        ///     ⛔ DISTINCT de `brick:dp_value_at_curv_correction` (dead) : celui-ci
        ///     corrigeait la COURBURE du lookup de V par un Taylor 2ᵈ ordre (reshaping
        ///     local de la table) ; ici la table n'est pas touchée, on pondère la
        ///     différence de valeur par un signal EXOGÈNE (l'utilisation des lignes).
        ///
        /// ⛔ DISTINCT de `brick:lp_objective_cfa_theta` (dead) : la CFA bakait un θ sur
        /// les coefficients de PRIX/tx du LP per-step (sites A/B/C), time-neutre et sans
        /// état ; ici on pondère la CONTINUATION, au seul site de la CG du LNS.
        pub cg_cont_scale: f64,
        pub cg_prox_rho: f64,
        pub cg_cong_haircut: f64,

        pub use_ldd_proximal: bool,
        pub ldd_momentum: f64,
        pub ldd_clip_fraction: f64,
        pub use_tail_quadrature: bool,
        
        // (sigma=0.15/rho_jump=0.02/alpha=3.5, generic paper defaults — never calibrated for t50 congested).
        // Defaults == previous hardcoded values → bit-exact baseline. Pilot V via build_tail_mixture_quadrature.
        pub dp_sigma: f64,
        pub dp_rho_jump: f64,
        pub dp_alpha: f64,
        pub lns_dual_smooth_alpha: f64,
        pub use_primal_refine: bool,
        pub use_lmp_premiums_kkt: bool,
        pub use_prime_admm: bool,
        pub dw_mu_damping_alpha: f64,
        pub dw_boxstep_delta: f64,
        // i40 (t50, Wentges auto-adaptive): clip+adaptive-smooth replaces pure box-step when enabled.
        // Convention: a = weight on OLD mu (a=0 → pure box-step/full-follow, a=1 → frozen).
        // Formula: a = alpha_min + (alpha_max - alpha_min) * misprice_rate
        // High misprice (slow conv.) → a → alpha_max (more damping, 20% of step at alpha_max=0.8).
        // Low misprice (near conv.) → a → alpha_min (less damping; alpha_min=0.0 = pure box-step).
        // 0.0/0.0 = disabled → pure box-step path (CTRL bit-exact to i36 baseline).
        pub dw_wentges_alpha_min: f64,
        pub dw_wentges_alpha_max: f64,
        pub use_adaptive_lines: bool,
        pub use_binary_congestion_premium: bool,
        pub congestion_quantize_levels: usize,
        pub use_action_aware_premium: bool,
        // i12 (t50, DFL-seed proxy): decision-regret shaping of the congestion-premium response.
        // The P7 premium is proba·base where proba is a LINEAR ramp of the congestion ratio
        // [thr,1]->[0,1]. gamma reshapes it to proba^gamma (build-time constant, ~0 runtime):
        // gamma>1 = convex, concentrates the premium on near-BINDING lines (ratio->1, where a
        // mispriced discharge carries the largest realized-congestion regret) and attenuates
        // gamma=1.0 = bit-exact linear baseline (fallback). NOT a scale multiplier (surgical,
        // ratio-dependent) and NOT an additive term (reshapes existing proba -> no double-count).
        pub premium_shape_gamma: f64,
        // i17 (t50, spatial reshape): analogous to i12 gamma (form on line), but on the SPATIAL
        // distribution to batteries. Distribution is -impact * sign_f * premium where impact =
        // PTDF battery->line (|impact| ≲ 1, bounded). Reshaping |impact|^delta with delta > 1
        // concentrates the premium on high-PTDF batteries (largest real congestion lever).
        // delta=1.0 = bit-exact P20 baseline (identity: signum(x)*|x|^1 = x).
        // 3eeb0a81-safe: |impact|≲1 → delta>1 ATTENUATES the low end, no unbounded amplification.
        pub premium_impact_delta: f64,
        // i19 (t50, L7 DFL): per-line PTDF-aggregate weight w_l = Σ_b|impact_{l,b}| normalized to
        // mean=1, clamped [w_min,w_max]. false (default) → all w_l=1.0 → CTRL bit-exact 603331.
        // Orthogonal to gamma (line shape i12) and delta (battery spatial i17).
        pub use_learned_line_weights: bool,
        pub line_weight_w_min: f64,
        pub line_weight_w_max: f64,
        // i20 (t50, L7 capacity-norm): "danger-score" per line = Σ_b|impact_{l,b}| / flow_limit[l]
        // normalized to mean=1, clamp reuses line_weight_w_min/w_max. Lines with high battery
        // footprint AND tight capacity → highest weight. COS proxy: r_l=footprint, y_l=1/limit →
        // w_l ∝ footprint/limit. Orthogonal to i19 (footprint-only, no /limit).
        // false (default) → all w_l=1.0 → CTRL bit-exact 603331.
        pub use_cap_norm_weights: bool,
        // i21 (t50, L7 sqrt-cap): geometric-mean interpolation between i19 (footprint, p=1) and
        // i20 (cap-norm, p=1). w_l = Σ_b|impact_{l,b}| / sqrt(flow_limit[l]), normalized mean=1,
        // clamp reuses line_weight_w_min/w_max. Half-power capacity adjustment: preserves the
        // bimodal separation of t50 congestion without the full /limit amplification of i20.
        // Analogy: geometric mean of footprint signal and cap-norm signal.
        // false (default) → all w_l=1.0 → CTRL bit-exact 603331.
        pub use_sqrt_cap_weights: bool,
        // i23 (t50, L7 COS): Decision-Focused Learning per-line weights calibrated offline via
        // cost-sensitive regression closed-form (Schutte et al. 2026, arXiv:2605.18005):
        //   w_l = Σ_s c_{l,s} r_{l,s} y_{l,s} / Σ_s c_{l,s} r_{l,s}²
        // r_{l,s} = PTDF-footprint Σ_b|impact_{l,b}| × congestion proba_l(s) (feature)
        // y_{l,s} = dispatch regret sensitivity to line l in scenario s (via KKT dual, e7dcefdc)
        // c_{l,s} = asymmetric COS cost α_under >> α_over for binding lines (Schutte 2026 §3)
        // Weights calibrated from historical challenge nonces via analyse/calib_cos_line_weights.py.
        // PLACEHOLDER: cos_line_weight_scale applies a global scale to the placeholder 1.0 vector.
        // CTRL: use_cos_weights=false → all 1.0 → bit-exact 603331. clamp reuses w_min/w_max.
        pub use_cos_weights: bool,
        // Scale applied to placeholder COS weights (1.0 = identity = CTRL for uniform weights).
        // After calibration, actual w_l values replace the uniform vector and this becomes a
        // fine-tuning knob. Default 1.0 = no-op.
        pub cos_line_weight_scale: f64,
        // i57 L7 COS asymmetry: multiplicative binding-proximity boost.
        // w_l = base(proba_l) * (1 + kappa * asym_l)
        // asym_l = mean_t(proba_linear_l) = mean linear excess over threshold (no gamma shaping).
        // kappa=0.0 → identity → P29 BIT-EXACT sentinel. Default 0.0.
        pub cos_asymmetry_kappa: f64,
        // i58 L7 complet COS: cost-sensitive weighted proba mean.
        // w_l = 0.5 + β3 × cs_mean_proba^γ, normalized to mean=1, clamped.
        // cs_mean = Σ_t c_{l,t} × proba^γ / Σ_t c_{l,t}
        // c_{l,t} = cos_alpha_under if ratio_l(t) > lmp_threshold, else 1.0.
        // Amplifies lines with EPISODIC near-binding congestion (rare but high-regret moments).
        // cos_alpha_under=1.0 → uniform weights → BIT-EXACT to use_cos_weights P29.
        // false (default) → falls through to use_cos_weights=true (P29 baseline) → CTRL.
        pub use_cos_cs_weights: bool,
        pub cos_alpha_under: f64,
        // 0 = off (bit-exact baseline) ; 1 = cheap reuse of mu_dw ; 2 = faithful lagged re-solve.
        pub coord_premium_mode: usize,
        pub coord_premium_scale: f64,
        // SLP (i46): replace exact quadratic deg eval in primal_refine c_fin with linear approx at incumbent
        pub use_slp_degradation: bool,
        // OCO constraint tracking (i55): adaptive Lagrange multiplier iterated on PTDF flow violations
        // μ_l ← [μ_l + ct_step_eta·(|flow_l|−limit_l)]_+ ; premium rebuilt → non-trivial fixed point ≠ i45
        pub use_ptdf_constraint_tracking: bool,
        pub ct_step_eta: f64,
        // OCO reference tracking (i59 — 3rd Huang ingredient): anchor LDD correction on unconstrained DP SOC trajectory
        pub ct_ref_kappa: f64,
        // OCO OC-reference (i65 — 4th Huang ingredient): anchor correction on ∂V/∂soc (water value) from unconstrained backward DP.
        // ep[t][b] -= kappa * max(0, oc_ref[t][b] - oc_implied[t][b]). Default 0.0 = iso-binaire i60.
        pub ct_oc_kappa: f64,
        // OCO adaptive per-line step (i60): η_l = ct_step_eta × (viol_l / max_viol_t) — concentrates mass on worst violator
        pub use_ct_adaptive_per_line: bool,
        // GDD non-linear regularizer (i67 DEAD: ct_gdd_rho MW-brut quadratique → over-penalization).
        // Default 0.0 = disabled (dead, do not set).
        pub ct_gdd_rho: f64,
        // GDD exponential regularizer (i68): normalized violation + exponential penalty (Sinha-Vaze 2024).
        // v_frac_l = (|flow_l| - limit_l) / limit_l  (dimensionless, O(0.01-0.5))
        // s_b = Σ_l v_frac_l * |ptdf_{b,l}|
        // ep_ct[t][b] -= exp(ct_gdd_alpha * s_b) - 1
        // Default 0.0 → zero-cost branch, iso-binaire i60 EXACT.
        pub ct_gdd_alpha: f64,
        // i2 (min_level=6 replace) — objective model of the per-step joint LP.
        // Baseline (0) is a doubly-biased surrogate of `eval_profit`: it DROPS the convex
        // quadratic degradation deg_c*u^2 and linearises the CONCAVE continuation value
        // V(soc') with the tangent slope taken at the current SOC. Both biases push the LP
        // toward bang-bang actions. Each mode repairs exactly one bias so the R7 cascade
        // can attribute the gain:
        //   0 = baseline  (segments=1, tangent dv, no degradation)   -> bit-exact control
        //   1 = chord     (segments=1, secant dv over the reachable SOC interval, free)
        //   2 = pwl_value (segments=S, concave-hull marginal continuation value)
        //   3 = pwl_deg   (segments=S, tangent dv + exact convex PWL degradation cost)
        pub lp_obj_mode: usize,
        pub lp_pwl_segments: usize,
        /// WHERE the `lp_pwl_segments` cuts sit inside the action interval (the COUNT is
        /// saturated: dead-list `hp:lp_pwl_segments`, bell with max at S=3). Placement is a
        /// different lever: with `lp_obj_mode=2` the LP consumes segments in order, so the
        /// cut positions ARE the reachable dispatch lattice.
        ///   0 = uniform   (baseline, `cap/segs` each)                 -> bit-exact control
        ///   1 = soc_anchor (cuts phase-locked to the ABSOLUTE SOC lattice, Zheng bids)
        ///   2 = dp_snap    (cuts snapped to the DP node grid -> exact secants)
        ///   3 = geo_rating (geometric, finest segment at the power rating)
        ///   4 = geo_zero   (geometric, finest segment at zero action) -- two-sided control
        pub lp_pwl_spacing: usize,
        pub lp_pwl_geo_ratio: f64,
        /// Placement mechanism for the CHARGE direction ONLY. `-1` (default) = "same as
        /// `lp_pwl_spacing`", i.e. the lattice stays symmetric and this knob is inert.
        /// Set to `0..=4` to give charge its own mechanism: the two directions share neither
        /// their conversion factor (`s_dis = dt/eta_d` vs `s_chg = eta_c*dt`) nor their side
        /// of the concave `V`, so nothing says one placement has to fit both.
        pub lp_pwl_spacing_chg: i64,
    }

    impl TrackHp {
        pub fn override_from_map(&mut self, h: &Option<Map<String, Value>>) {
            let Some(m) = h else { return };
            if let Some(v) = m.get("soc_levels").and_then(|v| v.as_u64()) { self.soc_levels = (v as usize).max(3); }
            if let Some(v) = m.get("action_grid").and_then(|v| v.as_u64()) { self.action_grid = (v as usize).max(4); }
            if let Some(v) = m.get("asca_iters").and_then(|v| v.as_u64()) { self.asca_iters = v as usize; }
            if let Some(v) = m.get("ternary_iters").and_then(|v| v.as_u64()) { self.ternary_iters = v as usize; }
            if let Some(v) = m.get("convergence_tol").and_then(|v| v.as_f64()) { self.convergence_tol = v; }
            if let Some(v) = m.get("anticipate_lmp").and_then(|v| v.as_bool()) { self.anticipate_lmp = v; }
            if let Some(v) = m.get("lmp_threshold").and_then(|v| v.as_f64()) { self.lmp_threshold = v; }
            if let Some(v) = m.get("lmp_premium_scale").and_then(|v| v.as_f64()) { self.lmp_premium_scale = v; }
            if let Some(v) = m.get("jump_premium").and_then(|v| v.as_f64()) { self.jump_premium = v; }
            if let Some(v) = m.get("prune_ratio").and_then(|v| v.as_f64()) { self.prune_ratio = v.clamp(0.0, 0.9); }
            if let Some(v) = m.get("deflator_iters").and_then(|v| v.as_u64()) { self.deflator_iters = v as usize; }
            if let Some(v) = m.get("flow_margin").and_then(|v| v.as_f64()) { self.flow_margin = v.max(0.0); }
            // Clamp to [0, EPS_FLOW]: an HP override may only make the acceptance check
            // STRICTER, never laxer than the framework — this makes the bug unreachable.
            if let Some(v) = m.get("flow_feas_tol").and_then(|v| v.as_f64()) { self.flow_feas_tol = v.clamp(0.0, 1e-6); }
            if let Some(v) = m.get("network_derating").and_then(|v| v.as_f64()) { self.network_derating = v.clamp(0.01, 1.0); }
            if let Some(v) = m.get("nd_mode").and_then(|v| v.as_u64()) { self.nd_mode = v as usize; }
            if let Some(v) = m.get("nd_free").and_then(|v| v.as_f64()) { self.nd_free = v.clamp(0.01, 1.0); }
            // i23 — QUANTILE, plus un seuil absolu : clamp [0,1] (cf note (2) sur la struct).
            if let Some(v) = m.get("nd_gate").and_then(|v| v.as_f64()) { self.nd_gate = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("nd_probe").and_then(|v| v.as_u64()) { self.nd_probe = v as usize; }
            if let Some(v) = m.get("nd_scalar_mask").and_then(|v| v.as_u64()) { self.nd_scalar_mask = v as usize; }
            if let Some(v) = m.get("dual_iters").and_then(|v| v.as_u64()) { self.dual_iters = v as usize; }
            if let Some(v) = m.get("da_step_size").and_then(|v| v.as_f64()) { self.da_step_size = v.max(0.0); }
            if let Some(v) = m.get("ldd_iters").and_then(|v| v.as_u64()) { self.ldd_iters = v as usize; }
            if let Some(v) = m.get("ldd_step_size").and_then(|v| v.as_f64()) { self.ldd_step_size = v.max(0.0); }
            if let Some(v) = m.get("use_kkt").and_then(|v| v.as_bool()) { self.use_kkt = v; }
            if let Some(v) = m.get("kkt_cong_threshold").and_then(|v| v.as_f64()) { self.kkt_cong_threshold = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("kkt_price_scale").and_then(|v| v.as_f64()) { self.kkt_price_scale = v.max(0.1); }
            if let Some(v) = m.get("max_admm_iters").and_then(|v| v.as_u64()) { self.max_admm_iters = v as usize; }
            if let Some(v) = m.get("admm_rho").and_then(|v| v.as_f64()) { self.admm_rho = v; }
            if let Some(v) = m.get("admm_primal_tol").and_then(|v| v.as_f64()) { self.admm_primal_tol = v; }
            if let Some(v) = m.get("use_lp").and_then(|v| v.as_bool()) { self.use_lp = v; }
            if let Some(v) = m.get("dantzig_in_dw").and_then(|v| v.as_bool()) { self.dantzig_in_dw = v; }
            if let Some(v) = m.get("dantzig_in_lns").and_then(|v| v.as_bool()) { self.dantzig_in_lns = v; }
            if let Some(v) = m.get("dantzig_in_kkt").and_then(|v| v.as_bool()) { self.dantzig_in_kkt = v; }
            if let Some(v) = m.get("dantzig_bland_degen").and_then(|v| v.as_bool()) { self.dantzig_bland_degen = v; }
            if let Some(v) = m.get("lp_soft_lambda").and_then(|v| v.as_f64()) { self.lp_soft_lambda = v.max(0.0); }
            if let Some(v) = m.get("lp_per_call_pivots").and_then(|v| v.as_u64()) { self.lp_per_call_pivots = v as usize; }
            if let Some(v) = m.get("lp_total_pivots").and_then(|v| v.as_u64()) { self.lp_total_pivots = v as usize; }
            if let Some(v) = m.get("use_policy").and_then(|v| v.as_bool()) { self.use_policy = v; }
            if let Some(v) = m.get("use_warmstart").and_then(|v| v.as_bool()) { self.use_warmstart = v; }
            if let Some(v) = m.get("use_mpc").and_then(|v| v.as_bool()) { self.use_mpc = v; }
            if let Some(v) = m.get("mpc_horizon").and_then(|v| v.as_u64()) { self.mpc_horizon = v as usize; }
            if let Some(v) = m.get("mpc_pivot_budget").and_then(|v| v.as_u64()) { self.mpc_pivot_budget = v as usize; }
            if let Some(v) = m.get("use_dw").and_then(|v| v.as_bool()) { self.use_dw = v; }
            if let Some(v) = m.get("dw_iters").and_then(|v| v.as_u64()) { self.dw_iters = v as usize; }
            if let Some(v) = m.get("dw_max_lines").and_then(|v| v.as_u64()) { self.dw_max_lines = v as usize; }
            if let Some(v) = m.get("dw_max_cols_per_batt").and_then(|v| v.as_u64()) { self.dw_max_cols_per_batt = v as usize; }
            if let Some(v) = m.get("dw_pivot_budget_per_solve").and_then(|v| v.as_u64()) { self.dw_pivot_budget_per_solve = v as usize; }
            if let Some(v) = m.get("dw_total_pivot_budget").and_then(|v| v.as_u64()) { self.dw_total_pivot_budget = v as usize; }
            if let Some(v) = m.get("use_dw_prescreen").and_then(|v| v.as_bool()) { self.use_dw_prescreen = v; }
            if let Some(v) = m.get("use_lns").and_then(|v| v.as_bool()) { self.use_lns = v; }
            if let Some(v) = m.get("lns_cg_iters").and_then(|v| v.as_u64()) { self.lns_cg_iters = v as usize; }
            if let Some(v) = m.get("lns_cg_column_limit").and_then(|v| v.as_u64()) { self.lns_cg_column_limit = (v as usize).max(2); }
            if let Some(v) = m.get("lns_max_lines").and_then(|v| v.as_u64()) { self.lns_max_lines = (v as usize).max(1); }
            if let Some(v) = m.get("lns_lp_pivots_total").and_then(|v| v.as_u64()) { self.lns_lp_pivots_total = v as usize; }
            if let Some(v) = m.get("use_pivot_reserve").and_then(|v| v.as_bool()) { self.use_pivot_reserve = v; }
            if let Some(v) = m.get("lp_max_lines").and_then(|v| v.as_u64()) { self.lp_max_lines = (v as usize).max(1); }
            if let Some(v) = m.get("use_parallel_dp").and_then(|v| v.as_bool()) { self.use_parallel_dp = v; }
            if let Some(v) = m.get("use_sdp").and_then(|v| v.as_bool()) { self.use_sdp = v; }
            if let Some(v) = m.get("sdp_k").and_then(|v| v.as_u64()) { self.sdp_k = (v as usize).max(2); }
            if let Some(v) = m.get("dp_prepass_mode").and_then(|v| v.as_u64()) { self.dp_prepass_mode = v as usize; }
            if let Some(v) = m.get("dw_prescreen_mode").and_then(|v| v.as_u64()) { self.dw_prescreen_mode = v as usize; }
            if let Some(v) = m.get("sdp_expect_mode").and_then(|v| v.as_u64()) { self.sdp_expect_mode = v as usize; }
            if let Some(v) = m.get("sdp_alloc_mode").and_then(|v| v.as_u64()) { self.sdp_alloc_mode = v as usize; }
            if let Some(v) = m.get("lns_cg_stop_mode").and_then(|v| v.as_u64()) { self.lns_cg_stop_mode = v as usize; }
            if let Some(v) = m.get("lns_probe").and_then(|v| v.as_u64()) { self.lns_probe = v as usize; }
            if let Some(v) = m.get("lns_gate_mode").and_then(|v| v.as_u64()) { self.lns_gate_mode = v as usize; }
            if let Some(v) = m.get("lns_gate_headroom_tol").and_then(|v| v.as_f64()) { self.lns_gate_headroom_tol = v.max(0.0); }
            if let Some(v) = m.get("lns_gate_streak_k").and_then(|v| v.as_u64()) { self.lns_gate_streak_k = v; }
            if let Some(v) = m.get("lns_gate_probe_period").and_then(|v| v.as_u64()) { self.lns_gate_probe_period = (v as usize).max(1); }
            if let Some(v) = m.get("lns_cg_probe").and_then(|v| v.as_u64()) { self.lns_cg_probe = v as usize; }
            if let Some(v) = m.get("lns_cg_mat_mode").and_then(|v| v.as_u64()) { self.lns_cg_mat_mode = v as usize; }
            if let Some(v) = m.get("lns_cg_mat_delta").and_then(|v| v.as_f64()) { self.lns_cg_mat_delta = v.max(0.0); }
            // i14 — pondération du surrogat de la CG. Défauts NEUTRES (1.0 / 0.0 / 0.0).
            if let Some(v) = m.get("cg_cont_scale").and_then(|v| v.as_f64()) { self.cg_cont_scale = v; }
            if let Some(v) = m.get("cg_prox_rho").and_then(|v| v.as_f64()) { self.cg_prox_rho = v.max(0.0); }
            if let Some(v) = m.get("cg_cong_haircut").and_then(|v| v.as_f64()) { self.cg_cong_haircut = v.max(0.0); }
            if let Some(v) = m.get("use_ldd_proximal").and_then(|v| v.as_bool()) { self.use_ldd_proximal = v; }
            if let Some(v) = m.get("ldd_momentum").and_then(|v| v.as_f64()) { self.ldd_momentum = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("ldd_clip_fraction").and_then(|v| v.as_f64()) { self.ldd_clip_fraction = v.clamp(0.01, 1.0); }
            if let Some(v) = m.get("use_tail_quadrature").and_then(|v| v.as_bool()) { self.use_tail_quadrature = v; }
            if let Some(v) = m.get("dp_sigma").and_then(|v| v.as_f64()) { self.dp_sigma = v; }
            if let Some(v) = m.get("dp_rho_jump").and_then(|v| v.as_f64()) { self.dp_rho_jump = v; }
            if let Some(v) = m.get("dp_alpha").and_then(|v| v.as_f64()) { self.dp_alpha = v; }
            if let Some(v) = m.get("lns_dual_smooth_alpha").and_then(|v| v.as_f64()) { self.lns_dual_smooth_alpha = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("use_primal_refine").and_then(|v| v.as_bool()) { self.use_primal_refine = v; }
            if let Some(v) = m.get("use_lmp_premiums_kkt").and_then(|v| v.as_bool()) { self.use_lmp_premiums_kkt = v; }
            if let Some(v) = m.get("use_prime_admm").and_then(|v| v.as_bool()) { self.use_prime_admm = v; }
            if let Some(v) = m.get("dw_mu_damping_alpha").and_then(|v| v.as_f64()) { self.dw_mu_damping_alpha = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("dw_boxstep_delta").and_then(|v| v.as_f64()) { self.dw_boxstep_delta = v.max(0.0); }
            if let Some(v) = m.get("dw_wentges_alpha_min").and_then(|v| v.as_f64()) { self.dw_wentges_alpha_min = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("dw_wentges_alpha_max").and_then(|v| v.as_f64()) { self.dw_wentges_alpha_max = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("use_adaptive_lines").and_then(|v| v.as_bool()) { self.use_adaptive_lines = v; }
            if let Some(v) = m.get("use_binary_congestion_premium").and_then(|v| v.as_bool()) { self.use_binary_congestion_premium = v; }
            if let Some(v) = m.get("congestion_quantize_levels").and_then(|v| v.as_u64()) { self.congestion_quantize_levels = v as usize; }
            if let Some(v) = m.get("use_action_aware_premium").and_then(|v| v.as_bool()) { self.use_action_aware_premium = v; }
            if let Some(v) = m.get("premium_shape_gamma").and_then(|v| v.as_f64()) { self.premium_shape_gamma = v.max(0.1); }
            if let Some(v) = m.get("premium_impact_delta").and_then(|v| v.as_f64()) { self.premium_impact_delta = v.max(0.1); }
            if let Some(v) = m.get("use_learned_line_weights").and_then(|v| v.as_bool()) { self.use_learned_line_weights = v; }
            if let Some(v) = m.get("line_weight_w_min").and_then(|v| v.as_f64()) { self.line_weight_w_min = v.max(0.01); }
            if let Some(v) = m.get("line_weight_w_max").and_then(|v| v.as_f64()) { self.line_weight_w_max = v.max(self.line_weight_w_min); }
            if let Some(v) = m.get("use_cap_norm_weights").and_then(|v| v.as_bool()) { self.use_cap_norm_weights = v; }
            if let Some(v) = m.get("use_sqrt_cap_weights").and_then(|v| v.as_bool()) { self.use_sqrt_cap_weights = v; }
            if let Some(v) = m.get("use_cos_weights").and_then(|v| v.as_bool()) { self.use_cos_weights = v; }
            if let Some(v) = m.get("cos_line_weight_scale").and_then(|v| v.as_f64()) { self.cos_line_weight_scale = v.max(0.01); }
            if let Some(v) = m.get("cos_asymmetry_kappa").and_then(|v| v.as_f64()) { self.cos_asymmetry_kappa = v.max(0.0); }
            if let Some(v) = m.get("use_cos_cs_weights").and_then(|v| v.as_bool()) { self.use_cos_cs_weights = v; }
            if let Some(v) = m.get("cos_alpha_under").and_then(|v| v.as_f64()) { self.cos_alpha_under = v.max(1.0); }
            if let Some(v) = m.get("coord_premium_mode").and_then(|v| v.as_u64()) { self.coord_premium_mode = v as usize; }
            if let Some(v) = m.get("coord_premium_scale").and_then(|v| v.as_f64()) { self.coord_premium_scale = v.max(0.0); }
            if let Some(v) = m.get("use_slp_degradation").and_then(|v| v.as_bool()) { self.use_slp_degradation = v; }
            if let Some(v) = m.get("use_ptdf_constraint_tracking").and_then(|v| v.as_bool()) { self.use_ptdf_constraint_tracking = v; }
            if let Some(v) = m.get("ct_step_eta").and_then(|v| v.as_f64()) { self.ct_step_eta = v.max(0.0); }
            if let Some(v) = m.get("ct_ref_kappa").and_then(|v| v.as_f64()) { self.ct_ref_kappa = v.max(0.0); }
            if let Some(v) = m.get("ct_oc_kappa").and_then(|v| v.as_f64()) { self.ct_oc_kappa = v.max(0.0); }
            if let Some(v) = m.get("use_ct_adaptive_per_line").and_then(|v| v.as_bool()) { self.use_ct_adaptive_per_line = v; }
            if let Some(v) = m.get("ct_gdd_rho").and_then(|v| v.as_f64()) { self.ct_gdd_rho = v.max(0.0); }
            if let Some(v) = m.get("ct_gdd_alpha").and_then(|v| v.as_f64()) { self.ct_gdd_alpha = v.max(0.0); }
            if let Some(v) = m.get("lp_obj_mode").and_then(|v| v.as_u64()) { self.lp_obj_mode = (v as usize).min(3); }
            if let Some(v) = m.get("lp_pwl_segments").and_then(|v| v.as_u64()) { self.lp_pwl_segments = (v as usize).clamp(1, 12); }
            if let Some(v) = m.get("lp_pwl_spacing").and_then(|v| v.as_u64()) { self.lp_pwl_spacing = (v as usize).min(4); }
            if let Some(v) = m.get("lp_pwl_geo_ratio").and_then(|v| v.as_f64()) { self.lp_pwl_geo_ratio = v.clamp(1.0, 8.0); }
            if let Some(v) = m.get("lp_pwl_spacing_chg").and_then(|v| v.as_i64()) { self.lp_pwl_spacing_chg = v.clamp(-1, 4); }
        }
    }

    pub struct AycdicdbCache {
        pub dp: Vec<Vec<Vec<f64>>>,
        pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
        pub b_to_lines: Vec<Vec<(usize, f64)>>,
        pub batt_nodes: Vec<usize>,
    }

    // i13 — accumulateur de la sonde SHADOW de MATÉRIALITÉ (1 par nonce). Champ dédié de
    // `Inner` ⇒ emprunt DISJOINT de `inner.cache` / `inner.hp` au site d'appel.
    // `fire[i]` = nombre d'itérations CG qui auraient été SAUTÉES par la famille `i`
    //             (= le TEMPS que la famille récupère, en unités d'itération CG) ;
    // `hits[i]` = nombre d'APPELS où la famille aurait mordu au moins une fois ;
    // `bad[i]`  = sous-ensemble de `hits` où `best_obj` s'améliorait ENCORE après le point
    //             d'arrêt (= la famille aurait coûté du Q, MESURÉ et non supposé).
    // `pool[i]` = colonnes qui n'auraient jamais été créées ⇒ le `use_primal_refine`
    //             post-boucle voit une pool PLUS PETITE : `bad == 0` ne suffit donc PAS à
    //             conclure « bit-exact », il faut le bench de l'arm appliqué.
    const CG_FAMS: usize = 9;
    const CG_MAXK: usize = 8;
    #[derive(Default)]
    struct CgProbe {
        calls: u64,
        iters: u64,
        iters_ge1: u64,
        fire: [u64; CG_FAMS],
        hits: [u64; CG_FAMS],
        bad: [u64; CG_FAMS],
        pool: [u64; CG_FAMS],
    }

    struct Inner {
        hp: TrackHp,
        cache: Option<AycdicdbCache>,
        lp_pivots_consumed: usize,
        lp_pivot_reserve: isize,
        policy_weights: Option<Vec<Vec<f64>>>,
        // i12 — compteurs de la sonde LNS (remis à zéro par nonce : `Inner` est
        // reconstruit à chaque `solve_with_hp`). `lns_reject_streak` est le SEUL
        // champ lu par une décision (prédicat F2, `lns_gate_mode == 2`).
        lns_calls: u64,
        lns_work: u64,
        lns_accepted: u64,
        lns_skipped: u64,
        lns_reject_streak: u64,
        sh1_fire: u64,
        sh1_bad: u64,
        sh2_fire: u64,
        sh2_bad: u64,
        sh3_fire: u64,
        sh3_bad: u64,
        cg: CgProbe,
    }

    thread_local! {
        static STATE: RefCell<Option<Inner>> = RefCell::new(None);
    }

    pub fn solve_with_hp(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: TrackHp,
    ) -> Result<()> {
        STATE.with(|s| *s.borrow_mut() = Some(Inner {
            hp, cache: None, lp_pivots_consumed: 0, lp_pivot_reserve: 0, policy_weights: None,
            lns_calls: 0, lns_work: 0, lns_accepted: 0, lns_skipped: 0, lns_reject_streak: 0,
            sh1_fire: 0, sh1_bad: 0, sh2_fire: 0, sh2_bad: 0, sh3_fire: 0, sh3_bad: 0,
            cg: CgProbe::default(),
        }));
        let out = challenge.grid_optimize(&policy_entry);
        STATE.with(|s| *s.borrow_mut() = None);
        let solution = out?;
        save_solution(&solution)?;
        Ok(())
    }

    fn policy_entry(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        STATE.with(|s| -> Result<Vec<f64>> {
            let mut guard = s.borrow_mut();
            let inner = guard.as_mut().expect("Aycdicdb: STATE not initialised");
            if inner.cache.is_none() {
                inner.cache = Some(build_cache(challenge, state, &inner.hp));
            }
            let cache = inner.cache.as_ref().unwrap();
            let hp = &inner.hp;

            let zero_action = vec![0.0_f64; challenge.num_batteries];
            let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base_cur);

            let uncons_actions: Vec<f64> = (0..challenge.num_batteries)
                .map(|b| optimal_unconstrained_action(challenge, state, cache, b))
                .collect();
            let warm_init: Vec<f64> = if hp.use_warmstart {
                uncons_actions.clone()
            } else {
                vec![0.0; challenge.num_batteries]
            };

            let base_actions = if hp.use_kkt {
                kkt_policy(challenge, state, cache, hp, &flows_base, &warm_init)
            } else {
                let mut a = warm_init.clone();
                run_asca(challenge, state, cache, hp, &flows_base, &mut a);
                if hp.dual_iters > 0 {
                    let asca_actions = a.clone();
                    let dual_actions = run_dual_ascent(challenge, state, cache, hp, &flows_base, &asca_actions);
                    let profit_asca: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, asca_actions[b]))
                        .sum();
                    let profit_dual: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, dual_actions[b]))
                        .sum();
                    if profit_dual >= profit_asca { a = dual_actions; }
                } else if hp.max_admm_iters > 0 {
                    if !run_admm_dispatch(challenge, state, cache, hp, &flows_base, &mut a) {
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut a);
                    }
                } else {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut a);
                }
                a
            };

            // i12 — copie de l'incumbent PRÉ-LP, uniquement si le prédicat F3 en a besoin
            // (sonde ou `lns_gate_mode == 3`). Sinon : 0 allocation ⇒ CTRL bit-exact.
            let pre_lp_actions: Option<Vec<f64>> =
                if hp.lns_probe != 0 || hp.lns_gate_mode == 3 { Some(base_actions.clone()) } else { None };

            let actions = if hp.use_lp {
                let max_util = (0..flows_base.len())
                    .map(|l| {
                        let limit = challenge.network.flow_limits[l];
                        if limit > 1e-6 { flows_base[l].abs() / limit } else { 0.0 }
                    })
                    .fold(0.0_f64, f64::max);

                let per_call_limit = if hp.lp_per_call_pivots > 0 {
                    hp.lp_per_call_pivots
                } else { 3000 };

                let budget = if hp.use_pivot_reserve {
                    compute_lp_budget_for_step(hp, inner.lp_pivots_consumed, &mut inner.lp_pivot_reserve)
                } else {
                    per_call_limit
                };

                let budget_ok = if hp.use_pivot_reserve {
                    budget > 0
                } else {
                    hp.lp_total_pivots == 0
                        || inner.lp_pivots_consumed.saturating_add(per_call_limit) <= hp.lp_total_pivots
                };

                if max_util >= hp.kkt_cong_threshold && budget_ok {
                    let lp_result = if hp.use_pivot_reserve {
                        joint_lp_dispatch_with_used(challenge, state, cache, &flows_base, &base_actions, hp.lp_soft_lambda, budget, hp)
                    } else {
                        let sol = joint_lp_dispatch(challenge, state, cache, &flows_base, &base_actions, hp.lp_soft_lambda, budget, hp);
                        (sol, budget)
                    };

                    if let Some(mut lp_act) = lp_result.0 {
                        let lp_used = lp_result.1;
                        if hp.use_pivot_reserve {
                            inner.lp_pivots_consumed = inner.lp_pivots_consumed.saturating_add(lp_used).min(hp.lp_total_pivots.max(1));
                            let allocated = budget;
                            let unused = allocated.saturating_sub(lp_used);
                            inner.lp_pivot_reserve = (inner.lp_pivot_reserve + unused as isize).min(hp.lp_per_call_pivots as isize);
                        } else {
                            inner.lp_pivots_consumed += per_call_limit;
                        }
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut lp_act);
                        let lp_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, lp_act[b])).sum();
                        let base_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, base_actions[b])).sum();
                        if lp_p >= base_p { lp_act } else { base_actions }
                    } else {
                        if hp.use_pivot_reserve {
                            inner.lp_pivot_reserve = (inner.lp_pivot_reserve + budget as isize).min(hp.lp_per_call_pivots as isize);
                        }
                        base_actions
                    }
                } else {
                    if hp.use_pivot_reserve {
                        inner.lp_pivot_reserve = (inner.lp_pivot_reserve + budget as isize).min(hp.lp_per_call_pivots as isize);
                    }
                    base_actions
                }
            } else { base_actions };

            let actions = if hp.use_mpc {
                if let Some(mut mpc_act) = mpc_dispatch_2step(challenge, state, cache, hp, &flows_base) {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut mpc_act);
                    // i13: two-step acceptance score (see `mpc_two_step_scores`). The former
                    // guard summed `eval_profit`, which is instantaneous profit PLUS the
                    // per-battery continuation V(t+1) -- i.e. exactly the objective that
                    // `joint_lp_dispatch` already maximises under the t-network. The base
                    // action is therefore the argmax of that criterion and a 2-step MPC
                    // (which deviates only to keep t+1 feasible) can essentially never win
                    // it: i12 measured 0 acceptance over ~3072 steps. Falls back to the
                    // historical guard when the stage-2 LP is unavailable.
                    let (mpc_p, base_p) =
                        match mpc_two_step_scores(challenge, state, cache, hp, &mpc_act, &actions) {
                            Some(pair) => pair,
                            None => (
                                (0..challenge.num_batteries)
                                    .map(|b| eval_profit(challenge, state, cache, b, mpc_act[b])).sum(),
                                (0..challenge.num_batteries)
                                    .map(|b| eval_profit(challenge, state, cache, b, actions[b])).sum(),
                            ),
                        };
                    if mpc_p >= base_p { mpc_act } else { actions }
                } else {
                    actions
                }
            } else {
                actions
            };

            let actions = if hp.use_policy {
                if let Some(ref weights) = inner.policy_weights {
                    let mut policy_act = policy_dispatch(challenge, state, weights);
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut policy_act);
                    policy_act
                } else {
                    if state.time_step == 0 {
                        let trained = train_policy_cmaes(challenge, state, cache, hp);
                        inner.policy_weights = Some(trained);
                        let weights = inner.policy_weights.as_ref().unwrap();
                        let mut policy_act = policy_dispatch(challenge, state, weights);
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut policy_act);
                        policy_act
                    } else {
                        actions
                    }
                }
            } else {
                actions
            };

            // ── i12 — ADMISSION GATE du bloc LNS + SONDE DE TAUX D'ACCEPTATION ──────────
            // Anti-pattern A_CUT_AT_THE_WRONG_LAYER (i11) : la CG du LNS ne fait PAS de
            // travail mort (RC-gate jamais mordant, Q bit-exact ×3) — le travail mort est
            // un cran AU-DESSUS, chez le CONSOMMATEUR : `lns_p > base_p` ci-dessous jette
            // le résultat. Coût MESURÉ du bloc (i11, étalon `use_lns=false`, 32/32) :
            // 1,5 s / 4,0 s = 3 quanta ⇒ un gate qui saute une fraction f du bloc rend
            // f × 1,5 s ; il franchit le quantum de 500 ms dès f ≥ 1/3 (gate pré-code
            // A_SITE_IS_NOT_A_COUNT : le COMPTE × COÛT UNITAIRE est ici mesuré, pas déduit
            // du site). Un skip rend `(actions, false)` = EXACTEMENT l'état aval d'un
            // rejet ⇒ seuls les pas où le LNS aurait été ACCEPTÉ peuvent coûter du Q.
            let (actions, used_lns) = if hp.use_lns {
                let probe_on = hp.lns_probe != 0;
                let gate_mode = hp.lns_gate_mode;
                if probe_on { inner.lns_calls = inner.lns_calls.saturating_add(1); }

                // F1 — BORNE DE HEADROOM (famille : borne supérieure du gain).
                let p1 = if probe_on || gate_mode == 1 {
                    let mut headroom = 0.0_f64;
                    for b in 0..challenge.num_batteries {
                        let (lo, hi) = state.action_bounds[b];
                        let u = uncons_actions[b].clamp(lo, hi);
                        let gain = eval_profit(challenge, state, cache, b, u)
                            - eval_profit(challenge, state, cache, b, actions[b]);
                        if gain.is_finite() && gain > 0.0 { headroom += gain; }
                    }
                    headroom <= hp.lns_gate_headroom_tol
                } else { false };

                // F2 — PERSISTANCE TEMPORELLE (famille : ordonnancement adaptatif).
                let p2 = if probe_on || gate_mode == 2 {
                    inner.lns_reject_streak >= hp.lns_gate_streak_k
                        && (state.time_step % hp.lns_gate_probe_period.max(1)) != 0
                } else { false };

                // F3 — REDONDANCE PRODUCTEUR (famille : consommateur du LP).
                let p3 = if probe_on || gate_mode == 3 {
                    match pre_lp_actions {
                        Some(ref prev) => (0..challenge.num_batteries)
                            .any(|b| (prev[b] - actions[b]).abs() > 1e-9),
                        None => false,
                    }
                } else { false };

                let skip = match gate_mode { 1 => p1, 2 => p2, 3 => p3, _ => false };

                if skip {
                    if probe_on { inner.lns_skipped = inner.lns_skipped.saturating_add(1); }
                    inner.lns_reject_streak = inner.lns_reject_streak.saturating_add(1);
                    (actions, false)
                } else if let Some(mut lns_act) = lns_dw_per_step(challenge, state, cache, hp, &flows_base, &actions, &uncons_actions, &mut inner.cg) {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut lns_act);
                    let lns_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, lns_act[b])).sum();
                    let base_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, actions[b])).sum();
                    let accepted = lns_p > base_p;
                    if probe_on {
                        // `lns_work` ne compte que les appels qui ont VRAIMENT travaillé
                        // (le gate `max_util` en tête de `lns_dw_per_step` sort à coût ~0) :
                        // sauter un `None` n'économise rien, seul `work` porte les 1,5 s.
                        inner.lns_work = inner.lns_work.saturating_add(1);
                        if accepted { inner.lns_accepted = inner.lns_accepted.saturating_add(1); }
                        if p1 { inner.sh1_fire = inner.sh1_fire.saturating_add(1);
                                if accepted { inner.sh1_bad = inner.sh1_bad.saturating_add(1); } }
                        if p2 { inner.sh2_fire = inner.sh2_fire.saturating_add(1);
                                if accepted { inner.sh2_bad = inner.sh2_bad.saturating_add(1); } }
                        if p3 { inner.sh3_fire = inner.sh3_fire.saturating_add(1);
                                if accepted { inner.sh3_bad = inner.sh3_bad.saturating_add(1); } }
                    }
                    if accepted {
                        inner.lns_reject_streak = 0;
                        (lns_act, true)
                    } else {
                        inner.lns_reject_streak = inner.lns_reject_streak.saturating_add(1);
                        (actions, false)
                    }
                } else {
                    inner.lns_reject_streak = inner.lns_reject_streak.saturating_add(1);
                    (actions, false)
                }
            } else {
                (actions, false)
            };

            let actions = if used_lns {
                actions
            } else {
                if let Some(polish) = targeted_tight_line_polish(
                    challenge,
                    state,
                    cache,
                    hp,
                    &flows_base,
                    &actions,
                ) {
                    polish
                } else {
                    actions
                }
            };
            let baseline_actions = actions;
            let is_last_step = state.time_step + 1 >= challenge.num_steps;

            // i12 — SONDE : 1 ligne par nonce en fin d'horizon. `println!` obligatoire —
            // `eprintln!` est MUET sur c008 (row `infra:eprintln_probe_not_relayed`).
            // `sh<i>=fire/bad` : `fire` = pas où le prédicat aurait sauté un appel qui a
            // VRAIMENT travaillé (= temps récupéré) ; `bad` = pas où il aurait sauté un
            // appel ACCEPTÉ (= Q perdu). Un prédicat n'est éligible que si `bad == 0`.
            if hp.lns_probe != 0 && is_last_step {
                println!(
                    "MI_PROBE lns calls={} work={} acc={} skip={} sh1={}/{} sh2={}/{} sh3={}/{}",
                    inner.lns_calls, inner.lns_work, inner.lns_accepted, inner.lns_skipped,
                    inner.sh1_fire, inner.sh1_bad,
                    inner.sh2_fire, inner.sh2_bad,
                    inner.sh3_fire, inner.sh3_bad,
                );
            }
            // i13 — SONDE DE MATÉRIALITÉ, 1 ligne par nonce. `println!` obligatoire
            // (row `infra:eprintln_probe_not_relayed`). Lecture :
            //   `it`   = itérations CG démarrées ; `it1` = celles d'index ≥ 1 (elles
            //            portent LES 1,0 s mesurées par l'étalon `lns_cg_iters=1`) ;
            //   `f<i>` = itérations d'index ≥ 1 que la famille aurait SAUTÉES ⇒ gain
            //            temps prédit = `f/it1 × 1,0 s`. Éligible seulement si ≥ 0,5 s ;
            //   `b<i>` = appels où `best_obj` progressait ENCORE après l'arrêt (Q menacé) ;
            //   `p<i>` = colonnes perdues pour `use_primal_refine` (risque non prouvé nul).
            if hp.lns_cg_probe != 0 && is_last_step {
                let c = &inner.cg;
                println!(
                    "MI_PROBE cg calls={} it={} it1={} m1={}/{}/{}|{}/{}/{}|{}/{}/{} m2={}/{}/{}|{}/{}/{}|{}/{}/{} m3={}/{}/{}|{}/{}/{}|{}/{}/{}",
                    c.calls, c.iters, c.iters_ge1,
                    c.fire[0], c.bad[0], c.pool[0], c.fire[1], c.bad[1], c.pool[1], c.fire[2], c.bad[2], c.pool[2],
                    c.fire[3], c.bad[3], c.pool[3], c.fire[4], c.bad[4], c.pool[4], c.fire[5], c.bad[5], c.pool[5],
                    c.fire[6], c.bad[6], c.pool[6], c.fire[7], c.bad[7], c.pool[7], c.fire[8], c.bad[8], c.pool[8],
                );
            }
            Ok(if is_last_step {
                let mut best_actions = baseline_actions.clone();
                let baseline_profit = challenge.compute_profit(state, &baseline_actions);
                let best_profit = baseline_profit;

                let all_lines_pivots = 8000usize;
                let mut lp_hp_all = hp.clone();
                lp_hp_all.lp_max_lines = challenge.network.flow_limits.len();
                if let Some(lp_actions) = joint_lp_dispatch(
                    challenge, state, cache,
                    &flows_base, &baseline_actions,
                    hp.lp_soft_lambda, all_lines_pivots, &lp_hp_all,
                ) {
                    let lp_profit = challenge.compute_profit(state, &lp_actions);
                    let inj = challenge.compute_total_injections(state, &lp_actions);
                    let flows = challenge.network.compute_flows(&inj);
                    // Feasibility acceptance aligned with the framework's verify_flows
                    // (EPS_FLOW=1e-6). flow_feas_tol defaults to 1e-6 and is clamped to
                    // [0, 1e-6] so the last-step LP refinement can never ship an action
                    // the framework rejects (root cause of the "schedule vide" prod bug).
                    let feasible = (0..flows.len()).all(|l| flows[l].abs() <= challenge.network.flow_limits[l] * (1.0 + hp.flow_feas_tol));
                    if lp_profit > best_profit && feasible {
                        best_actions = lp_actions;
                    }
                }

                best_actions
            } else {
                baseline_actions
            })
        })
    }

    fn build_dp_with_mu(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
    ) -> Vec<Vec<Vec<f64>>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let mut dp = vec![vec![vec![0.0_f64; soc_levels]; num_t + 1]; num_b];

        // i23 — SITE 1. Hoisté hors des DEUX nids (b et t) : `nd_congestion_ratio` coûte
        // `num_t` × `compute_flows` ≈ 96 appels par construction de DP, contre un nid en
        // `num_t × num_b × soc_levels × (action_grid+1)` ≈ 10⁶–10⁷ ⇒ 4-5 ordres de grandeur
        // sous le site chaud (gate `A_SITE_IS_NOT_A_COUNT`, i10). Sous le CTRL
        // (`nd_mode == 0`) rien n'est calculé ni alloué ⇒ bit-exactitude PAR CONSTRUCTION.
        let nd_on = hp.nd_mode != 0;
        let nd_ratio: Vec<f64> = if nd_on { nd_congestion_ratio(challenge) } else { Vec::new() };
        let nd_thr = if nd_on { nd_gate_threshold(&nd_ratio, hp.nd_gate) } else { 0.0 };
        let nd_expo: Vec<f64> = if nd_on { nd_ptdf_exposure(challenge, b_to_lines) } else { Vec::new() };
        let nd_scalar = nd_scalar_for_site(hp, 12);

        // SONDE (coût Q nul, `nd_probe != 0` seulement) : la distribution réelle de `ratio`
        // et le seuil que le quantile en tire. Si `thr` vaut q00 ou q100, la porte est un
        // alias d'une constante — ça se lit ici, pas dans un bench raté.
        if hp.nd_probe != 0 && !nd_ratio.is_empty() {
            let mut srt = nd_ratio.clone();
            srt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = srt.len();
            let q = |f: f64| srt[(((n - 1) as f64) * f).round() as usize];
            let above = nd_ratio.iter().filter(|&&r| r >= nd_thr).count();
            println!(
                "MI_PROBE nd n={} q00={:.4} q25={:.4} q50={:.4} q75={:.4} q90={:.4} q100={:.4} gate_q={:.2} thr={:.4} above={} expo_min={:.3} expo_med={:.3}",
                n, q(0.0), q(0.25), q(0.5), q(0.75), q(0.90), q(1.0),
                hp.nd_gate, nd_thr, above,
                nd_expo.iter().cloned().fold(f64::INFINITY, f64::min),
                { let mut e = nd_expo.clone(); e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); if e.is_empty() { 0.0 } else { e[e.len() / 2] } },
            );
        }

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let expo_b = if b < nd_expo.len() { nd_expo[b] } else { 0.0 };

            for t in (0..num_t).rev() {
                let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };

                let mu_adjust: f64 = b_to_lines[b].iter()
                    .map(|&(l, impact)| mu[t][l] * impact)
                    .sum();

                let extra = expected_premiums[t][b] - mu_adjust;
                let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
                let p_buy = p_da + extra;

                // i23 — SITE 1 : dérate état-dépendant. `nd_mode == 0` ⇒ `(nd_scalar, nd_scalar)`
                // = exactement i15 (`nd_scalar_mask == 0` ⇒ `nd_scalar == hp.network_derating`).
                let (der_c, der_d) = if nd_on {
                    let nd_r = if t < nd_ratio.len() { nd_ratio[t] } else { 0.0 };
                    nd_derate_pair(hp, nd_r, nd_thr, expo_b)
                } else {
                    (nd_scalar, nd_scalar)
                };
                let max_pwr_c = bat.power_charge_mw * der_c;
                let max_pwr_d = bat.power_discharge_mw * der_d;

                for i in 0..soc_levels {
                    let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                    let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                        (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                    } else { 0.0 };
                    let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                        (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                    } else { 0.0 };

                    let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0)));
                    let u_max = max_pwr_d.min(discharge_soc_limit.max(0.0));
                    let u_max = u_max.max(u_min);

                    let mut max_val = f64::NEG_INFINITY;
                    let span = u_max - u_min;
                    for j in 0..=action_grid {
                        let u = if span > 0.0 {
                            u_min + span * (j as f64) / (action_grid as f64)
                        } else { u_min };
                        let price = if u > 0.0 { p_sell } else { p_buy };
                        let abs_u = u.abs();
                        let revenue = u * price * dt;
                        let tx = 0.25 * abs_u * dt;
                        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                        let deg = deg_base * deg_base;
                        let profit = revenue - tx - deg;

                        let next_soc_raw = if u < 0.0 {
                            soc + bat.efficiency_charge * (-u) * dt
                        } else {
                            soc - u / bat.efficiency_discharge.max(1e-9) * dt
                        };
                        let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                        let idx0 = (idx_f.floor() as isize).max(0) as usize;
                        let idx0c = idx0.min(soc_levels - 1);
                        let idx1c = (idx0 + 1).min(soc_levels - 1);
                        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                        let v_next = dp[b][t + 1][idx0c] * (1.0 - frac)
                            + dp[b][t + 1][idx1c] * frac;

                        let val = profit + v_next;
                        if val > max_val { max_val = val; }
                    }
                    dp[b][t][i] = max_val;
                }
            }
        }
        dp
    }

    fn ldd_simulate_flows(
        challenge: &Challenge,
        state: &State,
        dp: &[Vec<Vec<f64>>],
        batt_nodes: &[usize],
        ptdf_sparse: &[Vec<(usize, f64)>],
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let sim_pts = 20usize;

        let mut socs: Vec<f64> = state.socs.clone();
        let mut flows_all = vec![vec![0.0_f64; num_l]; num_t];

        for t in 0..num_t {
            let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut bat_actions = vec![0.0_f64; num_b];

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let soc_levels = dp[b][0].len();
                let node = batt_nodes[b];

                let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let u_min = -(bat.power_charge_mw.min(charge_soc_limit.max(0.0)));
                let u_max = bat.power_discharge_mw.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut best_u = 0.0_f64;
                let mut best_val = f64::NEG_INFINITY;
                let span = u_max - u_min;

                for j in 0..=sim_pts {
                    let u = if span > 0.0 { u_min + span * (j as f64) / (sim_pts as f64) } else { u_min };
                    let abs_u = u.abs();
                    let revenue = u * p_da * dt;
                    let tx = 0.25 * abs_u * dt;
                    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                    let deg = deg_base * deg_base;
                    let profit = revenue - tx - deg;

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let t_next = (t + 1).min(num_t);
                    let v_next = dp[b][t_next][idx0c] * (1.0 - frac) + dp[b][t_next][idx1c] * frac;

                    let val = profit + v_next;
                    if val > best_val { best_val = val; best_u = u; }
                }

                bat_actions[b] = best_u;
                let next_soc_raw = if best_u < 0.0 {
                    soc + bat.efficiency_charge * (-best_u) * dt
                } else {
                    soc - best_u / bat.efficiency_discharge.max(1e-9) * dt
                };
                socs[b] = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }

            let mut bat_flows = vec![0.0_f64; num_l];
            for l in 0..num_l {
                for &(b, impact) in &ptdf_sparse[l] {
                    bat_flows[l] += impact * bat_actions[b];
                }
            }
            for l in 0..num_l {
                flows_all[t][l] = exo_flows[l] + bat_flows[l];
            }
        }
        flows_all
    }

    // Simulate SOC trajectories from DP policy, returns soc_traj[t][b].
    fn ldd_simulate_socs(
        challenge: &Challenge,
        state: &State,
        dp: &[Vec<Vec<f64>>],
        batt_nodes: &[usize],
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let dt = 0.25_f64;
        let sim_pts = 20usize;

        let mut socs: Vec<f64> = state.socs.clone();
        let mut soc_traj = vec![vec![0.0_f64; num_b]; num_t];

        for t in 0..num_t {
            for b in 0..num_b {
                soc_traj[t][b] = socs[b];
            }

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let soc_levels = dp[b][0].len();
                let node = batt_nodes[b];

                let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let u_min = -(bat.power_charge_mw.min(charge_soc_limit.max(0.0)));
                let u_max = bat.power_discharge_mw.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut best_u = 0.0_f64;
                let mut best_val = f64::NEG_INFINITY;
                let span = u_max - u_min;

                for j in 0..=sim_pts {
                    let u = if span > 0.0 { u_min + span * (j as f64) / (sim_pts as f64) } else { u_min };
                    let abs_u = u.abs();
                    let revenue = u * p_da * dt;
                    let tx = 0.25 * abs_u * dt;
                    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                    let deg = deg_base * deg_base;
                    let profit = revenue - tx - deg;

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let t_next = (t + 1).min(num_t);
                    let v_next = dp[b][t_next][idx0c] * (1.0 - frac) + dp[b][t_next][idx1c] * frac;

                    let val = profit + v_next;
                    if val > best_val { best_val = val; best_u = u; }
                }

                let next_soc_raw = if best_u < 0.0 {
                    soc + bat.efficiency_charge * (-best_u) * dt
                } else {
                    soc - best_u / bat.efficiency_discharge.max(1e-9) * dt
                };
                socs[b] = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }
        }

        soc_traj
    }

    #[inline]
    fn dp_lambda(dp_b: &[Vec<f64>], t_next: usize, soc: f64, soc_min: f64, soc_span: f64, soc_levels: usize) -> f64 {
        if soc_levels < 2 || t_next >= dp_b.len() {
            return 0.0;
        }
        let delta_s = soc_span / (soc_levels - 1) as f64;
        let idx_f = (soc - soc_min) / soc_span * (soc_levels - 1) as f64;
        let lo = ((idx_f.floor() as isize).max(0) as usize).min(soc_levels - 2);
        let hi = (lo + 1).min(soc_levels - 1);
        let v_lo = dp_b[t_next][lo];
        let v_hi = dp_b[t_next][hi];
        if !v_lo.is_finite() || !v_hi.is_finite() {
            return 0.0;
        }
        (v_hi - v_lo) / delta_s
    }

    #[inline]
    fn optimal_unconstrained_action(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
    ) -> f64 {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        if !rt.is_finite() { return 0.0; }

        let soc = state.socs[b];
        let (u_min, u_max) = state.action_bounds[b];
        if u_min >= u_max { return u_min; }

        let dt = 0.25_f64;
        let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);
        let two_deg = 2.0 * deg_coeff;
        let soc_levels = ca.dp[b][0].len();
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
        let lambda = dp_lambda(&ca.dp[b], t_next, soc, bat.soc_min_mwh, soc_span, soc_levels);
        if !lambda.is_finite() { return 0.0; }

        let mut best_u = 0.0_f64;
        let mut best_val = eval_profit(challenge, state, ca, b, 0.0);
        if !best_val.is_finite() { best_val = f64::NEG_INFINITY; }

        if u_min < -1e-12 {
            let hi = 0.0_f64.min(u_max);
            if u_min < hi {
                let b_c = dt * (lambda * bat.efficiency_charge - rt - 0.25);
                if two_deg > 1e-30 {
                    let raw_charge = (-b_c) / two_deg; 
                    let cand = (-raw_charge.clamp(0.0, (-u_min).max(0.0))).clamp(u_min, hi);
                    if cand >= u_min && cand <= hi {
                        let v = eval_profit(challenge, state, ca, b, cand);
                        if v.is_finite() && v > best_val { best_val = v; best_u = cand; }
                    }
                }
                {
                    let v_lo = eval_profit(challenge, state, ca, b, u_min);
                    if v_lo.is_finite() && v_lo > best_val { best_val = v_lo; best_u = u_min; }
                }
            }
        }

        if u_max > 1e-12 {
            let lo = 0.0_f64.max(u_min);
            if lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (rt - 0.25 - lambda / eff_d);
                if two_deg > 1e-30 {
                    let raw_disch = b_d / two_deg;
                    let cand = raw_disch.clamp(lo, u_max);
                    if cand >= lo && cand <= u_max {
                        let v = eval_profit(challenge, state, ca, b, cand);
                        if v.is_finite() && v > best_val { best_val = v; best_u = cand; }
                    }
                }
                {
                    let v_hi = eval_profit(challenge, state, ca, b, u_max);
                    if v_hi.is_finite() && v_hi > best_val { best_u = u_max; }
                }
            }
        }

        if best_u.is_finite() { best_u } else { 0.0 }
    }

    fn prescreen_binding_lines(
        challenge: &Challenge,
        flows_all: &[Vec<f64>],
        max_lines: usize,
    ) -> Vec<usize> {
        let num_lines = challenge.network.flow_limits.len();
        if num_lines == 0 || flows_all.is_empty() || max_lines == 0 {
            return Vec::new();
        }

        let mut max_viol = vec![0.0_f64; num_lines];
        for t_flows in flows_all.iter() {
            for l in 0..num_lines {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let absflow = t_flows.get(l).copied().unwrap_or(0.0).abs();
                if absflow > limit {
                    let v = absflow - limit;
                    if v > max_viol[l] {
                        max_viol[l] = v;
                    }
                }
            }
        }

        let threshold = 1e-6;
        let mut ranked: Vec<(usize, f64)> = (0..num_lines)
            .filter(|&l| max_viol[l] > threshold * challenge.network.flow_limits[l].max(threshold))
            .map(|l| (l, max_viol[l]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(max_lines);
        ranked.into_iter().map(|(l, _)| l).collect()
    }


    fn build_dw_dual_prices(
        challenge: &Challenge,
        state: &State,
        hp: &TrackHp,
        batt_nodes: &[usize],
        b_to_lines: &[Vec<(usize, f64)>],
        preselected_lines: Option<&[usize]>,
    ) -> Vec<Vec<f64>> {
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        if num_l == 0 || num_b == 0 || num_t == 0 {
            return vec![vec![0.0_f64; num_l]; num_t];
        }

        let active_lines: Vec<usize> = if let Some(preselected) = preselected_lines {
            let k_active = hp.dw_max_lines.min(preselected.len());
            preselected.iter().take(k_active).copied().collect()
        } else {
            let mut line_cong: Vec<(usize, f64)> = Vec::new();
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let mut sum_abs = 0.0_f64;
                for t in 0..num_t {
                    let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                    sum_abs += exo_flows[l].abs();
                }
                let avg = sum_abs / num_t as f64;
                let util = avg / limit;
                line_cong.push((l, util));
            }
            line_cong.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k_active = hp.dw_max_lines.min(line_cong.len());
            line_cong.iter().take(k_active).map(|(l, _)| *l).collect()
        };
        let n_active = active_lines.len();
        if n_active == 0 {
            return vec![vec![0.0_f64; num_l]; num_t];
        }

        let _line_idx_map: Vec<usize> = {
            let mut map = vec![usize::MAX; num_l];
            for (i, &l) in active_lines.iter().enumerate() {
                map[l] = i;
            }
            map
        };

        let ptdf_batt: Vec<Vec<f64>> = (0..num_b).map(|b| {
            (0..n_active).map(|ai| {
                let l = active_lines[ai];
                let val = b_to_lines[b].iter()
                    .find(|&&(ll, _)| ll == l)
                    .map(|&(_, coef)| coef)
                    .unwrap_or(0.0);
                val
            }).collect()
        }).collect();

        let exo_flows_all: Vec<Vec<f64>> = (0..num_t).map(|t| {
            challenge.network.compute_flows(&challenge.exogenous_injections[t])
        }).collect();

        let mut columns: Vec<Vec<f64>> = Vec::new();
        let mut col_batt_idx: Vec<usize> = Vec::new();

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let mut greedy = vec![0.0_f64; num_t];
            let mut soc = state.socs[b];
            let dt = 0.25_f64;

            for t in 0..num_t {
                let da_price = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };
                if !da_price.is_finite() { continue; }

                let eta_c = bat.efficiency_charge;
                let eta_d = bat.efficiency_discharge.max(1e-9);

                let charge_lim = if eta_c > 0.0 {
                    (bat.soc_max_mwh - soc) / (eta_c * dt)
                } else { 0.0 };
                let disch_lim = if eta_d > 0.0 {
                    (soc - bat.soc_min_mwh) * eta_d / dt
                } else { 0.0 };

                let u_min = -(bat.power_charge_mw.min(charge_lim.max(0.0)));
                let u_max = bat.power_discharge_mw.min(disch_lim.max(0.0));
                if u_min >= u_max { continue; }

                let median_price = state.rt_prices.get(node).copied().unwrap_or(da_price);
                let u: f64;
                if median_price > da_price + 5.0 && u_max > 0.0 {
                    u = u_max;
                } else if median_price < da_price - 5.0 && u_min < 0.0 {
                    u = u_min;
                } else {
                    u = 0.0;
                }

                greedy[t] = u;
                let next_soc_raw = if u < 0.0 {
                    soc + eta_c * (-u) * dt
                } else {
                    soc - u / eta_d * dt
                };
                soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }

            columns.push(greedy);
            col_batt_idx.push(b);
        }

        let n_constraints = n_active * num_t;
        let m_penalty: f64 = 1e7; 

        let mut total_pivots_used = 0usize;

        let mut mu = vec![vec![0.0_f64; num_l]; num_t];

        'dw_outer: for _dw_iter in 0..hp.dw_iters {
            let n_cols = columns.len();
            if n_cols == 0 { break 'dw_outer; }

            let n_slacks = n_constraints;
            let total_vars = n_cols + n_slacks;

            let mut c_lp = vec![0.0_f64; total_vars];
            for (j, col) in columns.iter().enumerate() {
                let b_idx = col_batt_idx[j];
                let profit = compute_schedule_profit(challenge, b_idx, col);
                c_lp[j] = if profit.is_finite() { profit } else { 0.0 };
            }
            for s in 0..n_slacks {
                c_lp[n_cols + s] = -m_penalty;
            }

            let mut a_lp: Vec<Vec<f64>> = vec![vec![0.0_f64; total_vars]; n_constraints];
            let mut b_lp = vec![0.0_f64; n_constraints];

            for ai in 0..n_active {
                let l = active_lines[ai];
                let limit = challenge.network.flow_limits[l];
                for t in 0..num_t {
                    let row = ai * num_t + t;
                    let exo = exo_flows_all[t][l];
                    b_lp[row] = (limit - exo).max(0.0);

                    for c in 0..n_cols {
                        let b = col_batt_idx[c];
                        let action_val = columns[c].get(t).copied().unwrap_or(0.0);
                        let ptdf_val = ptdf_batt[b][ai];
                        a_lp[row][c] += action_val * ptdf_val;
                    }

                    a_lp[row][n_cols + row] = -1.0;
                }
            }

            let pivot_budget = hp.dw_pivot_budget_per_solve;
            total_pivots_used += pivot_budget;
            if total_pivots_used > hp.dw_total_pivot_budget {
                break 'dw_outer;
            }

            let (sol, duals_opt, _) = super::lp::lp_solve_with_duals(
                total_vars, n_constraints, &c_lp, &a_lp, &b_lp, pivot_budget, hp.dantzig_in_dw, false,
            );

            let Some(duals) = duals_opt else { break 'dw_outer; };
            let Some(_x_sol) = sol else { break 'dw_outer; };

            let mut mu_new = vec![vec![0.0_f64; num_l]; num_t];
            let mut dual_valid = true;

            for ai in 0..n_active {
                let l = active_lines[ai];
                for t in 0..num_t {
                    let row = ai * num_t + t;
                    if row < duals.len() {
                        let d = duals[row];
                        if d.abs() > 1e6 {
                            dual_valid = false;
                            continue;
                        }
                        mu_new[t][l] = d;
                    }
                }
            }

            if !dual_valid { break 'dw_outer; }

            // Count mismatched pairs for Wentges adaptive alpha + convergence check.
            let total_pairs = n_active * num_t;
            let mut n_diff = 0usize;
            let mut converged = true;
            for t in 0..num_t {
                for ai in 0..n_active {
                    let l = active_lines[ai];
                    if (mu[t][l] - mu_new[t][l]).abs() > 1e-3 {
                        converged = false;
                        n_diff += 1;
                    }
                }
            }
            let delta = hp.dw_boxstep_delta;
            let wentges_on = hp.dw_wentges_alpha_min > 0.0
                || hp.dw_wentges_alpha_max > hp.dw_wentges_alpha_min;
            if wentges_on && delta > 0.0 {
                // Wentges adaptive: clip then scale-down based on misprice_rate.
                // a = alpha_min + (alpha_max - alpha_min) * misprice_rate
                // High misprice → a → alpha_max (more damping); low misprice → a → alpha_min (less).
                let misprice_rate = if total_pairs > 0 { n_diff as f64 / total_pairs as f64 } else { 0.0 };
                let a = (hp.dw_wentges_alpha_min
                    + (hp.dw_wentges_alpha_max - hp.dw_wentges_alpha_min) * misprice_rate)
                    .clamp(hp.dw_wentges_alpha_min.min(hp.dw_wentges_alpha_max),
                           hp.dw_wentges_alpha_min.max(hp.dw_wentges_alpha_max));
                for t in 0..num_t {
                    for ai in 0..n_active {
                        let l = active_lines[ai];
                        let clipped = mu[t][l] + (mu_new[t][l] - mu[t][l]).clamp(-delta, delta);
                        mu[t][l] = a * mu[t][l] + (1.0 - a) * clipped;
                    }
                }
            } else if delta > 0.0 {
                // CTRL: original pure box-step (bit-exact to i36 baseline)
                for t in 0..num_t {
                    for ai in 0..n_active {
                        let l = active_lines[ai];
                        let step = (mu_new[t][l] - mu[t][l]).clamp(-delta, delta);
                        mu[t][l] += step;
                    }
                }
            } else {
                let a = hp.dw_mu_damping_alpha;
                if a == 0.0 {
                    mu = mu_new;
                } else {
                    for t in 0..num_t {
                        for ai in 0..n_active {
                            let l = active_lines[ai];
                            mu[t][l] = a * mu[t][l] + (1.0 - a) * mu_new[t][l];
                        }
                    }
                }
            }
            if converged { break 'dw_outer; }

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = batt_nodes[b];
                let eta_c = bat.efficiency_charge;
                let eta_d = bat.efficiency_discharge.max(1e-9);
                let dt = 0.25_f64;

                let mut eff_prices = vec![0.0_f64; num_t];
                for t in 0..num_t {
                    let da = if node < challenge.market.day_ahead_prices[t].len() {
                        challenge.market.day_ahead_prices[t][node]
                    } else {
                        challenge.market.day_ahead_prices[t][0]
                    };
                    let cong_adj: f64 = (0..n_active).map(|ai| {
                        mu[t][active_lines[ai]] * ptdf_batt[b][ai]
                    }).sum();
                    eff_prices[t] = da - cong_adj;
                }

                let median_eff = eff_prices.iter().fold(0.0_f64, |a, &b| a + b) / num_t as f64;
                let mut cand_soc = state.socs[b];
                let mut candidate_col = vec![0.0_f64; num_t];

                for t in 0..num_t {
                    let eff = eff_prices[t];
                    let charge_lim = if eta_c > 0.0 {
                        (bat.soc_max_mwh - cand_soc) / (eta_c * dt)
                    } else { 0.0 };
                    let disch_lim = if eta_d > 0.0 {
                        (cand_soc - bat.soc_min_mwh) * eta_d / dt
                    } else { 0.0 };

                    let u_min = -(bat.power_charge_mw.min(charge_lim.max(0.0)));
                    let u_max = bat.power_discharge_mw.min(disch_lim.max(0.0));
                    if u_min >= u_max { continue; }

                    let u: f64;
                    if eff > median_eff + 3.0 && u_max > 0.0 {
                        u = u_max;
                    } else if eff < median_eff - 3.0 && u_min < 0.0 {
                        u = u_min;
                    } else {
                        u = 0.0;
                    }

                    candidate_col[t] = u;
                    let next_soc_raw = if u < 0.0 {
                        cand_soc + eta_c * (-u) * dt
                    } else {
                        cand_soc - u / eta_d * dt
                    };
                    cand_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
                }

                let is_duplicate = columns.iter().zip(col_batt_idx.iter())
                    .filter(|(_, &idx)| idx == b)
                    .any(|(col, _)| {
                        col.iter().zip(candidate_col.iter())
                            .all(|(a, c)| (a - c).abs() < 1e-4)
                    });

                if !is_duplicate {
                    columns.push(candidate_col);
                    col_batt_idx.push(b);

                    let new_count = col_batt_idx.iter().filter(|&&idx| idx == b).count();
                    if new_count > hp.dw_max_cols_per_batt {
                        if let Some(pos) = col_batt_idx.iter().position(|&idx| idx == b) {
                            columns.remove(pos);
                            col_batt_idx.remove(pos);
                        }
                    }
                }
            }

            let cols_added = columns.len() > n_cols;
            if !cols_added { break 'dw_outer; }
        }

        for t in 0..num_t {
            for l in 0..num_l {
                if !mu[t][l].is_finite() || mu[t][l].abs() > 1e6 {
                    mu[t][l] = 0.0;
                }
            }
        }

        mu
    }

    fn compute_schedule_profit(challenge: &Challenge, b: usize, schedule: &[f64]) -> f64 {
        let bat = &challenge.batteries[b];
        let dt = 0.25_f64;
        let mut profit = 0.0_f64;

        for t in 0..schedule.len().min(challenge.num_steps) {
            let u = schedule[t];
            if !u.is_finite() { continue; }

            let da_price = if challenge.batteries[b].node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][challenge.batteries[b].node]
            } else {
                challenge.market.day_ahead_prices[t].get(0).copied().unwrap_or(0.0)
            };

            let abs_u = u.abs();
            let revenue = u * da_price * dt;
            let tx = 0.25 * abs_u * dt;
            let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
            let deg = deg_base * deg_base;
            profit += revenue - tx - deg;
        }

        if profit.is_finite() { profit } else { 0.0 }
    }

    fn lns_dw_per_step(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        base_actions: &[f64],
        uncons: &[f64],
        cgs: &mut CgProbe,
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let _t = state.time_step;

        let mut max_util = 0.0_f64;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let util = flows_base[l].abs() / limit;
                if util > max_util { max_util = util; }
            }
        }
        if max_util < hp.kkt_cong_threshold {
            return None; 
        }

        let mut line_set = select_active_lines(challenge, flows_base, ca, Some(base_actions), hp.lns_max_lines);
        let mut n_lines = line_set.len();
        if n_lines == 0 {
            return None;
        }

        let mut col_pools: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            let cur = base_actions[b];
            let uncons_u = uncons[b].clamp(lo, hi);
            let mut seed_actions: Vec<f64> = vec![cur, 0.0, lo, hi, uncons_u];

            if hi - lo > 1e-12 {
                for &frac in &[0.25, 0.5, 0.75] {
                    seed_actions.push(lo + frac * (hi - lo));
                }
            }

            let mut unique: Vec<f64> = Vec::new();
            for &a in &seed_actions {
                if a.is_finite() && !unique.iter().any(|&u| (u - a).abs() < 1e-6) {
                    unique.push(a);
                }
            }
            if unique.len() > hp.lns_cg_column_limit {
                let mut keep = Vec::new();
                let mut has_zero = false;
                let mut has_lo = false;
                let mut has_hi = false;
                for &a in &unique {
                    if keep.is_empty() || (!has_zero && a.abs() < 1e-6) || (!has_lo && (a - lo).abs() < 1e-6) || (!has_hi && (a - hi).abs() < 1e-6) {
                        if !has_zero && a.abs() < 1e-6 { has_zero = true; }
                        if !has_lo && (a - lo).abs() < 1e-6 { has_lo = true; }
                        if !has_hi && (a - hi).abs() < 1e-6 { has_hi = true; }
                        keep.push(a);
                    }
                    if keep.len() >= hp.lns_cg_column_limit { break; }
                }
                unique = keep;
            }
            col_pools[b] = unique;
        }

        let mut best_actions: Option<Vec<f64>> = None;
        let mut best_obj = f64::NEG_INFINITY;
        let mut pivot_budget = hp.lns_lp_pivots_total;
        let mut mu_prev: Vec<f64> = Vec::new();

        // ── i13 — SEUIL DE MATÉRIALITÉ (shadow + appliqué) ───────────────────────────
        // `mat_on == false` (sonde OFF **et** mécanisme OFF) ⇒ aucune des lignes ci-dessous
        // n'exécute autre chose qu'un test de booléen ⇒ CTRL BIT-EXACT par construction.
        let mat_on = hp.lns_cg_probe != 0 || hp.lns_cg_mat_mode != 0;
        let mut mr1 = [f64::INFINITY; CG_MAXK]; // gap de dualité relatif  Σ max(0,rc)/|obj|
        let mut mr2 = [f64::INFINITY; CG_MAXK]; // amélioration marginale réalisée relative
        let mut mr3 = [f64::INFINITY; CG_MAXK]; // déplacement primal max des colonnes / span
        let mut objk = [f64::NEG_INFINITY; CG_MAXK]; // best_obj à la FIN de l'itération k
        let mut addk = [0u64; CG_MAXK];              // colonnes ajoutées à l'itération k
        let mut k_exec = 0usize;
        let mut prev_obj = f64::NEG_INFINITY;

        // ── i14 — CONTEXTE DE PONDÉRATION DU SURROGAT (invariant sur la boucle CG) ─────
        // Ancrage sur l'incumbent : `v_ref[b] = V̂(u_base_b)`, `span_b` pour adimensionner
        // le déplacement, `util_b` = exposition de la batterie à la congestion.
        // ⚠️ Coût STRICTEMENT NUL au défaut : `cg_obj_active` faux ⇒ 3 `Vec::new()` vides,
        // aucun appel à `cont_value`. Le compte est `num_b` (≈ 10¹) par pas, pas un nid.
        let cg_obj_on = cg_obj_active(hp);
        let (cg_vref, cg_span, cg_util): (Vec<f64>, Vec<f64>, Vec<f64>) = if cg_obj_on {
            let mut vr = vec![0.0_f64; num_b];
            let mut sp = vec![1.0_f64; num_b];
            let mut ut = vec![0.0_f64; num_b];
            for b in 0..num_b {
                let v = cont_value(challenge, state, ca, b, base_actions[b]);
                vr[b] = if v.is_finite() { v } else { 0.0 };
                let (lo, hi) = state.action_bounds[b];
                sp[b] = (hi - lo).abs().max(1e-9);
                let mut u_max = 0.0_f64;
                for &(l, _) in &ca.b_to_lines[b] {
                    if l >= num_l { continue; }
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let util = flows_base[l].abs() / limit;
                    if util > u_max { u_max = util; }
                }
                // Clamp à [0,1] : au-delà de la limite la décote est déjà maximale, et un
                // `util` non borné ferait exploser le coefficient sur les nonces violés.
                ut[b] = if u_max.is_finite() { u_max.min(1.0) } else { 0.0 };
            }
            (vr, sp, ut)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        // Accès sûrs : au défaut les Vec sont vides ⇒ valeurs neutres, jamais d'index OOB.
        let cg_vref_at = |b: usize| -> f64 { cg_vref.get(b).copied().unwrap_or(0.0) };
        let cg_span_at = |b: usize| -> f64 { cg_span.get(b).copied().unwrap_or(1.0) };
        let cg_util_at = |b: usize| -> f64 { cg_util.get(b).copied().unwrap_or(0.0) };

        let mut k_started = 0usize;
        for _cg_iter in 0..hp.lns_cg_iters {
            if pivot_budget < 50 { break; }
            k_started += 1;

            // P_ADAPT_LINES (i42): interleaved constraint generation. Re-select the active
            // line set from the CURRENT incumbent each CG iteration so lines that become
            // binding as the solution evolves (and were not stressed by base_actions) enter
            // the master. Capped at hp.lns_max_lines => n_lines <= k => matrix dims bounded
            // identically to the static path (iso-cost LP). Distinct from dual-pricing
            // perturbation (dead): we change WHICH constraints exist, not how duals are
            // computed. Flag OFF (default) => block skipped => line_set/n_lines unchanged
            // => bit-exact baseline i36. First iter (best_actions == None) keeps the base
            // selection so iter-0 is identical even with the flag on.
            if hp.use_adaptive_lines {
                if let Some(ref inc) = best_actions {
                    let new_set = select_active_lines(challenge, flows_base, ca, Some(inc.as_slice()), hp.lns_max_lines);
                    if !new_set.is_empty() {
                        line_set = new_set;
                        n_lines = line_set.len();
                    }
                }
            }

            let n_cols_total: usize = col_pools.iter().map(|v| v.len()).sum();
            if n_cols_total == 0 { break; }

            let m = 2 * num_b + 2 * n_lines;
            let mut c_vec = vec![0.0_f64; n_cols_total];
            let mut a_mat = vec![vec![0.0_f64; n_cols_total]; m];
            let mut b_vec = vec![0.0_f64; m];

            let mut col_offset = 0usize;
            for b in 0..num_b {
                for j in 0..col_pools[b].len() {
                    a_mat[2 * b][col_offset + j] = 1.0;
                }
                b_vec[2 * b] = 1.0;

                for j in 0..col_pools[b].len() {
                    a_mat[2 * b + 1][col_offset + j] = -1.0;
                }
                a_mat[2 * b + 1] = vec![0.0_f64; n_cols_total]; 
                b_vec[2 * b + 1] = 0.0;

                for j in 0..col_pools[b].len() {
                    let u = col_pools[b][j];
                    let node = ca.batt_nodes[b];
                    let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    // i14 SITE 1/2 — coût des colonnes du maître restreint.
                    // Au défaut `eval_cg_value` délègue à `eval_profit_with_price` ⇒
                    // expression flottante INCHANGÉE ⇒ Q bit-exact.
                    let mut profit = eval_cg_value(
                        challenge, state, ca, hp, b, u, rt,
                        cg_vref_at(b), base_actions[b], cg_span_at(b), cg_util_at(b),
                    );
                    if !profit.is_finite() { profit = 0.0; }
                    c_vec[col_offset + j] = profit;
                }

                col_offset += col_pools[b].len();
            }

            for (i, &l) in line_set.iter().enumerate() {
                let limit = challenge.network.flow_limits[l];
                let exo = flows_base[l];

                let row_p = 2 * num_b + 2 * i;
                let mut col_idx = 0usize;
                for b in 0..num_b {
                    let ptdf_coef = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    for j in 0..col_pools[b].len() {
                        a_mat[row_p][col_idx] = ptdf_coef * col_pools[b][j];
                        col_idx += 1;
                    }
                }
                b_vec[row_p] = (limit - exo).max(0.0);

                let row_n = 2 * num_b + 2 * i + 1;
                let mut col_idx = 0usize;
                for b in 0..num_b {
                    let ptdf_coef = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    for j in 0..col_pools[b].len() {
                        a_mat[row_n][col_idx] = -ptdf_coef * col_pools[b][j];
                        col_idx += 1;
                    }
                }
                b_vec[row_n] = (limit + exo).max(0.0);
            }

            let per_solve = (pivot_budget / (hp.lns_cg_iters.max(1))).min(pivot_budget);
            let per_solve = per_solve.max(50);
            let (primal, duals_opt, pivots_used) = super::lp::lp_solve_with_duals(
                n_cols_total, m, &c_vec, &a_mat, &b_vec, per_solve, hp.dantzig_in_lns, hp.dantzig_bland_degen,
            );
            pivot_budget = pivot_budget.saturating_sub(pivots_used);

            let Some(primal_x) = primal else { break; };

            let mut actions_t = vec![0.0_f64; num_b];
            let mut flows_t = flows_base.to_vec();
            let mut line_mask = vec![false; num_l];
            for &l in &line_set {
                if l < num_l { line_mask[l] = true; }
            }

            let mut polish_rank: Vec<(f64, usize)> = Vec::new();
            let mut col_idx = 0usize;
            for b in 0..num_b {
                let n_local = col_pools[b].len();
                let mut blended = 0.0_f64;
                let mut w_sum = 0.0_f64;
                let mut max_w = 0.0_f64;
                let mut support = 0usize;
                for j in 0..n_local {
                    let w = primal_x[col_idx + j].max(0.0);
                    if w > 1e-9 {
                        blended += w * col_pools[b][j];
                        w_sum += w;
                        support += 1;
                        if w > max_w { max_w = w; }
                    }
                }

                let (lo, hi) = state.action_bounds[b];
                let residual = (1.0 - w_sum).clamp(0.0, 1.0);
                let anchor = base_actions[b];
                let mut decoded = (blended + residual * anchor).clamp(lo, hi);
                if !decoded.is_finite() {
                    decoded = anchor.clamp(lo, hi);
                    if !decoded.is_finite() {
                        decoded = 0.0_f64.clamp(lo, hi);
                    }
                }
                actions_t[b] = decoded;

                let mix_mass = (w_sum - max_w).max(0.0);
                let mut touch = 0.0_f64;
                for &(l, imp) in &ca.b_to_lines[b] {
                    if l < num_l && line_mask[l] {
                        touch += imp.abs();
                    }
                }
                let move_mag = (decoded - base_actions[b]).abs();
                if touch > 0.0 && (support >= 2 || mix_mass > 1e-8 || residual > 1e-6 || move_mag > 1e-5) {
                    polish_rank.push((((mix_mass + 0.5 * residual) * (1.0 + move_mag)) + 0.05 * touch, b));
                }

                col_idx += n_local;
            }

            for l in 0..num_l {
                for &(b, imp) in &ca.ptdf_sparse[l] {
                    flows_t[l] += imp * actions_t[b];
                }
            }

            let decoded_feasible = (0..num_l).all(|l| {
                let limit = challenge.network.flow_limits[l];
                flows_t[l].abs() <= limit + 1e-6 * limit.max(1.0)
            });

            let mut obj_val = f64::NEG_INFINITY;
            if decoded_feasible {
                obj_val = (0..num_b)
                    .map(|b| eval_profit(challenge, state, ca, b, actions_t[b]))
                    .sum();

                if !polish_rank.is_empty() {
                    polish_rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let polish_cap = if num_b <= 24 { 12 } else { 8 };
                    let polish_set: Vec<usize> = polish_rank.into_iter().take(polish_cap).map(|(_, b)| b).collect();
                    let ternary_iters = (hp.ternary_iters / 2).max(6);
                    let sweeps = if polish_set.len() <= 8 { 2 } else { 1 };

                    let feasible_interval = |b: usize, cur_u: f64, cur_flows: &[f64]| -> (f64, f64) {
                        let (mut lo, mut hi) = state.action_bounds[b];
                        for &(l, p) in &ca.b_to_lines[b] {
                            if p.abs() < 1e-12 { continue; }
                            let limit = challenge.network.flow_limits[l];
                            let f_other = cur_flows[l] - p * cur_u;
                            let x1 = (-limit - f_other) / p;
                            let x2 = ( limit - f_other) / p;
                            let (llo, hhi) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
                            if llo > lo { lo = llo; }
                            if hhi < hi { hi = hhi; }
                        }
                        if lo > cur_u { lo = cur_u; }
                        if hi < cur_u { hi = cur_u; }
                        if lo > hi { (cur_u, cur_u) } else { (lo, hi) }
                    };

                    let mut polish_actions = actions_t.clone();
                    let mut polish_flows = flows_t.clone();
                    let mut polish_obj = obj_val;

                    for _ in 0..sweeps {
                        let mut improved = false;
                        for &b in &polish_set {
                            let cur = polish_actions[b];
                            let cur_val = eval_profit(challenge, state, ca, b, cur);
                            let (lo, hi) = feasible_interval(b, cur, &polish_flows);
                            if hi - lo < 1e-10 { continue; }

                            let mut best_u = cur;
                            let mut best_total = polish_obj;

                            {
                                let try_u = |u: f64, best_u: &mut f64, best_total: &mut f64| {
                                    if !u.is_finite() || (u - cur).abs() < 1e-10 { return; }
                                    let cand_total = polish_obj - cur_val + eval_profit(challenge, state, ca, b, u);
                                    if cand_total > *best_total + 1e-9 {
                                        *best_total = cand_total;
                                        *best_u = u;
                                    }
                                };

                                try_u(lo, &mut best_u, &mut best_total);
                                try_u(hi, &mut best_u, &mut best_total);
                                if lo <= 0.0 && 0.0 <= hi {
                                    try_u(0.0, &mut best_u, &mut best_total);
                                }
                                let orig = base_actions[b].clamp(lo, hi);
                                try_u(orig, &mut best_u, &mut best_total);

                                if lo < 0.0 {
                                    let hi_neg = 0.0_f64.min(hi);
                                    if lo < hi_neg {
                                        let (u, v) = ternary_search(
                                            |x| polish_obj - cur_val + eval_profit(challenge, state, ca, b, x),
                                            lo,
                                            hi_neg,
                                            ternary_iters,
                                        );
                                        if v > best_total + 1e-9 {
                                            best_total = v;
                                            best_u = u;
                                        }
                                    }
                                }
                                if hi > 0.0 {
                                    let lo_pos = 0.0_f64.max(lo);
                                    if lo_pos < hi {
                                        let (u, v) = ternary_search(
                                            |x| polish_obj - cur_val + eval_profit(challenge, state, ca, b, x),
                                            lo_pos,
                                            hi,
                                            ternary_iters,
                                        );
                                        if v > best_total + 1e-9 {
                                            best_total = v;
                                            best_u = u;
                                        }
                                    }
                                }
                            }

                            let delta = best_u - cur;
                            if delta.abs() > 1e-10 {
                                polish_actions[b] = best_u;
                                for &(l, p) in &ca.b_to_lines[b] {
                                    if l < num_l { polish_flows[l] += p * delta; }
                                }
                                polish_obj = best_total;
                                improved = true;
                            }
                        }
                        if !improved { break; }
                    }

                    let polish_feasible = (0..num_l).all(|l| {
                        let limit = challenge.network.flow_limits[l];
                        polish_flows[l].abs() <= limit + 1e-6 * limit.max(1.0)
                    });

                    if polish_feasible && polish_obj > obj_val + 1e-9 {
                        actions_t = polish_actions;
                        obj_val = polish_obj;
                    }
                }
            }

            if obj_val.is_finite() && obj_val > best_obj {
                best_obj = obj_val;
                best_actions = Some(actions_t.clone());
            }

            let Some(duals) = duals_opt else { break; };

            let line_dual_offset = 2 * num_b;
            if line_dual_offset >= duals.len() { break; }

            let pricing_duals: Vec<f64> = if hp.lns_dual_smooth_alpha > 0.0 && mu_prev.len() == duals.len() {
                duals.iter().zip(mu_prev.iter()).map(|(&d, &p)| {
                    let s = hp.lns_dual_smooth_alpha * p + (1.0 - hp.lns_dual_smooth_alpha) * d;
                    if s.is_finite() { s } else { d }
                }).collect()
            } else {
                duals.clone()
            };
            mu_prev = pricing_duals.clone();

            // i11 mode 2 — SAT-SHORTCIRCUIT. Si toutes les pools sont déjà au plafond, la
            // passe de pricing ci-dessous exécute `continue` pour CHAQUE `b` : elle ne peut
            // ni muter `col_pools` ni mettre `added_any` à vrai, et sort donc par le `break`
            // final. On saute une passe dont la sortie est PROUVÉE VIDE. `mu_prev` vient
            // d'être assigné et n'est plus relu après le break ⇒ Q bit-exact PAR CONSTRUCTION.
            if (hp.lns_cg_stop_mode & 2) != 0
                && (0..num_b).all(|b| col_pools[b].len() >= hp.lns_cg_column_limit)
            {
                break;
            }

            let mut added_any = false;
            // i13 — accumulateurs de MATÉRIALITÉ de l'itération courante (0 coût si OFF).
            let mut added_cnt = 0u64;
            let mut mat_gap = 0.0_f64;
            let mut mat_disp = 0.0_f64;
            for b in 0..num_b {
                if col_pools[b].len() >= hp.lns_cg_column_limit { continue; }

                let node = ca.batt_nodes[b];
                let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };

                let mut cong_adj = 0.0_f64;
                for (i, &l) in line_set.iter().enumerate() {
                    let dual_row = line_dual_offset + 2 * i;
                    if dual_row >= pricing_duals.len() { continue; }
                    let d = pricing_duals[dual_row];
                    if !d.is_finite() { continue; }
                    let ptdf_val = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    cong_adj += d * ptdf_val;
                }

                // GDD non-linear regularizer (i66): augment linear cong_adj with quadratic coupling.
                // gdd_sum = Σ_l max(0, |flow_l|-limit_l) * |ptdf_{b,l}| (per-battery flow violation exposure)
                // cong_adj += ct_gdd_rho * gdd_sum^2
                // "convexifying the sum is tighter than convexifying individually" (6e60694b).
                // When ct_gdd_rho=0.0 → zero-cost branch, iso-binaire i60 EXACT.
                if hp.ct_gdd_rho > 0.0 {
                    let mut gdd_sum = 0.0_f64;
                    for &l in &line_set {
                        let limit = if l < challenge.network.flow_limits.len() {
                            challenge.network.flow_limits[l]
                        } else { continue; };
                        if limit <= 1e-6 { continue; }
                        let viol_l = if l < flows_base.len() {
                            (flows_base[l].abs() - limit).max(0.0)
                        } else { 0.0 };
                        if viol_l <= 0.0 { continue; }
                        let ptdf_abs = ca.b_to_lines[b].iter()
                            .find(|&&(ll, _)| ll == l)
                            .map(|&(_, coef)| coef.abs())
                            .unwrap_or(0.0);
                        gdd_sum += viol_l * ptdf_abs;
                    }
                    cong_adj += hp.ct_gdd_rho * gdd_sum * gdd_sum;
                }

                let eff_price = rt - cong_adj;
                if !eff_price.is_finite() { continue; }

                let (lo, hi) = state.action_bounds[b];
                if hi - lo < 1e-12 { continue; }

                // i14 SITE 2/2 — pricing subproblem (prix lagrangien `eff_price`).
                // Le candidat retenu doit être cherché SOUS LE MÊME objectif que celui du
                // maître, sinon la CG génère des colonnes que le maître n'évalue pas comme
                // elle : le couple (:1786, ici) est indissociable.
                let cg_vref_b = cg_vref_at(b);
                let cg_span_b = cg_span_at(b);
                let cg_util_b = cg_util_at(b);
                let u_base_b = base_actions[b];
                let eval_action = |u: f64| -> f64 {
                    eval_cg_value(
                        challenge, state, ca, hp, b, u, eff_price,
                        cg_vref_b, u_base_b, cg_span_b, cg_util_b,
                    )
                };

                let mut candidate_u = 0.0_f64;
                let mut best_cand_val = eval_action(0.0);

                if lo < 0.0 {
                    let (u, v) = ternary_search(&eval_action, lo, 0.0_f64.min(hi), hp.ternary_iters);
                    if v > best_cand_val { best_cand_val = v; candidate_u = u; }
                }
                if hi > 0.0 {
                    let (u, v) = ternary_search(&eval_action, 0.0_f64.max(lo), hi, hp.ternary_iters);
                    if v > best_cand_val { candidate_u = u; }
                }

                // ── i13 — SIGNAUX DE MATÉRIALITÉ (lecture pure, aucun effet de bord) ──
                // `rc` est le MÊME coût réduit qu'i11 (`eval_action(u*) − σ_b`) : i11 en
                // testait le SIGNE (jamais mordant), on en cumule ici la MAGNITUDE.
                // `Σ_b max(0, rc_b)` majore l'amélioration restante du maître restreint.
                if mat_on && candidate_u.is_finite() {
                    if 2 * b < pricing_duals.len() {
                        let sigma_b = pricing_duals[2 * b];
                        let cand_val = eval_action(candidate_u);
                        if sigma_b.is_finite() && cand_val.is_finite() {
                            let rc = cand_val - sigma_b;
                            if rc > 0.0 { mat_gap += rc; }
                        }
                    }
                    let span = (hi - lo).abs().max(1e-9);
                    let d = (candidate_u - actions_t[b]).abs() / span;
                    if d.is_finite() && d > mat_disp { mat_disp = d; }
                }

                let mut exists = false;
                for &existing in &col_pools[b] {
                    if (existing - candidate_u).abs() < 1e-4 {
                        exists = true;
                        break;
                    }
                }

                // i11 mode 1 — RC-GATE : test d'optimalité canonique de la column generation.
                // La colonne candidate ne peut améliorer le maître que si son COÛT RÉDUIT est
                // strictement positif. Le maître est un `max Σ c_c λ_c` sous des lignes `≤` ;
                // `pricing_duals` sont les duaux ≥ 0 correspondants. Le coût réduit d'une
                // colonne neuve vaut `c_new − σᵀ a_new`, où :
                //   * le terme des lignes de réseau est DÉJÀ net dans `eval_action`, qui
                //     évalue au prix lagrangien `eff_price = rt − cong_adj` avec
                //     `cong_adj = Σ_i d_i · ptdf_{b,l_i}` ;
                //   * la ligne de convexité `Σ_j λ_bj ≤ 1` (indice `2*b`, cf assemblage)
                //     apporte `σ_b · 1` ;
                //   * la ligne `2*b+1` est mise à zéro à l'assemblage ⇒ contribution nulle.
                // D'où `rc = eval_action(candidate_u) − σ_b`. On réévalue `candidate_u`
                // explicitement (1 appel, contre ~40 pour les deux recherches ternaires) :
                // `best_cand_val` n'est pas mis à jour dans la seconde branche du CTRL et ne
                // vaut donc pas toujours la valeur du candidat retenu.
                let rc_ok = if (hp.lns_cg_stop_mode & 1) != 0 {
                    const LNS_RC_TOL: f64 = 1e-9;
                    if 2 * b < pricing_duals.len() {
                        let sigma_b = pricing_duals[2 * b];
                        let cand_val = eval_action(candidate_u);
                        sigma_b.is_finite()
                            && cand_val.is_finite()
                            && (cand_val - sigma_b) > LNS_RC_TOL
                    } else {
                        true
                    }
                } else {
                    true
                };

                if !exists && candidate_u.is_finite() && rc_ok {
                    col_pools[b].push(candidate_u);
                    added_any = true;
                    added_cnt += 1;
                }
            }

            // ── i13 — CLÔTURE DE L'ITÉRATION k : enregistrement puis décision ─────────
            // Les 3 signaux sont enregistrés pour TOUTE itération complète, que le
            // mécanisme soit appliqué ou seulement observé. L'index d'enregistrement
            // `k_exec` ne compte que les itérations qui atteignent ce point (les `break`
            // amont terminent la boucle de toute façon : aucune décision n'y est possible).
            if mat_on {
                if k_exec < CG_MAXK {
                    let denom = if best_obj.is_finite() { best_obj.abs().max(1e-9) } else { f64::INFINITY };
                    mr1[k_exec] = mat_gap / denom;
                    mr2[k_exec] = if k_exec >= 1 && prev_obj.is_finite() && best_obj.is_finite() {
                        (best_obj - prev_obj) / prev_obj.abs().max(1e-9)
                    } else {
                        f64::INFINITY
                    };
                    mr3[k_exec] = mat_disp;
                    objk[k_exec] = best_obj;
                    addk[k_exec] = added_cnt;
                }
                prev_obj = best_obj;
            }
            k_exec += 1;

            // Mécanisme APPLIQUÉ. `lns_cg_mat_mode == 0` ⇒ `false` sans lire un seul f64.
            let mat_fire = if hp.lns_cg_mat_mode != 0 && k_exec <= CG_MAXK {
                let v = match hp.lns_cg_mat_mode {
                    1 => mr1[k_exec - 1],
                    2 => mr2[k_exec - 1],
                    3 => mr3[k_exec - 1],
                    _ => f64::INFINITY,
                };
                v <= hp.lns_cg_mat_delta
            } else {
                false
            };

            if !added_any || mat_fire { break; }
        }

        // ── i13 — COMPTABILITÉ SHADOW : 9 familles jugées pour 1 bench, coût Q NUL ────
        // `final_obj` est pris AVANT `use_primal_refine` : c'est le seul contrefactuel
        // EXACT dont on dispose (le refine post-boucle relit `col_pools`, qu'un arrêt
        // précoce rétrécit ⇒ `pool` chiffre ce risque résiduel, que seul le bench tranche).
        if hp.lns_cg_probe != 0 {
            let final_obj = best_obj;
            let n_rec = k_exec.min(CG_MAXK);
            cgs.calls = cgs.calls.saturating_add(1);
            cgs.iters = cgs.iters.saturating_add(k_started as u64);
            cgs.iters_ge1 = cgs.iters_ge1.saturating_add(k_started.saturating_sub(1) as u64);
            const DG: [[f64; 3]; 3] = [
                [1e-6, 1e-4, 1e-2], // M1 gap de dualité relatif
                [1e-6, 1e-4, 1e-2], // M2 amélioration marginale réalisée
                [1e-4, 1e-2, 1e-1], // M3 déplacement primal / span
            ];
            for fam in 0..CG_FAMS {
                let mech = fam / 3;
                let d = DG[mech][fam % 3];
                let mrs: &[f64; CG_MAXK] = match mech {
                    0 => &mr1,
                    1 => &mr2,
                    _ => &mr3,
                };
                for k in 0..n_rec {
                    if mrs[k] <= d {
                        let skipped = k_started.saturating_sub(1).saturating_sub(k);
                        cgs.hits[fam] = cgs.hits[fam].saturating_add(1);
                        cgs.fire[fam] = cgs.fire[fam].saturating_add(skipped as u64);
                        if final_obj > objk[k] {
                            cgs.bad[fam] = cgs.bad[fam].saturating_add(1);
                        }
                        let mut lost = 0u64;
                        for kk in (k + 1)..n_rec {
                            lost = lost.saturating_add(addk[kk]);
                        }
                        cgs.pool[fam] = cgs.pool[fam].saturating_add(lost);
                        break;
                    }
                }
            }
        }

        // Primal face refinement: re-solve final LP with combined objective (primary + eps*SoC-lookahead)
        // to select among degenerate optimal vertices. Orthogonal to all dual-pricing axes (dead).
        if hp.use_primal_refine {
            let n_cols_fin: usize = col_pools.iter().map(|v| v.len()).sum();
            if n_cols_fin > 0 {
                let m_fin = 2 * num_b + 2 * n_lines;
                let mut c_fin = vec![0.0_f64; n_cols_fin];
                let mut c2_fin = vec![0.0_f64; n_cols_fin];
                let mut a_fin = vec![vec![0.0_f64; n_cols_fin]; m_fin];
                let mut b_fin = vec![0.0_f64; m_fin];

                // dp_lambda: marginal value of SoC at next step (proxy for tie-breaking)
                let dp_lam: Vec<f64> = (0..num_b).map(|b| {
                    let bat = &challenge.batteries[b];
                    let soc = state.socs[b];
                    let soc_sp = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                    let soc_lv = ca.dp[b].first().map(|r| r.len()).unwrap_or(2);
                    let t_nx = (state.time_step + 1).min(ca.dp[b].len().saturating_sub(1));
                    dp_lambda(&ca.dp[b], t_nx, soc, bat.soc_min_mwh, soc_sp, soc_lv)
                }).collect();

                let mut col_off = 0usize;
                for b in 0..num_b {
                    let node = ca.batt_nodes[b];
                    let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    let lam = dp_lam[b];
                    for j in 0..col_pools[b].len() {
                        let u = col_pools[b][j];
                        let mut profit = eval_profit_with_price(challenge, state, ca, b, u, rt);
                        if !profit.is_finite() { profit = 0.0; }
                        c_fin[col_off + j] = profit;
                        c2_fin[col_off + j] = if lam.is_finite() { lam * u } else { 0.0 };
                        a_fin[2 * b][col_off + j] = 1.0;
                    }
                    b_fin[2 * b] = 1.0;
                    if hp.use_slp_degradation {
                        // SLP: linearize deg(u)=(dt*u/cap)^2 around incumbent u_inc.
                        // eval_profit includes exact -deg_c*u^2; replace with linear approx -2*deg_c*u_inc*u.
                        // net correction: deg_c*u^2 - 2*deg_c*u_inc*u = deg_c*u*(u - 2*u_inc)
                        let cap = challenge.batteries[b].capacity_mwh.max(1e-9);
                        let deg_c = (0.25_f64 / cap).powi(2);
                        let u_inc = base_actions[b];
                        for j in 0..col_pools[b].len() {
                            let u = col_pools[b][j];
                            c_fin[col_off + j] += deg_c * u * (u - 2.0 * u_inc);
                        }
                    }
                    col_off += col_pools[b].len();
                }
                for (i, &l) in line_set.iter().enumerate() {
                    let limit = challenge.network.flow_limits[l];
                    let exo = flows_base[l];
                    let row_p = 2 * num_b + 2 * i;
                    let row_n = 2 * num_b + 2 * i + 1;
                    let mut col_idx = 0usize;
                    for b in 0..num_b {
                        let ptdf_coef = ca.b_to_lines[b].iter()
                            .find(|&&(ll, _)| ll == l)
                            .map(|&(_, coef)| coef)
                            .unwrap_or(0.0);
                        for j in 0..col_pools[b].len() {
                            let u = col_pools[b][j];
                            a_fin[row_p][col_idx] = ptdf_coef * u;
                            a_fin[row_n][col_idx] = -ptdf_coef * u;
                            col_idx += 1;
                        }
                    }
                    b_fin[row_p] = (limit - exo).max(0.0);
                    b_fin[row_n] = (limit + exo).max(0.0);
                }

                // Combined objective: primary + small secondary (tie-breaks degenerate vertices)
                const REFINE_EPS: f64 = 1e-7;
                let c_comb: Vec<f64> = c_fin.iter().zip(c2_fin.iter())
                    .map(|(p, s)| p + REFINE_EPS * s)
                    .collect();

                let (opt_x2, _) = super::lp::lp_solve_with_budget(
                    n_cols_fin, m_fin, &c_comb, &a_fin, &b_fin, 20,
                );
                if let Some(primal2) = opt_x2 {
                    let mut actions_ref = vec![0.0_f64; num_b];
                    let mut col_idx = 0usize;
                    for b in 0..num_b {
                        let n_loc = col_pools[b].len();
                        let mut blended = 0.0_f64;
                        let mut w_sum = 0.0_f64;
                        for j in 0..n_loc {
                            let w = primal2[col_idx + j].max(0.0);
                            if w > 1e-9 { blended += w * col_pools[b][j]; w_sum += w; }
                        }
                        let (lo, hi) = state.action_bounds[b];
                        let residual = (1.0 - w_sum).clamp(0.0, 1.0);
                        let decoded = (blended + residual * base_actions[b]).clamp(lo, hi);
                        actions_ref[b] = if decoded.is_finite() { decoded } else { base_actions[b].clamp(lo, hi) };
                        col_idx += n_loc;
                    }
                    let mut flows_ref = flows_base.to_vec();
                    for l in 0..num_l {
                        for &(b, imp) in &ca.ptdf_sparse[l] {
                            flows_ref[l] += imp * actions_ref[b];
                        }
                    }
                    let feasible_ref = (0..num_l).all(|l| {
                        let limit = challenge.network.flow_limits[l];
                        flows_ref[l].abs() <= limit + 1e-6 * limit.max(1.0)
                    });
                    if feasible_ref {
                        let obj_ref: f64 = (0..num_b)
                            .map(|b| eval_profit(challenge, state, ca, b, actions_ref[b]))
                            .sum();
                        if obj_ref > best_obj + 1e-9 {
                            best_obj = obj_ref;
                            best_actions = Some(actions_ref);
                        }
                    }
                }
            }
        }

        let mut actions = best_actions?;

        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            actions[b] = actions[b].clamp(lo, hi);
            if !actions[b].is_finite() {
                actions[b] = 0.0;
            }
        }

        Some(actions)
    }

    fn compute_lp_budget_for_step(hp: &TrackHp, consumed: usize, reserve: &mut isize) -> usize {
        let base = hp.lp_per_call_pivots;
        if base == 0 { return 0; }
        let total = hp.lp_total_pivots;
        let remaining = if total > 0 { total.saturating_sub(consumed) } else { base };
        if remaining == 0 { return 0; }

        let reserve_usable = (*reserve).min(base as isize).max(0) as usize;
        let budget = base + reserve_usable;
        budget.min(remaining).min(2 * base)
    }

    // i12 (t50, DFL-seed proxy): decision-regret reshaping of the congestion-premium response.
    // proba in [0,1] (linear congestion ramp) -> proba^gamma. gamma=1.0 is bit-exact identity
    // (fallback). gamma>1 (convex) keeps the premium low on marginally-congested lines and ramps
    // it up sharply as a line approaches binding (ratio->1), concentrating the anticipatory signal
    // where a mispriced dispatch carries the largest realized-congestion regret. Build-time
    
    #[inline(always)]
    fn shape_proba(proba: f64, gamma: f64) -> f64 {
        if gamma == 1.0 { proba } else { proba.powf(gamma) }
    }

    fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> AycdicdbCache {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;
        let num_n = challenge.network.num_nodes;

        let zero_action = vec![0.0_f64; num_b];
        let inj_base = challenge.compute_total_injections(state, &zero_action);
        let flows0 = challenge.network.compute_flows(&inj_base);

        let mut batt_nodes = vec![0usize; num_b];
        let mut ptdf_sparse: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
        let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
        let mut dummy = zero_action.clone();
        for b in 0..num_b {
            dummy[b] = 1.0;
            let inj1 = challenge.compute_total_injections(state, &dummy);
            let flows1 = challenge.network.compute_flows(&inj1);
            for k in 0..num_n {
                if (inj1[k] - inj_base[k]).abs() > 0.5 && k != challenge.network.slack_bus {
                    batt_nodes[b] = k;
                    break;
                }
            }
            for l in 0..num_l {
                let impact = flows1[l] - flows0[l];
                if impact.abs() > 1e-8 {
                    ptdf_sparse[l].push((b, impact));
                    b_to_lines[b].push((l, impact));
                }
            }
            dummy[b] = 0.0;
        }

        // i19/i20/i21/i24 L7: per-line weight normalized to mean=1.
        
        //      mean_proba_l = mean_t(proba_l(t)) from exogenous flows, per-nonce (topology-robust).
        //      β3 = cos_line_weight_scale (default 1.0). No footprint term → distinct from dead probes.
        
        
        
        // CTRL path (all false) → all 1.0, zero-cost branch.
        let line_weights: Vec<f64> = {
            if hp.use_cos_cs_weights {
                // L7 complet COS: cost-sensitive weighted proba^gamma mean.
                // w_l = 0.5 + β3 × (Σ_t c_{l,t} × proba^γ / Σ_t c_{l,t}), normalized to mean=1.
                // c_{l,t} = cos_alpha_under if ratio_l(t) > lmp_threshold, else 1.0.
                // Episodic near-binding timesteps get alpha_under× weight → amplifies lines
                // with rare but severe congestion (high regret moments). alpha_under=1.0 →
                // uniform c → Σ c×proba/Σ c = mean_proba → BIT-EXACT to use_cos_weights P29.
                let beta3 = hp.cos_line_weight_scale;
                let alpha_u = hp.cos_alpha_under;
                let mut wls_num = vec![0.0_f64; num_l];
                let mut wls_den = vec![0.0_f64; num_l];
                for t in 0..num_t {
                    let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        if limit <= 1e-6 { continue; }
                        let ratio = f_exo[l].abs() / limit;
                        let c = if ratio > hp.lmp_threshold { alpha_u } else { 1.0 };
                        let proba_gamma = if ratio > hp.lmp_threshold {
                            ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                                .clamp(0.0, 1.0)
                                .powf(hp.premium_shape_gamma)
                        } else { 0.0 };
                        wls_num[l] += c * proba_gamma;
                        wls_den[l] += c;
                    }
                }
                let raw: Vec<f64> = (0..num_l).map(|l| {
                    let cs_mean = if wls_den[l] > 1e-12 { wls_num[l] / wls_den[l] } else { 0.0 };
                    0.5 + beta3 * cs_mean
                }).collect();
                let mean_w = raw.iter().sum::<f64>() / raw.len().max(1) as f64;
                if mean_w < 1e-12 {
                    vec![1.0_f64; num_l]
                } else {
                    raw.iter()
                        .map(|&w| (w / mean_w).clamp(hp.line_weight_w_min, hp.line_weight_w_max))
                        .collect()
                }
            } else if hp.use_cos_weights {
                // Feature-parametric COS: w_l = base(proba_l) * (1 + kappa * asym_l), normalized to mean=1.
                // base(proba_l) = 0.5 + beta3 * mean_proba_l  (P29, gamma-shaped)
                // asym_l = mean_t(proba_linear_l) — linear excess over threshold (no gamma shaping),
                //   captures proximity-to-binding as a decision-aware asymmetry signal (COS §3.2).
                //   Under-pricing near-binding lines costs >> over-pricing lax lines (SPO+ asymmetry).
                // kappa=0.0 → w_l = base(proba_l) → P29 BIT-EXACT sentinel.
                let beta3 = hp.cos_line_weight_scale;
                let kappa = hp.cos_asymmetry_kappa;
                let n_t_f = num_t as f64;
                let mut mean_proba = vec![0.0_f64; num_l];   // Σ_t proba^gamma / T
                let mut mean_excess = vec![0.0_f64; num_l];  // Σ_t proba_linear / T (kappa term)
                for t in 0..num_t {
                    let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        if limit <= 1e-6 { continue; }
                        let ratio = f_exo[l].abs() / limit;
                        if ratio > hp.lmp_threshold {
                            let p_linear = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                                .clamp(0.0, 1.0);
                            mean_proba[l] += p_linear.powf(hp.premium_shape_gamma) / n_t_f;
                            mean_excess[l] += p_linear / n_t_f;
                        }
                    }
                }
                let raw: Vec<f64> = (0..num_l).map(|l| {
                    let base = 0.5 + beta3 * mean_proba[l];
                    let asym_boost = if kappa > 0.0 { 1.0 + kappa * mean_excess[l] } else { 1.0 };
                    base * asym_boost
                }).collect();
                let mean_w = raw.iter().sum::<f64>() / raw.len().max(1) as f64;
                if mean_w < 1e-12 {
                    vec![1.0_f64; num_l]
                } else {
                    raw.iter()
                        .map(|&w| (w / mean_w).clamp(hp.line_weight_w_min, hp.line_weight_w_max))
                        .collect()
                }
            } else if hp.use_sqrt_cap_weights {
                let raw: Vec<f64> = (0..num_l)
                    .map(|l| {
                        let limit = challenge.network.flow_limits[l];
                        if limit < 1e-6 { 0.0 }
                        else { ptdf_sparse[l].iter().map(|&(_, imp)| imp.abs()).sum::<f64>() / limit.sqrt() }
                    })
                    .collect();
                let mean = raw.iter().sum::<f64>() / (raw.len().max(1) as f64);
                if mean < 1e-12 { vec![1.0_f64; num_l] }
                else {
                    raw.iter()
                        .map(|&w| (w / mean).clamp(hp.line_weight_w_min, hp.line_weight_w_max))
                        .collect()
                }
            } else if hp.use_cap_norm_weights {
                let raw: Vec<f64> = (0..num_l)
                    .map(|l| {
                        let limit = challenge.network.flow_limits[l];
                        if limit < 1e-6 { 0.0 }
                        else { ptdf_sparse[l].iter().map(|&(_, imp)| imp.abs()).sum::<f64>() / limit }
                    })
                    .collect();
                let mean = raw.iter().sum::<f64>() / (raw.len().max(1) as f64);
                if mean < 1e-12 { vec![1.0_f64; num_l] }
                else {
                    raw.iter()
                        .map(|&w| (w / mean).clamp(hp.line_weight_w_min, hp.line_weight_w_max))
                        .collect()
                }
            } else if hp.use_learned_line_weights {
                let raw: Vec<f64> = (0..num_l)
                    .map(|l| ptdf_sparse[l].iter().map(|&(_, imp)| imp.abs()).sum::<f64>())
                    .collect();
                let mean = raw.iter().sum::<f64>() / (raw.len().max(1) as f64);
                if mean < 1e-12 { vec![1.0_f64; num_l] }
                else {
                    raw.iter()
                        .map(|&w| (w / mean).clamp(hp.line_weight_w_min, hp.line_weight_w_max))
                        .collect()
                }
            } else {
                vec![1.0_f64; num_l]
            }
        };

        let mut expected_premiums = vec![vec![0.0_f64; num_b]; num_t];
        if hp.use_action_aware_premium && hp.anticipate_lmp && num_l > 0 {
            // Lagged one-shot: cheap probe DP with f_exo premiums → ldd_simulate_flows
            // → total flows (exo + battery-induced) → rebuild premiums from total flows.
            // This aligns the premium signal with actual congestion including battery dispatch,
            // matching the lagged-indicator mechanism in the TIG challenge settlement.
            let base_premium = 20.0 * hp.lmp_premium_scale;
            // Step 1: build f_exo-based seed premiums
            let mut seed_premiums = vec![vec![0.0_f64; num_b]; num_t];
            for t in 0..num_t {
                let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let ratio = f_exo[l].abs() / limit;
                    if ratio > hp.lmp_threshold {
                        let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6)).clamp(0.0, 1.0);
                        let premium = base_premium * shape_proba(proba, hp.premium_shape_gamma);
                        let sign_f = f_exo[l].signum();
                        for &(b, impact) in &ptdf_sparse[l] {
                            if impact.abs() > 1e-6 {
                                seed_premiums[t][b] += line_weights[l] * (-impact.signum() * impact.abs().powf(hp.premium_impact_delta) * sign_f * premium);
                            }
                        }
                    }
                }
            }
            // Step 2: cheap probe DP (reduced resolution) to simulate actions
            let mut hp_probe = hp.clone();
            hp_probe.soc_levels = 31;
            hp_probe.action_grid = 15;
            let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
            let dp_probe = build_dp_parallel_or_serial(challenge, &hp_probe, &batt_nodes, &seed_premiums, &b_to_lines, &mu_zero);
            // Step 3: total flows = exo + battery-induced (from probe dispatch policy)
            let flows_total = ldd_simulate_flows(challenge, state, &dp_probe, &batt_nodes, &ptdf_sparse);
            // Step 4: rebuild premiums from total flows
            for t in 0..num_t {
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let ratio = flows_total[t][l].abs() / limit;
                    if ratio > hp.lmp_threshold {
                        let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6)).clamp(0.0, 1.0);
                        let premium = base_premium * shape_proba(proba, hp.premium_shape_gamma);
                        let sign_f = flows_total[t][l].signum();
                        for &(b, impact) in &ptdf_sparse[l] {
                            if impact.abs() > 1e-6 {
                                expected_premiums[t][b] += line_weights[l] * (-impact.signum() * impact.abs().powf(hp.premium_impact_delta) * sign_f * premium);
                            }
                        }
                    }
                }
            }
        } else if hp.anticipate_lmp && num_l > 0 {
            let base_premium = 20.0 * hp.lmp_premium_scale;
            for t in 0..num_t {
                let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let ratio = f_exo[l].abs() / limit;
                    if ratio > hp.lmp_threshold {
                        let proba = if hp.use_binary_congestion_premium {
                            1.0_f64
                        } else if hp.congestion_quantize_levels >= 2 {
                            let continuous = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6)).clamp(0.0, 1.0);
                            let levels = hp.congestion_quantize_levels as f64;
                            (continuous * (levels - 1.0)).round() / (levels - 1.0)
                        } else {
                            ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                                .clamp(0.0, 1.0)
                        };
                        let premium = base_premium * shape_proba(proba, hp.premium_shape_gamma);
                        let sign_f = f_exo[l].signum();
                        for &(b, impact) in &ptdf_sparse[l] {
                            if impact.abs() > 1e-6 {
                                let nodal_shift = line_weights[l] * (-impact.signum() * impact.abs().powf(hp.premium_impact_delta) * sign_f * premium);
                                expected_premiums[t][b] += nodal_shift;
                            }
                        }
                    }
                }
            }
        }

        if hp.use_kkt && !hp.use_lmp_premiums_kkt {
            for row in expected_premiums.iter_mut() {
                for v in row.iter_mut() { *v = 0.0; }
            }
        }

        let dp = if hp.use_dw && num_l > 0 {
            // One DW solve given a premium seed: prescreen → coordinated dual prices mu_dw → final DP.
            // Returns (final dp, mu_dw). mu_dw[t][l] = congestion component of the LMP at the
            let dw_solve = |ep: &Vec<Vec<f64>>| -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
                let prescreen_lines: Vec<usize> = if hp.use_dw_prescreen {
                    // i7 — LE PRESCREEN EST UNE SONDE, PAS UNE VALEUR.
                    //
                    // Ce `dp0` est construit à mu=0 puis simulé par `ldd_simulate_flows` ; son
                    // UNIQUE consommateur est `prescreen_binding_lines`, qui rend un ENSEMBLE
                    // ORDONNÉ de ≤ `dw_max_lines` (=10) INDICES de lignes. Aucun octet de `dp0`
                    // ni des magnitudes de flux ne survit : seul le RANG des lignes par violation
                    // max compte. Or `dp0` est bâti en PLEINE résolution (soc=101, action=40) ET
                    // en mode stochastique (quadrature à queue, ~3 nœuds) — la fidélité la plus
                    // chère du fichier pour produire 10 entiers.
                    //
                    // Le fichier fait DÉJÀ ce rôle de sonde en fidélité réduite à deux endroits :
                    //   - reconstruction OCO      : `soc/2`, `action/2`  (« ~4× cheaper »)
                    //   - sonde ADMM `dp_low`     : `soc=31`, `action=15`, déterministe
                    // Ces deux sites consomment pourtant des flux de façon plus exigeante que
                    // nous (magnitudes continues injectées dans des primes), alors qu'ici on ne
                    // consomme qu'un rang. La pleine résolution au prescreen est donc une
                    // ASYMÉTRIE non justifiée, pas un choix.
                    //
                    // ⚠️ Contrairement à i6, ce coup N'EST PAS Q-neutre par construction : si la
                    // grille grossière réordonne les lignes, l'ensemble sélectionné change et
                    // `build_dw_dual_prices` travaille sur d'autres contraintes ⇒ Q bouge. Le
                    // coup se juge donc À LA MESURE. Antécédent qui impose la prudence :
                    // Hindsight `f32d19ff` / `9fd8d743` (Pivot J, 10/06) — un DP basse résolution
                    // a produit des VIOLATIONS incorrectes qui ont désaligné l'init LDD. La
                    // différence discriminante ici : ce consommateur-ci ne lit pas la MAGNITUDE
                    // de la violation, seulement l'ORDRE de 10 lignes sur ~num_l, quantité bien
                    // plus robuste au grain de la grille. C'est exactement ce que le probe
                    // ci-dessous mesure au lieu de le supposer.
                    let hp_ps: TrackHp = match hp.dw_prescreen_mode {
                        // Famille A — grain de la GRILLE (soc × action).
                        1 => {
                            let mut h = hp.clone();
                            h.soc_levels = (hp.soc_levels / 2).max(31);
                            h.action_grid = (hp.action_grid / 2).max(15);
                            h
                        }
                        // Famille B — fidélité du MODÈLE D'INCERTITUDE (quadrature à queue).
                        // Grille pleine conservée : les magnitudes de flux restent fines, on ne
                        // retire que l'espérance sur les nœuds de prix (`num_q` ≈ 3 → 1 passe).
                        2 => {
                            let mut h = hp.clone();
                            h.use_sdp = false;
                            h
                        }
                        // Famille C — composition A+B, = configuration littérale de la sonde ADMM.
                        3 => {
                            let mut h = hp.clone();
                            h.soc_levels = 31;
                            h.action_grid = 15;
                            h.use_sdp = false;
                            h
                        }
                        _ => hp.clone(),
                    };
                    let zero_mu = vec![vec![0.0_f64; num_l]; num_t];
                    let dp0 = build_dp_parallel_or_serial(challenge, &hp_ps, &batt_nodes, ep, &b_to_lines, &zero_mu);
                    // `ldd_simulate_flows` lit `soc_levels` depuis `dp[b][0].len()` et échantillonne
                    // l'action sur `sim_pts=20` fixes : elle est agnostique à la forme de `dp0`.
                    let flows_all = ldd_simulate_flows(challenge, state, &dp0, &batt_nodes, &ptdf_sparse);
                    let lines = prescreen_binding_lines(challenge, &flows_all, hp.dw_max_lines);
                    // CONTRÔLE DU DÉTECTEUR — on journalise ce que la sonde PRODUIT, pas ce
                    // qu'elle coûte : l'ensemble ordonné exact. Si, nonce par nonce, l'arme rend
                    // la MÊME liste que le CTRL, alors la neutralité de Q est PROUVÉE et le temps
                    // rendu est gratuit ; si les listes diffèrent, on sait immédiatement où
                    // attribuer un éventuel écart de Q au lieu de l'inférer.
                    {
                        let mut ids = String::new();
                        for (i, &l) in lines.iter().enumerate() {
                            if i > 0 { ids.push(','); }
                            ids.push_str(&l.to_string());
                        }
                        eprintln!(
                            "MI_PROBE ps mode={} soc={} act={} sdp={} n={} ids=[{}]",
                            hp.dw_prescreen_mode, hp_ps.soc_levels, hp_ps.action_grid,
                            hp_ps.use_sdp as u8, lines.len(), ids
                        );
                    }
                    lines
                } else {
                    Vec::new()
                };
                let ps: Option<&[usize]> = if prescreen_lines.is_empty() { None } else { Some(&prescreen_lines) };
                let mu_dw = build_dw_dual_prices(challenge, state, hp, &batt_nodes, &b_to_lines, ps);
                let dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, ep, &b_to_lines, &mu_dw);
                (dp, mu_dw)
            };
            let (dp1, mu_dw1) = dw_solve(&expected_premiums);
            if hp.use_ptdf_constraint_tracking && hp.coord_premium_mode == 0 && num_l > 0 {
                // OCO constraint tracking (i55): adaptive Lagrange multiplier on PTDF flow violations.
                // Discriminant vs i45 (INERTE LP-dual fold): μ_l_oco tracks real DP-dispatch violations,
                // not LP shadow prices → non-trivial fixed point (sublinear regret, Huang OCO 78025f66).
                let flows_all = ldd_simulate_flows(challenge, state, &dp1, &batt_nodes, &ptdf_sparse);
                let eta = hp.ct_step_eta;
                let mut mu_oco = vec![vec![0.0_f64; num_l]; num_t];
                if hp.use_ct_adaptive_per_line {
                    
                    // 1) find max_viol_t = max_l(viol_l), then η_l = η × (viol_l / max_viol_t).
                    // Concentrates subgradient mass on the worst violator, attenuates marginal lines
                    // (SA-PD ablation `1eccc053`: adaptive step reduces violation 60 %).
                    for t in 0..num_t {
                        let mut max_viol_t = 0.0_f64;
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let v = flows_all[t][l].abs() - limit;
                            if v > max_viol_t { max_viol_t = v; }
                        }
                        if max_viol_t <= 0.0 { continue; }
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let viol = flows_all[t][l].abs() - limit;
                            if viol > 0.0 {
                                let eta_l = eta * (viol / max_viol_t);
                                mu_oco[t][l] = (eta_l * viol).min(limit * 0.5);
                            }
                        }
                    }
                } else {
                    for t in 0..num_t {
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let viol = flows_all[t][l].abs() - limit;
                            if viol > 0.0 {
                                mu_oco[t][l] = (eta * viol).min(limit * 0.5);
                            }
                        }
                    }
                }
                let mut ep_ct = expected_premiums.clone();
                
                // to prevent myopic over-correction. When ct_ref_kappa=0: full CT = iso-binaire i55.
                // When kappa>0: partial CT, anchors toward pre-CT premiums (mitigates over-redirection).
                let ct_scale = 1.0 - hp.ct_ref_kappa;
                for t in 0..num_t {
                    for l in 0..num_l {
                        let mu_l = mu_oco[t][l];
                        if mu_l <= 1e-12 { continue; }
                        let sign = flows_all[t][l].signum();
                        for &(b, impact) in &ptdf_sparse[l] {
                            if impact.abs() > 1e-6 {
                                ep_ct[t][b] -= impact * sign * mu_l * ct_scale;
                            }
                        }
                    }
                }
                // GDD exponential regularizer (i68): normalized violation + Sinha-Vaze exponential penalty.
                // i67 (DEAD) used MW-brut viol × quadratic → 100-10000× over-penalization.
                // Fix: v_frac_l = (|flow_l|-limit_l)/limit_l (dimensionless O(0.01-0.5))
                //      s_b = Σ_l v_frac_l * |ptdf_{b,l}|  (per-battery normalized exposure)
                //      Φ(s) = exp(α·s) - 1  (Sinha-Vaze 2024 `48397995`: exponential > quadratic in single-round m=0)
                // ep_ct[t][b] -= Φ(s_b) ; commensurable with premium scale (~$/MWh).
                // ct_gdd_alpha=0.0 → zero-cost branch, iso-binaire i60 EXACT.
                if hp.ct_gdd_alpha > 1e-12 {
                    let mut n_gdd_active = 0usize;
                    for t in 0..num_t {
                        for b in 0..num_b {
                            let s_b: f64 = b_to_lines[b].iter()
                                .map(|&(l, impact)| {
                                    let limit = challenge.network.flow_limits[l];
                                    if limit <= 1e-6 { return 0.0; }
                                    let v_frac = (flows_all[t][l].abs() - limit).max(0.0) / limit;
                                    v_frac * impact.abs()
                                })
                                .sum();
                            if s_b > 1e-9 { n_gdd_active += 1; }
                            ep_ct[t][b] -= (hp.ct_gdd_alpha * s_b).exp() - 1.0;
                        }
                    }
                    eprintln!("CT-GDD-EXP alpha={:.2} gdd_active={}/{}", hp.ct_gdd_alpha, n_gdd_active, num_t * num_b);
                }
                // Half-resolution rebuild: reuse mu_dw1, reduced soc_levels/action_grid (~4× cheaper).
                // Full-res adds 1.5s (7.0s > 6.0s); half-res adds ~0.38s (target: elapsed ≤ 6.0s).
                let mut hp_oco = hp.clone();
                hp_oco.soc_levels = (hp.soc_levels / 2).max(31);
                hp_oco.action_grid = (hp.action_grid / 2).max(15);
                build_dp_parallel_or_serial(challenge, &hp_oco, &batt_nodes, &ep_ct, &b_to_lines, &mu_dw1)
            } else if hp.coord_premium_mode == 0 {
                dp1
            } else {
                // P12 — ENDOGENOUS COORDINATED PREMIUM (lagged one-shot).
                // Unlike i44 (uncoordinated full-power probe → over-estimated congestion → -23.5%),
                // mu_dw1 already reflects the cross-battery coordination of the DW master, so it is a
                // high-quality, non-circular congestion signal. Fold it back into the anticipatory
                // premium seed (amont), attacking the proven t50 ceiling = signal quality (11f52610).
                // Sign mirrors the DP congestion term (-mu*|impact|, discourages dispatch on binding lines).
                let cs = hp.coord_premium_scale;
                let mut ep2 = expected_premiums.clone();
                for t in 0..num_t {
                    for l in 0..num_l {
                        let m = mu_dw1[t][l];
                        if m > 1e-9 {
                            for &(b, impact) in &ptdf_sparse[l] {
                                if impact.abs() > 1e-6 {
                                    ep2[t][b] += -impact.abs() * m * cs;
                                }
                            }
                        }
                    }
                }
                if hp.coord_premium_mode == 1 {
                    // Cheap: reuse mu_dw1, single extra DP (probes whether the converged dual is under-applied).
                    build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &ep2, &b_to_lines, &mu_dw1)
                } else {
                    // Faithful: full re-solve so DW re-prices its columns against the coordinated premium.
                    let (dp2, _mu_dw2) = dw_solve(&ep2);
                    dp2
                }
            }
        } else if hp.ldd_iters > 0 && num_l > 0 {
            let mut mu = vec![vec![0.0_f64; num_l]; num_t];
            let mut dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

            let active_lines_ldd: Vec<usize> = (0..num_l).collect();

            if hp.use_ldd_proximal {
                let mut prev_average = vec![vec![0.0_f64; num_l]; num_t];
                let mut fallback = false;

                
                // unconstrained DP ("expert") SOC trajectory to prevent myopic over-correction.
                // When ct_ref_kappa=0.0 → exactly iso-binaire i55 (zero-cost code path).
                let ep_base: Vec<Vec<f64>> = if hp.ct_ref_kappa > 0.0 { expected_premiums.clone() } else { Vec::new() };
                let soc_ref: Vec<Vec<f64>> = if hp.ct_ref_kappa > 0.0 {
                    ldd_simulate_socs(challenge, state, &dp, &batt_nodes)
                } else {
                    Vec::new()
                };

                
                // from the unconstrained backward DP. Budget-neutral: uses already-computed dp values,
                // no extra DP build. ep[t][b] = ep_base[t][b] − kappa·max(0, oc_ref[t] − oc_implied[t]).
                // When ct_oc_kappa=0.0 → exactly iso-binaire i60 (zero-cost code path).
                let ep_oc_base: Vec<Vec<f64>> = if hp.ct_oc_kappa > 0.0 { expected_premiums.clone() } else { Vec::new() };
                let dp0: Vec<Vec<Vec<f64>>> = if hp.ct_oc_kappa > 0.0 { dp.clone() } else { Vec::new() };

                for ldd_iter in 0..hp.ldd_iters {
                    let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse);
                    let step_base = hp.ldd_step_size;

                    let mut l2_per_t = vec![0.0_f64; num_t];
                    let mut max_viol = 0.0_f64;
                    for t in 0..num_t {
                        let mut l2_sq = 0.0_f64;
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let abs_f = f.abs();
                            if abs_f > limit {
                                let v = abs_f - limit;
                                l2_sq += v * v;
                                if v > max_viol { max_viol = v; }
                            }
                        }
                        l2_per_t[t] = l2_sq.sqrt();
                    }

                    for t in 0..num_t {
                        let raw_step = step_base / (1.0 + l2_per_t[t]);

                        for &l in &active_lines_ldd {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let abs_f = f.abs();

                            if abs_f > limit {
                                let v = abs_f - limit;
                                let sign = f.signum();
                                let delta = raw_step * v;
                                let max_delta = limit * hp.ldd_clip_fraction;
                                let clamped_delta = delta.min(max_delta);

                                let eff_delta = if ldd_iter == 0 {
                                    prev_average[t][l] = clamped_delta;
                                    clamped_delta
                                } else {
                                    let avg_delta = hp.ldd_momentum * prev_average[t][l] + (1.0 - hp.ldd_momentum) * clamped_delta;
                                    prev_average[t][l] = avg_delta;
                                    avg_delta
                                };

                                mu[t][l] = (mu[t][l] + sign * eff_delta).abs();
                            }
                        }
                    }

                    // Reference tracking: ep_eff[t][b] = ep_base[t][b] + kappa*(soc_ref[t][b] - soc_cur[t][b])
                    // soc_ref = unconstrained dp0 expert trajectory; soc_cur = current corrected dp.
                    if hp.ct_ref_kappa > 0.0 {
                        let soc_cur = ldd_simulate_socs(challenge, state, &dp, &batt_nodes);
                        for t in 0..num_t {
                            for b in 0..num_b {
                                expected_premiums[t][b] = ep_base[t][b]
                                    + hp.ct_ref_kappa * (soc_ref[t][b] - soc_cur[t][b]);
                            }
                        }
                    }

                    // OC-reference (i65): ep[t][b] = ep_oc_base[t][b] − kappa·max(0, oc_ref[t][b] − oc_implied[t][b]).
                    // oc_ref = ∂V/∂soc from unconstrained dp0 (value-dimension anchor).
                    // oc_implied = ∂V/∂soc from corrected dp (changes as μ accumulates).
                    // Anti-overshoot VALUE-dimension: correction fires only when OCO devalues future storage.
                    if hp.ct_oc_kappa > 0.0 && !dp0.is_empty() {
                        let soc_lv = hp.soc_levels;
                        for b in 0..num_b {
                            let bat = &challenge.batteries[b];
                            let soc_min = bat.soc_min_mwh;
                            let soc_sp = (bat.soc_max_mwh - soc_min).max(1e-9);
                            let soc = state.socs[b];
                            for t in 0..num_t {
                                let t_next = (t + 1).min(dp0[b].len().saturating_sub(1));
                                let oc_r = dp_lambda(&dp0[b], t_next, soc, soc_min, soc_sp, soc_lv);
                                let oc_i = dp_lambda(&dp[b], t_next, soc, soc_min, soc_sp, soc_lv);
                                let oc_diff = (oc_r - oc_i).max(0.0);
                                expected_premiums[t][b] = ep_oc_base[t][b] - hp.ct_oc_kappa * oc_diff;
                            }
                        }
                    }

                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

                    if ldd_iter == 0 {
                        let mut any_positive = 0usize;
                        for b in 0..num_b.min(20) {
                            let node = batt_nodes[b];
                            let da = if node < challenge.market.day_ahead_prices.len() && 0 < challenge.market.day_ahead_prices[node].len() {
                                challenge.market.day_ahead_prices[node][0]
                            } else { 0.0 };
                            let cong_adj: f64 = b_to_lines[b].iter()
                                .map(|&(l, impact)| if l < mu[0].len() { mu[0][l] * impact.abs() } else { 0.0 })
                                .sum();
                            let p_eff = da - cong_adj;
                            if p_eff > 0.0 { any_positive += 1; }
                        }
                        if any_positive < (num_b.min(20)).max(1) / 2 {
                            fallback = true;
                        }
                    }

                    if max_viol < 1e-4 { break; }
                }

                if fallback {
                    // Restore base premiums for the fallback path (undo reference tracking modifications)
                    if hp.ct_ref_kappa > 0.0 {
                        expected_premiums.clone_from(&ep_base);
                    } else if hp.ct_oc_kappa > 0.0 && !ep_oc_base.is_empty() {
                        expected_premiums.clone_from(&ep_oc_base);
                    }
                    mu = vec![vec![0.0_f64; num_l]; num_t];
                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);
                    for ldd_iter in 0..hp.ldd_iters {
                        let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse);
                        let alpha = hp.ldd_step_size / ((ldd_iter + 1) as f64);
                        let mut max_viol = 0.0_f64;
                        for t in 0..num_t {
                            for l in 0..num_l {
                                let limit = challenge.network.flow_limits[l];
                                if limit <= 1e-6 { continue; }
                                let f = flows_all[t][l];
                                if f > limit {
                                    let v = f - limit;
                                    mu[t][l] += alpha * v;
                                    if v > max_viol { max_viol = v; }
                                } else if f < -limit {
                                    let v = -f - limit;
                                    mu[t][l] -= alpha * v;
                                    if v > max_viol { max_viol = v; }
                                }
                            }
                        }
                        dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);
                        if max_viol < 1e-4 { break; }
                    }
                }
            } else {
                for ldd_iter in 0..hp.ldd_iters {
                    let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse);

                    let alpha = hp.ldd_step_size / ((ldd_iter + 1) as f64);
                    let mut max_viol = 0.0_f64;
                    for t in 0..num_t {
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            if f > limit {
                                let v = f - limit;
                                mu[t][l] += alpha * v;
                                if v > max_viol { max_viol = v; }
                            } else if f < -limit {
                                let v = -f - limit;
                                mu[t][l] -= alpha * v;
                                if v > max_viol { max_viol = v; }
                            }
                        }
                    }

                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

                    if max_viol < 1e-4 { break; }
                }
            }
            dp
        } else if hp.max_admm_iters > 0 && hp.anticipate_lmp && num_l > 0 {
            let mut hp_cheap = hp.clone();
            hp_cheap.soc_levels = 31;
            hp_cheap.action_grid = 15;
            let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
            let dp_low = build_dp_parallel_or_serial(challenge, &hp_cheap, &batt_nodes, &expected_premiums, &b_to_lines, &mu_zero);
            let flows_all = ldd_simulate_flows(challenge, state, &dp_low, &batt_nodes, &ptdf_sparse);
            for t in 0..num_t {
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let f = flows_all[t][l];
                    let viol = f.abs() - limit;
                    if viol > 0.0 {
                        let sign = f.signum();
                        let scale = (viol / limit) * 0.2;
                        for &(b, imp) in &ptdf_sparse[l] {
                            expected_premiums[t][b] -= imp * sign * scale;
                        }
                    }
                }
            }
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        } else {
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        };

        AycdicdbCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
    }

    #[inline]
    fn eval_profit(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize, u: f64) -> f64 {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let dt = 0.25_f64;
        let abs_u = u.abs();
        let revenue = u * rt_price * dt;
        let tx = 0.25 * abs_u * dt;
        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
        let deg = deg_base * deg_base;
        let profit = revenue - tx - deg;

        let soc = state.socs[b];
        let next_soc_raw = if u < 0.0 {
            soc + bat.efficiency_charge * (-u) * dt
        } else {
            soc - u / bat.efficiency_discharge.max(1e-9) * dt
        };
        let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp[b][0].len();
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as isize).max(0) as usize;
        let idx0c = idx0.min(soc_levels - 1);
        let idx1c = (idx0 + 1).min(soc_levels - 1);
        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
        let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
        profit + ca.dp[b][t_next][idx0c] * (1.0 - frac) + ca.dp[b][t_next][idx1c] * frac
    }

    fn run_asca(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mut flows: Vec<f64> = flows_base.to_vec();
        for l in 0..num_l {
            for &(b, p) in &ca.ptdf_sparse[l] {
                flows[l] += p * actions[b];
            }
        }

        let mut active = vec![true; num_b];
        if hp.prune_ratio > 0.0 && num_b >= 2 {
            let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
            if cutoff > 0 {
                let mut caps: Vec<(usize, f64)> = challenge.batteries.iter().enumerate().map(|(i, b)| (i, b.capacity_mwh)).collect();
                caps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..cutoff.min(num_b) { active[caps[i].0] = false; }
            }
        }

        let base_potential: Vec<f64> = (0..num_b).map(|b| {
            if active[b] { potential(challenge, state, ca, b) } else { 0.0 }
        }).collect();

        let stress_score = |batt: usize, cur_flows: &[f64]| -> f64 {
            let mut score = 1e-4;
            let mut tight = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[batt] {
                let limit = challenge.network.flow_limits[l];
                if limit > 1e-6 {
                    let util = (cur_flows[l].abs() / limit).min(2.0);
                    score += p.abs() * (0.25 + util * util);
                    if util > tight { tight = util; }
                }
            }
            score * (1.0 + 0.25 * tight)
        };

        let avg_rt = if num_b > 0 {
            (0..num_b)
                .map(|b| {
                    let node = ca.batt_nodes[b];
                    if node < state.rt_prices.len() { state.rt_prices[node].abs() } else { 0.0 }
                })
                .sum::<f64>() / num_b as f64
        } else { 0.0 };
        let penalty_scale = avg_rt.max(10.0) * (0.15 + 0.20 * hp.kkt_price_scale);
        let penalty_gate = (hp.kkt_cong_threshold - 0.08).max(0.55);

        for _sweep in 0..hp.asca_iters {
            let mut line_penalty = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 { continue; }
                let util = flows[l].abs() / limit;
                let excess = (util - penalty_gate).max(0.0);
                if excess <= 0.0 { continue; }
                let smooth = (excess * excess) / (0.20 + excess);
                line_penalty[l] = flows[l].signum() * penalty_scale * smooth.min(1.5);
            }

            let mut order: Vec<usize> = (0..num_b).filter(|&b| active[b]).collect();
            order.sort_by(|&a, &b| {
                let sa = base_potential[a] * (1.0 + stress_score(a, &flows));
                let sb = base_potential[b] * (1.0 + stress_score(b, &flows));
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut max_change = 0.0_f64;

            for &b in &order {
                let cur = actions[b];
                let (mut u_min, mut u_max) = state.action_bounds[b];

                for &(l, p) in &ca.b_to_lines[b] {
                    if p.abs() < 1e-9 { continue; }
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    let f_other = flows[l] - p * cur;
                    let b1 = (-limit - f_other) / p;
                    let b2 = (limit - f_other) / p;
                    let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                    if lo > u_min { u_min = lo; }
                    if hi < u_max { u_max = hi; }
                }

                if u_min > cur { u_min = cur; }
                if u_max < cur { u_max = cur; }
                if u_min > u_max { continue; }

                let node = ca.batt_nodes[b];
                let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                let shadow_adj: f64 = ca.b_to_lines[b].iter()
                    .map(|&(l, p)| line_penalty[l] * p)
                    .sum();
                let eff_price = {
                    let p = rt - shadow_adj;
                    if p.is_finite() { p } else { rt }
                };
                let eval = |u: f64| eval_profit_with_price(challenge, state, ca, b, u, eff_price);

                let mut best_u = cur;
                let mut best_v = eval(cur);

                let v_lo = eval(u_min);
                if v_lo > best_v { best_v = v_lo; best_u = u_min; }

                let v_hi = eval(u_max);
                if v_hi > best_v { best_v = v_hi; best_u = u_max; }

                if u_min <= 0.0 && 0.0 <= u_max {
                    let v0 = eval(0.0);
                    if v0 > best_v { best_v = v0; best_u = 0.0; }
                }

                if u_min < 0.0 {
                    let lo = u_min;
                    let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let (u, v) = ternary_search(|u| eval(u), lo, hi, hp.ternary_iters);
                        if v > best_v { best_v = v; best_u = u; }
                    }
                }

                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min);
                    let hi = u_max;
                    if lo < hi {
                        let (u, v) = ternary_search(|u| eval(u), lo, hi, hp.ternary_iters);
                        if v > best_v { best_u = u; }
                    }
                }

                let delta = best_u - cur;
                if delta.abs() > 1e-6 {
                    actions[b] = best_u;
                    for &(l, p) in &ca.b_to_lines[b] {
                        if l < num_l { flows[l] += p * delta; }
                    }
                    if delta.abs() > max_change { max_change = delta.abs(); }
                }
            }

            if max_change < hp.convergence_tol { break; }
        }
    }

    fn targeted_tight_line_polish(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        if num_b == 0 || num_l == 0 || num_b > 60 || actions.len() != num_b {
            return None;
        }

        let limits = &challenge.network.flow_limits;
        let mut flows = flows_base.to_vec();
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] {
                flows[l] += imp * actions[b];
            }
        }

        let base_profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions[b])).sum();

        let gate = (hp.kkt_cong_threshold - 0.05).max(0.55);
        let relaxed_gate = (gate - 0.10).max(0.45);

        let mut tight_lines: Vec<(f64, usize)> = (0..num_l)
            .filter_map(|l| {
                let limit = limits[l];
                if limit <= 1e-6 {
                    None
                } else {
                    let util = flows[l].abs() / limit;
                    if util >= gate { Some((util, l)) } else { None }
                }
            })
            .collect();

        if tight_lines.is_empty() {
            tight_lines = (0..num_l)
                .filter_map(|l| {
                    let limit = limits[l];
                    if limit <= 1e-6 {
                        None
                    } else {
                        let util = flows[l].abs() / limit;
                        if util >= relaxed_gate { Some((util, l)) } else { None }
                    }
                })
                .collect();
        }

        if tight_lines.is_empty() {
            return None;
        }

        tight_lines.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        tight_lines.truncate(if num_b <= 30 { 6 } else { 4 });

        let subset_cap = if num_b <= 24 { 8.min(num_b) } else { 6.min(num_b) };
        let mut ranked: Vec<(f64, usize)> = Vec::new();
        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            let room = (hi - actions[b]).abs() + (actions[b] - lo).abs();
            let mut score = 0.0_f64;
            for &(_, l) in &tight_lines {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == l) {
                    score += p.abs();
                }
            }
            if score > 1e-8 {
                ranked.push((score * (1.0 + 0.10 * room), b));
            }
        }

        if ranked.len() < 2 {
            return None;
        }

        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(subset_cap);
        let subset: Vec<usize> = ranked.iter().map(|&(_, b)| b).collect();

        let n_active = tight_lines.len();
        let n_vars = 2 * subset.len() + n_active;
        let m = 2 * subset.len() + 2 * n_active;

        let mut c_vec = vec![0.0_f64; n_vars];
        let mut a_mat = vec![vec![0.0_f64; n_vars]; m];
        let mut b_vec = vec![0.0_f64; m];

        let dt = 0.25_f64;
        let lp_lambda = hp.lp_soft_lambda;

        for ai in 0..n_active {
            c_vec[2 * subset.len() + ai] = -lp_lambda;
        }

        for (i, &b) in subset.iter().enumerate() {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (lo, hi) = state.action_bounds[b];
            let soc = state.socs[b];

            let soc_levels = ca.dp[b][0].len();
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = (idx_f.floor() as isize).max(0) as usize;
            let lo_idx = lo_idx.min(soc_levels - 1);
            let hi_idx = (lo_idx + 1).min(soc_levels - 1);
            let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
            let dv = (ca.dp[b][t_next][hi_idx] - ca.dp[b][t_next][lo_idx]) / delta_s;

            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);

            c_vec[i]                 = (rt - 0.25) * dt - dv * dt / eta_d;
            c_vec[subset.len() + i]  = -(rt + 0.25) * dt + dv * eta_c * dt;

            let r = 2 * i;
            a_mat[r][i] = 1.0;
            b_vec[r] = hi.max(0.0);
            a_mat[r + 1][subset.len() + i] = 1.0;
            b_vec[r + 1] = (-lo).max(0.0);
        }

        let row_flow = 2 * subset.len();
        for (ai, &(_, li)) in tight_lines.iter().enumerate() {
            let limit = limits[li];
            let mut flow_others = flows[li];
            for &b in &subset {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == li) {
                    flow_others -= p * actions[b];
                }
            }
            let rp = row_flow + 2 * ai;
            let rn = rp + 1;

            for (i, &b) in subset.iter().enumerate() {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == li) {
                    a_mat[rp][i]                 += p;
                    a_mat[rp][subset.len() + i]  -= p;
                    a_mat[rn][i]                 -= p;
                    a_mat[rn][subset.len() + i]  += p;
                }
            }

            let viol_idx = 2 * subset.len() + ai;
            a_mat[rp][viol_idx] = -1.0;
            a_mat[rn][viol_idx] = -1.0;

            b_vec[rp] = (limit - flow_others).max(0.0);
            b_vec[rn] = (limit + flow_others).max(0.0);
        }

        let max_pivots = 500;
        let (opt_x, _) = super::lp::lp_solve_with_budget(n_vars, m, &c_vec, &a_mat, &b_vec, max_pivots);

        if let Some(opt_x) = opt_x {
            let mut new_actions = actions.to_vec();
            for (i, &b) in subset.iter().enumerate() {
                let u = opt_x[i] - opt_x[subset.len() + i];
                let (lo, hi) = state.action_bounds[b];
                new_actions[b] = u.clamp(lo, hi);
            }

            let new_profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, new_actions[b])).sum();
            if new_profit > base_profit + 1e-9 {
                let mut check_flows = flows_base.to_vec();
                for l in 0..num_l {
                    for &(b, imp) in &ca.ptdf_sparse[l] {
                        check_flows[l] += imp * new_actions[b];
                    }
                }
                let feasible = (0..num_l).all(|l| check_flows[l].abs() <= limits[l] + 1e-6 * limits[l].max(1.0));
                if feasible {
                    return Some(new_actions);
                }
            }
        }

        None
    }

    #[inline]
    fn potential(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize) -> f64 {
        let (u_lo, u_hi) = state.action_bounds[b];
        let v_lo = eval_profit(challenge, state, ca, b, u_lo);
        let v_hi = eval_profit(challenge, state, ca, b, u_hi);
        let v0 = eval_profit(challenge, state, ca, b, 0.0);
        (v_lo.max(v_hi) - v0).max(0.0)
    }

    fn ternary_search<F: Fn(f64) -> f64>(f: F, mut l: f64, mut r: f64, iters: usize) -> (f64, f64) {
        if l >= r { return (l, f(l)); }
        for _ in 0..iters {
            let m1 = l + (r - l) / 3.0;
            let m2 = r - (r - l) / 3.0;
            if f(m1) < f(m2) { l = m1; } else { r = m2; }
        }
        let u = 0.5 * (l + r);
        (u, f(u))
    }

    fn run_dual_ascent(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        warm_start: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mut actions = warm_start.to_vec();

        let mut flows = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let mut sum = 0.0;
            for &(b, imp) in &ca.ptdf_sparse[l] { sum += imp * actions[b]; }
            flows[l] = flows_base[l] + sum;
        }

        let mut nu = vec![0.0_f64; num_l];

        for k in 1..=hp.dual_iters {
            let alpha = hp.da_step_size / (k as f64);

            for b in 0..num_b {
                let nu_dot_ptdf: f64 = ca.b_to_lines[b].iter()
                    .map(|&(l, impact)| nu[l] * impact)
                    .sum();

                let (mut u_min, mut u_max) = state.action_bounds[b];
                for &(l, p) in &ca.b_to_lines[b] {
                    if p.abs() < 1e-9 { continue; }
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    let f_other = flows[l] - p * actions[b];
                    let b1 = (-limit - f_other) / p;
                    let b2 = (limit - f_other) / p;
                    let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                    if lo > u_min { u_min = lo; }
                    if hi < u_max { u_max = hi; }
                }
                if u_min > u_max { u_min = actions[b]; u_max = actions[b]; }

                let mut best_u = actions[b];
                let mut best_v = eval_profit(challenge, state, ca, b, best_u) - nu_dot_ptdf * best_u;

                if u_min <= 0.0 && 0.0 <= u_max {
                    let v0 = eval_profit(challenge, state, ca, b, 0.0);
                    if v0 > best_v { best_v = v0; best_u = 0.0; }
                }
                if u_min < 0.0 {
                    let lo = u_min; let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let (u, v) = ternary_search(
                            |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                            lo, hi, hp.ternary_iters,
                        );
                        if v > best_v { best_v = v; best_u = u; }
                    }
                }
                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let (u, v) = ternary_search(
                            |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                            lo, hi, hp.ternary_iters,
                        );
                        if v > best_v { best_u = u; }
                    }
                }

                let delta = best_u - actions[b];
                if delta.abs() > 1e-10 {
                    actions[b] = best_u;
                    for &(l, p) in &ca.b_to_lines[b] {
                        if l < num_l { flows[l] += p * delta; }
                    }
                }
            }

            let mut max_viol = 0.0_f64;
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let v = flows[l].abs() - limit;
                if v > max_viol { max_viol = v; }
            }
            if max_viol < 1e-6 { break; }

            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if flows[l] > limit {
                    nu[l] += alpha * (flows[l] - limit);
                } else if flows[l] < -limit {
                    nu[l] -= alpha * (-flows[l] - limit);
                }
            }
        }

        actions
    }

    fn run_deflator(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) {
        let num_l = challenge.network.flow_limits.len();
        let num_b = challenge.num_batteries;
        if num_l == 0 || num_b == 0 { return; }

        let recompute_flows = |act: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let mut f = 0.0_f64;
                for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * act[b]; }
                out[l] = flows_base[l] + f;
            }
            out
        };

        let original_actions = actions.to_vec();
        let mut flows = recompute_flows(actions);

        let mut is_safe = true;
        let mut ever_violated = false;
        for _ in 0..hp.deflator_iters {
            is_safe = true;
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if flows[l].abs() <= limit { continue; }
                is_safe = false;
                ever_violated = true;
                let overflow = flows[l].abs() - limit;
                let sign = flows[l].signum();

                let mut culprits: Vec<(usize, f64, f64)> = Vec::new();
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let cur = actions[b];
                    let contrib = impact * cur;
                    if contrib * sign > 1e-9 {
                        let shrink = (0.35 * cur.abs()).max(0.05).min(cur.abs());
                        let trial = if shrink > 0.0 {
                            cur - cur.signum() * shrink
                        } else {
                            0.0
                        };

                        let val_curr = eval_profit(challenge, state, ca, b, cur);
                        let val_trial = eval_profit(challenge, state, ca, b, trial);
                        let value_loss = (val_curr - val_trial).max(0.0);

                        let delta = trial - cur;
                        let mut relief = 0.0_f64;
                        for &(ll, pp) in &ca.b_to_lines[b] {
                            let limit_ll = (challenge.network.flow_limits[ll] - hp.flow_margin).max(0.0);
                            if flows[ll].abs() <= limit_ll + 1e-9 { continue; }
                            let line_relief = -(flows[ll].signum()) * pp * delta;
                            if line_relief > 0.0 {
                                relief += line_relief;
                            }
                        }

                        let score = if relief > 1e-9 {
                            value_loss / relief
                        } else {
                            f64::INFINITY
                        };
                        culprits.push((b, contrib, score));
                    }
                }
                culprits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

                let mut remaining = overflow;
                for (b, contrib, _) in culprits {
                    if remaining <= 1e-9 { break; }
                    let contrib_abs = contrib.abs();
                    if contrib_abs < 1e-12 { continue; }
                    let reduction = contrib_abs.min(remaining);
                    let ratio = 1.0 - (reduction / contrib_abs);
                    let new_action = actions[b] * ratio;
                    let delta = new_action - actions[b];
                    actions[b] = new_action;
                    for &(ll, pp) in &ca.b_to_lines[b] {
                        if ll < num_l { flows[ll] += pp * delta; }
                    }
                    remaining -= reduction;
                }
            }
            if is_safe { break; }
        }

        if !is_safe {
            let f_act: Vec<f64> = (0..num_l).map(|l| {
                let mut s = 0.0;
                for &(b, imp) in &ca.ptdf_sparse[l] { s += imp * actions[b]; }
                s
            }).collect();

            let mut beta = 1.0_f64;
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let total = flows_base[l] + f_act[l];
                if total.abs() <= limit { continue; }
                if f_act[l].abs() < 1e-9 { continue; }
                let target = if total > 0.0 { limit } else { -limit };
                let candidate = (target - flows_base[l]) / f_act[l];
                if candidate < beta { beta = candidate; }
            }
            let beta = beta.clamp(0.0, 1.0);
            for b in 0..num_b { actions[b] *= beta; }
        }

        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            if actions[b] < lo { actions[b] = lo; }
            if actions[b] > hi { actions[b] = hi; }
        }

        flows = recompute_flows(actions);
        let repaired_actions = actions.to_vec();
        let repaired_flows = flows.clone();
        let repaired_feasible = (0..num_l).all(|l| {
            repaired_flows[l].abs()
                <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-6
        });
        if !repaired_feasible { return; }

        let finite_bounds = state.action_bounds.iter().all(|(lo, hi)| lo.is_finite() && hi.is_finite());
        if !finite_bounds { return; }

        let material_reduction: f64 = original_actions.iter().zip(repaired_actions.iter())
            .map(|(a0, a1)| (a0 - a1).abs())
            .sum();

        let tight_threshold = if ever_violated { 0.80 } else { 0.92 };
        let mut tight_lines = vec![false; num_l];
        let mut line_rank: Vec<(f64, usize)> = Vec::new();
        let mut any_tight = false;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 { continue; }
            let util = repaired_flows[l].abs() / limit;
            if util > tight_threshold {
                tight_lines[l] = true;
                any_tight = true;
                line_rank.push((util, l));
            }
        }
        if !ever_violated && !any_tight { return; }
        if material_reduction < 1e-5 && !any_tight { return; }

        line_rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let line_cap = if num_b <= 24 { 4 } else { 3 };
        line_rank.truncate(line_cap);
        if line_rank.is_empty() { return; }

        let bundle_cap = if num_b <= 24 { 4 } else { 3 };
        let rounds = if ever_violated && num_b <= 32 { 2 } else { 1 };

        for _ in 0..rounds {
            let mut improved = false;

            for &(_, l_anchor) in &line_rank {
                let mut cand: Vec<(f64, usize)> = Vec::new();
                for &(b, imp) in &ca.ptdf_sparse[l_anchor] {
                    let desired = original_actions[b] - actions[b];
                    let reduced = (original_actions[b] - repaired_actions[b]).abs();
                    let mut multi_tight_touch = 0.0_f64;
                    for &(ll, pp) in &ca.b_to_lines[b] {
                        if ll < num_l && tight_lines[ll] {
                            multi_tight_touch += pp.abs();
                        }
                    }
                    let score = desired.abs() * (1.0 + 0.25 * multi_tight_touch)
                        + 0.20 * reduced
                        + 0.05 * imp.abs();
                    if score > 1e-8 {
                        cand.push((score, b));
                    }
                }

                if cand.len() < 2 { continue; }
                cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                cand.truncate(bundle_cap);
                let bundle: Vec<usize> = cand.into_iter().map(|(_, b)| b).collect();
                if bundle.len() < 2 { continue; }

                let base_dir: Vec<f64> = bundle.iter()
                    .map(|&b| original_actions[b] - actions[b])
                    .collect();
                let base_mass: f64 = base_dir.iter().map(|d| d.abs()).sum();
                if base_mass <= 1e-8 { continue; }

                let mut directions: Vec<Vec<f64>> = vec![base_dir.clone()];
                let mut dot_pd = 0.0_f64;
                let mut dot_pp = 0.0_f64;
                for (i, &b) in bundle.iter().enumerate() {
                    let p = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l_anchor)
                        .map(|&(_, p)| p)
                        .unwrap_or(0.0);
                    dot_pd += p * base_dir[i];
                    dot_pp += p * p;
                }
                if dot_pp > 1e-12 {
                    let corr = dot_pd / dot_pp;
                    let mut projected = base_dir.clone();
                    for (i, &b) in bundle.iter().enumerate() {
                        let p = ca.b_to_lines[b].iter()
                            .find(|&&(ll, _)| ll == l_anchor)
                            .map(|&(_, p)| p)
                            .unwrap_or(0.0);
                        projected[i] -= p * corr;
                    }
                    let proj_mass: f64 = projected.iter().map(|d| d.abs()).sum();
                    let diff_mass: f64 = projected.iter().zip(base_dir.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    if proj_mass > 1e-8 && diff_mass > 1e-6 {
                        directions.push(projected);
                    }
                }

                let mut best_gain = 0.0_f64;
                let mut best_move: Option<Vec<(usize, f64)>> = None;

                for dir in directions {
                    let mut alpha_hi = 1.0_f64;
                    let mut touched_lines = vec![false; num_l];

                    for (i, &b) in bundle.iter().enumerate() {
                        let d = dir[i];
                        if d.abs() < 1e-10 { continue; }
                        let (lo, hi) = state.action_bounds[b];
                        let bound = if d > 0.0 {
                            (hi - actions[b]) / d
                        } else {
                            (lo - actions[b]) / d
                        };
                        if !bound.is_finite() || bound <= 0.0 {
                            alpha_hi = 0.0;
                            break;
                        }
                        if bound < alpha_hi { alpha_hi = bound; }
                        for &(ll, _) in &ca.b_to_lines[b] {
                            if ll < num_l { touched_lines[ll] = true; }
                        }
                    }

                    if alpha_hi <= 1e-8 { continue; }

                    for ll in 0..num_l {
                        if !touched_lines[ll] { continue; }
                        let mut c = 0.0_f64;
                        for (i, &b) in bundle.iter().enumerate() {
                            let d = dir[i];
                            if d.abs() < 1e-10 { continue; }
                            for &(line_idx, p) in &ca.b_to_lines[b] {
                                if line_idx == ll {
                                    c += p * d;
                                    break;
                                }
                            }
                        }
                        if c.abs() < 1e-12 { continue; }

                        let limit = (challenge.network.flow_limits[ll] - hp.flow_margin).max(0.0);
                        let x1 = (-limit - flows[ll]) / c;
                        let x2 = ( limit - flows[ll]) / c;
                        let (_, hhi) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
                        if hhi <= 0.0 {
                            alpha_hi = 0.0;
                            break;
                        }
                        if hhi < alpha_hi { alpha_hi = hhi; }
                    }

                    if alpha_hi <= 1e-8 { continue; }

                    for &frac in &[0.35_f64, 0.7_f64, 1.0_f64] {
                        let alpha = alpha_hi * frac;
                        if alpha <= 1e-8 { continue; }

                        let mut move_vec: Vec<(usize, f64)> = Vec::with_capacity(bundle.len());
                        let mut gain = 0.0_f64;
                        let mut valid = true;

                        for (i, &b) in bundle.iter().enumerate() {
                            let new_u = (actions[b] + alpha * dir[i])
                                .clamp(state.action_bounds[b].0, state.action_bounds[b].1);
                            if !new_u.is_finite() {
                                valid = false;
                                break;
                            }
                            gain += eval_profit(challenge, state, ca, b, new_u)
                                - eval_profit(challenge, state, ca, b, actions[b]);
                            move_vec.push((b, new_u));
                        }
                        if !valid || gain <= best_gain + 1e-9 { continue; }

                        for ll in 0..num_l {
                            if !touched_lines[ll] { continue; }
                            let limit = (challenge.network.flow_limits[ll] - hp.flow_margin).max(0.0);
                            let mut flow_new = flows[ll];
                            for &(b, new_u) in &move_vec {
                                let delta = new_u - actions[b];
                                if delta.abs() < 1e-12 { continue; }
                                for &(line_idx, p) in &ca.b_to_lines[b] {
                                    if line_idx == ll {
                                        flow_new += p * delta;
                                        break;
                                    }
                                }
                            }
                            if flow_new.abs() > limit + 1e-7 {
                                valid = false;
                                break;
                            }
                        }

                        if valid {
                            best_gain = gain;
                            best_move = Some(move_vec);
                        }
                    }
                }

                if let Some(move_vec) = best_move {
                    if best_gain > 1e-9 {
                        for &(b, new_u) in &move_vec {
                            let delta = new_u - actions[b];
                            if delta.abs() < 1e-12 { continue; }
                            actions[b] = new_u;
                            for &(ll, p) in &ca.b_to_lines[b] {
                                if ll < num_l { flows[ll] += p * delta; }
                            }
                        }
                        improved = true;
                    }
                }
            }

            if !improved { break; }
        }

        let final_feasible = (0..num_l).all(|l| {
            flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-6
        });
        if !final_feasible {
            actions.copy_from_slice(&repaired_actions);
        }
    }

    #[inline]
    fn eval_profit_with_price(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        u: f64,
        price: f64,
    ) -> f64 {
        let bat = &challenge.batteries[b];
        let dt = 0.25_f64;
        let abs_u = u.abs();
        let revenue = u * price * dt;
        let tx = 0.25 * abs_u * dt;
        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
        let deg = deg_base * deg_base;
        let profit = revenue - tx - deg;
        let soc = state.socs[b];
        let next_soc_raw = if u < 0.0 {
            soc + bat.efficiency_charge * (-u) * dt
        } else {
            soc - u / bat.efficiency_discharge.max(1e-9) * dt
        };
        let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp[b][0].len();
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as isize).max(0) as usize;
        let idx0c = idx0.min(soc_levels - 1);
        let idx1c = (idx0 + 1).min(soc_levels - 1);
        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
        let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
        profit + ca.dp[b][t_next][idx0c] * (1.0 - frac) + ca.dp[b][t_next][idx1c] * frac
    }

    // ── i14 — PONDÉRATION DU SURROGAT DE LA CG ────────────────────────────────────────
    /// Vrai dès qu'au moins une des 3 familles est armée. Faux ⇒ chemin CTRL bit-exact.
    #[inline(always)]
    fn cg_obj_active(hp: &TrackHp) -> bool {
        hp.cg_cont_scale != 1.0 || hp.cg_prox_rho > 0.0 || hp.cg_cong_haircut > 0.0
    }

    /// Partie CONTINUATION seule de `eval_profit_with_price` : `V̂(t+1, soc'(u))`.
    /// Reproduit à l'identique l'interpolation de `eval_profit_with_price` (:3970) et de
    /// `eval_profit` (:3121) — mêmes bornes, même `frac`, même `t_next`. N'est appelée que
    /// lorsque `cg_obj_active` est vrai (zéro coût au défaut).
    #[inline]
    fn cont_value(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        u: f64,
    ) -> f64 {
        let bat = &challenge.batteries[b];
        let dt = 0.25_f64;
        let soc = state.socs[b];
        let next_soc_raw = if u < 0.0 {
            soc + bat.efficiency_charge * (-u) * dt
        } else {
            soc - u / bat.efficiency_discharge.max(1e-9) * dt
        };
        let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp[b][0].len();
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as isize).max(0) as usize;
        let idx0c = idx0.min(soc_levels - 1);
        let idx1c = (idx0 + 1).min(soc_levels - 1);
        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
        let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
        ca.dp[b][t_next][idx0c] * (1.0 - frac) + ca.dp[b][t_next][idx1c] * frac
    }

    /// Valeur d'une colonne / d'un candidat de pricing SOUS L'OBJECTIF PONDÉRÉ.
    ///
    /// ⚠️ **Contrat de bit-exactitude** : quand aucune famille n'est armée, on RETOURNE
    /// l'appel original SANS le réécrire. Recomposer `profit + (v_lo + v_hi)` au lieu de
    /// `(profit + v_lo) + v_hi` suffirait à dériver d'1 ULP et à détruire le contrôle de
    /// détecteur — d'où la délégation stricte plutôt qu'une réécriture « équivalente ».
    ///
    /// ⚠️ **Portée** : ce wrapper ne remplace `eval_profit_with_price` QU'AUX DEUX SITES DE
    /// RECHERCHE de la CG (coût des colonnes :1786 et pricing `eval_action` :2103). Les
    /// points de SCORE (`obj_val`/`best_obj` :1898-2015, le test d'acceptation du
    /// consommateur `lns_p > base_p` :762, le refine post-boucle) restent sur
    /// `eval_profit` NON pondéré : la pondération est un RÉGULARISEUR DE RECHERCHE, jamais
    /// un changement de métrique. Sans cette séparation on ne mesurerait plus Q mais le
    /// surrogat lui-même.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn eval_cg_value(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        b: usize,
        u: f64,
        price: f64,
        v_ref: f64,
        u_base: f64,
        span: f64,
        util_b: f64,
    ) -> f64 {
        let base = eval_profit_with_price(challenge, state, ca, b, u, price);
        if !cg_obj_active(hp) {
            return base;
        }
        if !base.is_finite() {
            return base;
        }

        let mut out = base;

        // (A) + (C) portent tous deux sur la DIFFÉRENTIELLE du surrogat autour de
        // l'incumbent ⇒ un seul appel à `cont_value`, un seul coefficient composé.
        let mut dv_coeff = hp.cg_cont_scale - 1.0;
        if hp.cg_cong_haircut > 0.0 {
            dv_coeff -= hp.cg_cong_haircut * util_b;
        }
        if dv_coeff != 0.0 {
            let dv = cont_value(challenge, state, ca, b, u) - v_ref;
            if dv.is_finite() {
                out += dv_coeff * dv;
            }
        }

        // (B) terme proximal. `|price|·dt/span` porte les unités de profit ⇒ ρ est
        // adimensionné et sweepable sur plusieurs décades.
        if hp.cg_prox_rho > 0.0 {
            let d = u - u_base;
            let pen = hp.cg_prox_rho * price.abs() * 0.25 * d * d / span;
            if pen.is_finite() {
                out -= pen;
            }
        }

        if out.is_finite() { out } else { base }
    }

    fn kkt_best_action(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        price_override: Option<f64>,
        ternary_iters: usize,
    ) -> f64 {
        let (u_min, u_max) = state.action_bounds[b];
        if u_min >= u_max { return u_min; }
        let eval = |u: f64| -> f64 {
            match price_override {
                Some(p) => eval_profit_with_price(challenge, state, ca, b, u, p),
                None => eval_profit(challenge, state, ca, b, u),
            }
        };
        let mut best_u = 0.0_f64.clamp(u_min, u_max);
        let mut best_v = eval(best_u);
        if u_min < 0.0 {
            let lo = u_min; let hi = 0.0_f64.min(u_max);
            if lo < hi {
                let (u, v) = ternary_search(|x| eval(x), lo, hi, ternary_iters);
                if v > best_v { best_v = v; best_u = u; }
            }
        }
        if u_max > 0.0 {
            let lo = 0.0_f64.max(u_min); let hi = u_max;
            if lo < hi {
                let (u, v) = ternary_search(|x| eval(x), lo, hi, ternary_iters);
                if v > best_v { best_u = u; }
            }
        }
        best_u
    }

    fn kkt_pass(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        mu_override: &[f64],
        price_scale: f64,
    ) -> (Vec<f64>, f64) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mu: Vec<f64> = if mu_override.is_empty() {
            let mut m = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l];
                if lim < 1e-6 { continue; }
                let f_exo = flows_base[l];
                let util = f_exo.abs() / lim;
                if util > hp.kkt_cong_threshold {
                    let excess_frac = ((util - hp.kkt_cong_threshold)
                        / (1.0 - hp.kkt_cong_threshold).max(1e-6)).min(1.0);
                    m[l] = excess_frac * price_scale * f_exo.signum();
                }
            }
            m
        } else {
            mu_override.to_vec()
        };

        let mut p_eff = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let cong_adj: f64 = ca.b_to_lines[b].iter().map(|&(l, imp)| mu[l] * imp).sum();
            p_eff[b] = rt - cong_adj;
        }

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            actions[b] = kkt_best_action(challenge, state, ca, b, Some(p_eff[b]), hp.ternary_iters);
        }

        run_deflator(challenge, state, ca, hp, flows_base, &mut actions);

        let profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions[b])).sum();
        (actions, profit)
    }

    pub fn kkt_policy(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        warm_init: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let avg_price: f64 = if num_b > 0 {
            (0..num_b)
                .map(|b| {
                    let n = ca.batt_nodes[b];
                    if n < state.rt_prices.len() { state.rt_prices[n].abs() } else { 0.0 }
                })
                .sum::<f64>() / num_b as f64
        } else { 0.0 };
        let price_scale = avg_price.max(10.0) * hp.kkt_price_scale;

        let (actions1, profit1) = kkt_pass(challenge, state, ca, hp, flows_base, &[], price_scale);

        let mut total_flow1 = flows_base.to_vec();
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] {
                total_flow1[l] += imp * actions1[b];
            }
        }

        let mut mu2 = vec![0.0_f64; num_l];
        {
            let mut active_for_mu: Vec<usize> = Vec::new();
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l];
                if lim < 1e-6 { continue; }
                let util = total_flow1[l].abs() / lim;
                if util > 0.80 {
                    active_for_mu.push(l);
                }
            }
            active_for_mu.truncate(12); 
            if active_for_mu.len() > 0 {
                let n_active = active_for_mu.len();
                let dt = 0.25_f64;
                let num_b = challenge.num_batteries;
                let n = 2 * num_b + n_active;
                let m = 4 * num_b + 2 * n_active;
                let mut c_vec = vec![0.0_f64; n];
                let mut a_mat = vec![vec![0.0_f64; n]; m];
                let mut b_vec = vec![0.0_f64; m];
                let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);
                let lp_soft = 1e5;

                for ai in 0..active_for_mu.len() {
                    c_vec[2 * num_b + ai] = -lp_soft;
                }
                for b in 0..num_b {
                    let bat = &challenge.batteries[b];
                    let node = ca.batt_nodes[b];
                    let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    let (u_min, u_max) = state.action_bounds[b];
                    let eta_c = bat.efficiency_charge;
                    let eta_d = bat.efficiency_discharge.max(1e-9);
                    let soc = state.socs[b];
                    let soc_levels = ca.dp[b][0].len();
                    let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                    let delta_s = span / (soc_levels - 1) as f64;
                    let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
                    let lo_idx = (idx_f.floor() as isize).max(0) as usize;
                    let lo_idx = lo_idx.min(soc_levels - 1);
                    let hi_idx = (lo_idx + 1).min(soc_levels - 1);
                    let dv = (ca.dp[b][t_next][hi_idx] - ca.dp[b][t_next][lo_idx]) / delta_s;
                    c_vec[b]           = (rt - 0.25) * dt - dv * dt / eta_d;
                    c_vec[num_b + b]   = -(rt + 0.25) * dt + dv * eta_c * dt;
                    let r = 4 * b;
                    a_mat[r][b] = 1.0;
                    b_vec[r] = u_max.max(0.0);
                    a_mat[r + 1][num_b + b] = 1.0;
                    b_vec[r + 1] = (-u_min).max(0.0);
                    a_mat[r + 2][b]         =  dt / eta_d;
                    a_mat[r + 2][num_b + b] = -eta_c * dt;
                    b_vec[r + 2] = (soc - bat.soc_min_mwh).max(0.0);
                    a_mat[r + 3][b]         = -dt / eta_d;
                    a_mat[r + 3][num_b + b] =  eta_c * dt;
                    b_vec[r + 3] = (bat.soc_max_mwh - soc).max(0.0);
                }
                let row_f = 4 * num_b;
                for (ai, &l) in active_for_mu.iter().enumerate() {
                    let limit = challenge.network.flow_limits[l];
                    let exo = flows_base[l];
                    let viol_idx = 2 * num_b + ai;
                    let rp = row_f + 2 * ai;
                    let rn = rp + 1;
                    for &(b, impact) in &ca.ptdf_sparse[l] {
                        a_mat[rp][b]         += impact;
                        a_mat[rp][num_b + b] -= impact;
                        a_mat[rn][b]         -= impact;
                        a_mat[rn][num_b + b] += impact;
                    }
                    a_mat[rp][viol_idx] = -1.0;
                    a_mat[rn][viol_idx] = -1.0;
                    b_vec[rp] = (limit - exo).max(0.0);
                    b_vec[rn] = (limit + exo).max(0.0);
                }

                let (_, duals_opt, _) = super::lp::lp_solve_with_duals(n, m, &c_vec, &a_mat, &b_vec, 1500, hp.dantzig_in_kkt, false);
                if let Some(duals) = duals_opt {
                    for (ai, &l) in active_for_mu.iter().enumerate() {
                        let rp = row_f + 2 * ai;
                        let rn = rp + 1;
                        if rp < duals.len() && rn < duals.len() {
                            let dp = duals[rp];
                            let dn = duals[rn];
                            mu2[l] = dp - dn;
                            if !mu2[l].is_finite() || mu2[l].abs() > 1e6 { mu2[l] = 0.0; }
                        }
                    }
                } else {
                    for l in 0..num_l {
                        let lim = challenge.network.flow_limits[l];
                        if lim < 1e-6 { continue; }
                        let util = total_flow1[l].abs() / lim;
                        if util > 0.85 {
                            let excess_frac = ((util - 0.85_f64)
                                / (1.0_f64 - 0.85_f64).max(1e-6_f64)).min(1.0_f64);
                            mu2[l] = excess_frac * price_scale * total_flow1[l].signum();
                        }
                    }
                }
            } else {
            }
        }

        let (actions2, profit2) = kkt_pass(challenge, state, ca, hp, flows_base, &mu2, price_scale);

        let best_kkt = if profit2 >= profit1 * 0.98 && profit2 >= profit1 {
            actions2
        } else {
            actions1
        };
        let profit_kkt = profit1.max(profit2);

        let mut actions_asca = if warm_init.len() == num_b { warm_init.to_vec() } else { vec![0.0_f64; num_b] };
        run_asca(challenge, state, ca, hp, flows_base, &mut actions_asca);
        if hp.dual_iters > 0 {
            let dual = run_dual_ascent(challenge, state, ca, hp, flows_base, &actions_asca);
            let pa: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions_asca[b])).sum();
            let pd: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, dual[b])).sum();
            if pd > pa { actions_asca = dual; }
        } else {
            run_deflator(challenge, state, ca, hp, flows_base, &mut actions_asca);
        }

        let profit_asca: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions_asca[b])).sum();

        if profit_kkt >= profit_asca {
            let mut out = best_kkt;
            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                out[b] = out[b].clamp(lo, hi);
            }
            out
        } else {
            actions_asca
        }
    }

    fn run_admm_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) -> bool {
        let num_l = challenge.network.flow_limits.len();
        let num_b = challenge.num_batteries;
        let rho = hp.admm_rho;
        let tol = hp.admm_primal_tol;

        let mut any_violated = false;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let mut bat_f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_f += imp * actions[b]; }
            if (flows_base[l] + bat_f).abs() > limit { any_violated = true; break; }
        }
        if !any_violated { return true; }

        let mut y = vec![0.0_f64; num_l];
        let mut s: Vec<f64> = (0..num_l).map(|l| {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let mut bat_f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_f += imp * actions[b]; }
            (flows_base[l] + bat_f).clamp(-limit, limit)
        }).collect();

        for _iter in 0..hp.max_admm_iters {
            let prev_actions = actions.to_vec();

            let mut bat_flow = vec![0.0_f64; num_l];
            for l in 0..num_l {
                for &(b, imp) in &ca.ptdf_sparse[l] { bat_flow[l] += imp * actions[b]; }
            }

            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                if (hi - lo).abs() < 1e-12 { continue; }

                let lines_b = &ca.b_to_lines[b];
                let offsets: Vec<(f64, f64)> = lines_b.iter().map(|&(l, a_lb)| {
                    let off = s[l] - flows_base[l] + y[l] / rho - (bat_flow[l] - a_lb * actions[b]);
                    (off, a_lb)
                }).collect();

                const GRID: usize = 200;
                let step = (hi - lo) / GRID as f64;
                let mut best_u = actions[b];
                let mut best_val = f64::NEG_INFINITY;
                for k in 0..=GRID {
                    let u = (lo + k as f64 * step).clamp(lo, hi);
                    let profit = eval_profit(challenge, state, ca, b, u);
                    let penalty: f64 = offsets.iter().map(|&(off, a_lb)| {
                        let err = off - a_lb * u;
                        (rho / 2.0) * err * err
                    }).sum();
                    let val = profit - penalty;
                    if val > best_val { best_val = val; best_u = u; }
                }

                let delta = best_u - actions[b];
                for &(l, a_lb) in lines_b { bat_flow[l] += a_lb * delta; }
                actions[b] = best_u;
            }

            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                s[l] = (bat_flow[l] + flows_base[l] - y[l] / rho).clamp(-limit, limit);
            }

            let mut max_resid = 0.0_f64;
            if hp.use_prime_admm {
                // PRIME proximal dual: normalize step by L2-norm of residuals (noise rejection)
                let mut resid_sq = 0.0_f64;
                let mut residuals = vec![0.0_f64; num_l];
                for l in 0..num_l {
                    let r = s[l] - bat_flow[l] - flows_base[l];
                    residuals[l] = r;
                    resid_sq += r * r;
                    max_resid = max_resid.max(r.abs());
                }
                let norm = 1.0 + resid_sq.sqrt();
                for l in 0..num_l {
                    y[l] += rho * residuals[l] / norm;
                }
            } else {
                for l in 0..num_l {
                    let resid = s[l] - bat_flow[l] - flows_base[l];
                    y[l] += rho * resid;
                    max_resid = max_resid.max(resid.abs());
                }
            }
            let max_du = (0..num_b).map(|b| (actions[b] - prev_actions[b]).abs()).fold(0.0_f64, f64::max);
            if max_resid < tol && max_du < tol { return true; }
        }

        let mut bat_flow_final = vec![0.0_f64; num_l];
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_flow_final[l] += imp * actions[b]; }
        }
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if (bat_flow_final[l] + flows_base[l]).abs() > limit + 1.0 { return false; }
        }
        true
    }

    fn select_active_lines(
        challenge: &Challenge,
        flows_base: &[f64],
        ca: &AycdicdbCache,
        candidate_actions: Option<&[f64]>,
        k: usize,
    ) -> Vec<usize> {
        let num_l = challenge.network.flow_limits.len();
        if k >= num_l {
            return (0..num_l).collect();
        }

        let action_hint = candidate_actions.filter(|a| a.len() == challenge.num_batteries);

        #[derive(Clone, Copy)]
        struct LineInfo {
            line: usize,
            stress: f64,
            norm_sq: f64,
            violated: bool,
            violation_mag: f64,
        }

        let mut batt_weight = vec![1.0_f64; challenge.num_batteries];
        if let Some(a) = action_hint {
            let avg_abs = a.iter().map(|u| u.abs()).sum::<f64>() / a.len().max(1) as f64;
            for b in 0..challenge.num_batteries {
                let abs_a = a[b].abs();
                batt_weight[b] = if avg_abs > 1e-6 && abs_a < 0.5 * avg_abs {
                    0.15
                } else {
                    0.25 + abs_a
                };
            }
        }

        let mut raw: Vec<(usize, f64, f64, f64, bool, f64, f64)> = Vec::new();
        let mut mass_sum = 0.0_f64;

        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit <= 1e-6 { continue; }

            let exo_util = flows_base.get(l).copied().unwrap_or(0.0).abs() / limit;
            let mut hinted_util = exo_util;
            let mut sens_mass = 0.0_f64;
            let mut norm_sq = 0.0_f64;
            let mut violated = false;
            let mut violation_mag = 0.0_f64;

            if let Some(a) = action_hint {
                let mut batt_flow = 0.0_f64;
                for &(b, imp) in &ca.ptdf_sparse[l] {
                    let w = batt_weight[b];
                    batt_flow += imp * a[b];
                    sens_mass += imp.abs() * w;
                    norm_sq += imp * imp * w;
                }
                let total = flows_base.get(l).copied().unwrap_or(0.0) + batt_flow;
                hinted_util = total.abs() / limit;
                if total.abs() > limit + 1e-6 {
                    violated = true;
                    violation_mag = (total.abs() - limit) / limit.max(1e-6);
                }
            } else {
                for &(b, imp) in &ca.ptdf_sparse[l] {
                    let w = batt_weight[b];
                    sens_mass += imp.abs() * w;
                    norm_sq += imp * imp * w;
                }
            }

            mass_sum += sens_mass;
            raw.push((l, exo_util, hinted_util, sens_mass, violated, norm_sq, violation_mag));
        }

        if raw.is_empty() {
            return Vec::new();
        }

        let mass_scale = (mass_sum / raw.len().max(1) as f64).max(1e-9);

        let independent_rank = |mut lines: Vec<LineInfo>| -> Vec<usize> {
            lines.sort_by(|a, b| {
                b.stress
                    .partial_cmp(&a.stress)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line.cmp(&b.line))
            });
            lines.truncate(k);
            lines.into_iter().map(|x| x.line).collect()
        };

        let mut infos: Vec<LineInfo> = raw.into_iter()
            .map(|(l, exo_util, hinted_util, sens_mass, violated, norm_sq, violation_mag)| {
                let mass_norm = (sens_mass / mass_scale).min(5.0);
                let mut stress = exo_util.max(hinted_util) + 0.40 * hinted_util + 0.10 * mass_norm;
                if hinted_util > 0.95 {
                    stress += 1.5 * (hinted_util - 0.95);
                } else if exo_util > 0.95 {
                    stress += 0.25 * (exo_util - 0.95);
                }
                if violated {
                    stress += 10.0 + 20.0 * violation_mag.min(2.0);
                }
                LineInfo { line: l, stress, norm_sq, violated, violation_mag }
            })
            .collect();

        if action_hint.is_none() || k >= 24 {
            return independent_rank(infos);
        }

        let mut violated: Vec<LineInfo> = infos.iter().copied().filter(|x| x.violated).collect();
        violated.sort_by(|a, b| {
            b.violation_mag
                .partial_cmp(&a.violation_mag)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.line.cmp(&b.line))
        });
        if violated.len() >= k {
            violated.truncate(k);
            return violated.into_iter().map(|x| x.line).collect();
        }

        let pool_cap = num_l.min(k.saturating_mul(4).max(k + 8));
        infos.sort_by(|a, b| {
            b.stress
                .partial_cmp(&a.stress)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.line.cmp(&b.line))
        });
        infos.truncate(pool_cap);

        let redundancy = |la: usize, lb: usize, norm_a: f64, norm_b: f64| -> f64 {
            if norm_a <= 1e-12 || norm_b <= 1e-12 {
                return 0.0;
            }
            let va = &ca.ptdf_sparse[la];
            let vb = &ca.ptdf_sparse[lb];
            let mut ia = 0usize;
            let mut ib = 0usize;
            let mut dot_abs = 0.0_f64;

            while ia < va.len() && ib < vb.len() {
                let (ba, pa) = va[ia];
                let (bb, pb) = vb[ib];
                if ba == bb {
                    dot_abs += batt_weight[ba] * pa.abs() * pb.abs();
                    ia += 1;
                    ib += 1;
                } else if ba < bb {
                    ia += 1;
                } else {
                    ib += 1;
                }
            }

            (dot_abs / (norm_a.sqrt() * norm_b.sqrt())).clamp(0.0, 1.0)
        };

        let mut selected: Vec<usize> = violated.iter().map(|x| x.line).collect();
        let mut selected_meta: Vec<(usize, f64)> = violated.iter().map(|x| (x.line, x.norm_sq)).collect();
        let mut picked = vec![false; num_l];
        for &l in &selected {
            if l < picked.len() { picked[l] = true; }
        }

        while selected.len() < k {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_stress = f64::NEG_INFINITY;

            for (idx, info) in infos.iter().enumerate() {
                if picked[info.line] { continue; }

                let mut max_red = 0.0_f64;
                for &(sl, snorm) in &selected_meta {
                    let r = redundancy(info.line, sl, info.norm_sq, snorm);
                    if r > max_red { max_red = r; }
                }

                let mmr_score = info.stress * (1.0 - 0.55 * max_red);
                if mmr_score > best_score + 1e-12
                    || ((mmr_score - best_score).abs() <= 1e-12
                        && (info.stress > best_stress + 1e-12
                            || ((info.stress - best_stress).abs() <= 1e-12
                                && best_idx.map(|j| info.line < infos[j].line).unwrap_or(true))))
                {
                    best_idx = Some(idx);
                    best_score = mmr_score;
                    best_stress = info.stress;
                }
            }

            let Some(idx) = best_idx else { break; };
            let info = infos[idx];
            picked[info.line] = true;
            selected.push(info.line);
            selected_meta.push((info.line, info.norm_sq));
        }

        if selected.len() < k {
            for info in &infos {
                if !picked[info.line] {
                    picked[info.line] = true;
                    selected.push(info.line);
                    if selected.len() >= k { break; }
                }
            }
        }

        selected.truncate(k);
        selected
    }

    /// Linear interpolation of the DP continuation value V(soc) on the SOC grid.
    /// Boundary-safe (the inline `dv` of the baseline LP collapses to 0 at soc_max
    /// because lo_idx == hi_idx there; `dp_lambda` already clamps to soc_levels-2).
    #[inline(always)]
    fn dp_value_at(dp_t: &[f64], soc: f64, soc_min: f64, soc_span: f64) -> f64 {
        let levels = dp_t.len();
        if levels == 0 { return 0.0; }
        if levels == 1 { return dp_t[0]; }
        let idx_f = ((soc - soc_min) / soc_span * ((levels - 1) as f64))
            .clamp(0.0, (levels - 1) as f64);
        let i0 = (idx_f.floor() as usize).min(levels - 1);
        let i1 = (i0 + 1).min(levels - 1);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        dp_t[i0] * (1.0 - frac) + dp_t[i1] * frac
    }

    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        flows_base: &[f64],
        action_hint: &[f64],
        lp_lambda: f64,
        max_pivots: usize,
        lp_hp: &TrackHp,
    ) -> Option<Vec<f64>> {
        joint_lp_dispatch_with_used(
            challenge, state, ca, flows_base, action_hint, lp_lambda, max_pivots, lp_hp,
        ).0
    }

    /// Places the `segs` PWL cuts inside the action interval `[0, cap]` for ONE direction.
    ///
    /// `wid[s]` (s in 0..segs) = width of segment s in ACTION units (they sum to `cap`);
    /// `dep[s]` (s in 0..=segs) = depth of cut s in SOC units (`s_conv * cumulative width`),
    /// i.e. the SOC displacement the action has produced once segments `0..s` are full.
    ///
    /// Only reached for `spacing != 0`: the uniform case keeps the literal baseline
    /// expressions at the call site so that `lp_pwl_spacing=0` is a bit-exact control.
    #[inline(always)]
    fn pwl_place(
        spacing: usize,
        segs: usize,
        cap: f64,
        s_conv: f64,
        ratio: f64,
        soc: f64,
        soc_min: f64,
        delta_s: f64,
        charging: bool,
        wid: &mut [f64; 13],
        dep: &mut [f64; 13],
    ) {
        let w_uni = cap / segs as f64;
        let span_soc = s_conv * cap; // total SOC displacement at full rating
        // Degenerate cases fall back to the uniform lattice.
        if segs < 2 || cap <= 1e-12 || span_soc <= 1e-12 {
            for s in 0..segs { wid[s] = w_uni; }
            for s in 0..=segs { dep[s] = s_conv * w_uni * s as f64; }
            return;
        }
        let h = span_soc / segs as f64; // uniform SOC step of one segment

        dep[0] = 0.0;
        dep[segs] = span_soc;
        match spacing {
            1 => {
                // SOC-ANCHOR (Zheng SoC-segment bids, `4d155575`/`54e10296`): the cuts belong
                // to an ABSOLUTE SOC lattice of step `h` anchored at `soc_min`, not to the
                // action interval. Same granularity as uniform, different PHASE: a battery
                // bids on the same SOC bands whatever its current SOC.
                let p = (soc - soc_min).max(0.0);
                let frac = p - h * (p / h).floor();
                // distance to the first anchor in the direction of travel
                let mut d0 = if charging { h - frac } else { frac };
                if !(d0 > 1e-12) || d0 > h { d0 = h; }
                for s in 1..segs {
                    dep[s] = (d0 + h * (s - 1) as f64).min(span_soc);
                }
            }
            2 => {
                // DP-SNAP: pull each uniform cut onto the nearest node of the DP SOC grid, so
                // every segment secant `V(a)-V(e)` spans WHOLE `dp` cells instead of straddling
                // them. Targets the discretisation jitter measured on the `soc_levels` sweep
                // (i7: Q non-monotone, 4 100 Q spread over {51..201}).
                let mut prev = 0.0_f64;
                for s in 1..segs {
                    let target = h * s as f64;
                    let soc_at = if charging { soc + target } else { soc - target };
                    let k = ((soc_at - soc_min) / delta_s).round();
                    let snapped = soc_min + k * delta_s;
                    let d = if charging { snapped - soc } else { soc - snapped };
                    let d = if d.is_finite() { d } else { target };
                    dep[s] = d.clamp(prev, span_soc);
                    prev = dep[s];
                }
            }
            _ => {
                // GEOMETRIC: widths in geometric progression. `3` = finest segment AT the power
                // rating (the directive's prior, Zheng "finer near rating"); `4` = finest at
                // zero action = the two-sided control that keeps the sweep from being read
                // one-sided (the S-count bell of i3 is exactly what a one-sided read missed).
                let r = if ratio.is_finite() && ratio >= 1.0 { ratio } else { 2.0 };
                let mut pw = [0.0_f64; 13];
                let mut tot = 0.0_f64;
                for s in 0..segs {
                    let e = if spacing == 3 { (segs - 1 - s) as i32 } else { s as i32 };
                    pw[s] = r.powi(e);
                    tot += pw[s];
                }
                if !(tot > 1e-12) { tot = segs as f64; for s in 0..segs { pw[s] = 1.0; } }
                let mut acc = 0.0_f64;
                for s in 1..segs {
                    acc += pw[s - 1] / tot;
                    dep[s] = (span_soc * acc).clamp(0.0, span_soc);
                }
            }
        }
        // Widths derive from the cuts, so they sum to `cap` by construction (no drift).
        for s in 0..segs {
            wid[s] = ((dep[s + 1] - dep[s]) / s_conv).max(0.0);
        }
    }

    /// Per-step joint battery+network LP.
    ///
    /// Variable layout (`segs` segments per direction and per battery):
    ///   discharge b, segment s -> b*segs + s
    ///   charge    b, segment s -> num_b*segs + b*segs + s
    ///   line ai violation slack -> 2*num_b*segs + ai
    /// Row layout (block of `2*segs + 2` rows per battery, then 2 rows per active line):
    ///   r0+s            : x_dis[s] <= w_dis
    ///   r0+segs+s       : x_chg[s] <= w_chg
    ///   r0+2*segs       : SOC lower bound
    ///   r0+2*segs+1     : SOC upper bound
    /// With `segs == 1` this is *bit-for-bit* the baseline layout (same indices, same
    /// order, same floating-point expression shapes) -> `lp_obj_mode=0` is an exact
    /// control for codegen drift.
    fn joint_lp_dispatch_with_used(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        flows_base: &[f64],
        action_hint: &[f64],
        lp_lambda: f64,
        max_pivots: usize,
        lp_hp: &TrackHp,
    ) -> (Option<Vec<f64>>, usize) {
        let num_b = challenge.num_batteries;
        let dt = 0.25_f64;

        let active_lines = select_active_lines(challenge, flows_base, ca, Some(action_hint), lp_hp.lp_max_lines);
        let n_active = active_lines.len();

        let mode = lp_hp.lp_obj_mode;
        let segs: usize = if mode >= 2 { lp_hp.lp_pwl_segments.max(1) } else { 1 };
        let blk = 2 * segs + 2;

        let n = 2 * num_b * segs + n_active;
        let m = num_b * blk + 2 * n_active;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);

        for ai in 0..n_active {
            c_vec[2 * num_b * segs + ai] = -lp_lambda;
        }

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (u_min, u_max) = state.action_bounds[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc = state.socs[b];

            let soc_levels = ca.dp[b][0].len();
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = (idx_f.floor() as isize).max(0) as usize;
            let lo_idx = lo_idx.min(soc_levels - 1);
            let hi_idx = (lo_idx + 1).min(soc_levels - 1);
            let dv = (ca.dp[b][t_next][hi_idx] - ca.dp[b][t_next][lo_idx]) / delta_s;

            let cap_dis = u_max.max(0.0);
            let cap_chg = (-u_min).max(0.0);
            let w_dis = cap_dis / segs as f64;
            let w_chg = cap_chg / segs as f64;
            // per-MW SOC deltas (identical expressions to the baseline matrix entries)
            let s_dis = dt / eta_d;
            let s_chg = eta_c * dt;

            let dp_t = &ca.dp[b][t_next];
            let soc_min = bat.soc_min_mwh;
            let soc_max = bat.soc_max_mwh;

            // Cut lattice. `wid_*[s]` = segment width in action units, `dep_*[s]` = SOC
            // displacement at cut s. `spacing == 0` reproduces the baseline expressions
            // LITERALLY (`s_dis * w_dis * s as f64`, same association order) -> bit-exact.
            let sp_dis = lp_hp.lp_pwl_spacing;
            // Charge inherits the discharge mechanism unless it is given its own
            // (`lp_pwl_spacing_chg = -1` = inert -> the lattice stays symmetric).
            let sp_chg = if lp_hp.lp_pwl_spacing_chg < 0 {
                sp_dis
            } else {
                lp_hp.lp_pwl_spacing_chg as usize
            };
            let mut wid_d = [0.0_f64; 13];
            let mut wid_c = [0.0_f64; 13];
            let mut dep_d = [0.0_f64; 13];
            let mut dep_c = [0.0_f64; 13];
            if sp_dis == 0 {
                for s in 0..segs { wid_d[s] = w_dis; }
                for s in 0..=segs { dep_d[s] = s_dis * w_dis * s as f64; }
            } else {
                pwl_place(sp_dis, segs, cap_dis, s_dis, lp_hp.lp_pwl_geo_ratio,
                          soc, soc_min, delta_s, false, &mut wid_d, &mut dep_d);
            }
            if sp_chg == 0 {
                for s in 0..segs { wid_c[s] = w_chg; }
                for s in 0..=segs { dep_c[s] = s_chg * w_chg * s as f64; }
            } else {
                pwl_place(sp_chg, segs, cap_chg, s_chg, lp_hp.lp_pwl_geo_ratio,
                          soc, soc_min, delta_s, true, &mut wid_c, &mut dep_c);
            }

            // marginal continuation value per segment (mode 0/3 keep the tangent slope)
            let mut mv_dis = vec![dv; segs];
            let mut mv_chg = vec![dv; segs];
            match mode {
                1 => {
                    // CHORD: secant of the concave V over the SOC interval the action can
                    // actually traverse. Zero extra LP variables -> time-neutral.
                    if cap_dis > 1e-12 {
                        let d = (s_dis * cap_dis).min(soc - soc_min).max(0.0);
                        if d > 1e-12 {
                            let sl = (dp_value_at(dp_t, soc, soc_min, span)
                                - dp_value_at(dp_t, soc - d, soc_min, span)) / d;
                            if sl.is_finite() { for x in mv_dis.iter_mut() { *x = sl; } }
                        }
                    }
                    if cap_chg > 1e-12 {
                        let d = (s_chg * cap_chg).min(soc_max - soc).max(0.0);
                        if d > 1e-12 {
                            let sl = (dp_value_at(dp_t, soc + d, soc_min, span)
                                - dp_value_at(dp_t, soc, soc_min, span)) / d;
                            if sl.is_finite() { for x in mv_chg.iter_mut() { *x = sl; } }
                        }
                    }
                }
                2 => {
                    // PWL VALUE: per-segment secant + concave hull. Because the marginal
                    // value lost by discharging is non-decreasing in depth (and the value
                    // gained by charging non-increasing), the LP consumes the segments in
                    // order without any integer variable.
                    if cap_dis > 1e-12 {
                        let mut run = f64::NEG_INFINITY;
                        for s in 0..segs {
                            let a = (soc - dep_d[s]).max(soc_min);
                            let e = (soc - dep_d[s + 1]).max(soc_min);
                            let d = a - e;
                            let sl = if d > 1e-12 {
                                (dp_value_at(dp_t, a, soc_min, span)
                                    - dp_value_at(dp_t, e, soc_min, span)) / d
                            } else { dv };
                            if sl.is_finite() && sl > run { run = sl; }
                            mv_dis[s] = if run.is_finite() { run } else { dv };
                        }
                    }
                    if cap_chg > 1e-12 {
                        let mut run = f64::INFINITY;
                        for s in 0..segs {
                            let a = (soc + dep_c[s]).min(soc_max);
                            let e = (soc + dep_c[s + 1]).min(soc_max);
                            let d = e - a;
                            let sl = if d > 1e-12 {
                                (dp_value_at(dp_t, e, soc_min, span)
                                    - dp_value_at(dp_t, a, soc_min, span)) / d
                            } else { dv };
                            if sl.is_finite() && sl < run { run = sl; }
                            mv_chg[s] = if run.is_finite() { run } else { dv };
                        }
                    }
                }
                _ => {}
            }

            let off_d = b * segs;
            let off_c = num_b * segs + b * segs;
            // deg(u) = (|u|*dt/cap)^2 = deg_c * u^2 -- convex, so its marginal cost over
            // segment s of width w is deg_c*w*(2s+1) per MW: an EXACT PWL for a max-LP.
            let deg_c = (dt / bat.capacity_mwh.max(1e-9)).powi(2);

            for s in 0..segs {
                let mut cd = (rt - 0.25) * dt - mv_dis[s] * dt / eta_d;
                let mut cc = -(rt + 0.25) * dt + mv_chg[s] * eta_c * dt;
                if mode == 3 {
                    let k = (2 * s + 1) as f64;
                    cd -= deg_c * wid_d[s] * k;
                    cc -= deg_c * wid_c[s] * k;
                }
                c_vec[off_d + s] = cd;
                c_vec[off_c + s] = cc;
            }

            let r0 = b * blk;
            for s in 0..segs {
                a_mat[r0 + s][off_d + s] = 1.0;
                b_vec[r0 + s] = wid_d[s];
                a_mat[r0 + segs + s][off_c + s] = 1.0;
                b_vec[r0 + segs + s] = wid_c[s];
            }

            let r_lo = r0 + 2 * segs;
            let r_hi = r_lo + 1;
            for s in 0..segs {
                a_mat[r_lo][off_d + s] =  s_dis;
                a_mat[r_lo][off_c + s] = -s_chg;
                a_mat[r_hi][off_d + s] = -s_dis;
                a_mat[r_hi][off_c + s] =  s_chg;
            }
            b_vec[r_lo] = (soc - bat.soc_min_mwh).max(0.0);
            b_vec[r_hi] = (bat.soc_max_mwh - soc).max(0.0);
        }

        let row_f = num_b * blk;
        for (ai, &l) in active_lines.iter().enumerate() {
            let limit = challenge.network.flow_limits[l];
            let exo = flows_base[l];
            let viol_idx = 2 * num_b * segs + ai;
            let rp = row_f + 2 * ai;
            let rn = rp + 1;

            for &(b, impact) in &ca.ptdf_sparse[l] {
                let off_d = b * segs;
                let off_c = num_b * segs + b * segs;
                for s in 0..segs {
                    a_mat[rp][off_d + s] += impact;
                    a_mat[rp][off_c + s] -= impact;
                    a_mat[rn][off_d + s] -= impact;
                    a_mat[rn][off_c + s] += impact;
                }
            }
            a_mat[rp][viol_idx] = -1.0;
            a_mat[rn][viol_idx] = -1.0;

            b_vec[rp] = (limit - exo).max(0.0);
            b_vec[rn] = (limit + exo).max(0.0);
        }

        let (opt_x, pivots_used) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, max_pivots);
        let Some(opt_x) = opt_x else { return (None, 0); };

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let off_d = b * segs;
            let off_c = num_b * segs + b * segs;
            // segs==1 -> exactly `opt_x[b] - opt_x[num_b + b]` (no extra rounding step)
            let mut u = opt_x[off_d];
            for s in 1..segs { u += opt_x[off_d + s]; }
            u -= opt_x[off_c];
            for s in 1..segs { u -= opt_x[off_c + s]; }
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }
        (Some(actions), pivots_used)
    }

    /// Instantaneous step reward of action `u` for battery `b` (revenue - transaction cost
    /// - degradation). Identical expression shape to the first four lines of `eval_profit`
    /// but WITHOUT its `V(t+1)` continuation term: the two-step score below replaces that
    /// term with a network-constrained stage-2 LP value.
    #[inline(always)]
    fn inst_profit(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize, u: f64) -> f64 {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let dt = 0.25_f64;
        let abs_u = u.abs();
        let revenue = u * rt_price * dt;
        let tx = 0.25 * abs_u * dt;
        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
        let deg = deg_base * deg_base;
        revenue - tx - deg
    }

    /// Optimal stage-(t+1) value reachable from the SOC vector `socs_next`, under the t+1
    /// exogenous line loading and the t+1 day-ahead prices, with the DP continuation V(t+2)
    /// as terminal value. Same LP shape as the h=1 block of `mpc_dispatch_2step` (one step,
    /// no cross-step coupling), so both candidates are scored on the SAME network-aware
    /// objective. Returns the LP objective value.
    fn stage2_lp_value(
        challenge: &Challenge,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        socs_next: &[f64],
        exo_flows_t1: &[f64],
        da_t1: &[f64],
        t2: usize,
    ) -> Option<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let lambda_soft = hp.lp_soft_lambda;

        let n = 2 * num_b + 2 * num_l;
        let m = 2 * num_b + 2 * num_l;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let soc = socs_next[b];
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let t2_idx = t2.min(ca.dp[b].len() - 1);
            let mut lambda_b = dp_lambda(&ca.dp[b], t2_idx, soc, bat.soc_min_mwh, span, soc_levels);
            if !lambda_b.is_finite() {
                lambda_b = 0.0;
            }

            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            // t+1 action bounds: nameplate power capped by SOC availability/headroom --
            // same formulas as `Battery::compute_action_bounds`, which is crate-private.
            let ub_d = ((soc - bat.soc_min_mwh).max(0.0) * eta_d / dt)
                .min(bat.power_discharge_mw).max(0.0);
            let ub_c = if eta_c > 0.0 {
                ((bat.soc_max_mwh - soc).max(0.0) / (eta_c * dt))
                    .min(bat.power_charge_mw).max(0.0)
            } else {
                0.0
            };

            let da_p = if node < da_t1.len() { da_t1[node] } else { da_t1[0] };

            let d_idx = b;
            let c_idx = num_b + b;
            c_vec[d_idx] = (da_p - 0.25) * dt - lambda_b * dt / eta_d;
            c_vec[c_idx] = (-da_p - 0.25) * dt + lambda_b * eta_c * dt;

            a_mat[b][d_idx] = 1.0;
            b_vec[b] = ub_d;
            a_mat[num_b + b][c_idx] = 1.0;
            b_vec[num_b + b] = ub_c;
        }

        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            let exo = exo_flows_t1[l];

            let v_plus_idx = 2 * num_b + l;
            let v_minus_idx = 2 * num_b + num_l + l;
            let rp = 2 * num_b + 2 * l;
            let rn = rp + 1;

            for &(b, impact) in &ca.ptdf_sparse[l] {
                a_mat[rp][b] += impact;
                a_mat[rp][num_b + b] -= impact;
                a_mat[rn][b] -= impact;
                a_mat[rn][num_b + b] += impact;
            }

            a_mat[rp][v_plus_idx] = -1.0;
            a_mat[rn][v_minus_idx] = -1.0;

            c_vec[v_plus_idx] = -lambda_soft;
            c_vec[v_minus_idx] = -lambda_soft;

            b_vec[rp] = (limit - exo).max(0.0);
            b_vec[rn] = (limit + exo).max(0.0);
        }

        let pivots = hp.mpc_pivot_budget.max(100);
        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, pivots);
        let opt_x = opt_x?;

        let mut obj = 0.0_f64;
        for j in 0..n {
            obj += c_vec[j] * opt_x[j];
        }
        if obj.is_finite() { Some(obj) } else { None }
    }

    /// Score the MPC candidate and the incumbent on the SAME two-step objective:
    /// instantaneous reward at `t` (true RT prices, degradation included) plus the optimal
    /// network-constrained stage value at `t+1`. `None` when the horizon or the stage-2 LP
    /// makes the comparison unavailable -- the caller then falls back to the historical
    /// single-step guard.
    fn mpc_two_step_scores(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        mpc_act: &[f64],
        base_act: &[f64],
    ) -> Option<(f64, f64)> {
        let num_b = challenge.num_batteries;
        let t = state.time_step;
        if challenge.num_steps.saturating_sub(t) < 2 {
            return None;
        }
        let dt = 0.25_f64;
        let t2 = (t + 2).min(challenge.num_steps);

        let exo_inj = if t + 1 < challenge.exogenous_injections.len() {
            &challenge.exogenous_injections[t + 1]
        } else {
            challenge.exogenous_injections.last()?
        };
        let exo_flows_t1 = challenge.network.compute_flows(exo_inj);

        let da_t1 = if t + 1 < challenge.market.day_ahead_prices.len() {
            &challenge.market.day_ahead_prices[t + 1]
        } else {
            challenge.market.day_ahead_prices.last()?
        };

        let soc_after = |acts: &[f64]| -> Vec<f64> {
            (0..num_b)
                .map(|b| {
                    let bat = &challenge.batteries[b];
                    let u = acts[b];
                    let soc = state.socs[b];
                    let raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh)
                })
                .collect()
        };

        let socs_mpc = soc_after(mpc_act);
        let socs_base = soc_after(base_act);

        let v_mpc = stage2_lp_value(challenge, ca, hp, &socs_mpc, &exo_flows_t1, da_t1, t2)?;
        let v_base = stage2_lp_value(challenge, ca, hp, &socs_base, &exo_flows_t1, da_t1, t2)?;

        let p_mpc: f64 = (0..num_b).map(|b| inst_profit(challenge, state, ca, b, mpc_act[b])).sum();
        let p_base: f64 = (0..num_b).map(|b| inst_profit(challenge, state, ca, b, base_act[b])).sum();

        Some((p_mpc + v_mpc, p_base + v_base))
    }

    fn mpc_dispatch_2step(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let t = state.time_step;
        let hz = hp.mpc_horizon.min(2);
        let remaining = challenge.num_steps.saturating_sub(t);
        if remaining < 2 || hz < 2 {
            return None;
        }
        let dt = 0.25_f64;
        let lambda_soft = hp.lp_soft_lambda;
        let t2 = (t + 2).min(challenge.num_steps);

        let n = 4 * num_b + 4 * num_l;

        let m = 4 * num_b + 4 * num_l;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let mut lambdas = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let t2_idx = t2.min(ca.dp[b].len() - 1);
            lambdas[b] = dp_lambda(&ca.dp[b], t2_idx, soc, bat.soc_min_mwh, soc_span, soc_levels);
            if !lambdas[b].is_finite() {
                lambdas[b] = 0.0;
            }
        }

        let da_t = if t < challenge.market.day_ahead_prices.len() {
            &challenge.market.day_ahead_prices[t]
        } else {
            challenge.market.day_ahead_prices.last().unwrap_or(&challenge.market.day_ahead_prices[0])
        };
        let da_t1 = if t + 1 < challenge.market.day_ahead_prices.len() {
            &challenge.market.day_ahead_prices[t + 1]
        } else {
            da_t
        };

        let exo_flows_t = flows_base;
        let exo_flows_t1;
        {
            let exo_inj = if t + 1 < challenge.exogenous_injections.len() {
                &challenge.exogenous_injections[t + 1]
            } else {
                challenge.exogenous_injections.last().unwrap_or(&challenge.exogenous_injections[t])
            };
            exo_flows_t1 = challenge.network.compute_flows(exo_inj);
        }
        let exo_for_h = [exo_flows_t, &exo_flows_t1];

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let (u_min, u_max) = state.action_bounds[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let ub_d = u_max.max(0.0);
            let ub_c = (-u_min).max(0.0);

            let lambda_b = lambdas[b];

            for h in 0..2 {
                let da_p = if node < da_t.len() {
                    if h == 0 { da_t[node] } else { da_t1[node] }
                } else {
                    if h == 0 { da_t[0] } else { da_t1[0] }
                };

                let h_base = 2 * num_b * h;
                let d_idx = h_base + b;
                let c_idx = h_base + num_b + b;

                c_vec[d_idx] = (da_p - 0.25) * dt - lambda_b * dt / eta_d;
                c_vec[c_idx] = (-da_p - 0.25) * dt + lambda_b * eta_c * dt;

                let r_d = 2 * h * num_b + b;
                let r_c = 2 * h * num_b + num_b + b;

                a_mat[r_d][d_idx] = 1.0;
                b_vec[r_d] = ub_d;

                a_mat[r_c][c_idx] = 1.0;
                b_vec[r_c] = ub_c;
            }
        }

        for h in 0..2 {
            let exo_h = exo_for_h[h];
            let constraint_base = 4 * num_b + 2 * h * num_l;

            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                let exo = exo_h[l];

                let v_plus_idx = 4 * num_b + 2 * h * num_l + l;
                let v_minus_idx = 4 * num_b + (2 * h + 1) * num_l + l;

                let rp = constraint_base + 2 * l;
                let rn = constraint_base + 2 * l + 1;

                let h_base = 2 * num_b * h;
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let d_idx = h_base + b;
                    let c_idx = h_base + num_b + b;
                    a_mat[rp][d_idx] += impact;
                    a_mat[rp][c_idx] -= impact;
                    a_mat[rn][d_idx] -= impact;
                    a_mat[rn][c_idx] += impact;
                }

                a_mat[rp][v_plus_idx] = -1.0;
                a_mat[rn][v_minus_idx] = -1.0;

                c_vec[v_plus_idx] = -lambda_soft;
                c_vec[v_minus_idx] = -lambda_soft;

                b_vec[rp] = (limit - exo).max(0.0);
                b_vec[rn] = (limit + exo).max(0.0);
            }
        }

        let pivots = hp.mpc_pivot_budget.max(100);
        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, pivots);
        let opt_x = opt_x?;

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let d0 = opt_x[b];
            let c0 = opt_x[num_b + b];
            let u = d0 - c0;
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }

        Some(actions)
    }

    const POLICY_NUM_SEGMENTS: usize = 4;
    const POLICY_NUM_FEATURES: usize = 6;
    const POLICY_PARAMS_PER_BATT: usize = POLICY_NUM_SEGMENTS * (POLICY_NUM_FEATURES + 1);

    fn da_price_norm(p: f64) -> f64 {
        (p - 100.0) / 50.0
    }

    fn rt_da_delta_norm(d: f64) -> f64 {
        d / 30.0
    }

    fn eval_policy_for_battery(weights: &[f64], features: &[f64], norm_soc: f64, action_lo: f64, action_hi: f64) -> f64 {
        let seg_f = (norm_soc * POLICY_NUM_SEGMENTS as f64).floor();
        let seg = (seg_f as usize).min(POLICY_NUM_SEGMENTS - 1);
        let base = seg * (POLICY_NUM_FEATURES + 1);

        let mut action = weights[base + POLICY_NUM_FEATURES];
        for f in 0..POLICY_NUM_FEATURES {
            action += weights[base + f] * features[f];
        }

        action.clamp(action_lo, action_hi)
    }

    pub fn policy_dispatch(
        challenge: &Challenge,
        state: &State,
        weights: &[Vec<f64>],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let mut actions = vec![0.0_f64; num_b];

        let exo_inj = &challenge.exogenous_injections[state.time_step.min(challenge.exogenous_injections.len().saturating_sub(1))];
        let exo_flows = challenge.network.compute_flows(exo_inj);
        let mut max_util = 0.0_f64;
        for l in 0..challenge.network.flow_limits.len() {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let u = exo_flows[l].abs() / limit;
                if u > max_util { max_util = u; }
            }
        }
        let congestion_bin = if max_util < 0.5 { 0.0 } else if max_util < 0.8 { 0.5 } else { 1.0 };

        for b in 0..num_b {
            if b >= weights.len() || weights[b].len() < POLICY_PARAMS_PER_BATT {
                continue;
            }

            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let soc_range = bat.soc_max_mwh - bat.soc_min_mwh;
            let norm_soc = if soc_range > 1e-9 { (soc - bat.soc_min_mwh) / soc_range } else { 0.5 };

            let time_fraction = if challenge.num_steps > 1 {
                state.time_step as f64 / challenge.num_steps as f64
            } else { 0.0 };

            let node = challenge.batteries[b].node;
            let node_idx = if node < state.rt_prices.len() { node } else { state.rt_prices.len() - 1 };
            let rt_price = state.rt_prices[node_idx];
            let da_node = challenge.market.day_ahead_prices.len().saturating_sub(1);
            let da_node_idx = if node < da_node || node == 0 { node.min(da_node) } else { da_node };
            let da_vec = &challenge.market.day_ahead_prices[da_node_idx];
            let t_idx = if state.time_step < da_vec.len() { state.time_step } else { da_vec.len() - 1 };
            let da_price = da_vec[t_idx];
            let rt_da_delta = rt_price - da_price;

            let (lo, hi) = state.action_bounds[b];
            let action_range = hi - lo;

            let features: [f64; POLICY_NUM_FEATURES] =
                [norm_soc, time_fraction, da_price_norm(da_price), rt_da_delta_norm(rt_da_delta), congestion_bin, action_range];

            actions[b] = eval_policy_for_battery(&weights[b], &features, norm_soc, lo, hi);
        }

        actions
    }

    fn policy_rand_f64() -> f64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEED: AtomicU64 = AtomicU64::new(12345);
        let mut s = SEED.fetch_add(7, Ordering::Relaxed).wrapping_add(1);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f64) / (u32::MAX as f64)
    }

    fn policy_randn() -> f64 {
        thread_local! { static SPARE: RefCell<Option<f64>> = RefCell::new(None); }
        SPARE.with(|sp| {
            let mut guard = sp.borrow_mut();
            if let Some(s) = guard.take() { return s; }
            let u1 = policy_rand_f64().max(1e-15);
            let u2 = policy_rand_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let n1 = r * (2.0 * std::f64::consts::PI * u2).cos();
            let n2 = r * (2.0 * std::f64::consts::PI * u2).sin();
            *guard = Some(n2);
            n1
        })
    }

    fn simulate_policy(
        challenge: &Challenge,
        initial_state: &State,
        cache: &AycdicdbCache,
        hp: &TrackHp,
        weights: &[Vec<f64>],
    ) -> f64 {
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        let mut socs = initial_state.socs.clone();
        let mut total_profit = 0.0_f64;

        for t in 0..num_t {
            let rt_prices: Vec<f64> = challenge.market.day_ahead_prices.iter().map(|np| {
                if t < np.len() { np[t] } else { *np.last().unwrap_or(&0.0) }
            }).collect();

            let mut actions = policy_dispatch(challenge, &State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: (0..num_b).map(|b| {
                    let bat = &challenge.batteries[b];
                    let soc = socs[b];
                    let lo = (-(bat.power_charge_mw * 0.25)).max(-(soc - bat.soc_min_mwh) / bat.efficiency_charge.max(1e-9));
                    let hi = (bat.power_discharge_mw * 0.25).min((bat.soc_max_mwh - soc) * bat.efficiency_discharge.max(1e-9) / 0.25);
                    (lo, hi)
                }).collect(),
                total_profit,
            }, weights);

            let zero_action = vec![0.0_f64; num_b];
            let inj_base = challenge.compute_total_injections(&State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: actions.iter().map(|&a| (a, a)).collect(),
                total_profit,
            }, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base);
            run_deflator(challenge, &State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: (0..num_b).map(|_| (f64::NEG_INFINITY, f64::INFINITY)).collect(),
                total_profit,
            }, cache, hp, &flows_base, &mut actions);

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = cache.batt_nodes[b].min(rt_prices.len() - 1);
                let rt_price = rt_prices[node];
                let u = actions[b];
                let dt = 0.25_f64;
                let revenue = u * rt_price * dt;
                let tx = 0.25 * u.abs() * dt;
                let deg_base = (u.abs() * dt) / bat.capacity_mwh.max(1e-9);
                total_profit += revenue - tx - deg_base * deg_base;

                let next_soc = if u < 0.0 {
                    socs[b] + bat.efficiency_charge * (-u) * dt
                } else {
                    socs[b] - u / bat.efficiency_discharge.max(1e-9) * dt
                };
                socs[b] = next_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }
        }

        total_profit
    }

    pub fn train_policy_cmaes(
        challenge: &Challenge,
        initial_state: &State,
        cache: &AycdicdbCache,
        hp: &TrackHp,
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let dpb = POLICY_PARAMS_PER_BATT;
        let dim = num_b * dpb;

        let mut mean = vec![0.0; dim];
        let mut sigma = 1.0;
        let popsize = (20 + (3.0 * dim as f64).sqrt() as usize).min(60);
        let mu = popsize / 2;
        let max_gens = 15;

        let mut w: Vec<f64> = vec![0.0; mu];
        let mut w_sum = 0.0;
        for i in 0..mu {
            w[i] = (mu as f64 + 1.0 - i as f64).ln() / (mu as f64 + 1.0 - (i + 1) as f64).ln();
            w[i] = 1.0 / (mu as f64 / 2.0 + w[i].exp());
            w_sum += w[i];
        }
        for i in 0..mu { w[i] /= w_sum; }
        let eff_popsize: f64 = w_sum * w_sum / (0..mu as usize).map(|i| w[i] * w[i]).sum::<f64>();

        let cs = (eff_popsize + 2.0) / (dim as f64 + eff_popsize + 2.0);
        let ca = cs / (cs + 2.0);
        let dc = (1.0 + 2.0 * ((eff_popsize - 1.0) / (dim as f64 + 1.0)).sqrt() + cs) / (dim as f64 + cs);
        let ds = (1.0 + 2.0 * ((eff_popsize - 1.0) / (dim as f64 + 1.0)).sqrt() + cs) / (dim as f64 + cs);

        let mut ds_acc = 1.0_f64;
        let mut pc = vec![0.0; dim];

        let mut best_weights = mean.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let ec = 2.0 + 2.0 / (dim as f64 + 1.0);

        for _gen in 0..max_gens {
            let mut fitnesses = Vec::with_capacity(popsize);
            for _ in 0..popsize {
                let mut ind = mean.clone();
                for d in 0..dim {
                    ind[d] += sigma * policy_randn();
                }
                let weights_batt: Vec<Vec<f64>> = (0..num_b)
                    .map(|b| ind[b * dpb..(b + 1) * dpb].to_vec())
                    .collect();
                let fit = simulate_policy(challenge, initial_state, cache, hp, &weights_batt);
                fitnesses.push((fit, ind));
            }

            fitnesses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).reverse());

            if fitnesses[0].0 > best_fitness {
                best_fitness = fitnesses[0].0;
                best_weights = fitnesses[0].1.clone();
            }

            let mut m_new = mean.clone();
            for i in 0..mu {
                for d in 0..dim {
                    m_new[d] += w[i] * fitnesses[i].1[d];
                }
            }

            let chi_n = (dim as f64).sqrt() * (1.0 - 1.0 / dim as f64 + (dim as f64) * (1.0 - 2.0 / (dim as f64 + 4.0)).sqrt() / (dim as f64 - 1.0)).sqrt();
            for d in 0..dim {
                pc[d] = (1.0 - dc) * pc[d] + (dc * (eff_popsize / ec).sqrt()).min(1.0) * (m_new[d] - mean[d]) / sigma;
            }

            let pc_norm: f64 = pc.iter().map(|&x| x * x).sum::<f64>().sqrt();
            ds_acc = (1.0 - ds) * ds_acc + ds * (pc_norm / chi_n);
            sigma *= (ds_acc - 1.0) * cs * ca;
            sigma = sigma.max(1e-10).min(100.0);

            mean = m_new;
        }

        (0..num_b)
            .map(|b| best_weights[b * dpb..(b + 1) * dpb].to_vec())
            .collect()
    }

    /// i21 — SIGNAL DE CONGESTION PAR PAS DE TEMPS, pour la borne d'action du DP.
    ///
    /// `ratio[t] = max_l |f_exo(t,l)| / limit_l` : l'utilisation de la ligne la plus
    /// chargée par les seuls flux EXOGÈNES au pas `t`. C'est exactement la grandeur dont
    /// la memory `09d54049` dit que la distribution est BIMODALE sur t50 (« highly
    /// congested or low »), et celle que la row dead `engine:action_aware_premium` dit
    /// rester sous 1,0 (« flux exo < limites sur t50 ») — d'où un seuil `nd_gate` en HP
    /// plutôt qu'un « ≥ 1 » codé en dur, et une SONDE qui mesure la distribution réelle
    /// avant qu'on choisisse le seuil.
    ///
    /// ⚠️ Q-NEUTRE quand `nd_mode == 0 && nd_probe == 0` : on ne l'appelle même pas
    /// (le CTRL n'alloue rien ⇒ bit-exactitude par construction, pas par espérance).
    /// Coût : `num_t` × `compute_flows` = 96 appels par construction de DP, contre un nid
    /// DP en `num_t × soc_levels × (action_grid+1) × num_b` ≈ 10⁶–10⁷ ⇒ 4 ordres de
    /// grandeur sous le site chaud (gate `A_SITE_IS_NOT_A_COUNT`, i10).
    fn nd_congestion_ratio(challenge: &Challenge) -> Vec<f64> {
        let num_t = challenge.num_steps;
        let num_l = challenge.network.flow_limits.len();
        let mut ratio = vec![0.0_f64; num_t];
        if num_l == 0 {
            return ratio;
        }
        for t in 0..num_t {
            let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut worst = 0.0_f64;
            for l in 0..num_l.min(exo_flows.len()) {
                let limit = challenge.network.flow_limits[l];
                if limit > 1e-9 {
                    let u = exo_flows[l].abs() / limit;
                    if u > worst { worst = u; }
                }
            }
            ratio[t] = worst;
        }
        ratio
    }

    /// i23 — SEUIL RELATIF (quantile empirique) de la distribution de `ratio[t]`.
    ///
    /// Correction dure d'i21, dont le seuil ABSOLU (clampé 10.0) était INATTEIGNABLE :
    /// `ratio` reste sous 1,0 sur t50 (row dead `engine:action_aware_premium`), donc
    /// `nd_gate=99` dégénérait en `d = nd_free` partout — une constante déguisée en porte.
    /// Un quantile garantit qu'une fraction `1 − q` des pas est AU-DESSUS du seuil,
    /// quelle que soit l'échelle des flux. `q=0` ⇒ tous les pas congestionnés (alias du
    /// CTRL scalaire) ; `q=1` ⇒ aucun (alias de `nd_free`) : les deux bornes sont des
    /// contrôles internes lisibles dans la table de screening.
    fn nd_gate_threshold(ratio: &[f64], q: f64) -> f64 {
        if ratio.is_empty() {
            return f64::INFINITY;
        }
        let mut srt: Vec<f64> = ratio.to_vec();
        srt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = srt.len();
        let idx = (((n - 1) as f64) * q.clamp(0.0, 1.0)).round() as usize;
        srt[idx.min(n - 1)]
    }

    /// i23 — EXPOSITION PTDF NORMALISÉE par batterie (famille SPATIALE, mode 3).
    ///
    /// `expo[b] = max_l |impact_{l,b}| / limit_l`, renormalisé par le max sur `b` ⇒ dans
    /// [0,1]. C'est la sensibilité STRUCTURELLE des lignes contraintes à la puissance de
    /// `b` : une batterie électriquement loin de toute ligne serrée ne congestionne rien
    /// et est bridée à tort par un scalaire global.
    ///
    /// ⚠️ Discriminant vs `brick:par_batterie` (SATURATED) : cette row ferme la modulation
    /// par PROFIL de batterie (`ct_ref_kappa`, `dur_w`) sur un parc HOMOGÈNE (n ≤ 30) — et
    /// elle a raison, les batteries sont interchangeables. Ici la dissymétrie ne vient PAS
    /// du profil mais de la GÉOMÉTRIE du réseau (position PTDF du nœud), qui reste
    /// hétérogène même à parc parfaitement homogène. Familles disjointes.
    ///
    /// ⚠️ Discriminant vs le mode 3 d'i21 (dérate DIRECTIONNEL sur le signe de `mu_adjust`) :
    /// aux sites 1/2 `mu = zero_mu` (L2878) ⇒ `mu_adjust ≡ 0` ⇒ le mode 3 d'i21 y serait
    /// un NO-OP PAR CONSTRUCTION. Le porter tel quel aurait re-brûlé un iter sur un site
    /// mort. `b_to_lines`, lui, est structurel et NON nul sous `zero_mu`.
    fn nd_ptdf_exposure(challenge: &Challenge, b_to_lines: &[Vec<(usize, f64)>]) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let mut expo = vec![0.0_f64; num_b];
        for b in 0..num_b.min(b_to_lines.len()) {
            let mut worst = 0.0_f64;
            for &(l, impact) in b_to_lines[b].iter() {
                if l < challenge.network.flow_limits.len() {
                    let limit = challenge.network.flow_limits[l];
                    if limit > 1e-9 {
                        let e = impact.abs() / limit;
                        if e > worst { worst = e; }
                    }
                }
            }
            expo[b] = worst;
        }
        let peak = expo.iter().cloned().fold(0.0_f64, f64::max);
        if peak > 1e-12 {
            for e in expo.iter_mut() { *e /= peak; }
        }
        expo
    }

    /// i23 — les 3 familles de dérate état-dépendant, appliquées aux sites 1/2.
    ///
    /// Rend `(der_charge, der_decharge)`. `nd_mode == 0` retombe MOT POUR MOT sur le
    /// scalaire (bit-exactitude du CTRL par construction, pas par espérance).
    #[inline(always)]
    fn nd_derate_pair(hp: &TrackHp, nd_r: f64, thr: f64, expo_b: f64) -> (f64, f64) {
        match hp.nd_mode {
            // 1 — PORTE BIMODALE (TEMPORELLE) : dérate historique aux pas dont la
            // congestion exogène est au-dessus du quantile `nd_gate`, pleine puissance
            // (`nd_free`) ailleurs. Attaque la bimodalité mesurée de f_exo/limit (`09d54049`).
            1 => {
                let d = if nd_r >= thr { hp.network_derating } else { hp.nd_free };
                (d, d)
            }
            // 2 — RAMPE CONTINUE (TEMPORELLE) : interpolation `nd_free → network_derating`
            // sur [0, thr]. Famille distincte de 1 (aucune discontinuité) : si la
            // bimodalité est franche, 1 gagne ; si le signal est graduel, 2 gagne.
            2 => {
                let x = if thr > 1e-9 { (nd_r / thr).clamp(0.0, 1.0) } else { 1.0 };
                let d = hp.nd_free + (hp.network_derating - hp.nd_free) * x;
                (d, d)
            }
            // 3 — EXPOSITION PTDF (SPATIALE) : dérate proportionnel à la capacité de la
            // batterie à charger les lignes serrées. Axe ORTHOGONAL à 1/2 (espace vs temps).
            3 => {
                let d = hp.nd_free + (hp.network_derating - hp.nd_free) * expo_b.clamp(0.0, 1.0);
                (d, d)
            }
            // 0 — CTRL : mot pour mot i15.
            _ => (hp.network_derating, hp.network_derating),
        }
    }

    /// i23 — dérate SCALAIRE effectif à un site donné, sous le masque d'attribution.
    /// `site` : 12 = `build_dp_with_mu*` (prescreen) · 3 = `build_dp_stochastic_for_battery`.
    /// Un site MASQUÉ reçoit 1.0 (= aucun bridage), pas 0 : on retire le dérate, on ne
    /// coupe pas la batterie.
    #[inline(always)]
    fn nd_scalar_for_site(hp: &TrackHp, site: usize) -> f64 {
        match hp.nd_scalar_mask {
            1 => if site == 12 { hp.network_derating } else { 1.0 },
            2 => if site == 3 { hp.network_derating } else { 1.0 },
            _ => hp.network_derating,
        }
    }

    fn build_dp_parallel_or_serial(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
    ) -> Vec<Vec<Vec<f64>>> {
        let num_b = challenge.num_batteries;
        let mut dp: Vec<Vec<Vec<f64>>>;

        // i6 — ÉLIMINATION DU PRÉPASSE DP DÉTERMINISTE MORT.
        //
        // Le bloc `if hp.use_sdp` ci-dessous fait `dp[b] = build_dp_stochastic_for_battery(..)`
        // pour TOUT b de `0..num_b` : c'est une ÉCRASE TOTALE, pas un mélange. Et
        // `build_dp_stochastic_for_battery` ne prend PAS `dp` en entrée — elle alloue sa
        // propre table (`let mut dp = vec![vec![0.0; soc_levels]; num_t + 1]`). Le prépasse
        // déterministe construit juste au-dessus est donc, quand `use_sdp` est vrai,
        // intégralement JETÉ : 100 % de son coût est du travail mort.
        //
        // Le prépasse et la passe stochastique parcourent le MÊME triple nid
        // (num_t × soc_levels × (action_grid+1)) ; le stochastique y ajoute seulement une
        // boucle interne de `num_q = 3` nœuds de quadrature (mélange à queue : 2 nœuds GH
        // + 1 nœud de saut). Le prépasse pèse donc ~1/3 du temps DP, et le DP pèse ~70 %
        // du cycle (analyse/bricks.md) — sur le chemin prod il est payé 2× par nonce
        // (prescreen DW + DP final), aux deux appels en pleine résolution.
        //
        // Sauter le prépasse est Q-NEUTRE PAR CONSTRUCTION (aucun octet consommé en aval).
        // C'est la prédiction falsifiable de cet iter : Q doit rester BIT-EXACT à 586 194.
        // `dp` est alloué à la forme exacte que produit la passe stochastique.
        let skip_dead_prepass = hp.use_sdp && hp.dp_prepass_mode == 1;

        if skip_dead_prepass {
            dp = vec![vec![vec![0.0_f64; hp.soc_levels]; challenge.num_steps + 1]; num_b];
        } else if num_b <= 1 || !hp.use_parallel_dp {
            dp = build_dp_with_mu(challenge, hp, batt_nodes, expected_premiums, b_to_lines, mu);
        } else {
            let num_workers = num_b.min(32);
            let chunk_size = (num_b + num_workers - 1) / num_workers;

            let arc_challenge: Arc<Challenge> = Arc::new(challenge.clone());
            let arc_hp: Arc<TrackHp> = Arc::new(hp.clone());
            let arc_batt_nodes: Arc<Vec<usize>> = Arc::new(batt_nodes.to_vec());
            let arc_expected_premiums: Arc<Vec<Vec<f64>>> = Arc::new(expected_premiums.to_vec());
            let arc_b_to_lines: Arc<Vec<Vec<(usize, f64)>>> = Arc::new(b_to_lines.to_vec());
            let arc_mu: Arc<Vec<Vec<f64>>> = Arc::new(mu.to_vec());
            let _ = chunk_size;

            dp = vec![vec![vec![0.0_f64; hp.soc_levels]; challenge.num_steps + 1]; num_b];
            for b in 0..num_b {
                let tbl = build_dp_with_mu_for_battery(
                    &*arc_challenge, &*arc_hp, &*arc_batt_nodes,
                    &*arc_expected_premiums, &*arc_b_to_lines, &*arc_mu, b,
                );
                if !tbl.is_empty() {
                    dp[b] = tbl;
                }
            }
        }

        if hp.use_sdp {
            for b in 0..num_b {
                dp[b] = build_dp_stochastic_for_battery(
                    challenge, hp, batt_nodes, expected_premiums, b_to_lines, mu, b,
                );
            }
        }

        dp
    }

    fn build_dp_with_mu_for_battery(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
        b: usize,
    ) -> Vec<Vec<f64>> {
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let mut dp = vec![vec![0.0_f64; soc_levels]; num_t + 1];

        let bat = &challenge.batteries[b];
        let node = batt_nodes[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);

        // i23 — SITE 2, même mécanisme qu'au site 1. Hoisté hors du nid `t`. Cette fonction
        // est appelée PAR BATTERIE (L5877), donc `nd_congestion_ratio` y coûte `num_b × num_t`
        // ≈ 30 × 96 = 2 880 `compute_flows` au lieu de 96 — toujours 3 ordres de grandeur sous
        // le nid `num_t × soc_levels × (action_grid+1)`, et STRICTEMENT NUL sous le CTRL.
        let nd_on = hp.nd_mode != 0;
        let nd_ratio: Vec<f64> = if nd_on { nd_congestion_ratio(challenge) } else { Vec::new() };
        let nd_thr = if nd_on { nd_gate_threshold(&nd_ratio, hp.nd_gate) } else { 0.0 };
        let expo_b = if nd_on {
            let e = nd_ptdf_exposure(challenge, b_to_lines);
            if b < e.len() { e[b] } else { 0.0 }
        } else { 0.0 };
        let nd_scalar = nd_scalar_for_site(hp, 12);

        for t in (0..num_t).rev() {
            let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][node]
            } else {
                challenge.market.day_ahead_prices[t][0]
            };

            let mu_adjust: f64 = b_to_lines[b].iter()
                .map(|&(l, impact)| mu[t][l] * impact)
                .sum();

            let extra = expected_premiums[t][b] - mu_adjust;
            let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
            let p_buy = p_da + extra;

            // i23 — SITE 2 : identique au site 1 (même `nd_derate_pair`, mêmes familles).
            let (der_c, der_d) = if nd_on {
                let nd_r = if t < nd_ratio.len() { nd_ratio[t] } else { 0.0 };
                nd_derate_pair(hp, nd_r, nd_thr, expo_b)
            } else {
                (nd_scalar, nd_scalar)
            };
            let max_pwr_c = bat.power_charge_mw * der_c;
            let max_pwr_d = bat.power_discharge_mw * der_d;

            for i in 0..soc_levels {
                let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0)));
                let u_max = max_pwr_d.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut max_val = f64::NEG_INFINITY;
                let span = u_max - u_min;
                for j in 0..=action_grid {
                    let u = if span > 0.0 {
                        u_min + span * (j as f64) / (action_grid as f64)
                    } else { u_min };
                    let price = if u > 0.0 { p_sell } else { p_buy };
                    let abs_u = u.abs();
                    let revenue = u * price * dt;
                    let tx = 0.25 * abs_u * dt;
                    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                    let deg = deg_base * deg_base;
                    let profit = revenue - tx - deg;

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let v_next = dp[t + 1][idx0c] * (1.0 - frac)
                        + dp[t + 1][idx1c] * frac;

                    let val = profit + v_next;
                    if val > max_val { max_val = val; }
                }
                dp[t][i] = max_val;
            }
        }
        dp
    }

    fn gh_quadrature(k: usize) -> (Vec<f64>, Vec<f64>) {
        match k {
            3 => (
                vec![-1.732_050_807_568_877_2, 0.0, 1.732_050_807_568_877_2],
                vec![0.166_666_666_666_666_7, 0.666_666_666_666_666_6, 0.166_666_666_666_666_7],
            ),
            _ => (
                vec![-1.0, 1.0],
                vec![0.5, 0.5],
            ),
        }
    }

    fn build_dp_stochastic_for_battery(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
        b: usize,
    ) -> Vec<Vec<f64>> {
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let bat = &challenge.batteries[b];
        let node = batt_nodes[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);

        let sigma: f64 = hp.dp_sigma;
        let rho_jump: f64 = hp.dp_rho_jump;
        let alpha: f64 = hp.dp_alpha;

        let jump_prem_factor = if rho_jump > 0.0 && alpha > 1.0 {
            rho_jump * alpha / (alpha - 1.0)
        } else {
            0.0
        };

        let mut dp = vec![vec![0.0_f64; soc_levels]; num_t + 1];

        let std_gh = gh_quadrature(hp.sdp_k);

        for t in (0..num_t).rev() {
            let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][node]
            } else {
                challenge.market.day_ahead_prices[t][0]
            };

            // i10 — CHEMIN CTRL (`sdp_alloc_mode == 0`) : les `Vec` heap de i9, inchangés.
            // `Vec::new()` n'alloue PAS ⇒ sous le mode 1 ces liaisons ne coûtent rien.
            let (nodes_q_heap, weights_q_heap) = if hp.sdp_alloc_mode == 0 {
                if hp.use_tail_quadrature {
                    build_tail_mixture_quadrature(p_da, sigma, rho_jump, alpha)
                } else {
                    std_gh.clone()
                }
            } else {
                (Vec::new(), Vec::new())
            };

            let mu_adjust: f64 = b_to_lines[b].iter()
                .map(|&(l, impact)| {
                    if l < mu[t].len() { mu[t][l] * impact } else { 0.0 }
                })
                .sum();

            let extra = expected_premiums[t][b] - mu_adjust;
            let p_sell_base = p_da * (1.0 + hp.jump_premium) + extra;
            let p_buy_base = p_da + extra;

            // i23 — SITE 3 REMIS AU SCALAIRE. i21 a greffé les 3 modes ICI en croyant que
            // c'était le seul site vivant ; la mesure a rendu 5 arms BIT-EXACTS avec le CTRL,
            // Y COMPRIS `nd_free=0.05` (dérate ×20 sur toutes les batteries et tous les pas).
            // Une perturbation de cette violence qui ne déplace pas UN SEUL des 32 nonces
            // prouve que ce site n'influence pas Q ⇒ y remettre un mécanisme serait re-brûler
            // un iter sur un site mort (row `brick:dp_stochastic_power_bound_site`, dead).
            // Le mécanisme part aux sites 1/2 ; ici on ne garde que le masque d'attribution,
            // qui rend ce site MESURABLE sans le rendre porteur.
            let der_c = nd_scalar_for_site(hp, 3);
            let der_d = der_c;

            let max_pwr_c = bat.power_charge_mw * der_c;
            let max_pwr_d = bat.power_discharge_mw * der_d;

            let base_abs = p_da.abs().max(1e-6);

            // i10 — CHEMIN INLINE (`sdp_alloc_mode != 0`) : buffers PILE, 0 allocation heap.
            // Mêmes nœuds, mêmes poids, MÊME ORDRE (gh2[0] → gh2[1] → jump), mêmes gardes.
            let mut nodes_buf = [0.0_f64; 3];
            let mut weights_buf = [0.0_f64; 3];
            let mut perturb_buf = [0.0_f64; 3];
            let n_inline = if hp.sdp_alloc_mode != 0 {
                let n = if hp.use_tail_quadrature {
                    tail_mixture_quadrature_inline(sigma, rho_jump, alpha, &mut nodes_buf, &mut weights_buf)
                } else {
                    gh_quadrature_inline(hp.sdp_k, &mut nodes_buf, &mut weights_buf)
                };
                for k in 0..n {
                    perturb_buf[k] = sigma * nodes_buf[k] * base_abs + base_abs * jump_prem_factor;
                }
                n
            } else {
                0
            };

            let scene_perturb_heap: Vec<f64> = if hp.sdp_alloc_mode == 0 {
                nodes_q_heap.iter().map(|&z| {
                    sigma * z * base_abs + base_abs * jump_prem_factor
                }).collect()
            } else {
                Vec::new()
            };

            // Les deux chemins convergent sur des SLICES : le reste du corps est identique.
            let (weights_q, scene_perturb): (&[f64], &[f64]) = if hp.sdp_alloc_mode != 0 {
                (&weights_buf[..n_inline], &perturb_buf[..n_inline])
            } else {
                (&weights_q_heap[..], &scene_perturb_heap[..])
            };
            let num_q = weights_q.len();

            // i9 — COLLAPSE ALGÉBRIQUE DE LA QUADRATURE (mode 2).
            // Le seul terme dépendant de `k` dans l'espérance est `scene_perturb[k]`, et il
            // entre LINÉAIREMENT dans le revenu. Donc, pour tout (i, j) :
            //   Σ_k w_k·( u·dt·(base + p_k) − tx − deg + v_next )
            //     = u·dt·( W·base + P̄ ) − W·(tx + deg) + W·v_next
            // avec W = Σ_k w_k et P̄ = Σ_k w_k·p_k, tous deux constants SUR LE PAS t.
            // `sell_eff` / `buy_eff` absorbent `W·base + P̄` ⇒ la branche `u > 0` reste, mais
            // la boucle interne disparaît entièrement du nid le plus profond.
            let w_sum: f64 = weights_q.iter().sum();
            let pbar: f64 = weights_q.iter().zip(scene_perturb.iter())
                .map(|(&w, &p)| w * p)
                .sum();
            let sell_eff = w_sum * p_sell_base + pbar;
            let buy_eff = w_sum * p_buy_base + pbar;

            for i in 0..soc_levels {
                let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0)));
                let u_max = max_pwr_d.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut max_val = f64::NEG_INFINITY;
                let span = u_max - u_min;

                for j in 0..=action_grid {
                    let u = if span > 0.0 {
                        u_min + span * (j as f64) / (action_grid as f64)
                    } else { u_min };

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let v_next = dp[t + 1][idx0c] * (1.0 - frac) + dp[t + 1][idx1c] * frac;
                    if !v_next.is_finite() {
                        continue;
                    }

                    let exp_val = match hp.sdp_expect_mode {
                        // --- mode 2 : COLLAPSE. Boucle `k` éliminée (voir dérivation ci-dessus).
                        2 => {
                            let abs_u = u.abs();
                            let tx = 0.25 * abs_u * dt;
                            let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                            let deg = deg_base * deg_base;
                            let eff = if u > 0.0 { sell_eff } else { buy_eff };
                            u * eff * dt - w_sum * (tx + deg) + w_sum * v_next
                        }
                        // --- mode 1 : HOIST. Invariants en `k` sortis de la boucle ; l'ORDRE
                        // d'accumulation est identique au CTRL ⇒ Q bit-exact par construction.
                        1 => {
                            let abs_u = u.abs();
                            let tx = 0.25 * abs_u * dt;
                            let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                            let deg = deg_base * deg_base;
                            let base_price = if u > 0.0 { p_sell_base } else { p_buy_base };
                            let mut acc = 0.0_f64;
                            for k in 0..num_q {
                                let revenue = u * (base_price + scene_perturb[k]) * dt;
                                acc += weights_q[k] * (revenue - tx - deg + v_next);
                            }
                            acc
                        }
                        // --- mode 0 : CTRL historique, inchangé octet pour octet.
                        _ => {
                            let mut acc = 0.0_f64;
                            for k in 0..num_q {
                                let perturbation = scene_perturb[k];
                                let price = if u > 0.0 {
                                    p_sell_base + perturbation
                                } else {
                                    p_buy_base + perturbation
                                };
                                let abs_u = u.abs();
                                let revenue = u * price * dt;
                                let tx = 0.25 * abs_u * dt;
                                let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                                let deg = deg_base * deg_base;
                                let profit_scenario = revenue - tx - deg;
                                acc += weights_q[k] * (profit_scenario + v_next);
                            }
                            acc
                        }
                    };

                    if exp_val.is_finite() && exp_val > max_val {
                        max_val = exp_val;
                    }
                }

                if max_val == f64::NEG_INFINITY {
                    let single_price = if p_sell_base.is_sign_positive() {
                        p_sell_base
                    } else {
                        p_buy_base
                    };
                    let mid_u = (u_max + u_min) * 0.5;
                    let abs_mid = mid_u.abs();
                    let rev = mid_u * single_price * dt;
                    let tx = 0.25 * abs_mid * dt;
                    let deg_base = (abs_mid * dt) / bat.capacity_mwh.max(1e-9);
                    max_val = (rev - tx - deg_base * deg_base).max(0.0);
                }

                dp[t][i] = max_val;
            }
        }

        dp
    }

    fn build_tail_mixture_quadrature(
        _da_price: f64,
        sigma: f64,
        rho_jump: f64,
        alpha: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        if alpha <= 1.0 || sigma <= 0.0 || rho_jump <= 0.0 || rho_jump >= 1.0 {
            return gh_quadrature(3);
        }

        let mean_pareto = alpha / (alpha - 1.0);
        if !mean_pareto.is_finite() {
            return gh_quadrature(3);
        }

        let w_norm = 1.0 - rho_jump;
        let w_jump = rho_jump;

        let (gh2_nodes, gh2_weights) = gh_quadrature(2);

        let mut nodes = Vec::with_capacity(3);
        let mut weights = Vec::with_capacity(3);

        for (n, w) in gh2_nodes.iter().zip(gh2_weights.iter()) {
            nodes.push(*n);
            weights.push(w_norm * w);
        }

        let jump_node = mean_pareto / sigma;
        let jump_node = if jump_node.is_finite() && jump_node > 0.0 { jump_node } else { 2.0 };
        nodes.push(jump_node);
        weights.push(w_jump);

        let wsum: f64 = weights.iter().sum();
        if !wsum.is_finite() || wsum.abs() < 1e-12 {
            return gh_quadrature(3);
        }

        (nodes, weights)
    }

    /// i10 — variante ZÉRO-ALLOCATION de `gh_quadrature`, écrite dans des buffers PILE.
    /// Renvoie le nombre de nœuds. Valeurs et ORDRE strictement identiques à la version
    /// `Vec` (littéraux recopiés à l'identique) ⇒ aucune dérive possible.
    #[inline(always)]
    fn gh_quadrature_inline(k: usize, nodes: &mut [f64; 3], weights: &mut [f64; 3]) -> usize {
        match k {
            3 => {
                nodes[0] = -1.732_050_807_568_877_2;
                nodes[1] = 0.0;
                nodes[2] = 1.732_050_807_568_877_2;
                weights[0] = 0.166_666_666_666_666_7;
                weights[1] = 0.666_666_666_666_666_6;
                weights[2] = 0.166_666_666_666_666_7;
                3
            }
            _ => {
                nodes[0] = -1.0;
                nodes[1] = 1.0;
                weights[0] = 0.5;
                weights[1] = 0.5;
                2
            }
        }
    }

    /// i10 — variante ZÉRO-ALLOCATION de `build_tail_mixture_quadrature`.
    /// Reproduit À L'IDENTIQUE : les 3 branches de garde (fallback `gh_quadrature(3)`),
    /// l'ordre d'insertion (gh2[0], gh2[1], jump), les poids NON normalisés
    /// (`w_norm * 0.5` deux fois, puis `w_jump`), le garde-fou `jump_node > 0` → 2.0,
    /// et le contrôle `wsum` sommé dans le MÊME ordre (fold depuis 0.0).
    /// `_da_price` du parent est inutilisé ⇒ non repris ici.
    #[inline(always)]
    fn tail_mixture_quadrature_inline(
        sigma: f64,
        rho_jump: f64,
        alpha: f64,
        nodes: &mut [f64; 3],
        weights: &mut [f64; 3],
    ) -> usize {
        if alpha <= 1.0 || sigma <= 0.0 || rho_jump <= 0.0 || rho_jump >= 1.0 {
            return gh_quadrature_inline(3, nodes, weights);
        }

        let mean_pareto = alpha / (alpha - 1.0);
        if !mean_pareto.is_finite() {
            return gh_quadrature_inline(3, nodes, weights);
        }

        let w_norm = 1.0 - rho_jump;
        let w_jump = rho_jump;

        // gh_quadrature(2) : nodes [-1.0, 1.0], weights [0.5, 0.5]
        nodes[0] = -1.0;
        nodes[1] = 1.0;
        weights[0] = w_norm * 0.5;
        weights[1] = w_norm * 0.5;

        let jump_node = mean_pareto / sigma;
        let jump_node = if jump_node.is_finite() && jump_node > 0.0 { jump_node } else { 2.0 };
        nodes[2] = jump_node;
        weights[2] = w_jump;

        let wsum: f64 = 0.0_f64 + weights[0] + weights[1] + weights[2];
        if !wsum.is_finite() || wsum.abs() < 1e-12 {
            return gh_quadrature_inline(3, nodes, weights);
        }

        3
    }


}
mod lp {
    const LP_EPS: f64 = 1e-9;

    /// Entering-variable selection. Bland-like = first column with reduced cost < -eps (default).
    /// Dantzig = column with the most-negative reduced cost (use_dantzig=true).
    
    #[inline(always)]
    fn pick_entering(row: &[f64], n_vars: usize, use_dantzig: bool) -> Option<usize> {
        if use_dantzig {
            let mut best_j: Option<usize> = None;
            let mut best_val = -LP_EPS;
            for j in 0..n_vars {
                if row[j] < best_val {
                    best_val = row[j];
                    best_j = Some(j);
                }
            }
            best_j
        } else {
            (0..n_vars).find(|&j| row[j] < -LP_EPS)
        }
    }

    /// ITER i88 (2026-08-21, port i80) : meme algorithme (Bland premier-negatif, memes tie-breaks, memes flottants dans le
    /// meme ordre), tableau PLAT + saut des colonnes ou la ligne pivot vaut exactement 0.0 => resultat bit-identique.
    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, 0);
        }
        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1;
        let mut tab = vec![0.0_f64; n_cols * (m + 1)];
        for i in 0..m {
            let row = &mut tab[i * n_cols..(i + 1) * n_cols];
            row[..n].copy_from_slice(&a[i][..n]);
            row[n + i] = 1.0;
            row[rhs_col] = b[i].max(0.0);
        }
        for j in 0..n { tab[m * n_cols + j] = -c[j]; }
        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut pivots_done: usize = 0;
        let mut nz: Vec<usize> = Vec::with_capacity(n_cols);
        let mut prow: Vec<f64> = vec![0.0; n_cols];
        for pivot in 0..max_pivots {
            pivots_done = pivot + 1;
            let entering = match pick_entering(&tab[m * n_cols..(m + 1) * n_cols], n_vars, false) {
                Some(j) => j,
                None => break,
            };
            let leaving_row = (0..m)
                .filter(|&i| tab[i * n_cols + entering] > LP_EPS)
                .min_by(|&i1, &i2| {
                    let r1 = tab[i1 * n_cols + rhs_col] / tab[i1 * n_cols + entering];
                    let r2 = tab[i2 * n_cols + rhs_col] / tab[i2 * n_cols + entering];
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                });
            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, 0),
            };
            let pivot_val = tab[leaving_row * n_cols + entering];
            if pivot_val.abs() < LP_EPS {
                return (None, 0);
            }
            {
                let row = &mut tab[leaving_row * n_cols..(leaving_row + 1) * n_cols];
                for j in 0..n_cols { row[j] /= pivot_val; }
                prow.copy_from_slice(row);
            }
            nz.clear();
            for j in 0..n_cols { if prow[j] != 0.0 { nz.push(j); } }
            for i in 0..=m {
                if i != leaving_row {
                    let base = i * n_cols;
                    let factor = tab[base + entering];
                    if factor.abs() > 1e-15 {
                        let row = &mut tab[base..base + n_cols];
                        for &j in nz.iter() { row[j] -= factor * prow[j]; }
                    }
                }
            }
            basis[leaving_row] = entering;
        }
        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i * n_cols + rhs_col].max(0.0);
            }
        }
        (Some(x), pivots_done)
    }

    pub fn lp_solve_with_duals(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
        use_dantzig: bool, bland_degen: bool,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, None, 0);
        }

        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1;

        let mut tab = vec![vec![0.0_f64; n_cols]; m + 1];

        for i in 0..m {
            for j in 0..n {
                tab[i][j] = a[i][j];
            }
            tab[i][n + i] = 1.0;
            tab[i][rhs_col] = b[i].max(0.0);
        }

        for j in 0..n {
            tab[m][j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();

        for pivot_count in 0..max_pivots {
            let entering_d = match pick_entering(&tab[m], n_vars, use_dantzig) {
                Some(j) => j,
                None => {
                    let duals = (0..m).map(|i| {
                        let slack_col = n + i;
                        let val = -tab[m][slack_col];
                        if val.is_finite() { val } else { 0.0 }
                    }).collect();
                    let mut x = vec![0.0_f64; n];
                    for (i, &bv) in basis.iter().enumerate() {
                        if bv < n {
                            x[bv] = tab[i][rhs_col].max(0.0);
                        }
                    }
                    return (Some(x), Some(duals), pivot_count);
                }
            };

            // Find leaving row for Dantzig entering
            let lr_d = (0..m)
                .filter(|&i| tab[i][entering_d] > LP_EPS)
                .min_by(|&i1, &i2| {
                    let r1 = tab[i1][rhs_col] / tab[i1][entering_d];
                    let r2 = tab[i2][rhs_col] / tab[i2][entering_d];
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                });

            // Dantzig-Bland hybrid: on degenerate Dantzig steps, fall back to Bland entering.
            // Degenerate step = RHS of leaving row ≈ 0 (ratio = 0, no primal progress).
            
            let (entering, leaving_row) = if use_dantzig && bland_degen {
                if let Some(lr) = lr_d {
                    if tab[lr][rhs_col] <= LP_EPS {
                        if let Some(bland_j) = (0..n_vars).find(|&j| tab[m][j] < -LP_EPS) {
                            let lr_b = (0..m)
                                .filter(|&i| tab[i][bland_j] > LP_EPS)
                                .min_by(|&i1, &i2| {
                                    let r1 = tab[i1][rhs_col] / tab[i1][bland_j];
                                    let r2 = tab[i2][rhs_col] / tab[i2][bland_j];
                                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                                });
                            (bland_j, lr_b)
                        } else {
                            (entering_d, lr_d)
                        }
                    } else {
                        (entering_d, lr_d)
                    }
                } else {
                    (entering_d, lr_d)
                }
            } else {
                (entering_d, lr_d)
            };

            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, None, 0),
            };

            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, None, 0);
            }
            for j in 0..n_cols {
                tab[leaving_row][j] /= pivot_val;
            }

            for i in 0..=m {
                if i != leaving_row {
                    let factor = tab[i][entering];
                    if factor.abs() > 1e-15 {
                        for j in 0..n_cols {
                            tab[i][j] -= factor * tab[leaving_row][j];
                        }
                    }
                }
            }

            basis[leaving_row] = entering;
        }

        let duals = (0..m).map(|i| {
            let slack_col = n + i;
            let val = -tab[m][slack_col];
            if val.is_finite() { val } else { 0.0 }
        }).collect();
        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }
        (Some(x), Some(duals), max_pivots)
    }
}
mod track_congested {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 201,
            action_grid: 40,
            asca_iters: 4,
            ternary_iters: 20,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            flow_feas_tol: 1e-6,
            network_derating: 0.22,
            nd_mode: 0,
            nd_free: 1.0,
            nd_gate: 0.5,
            nd_probe: 0,
            nd_scalar_mask: 0,
            dual_iters: 0,
            da_step_size: 0.01,
            ldd_iters: 4,
            ldd_step_size: 0.25,
            use_kkt: true,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 10,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: true,
            dantzig_in_dw: false,
            dantzig_in_lns: false,
            dantzig_in_kkt: false,
            dantzig_bland_degen: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 800,
            lp_total_pivots: 15000,
            use_policy: false,
            use_warmstart: true,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: true,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: true,
            use_lns: true,
            // i15 — BAKE du point DOMINANT-2-AXES. 3 → 1.
            // Mesuré 4× (jobs 26376 v4577 · 26411/26412/26413 v4582, m13, 32/32, 0-invalide) :
            // `lns_cg_iters=1` rend **586336 @ 3,0 s**, CTRL `hp={}` rend **586194 @ 4,0 s**
            // (10 benchs, 6 binaires distincts, variance NULLE des deux côtés) ⇒ disjoint sur
            // les DEUX axes. Ce n'est PAS une troncature heuristique : i13 a mesuré
            // `it/calls = 1,907` (la boucle s'auto-termine déjà sur `!added_any`), et i11 a
            // prouvé que les colonnes des iters CG 2-3 sont légitimes au sens Dantzig-Wolfe
            // mais SANS VALEUR en Q. i11→i14 ont cherché un PRÉDICAT adaptatif pour récupérer
            // ces 2 quanta (4 iters, 4 nulls) ; la valeur statique les rend gratuitement.
            // ⚠️ Le +142 Q est SOUS le plancher de bruit du track (~1500) : il n'est PAS
            // revendiqué comme un gain et n'est attribué à AUCUN mécanisme (cf i14, anti-pattern
            // A_SUB_NOISE_DELTA_CANNOT_NAME_A_MECHANISM). La revendication est l'axe TEMPS
            // (−25 %, 2 quanta) à Q ≥ baseline.
            lns_cg_iters: 1,
            lns_cg_column_limit: 6,
            lns_max_lines: 12,
            lns_lp_pivots_total: 6000,
            use_pivot_reserve: true,
            lp_max_lines: 12,
            use_parallel_dp: true,
            use_sdp: true,
            sdp_k: 3,
            dp_prepass_mode: 1,
            // BAKE i8 = mode 3 (grille 31/15 + déterministe), choisi À LA MESURE, pas a priori.
            // Cascade v4570 (binaire unique, m13, 32/32 nonces, 0 invalide) — Q BIT-EXACT sur
            // les 5 configurations aux DEUX points de fonctionnement :
            //   point prod  (soc=101) : Q=586194 — CTRL 4,5 s | modes 1/2/3 et prescreen OFF 4,0 s
            //   calibration (soc=401) : Q=581535 — CTRL 7,5 s | mode1 6,5 | mode2 6,5 | mode3 5,5
            //                                     | prescreen OFF (étalon D) 5,5
            // Le mode 3 ATTEINT l'étalon `use_dw_prescreen=false` (5,5 s = 5,5 s) : le coût
            // résiduel de la sonde passe SOUS le quantum de 500 ms ⇒ la sonde devient gratuite
            // sans être RETIRÉE (on garde sa robustesse sur des instances non vues).
            dw_prescreen_mode: 3,
            // i9 — défaut = 2 (COLLAPSE algébrique de la boucle de quadrature).
            // Adjugé À LA MESURE sur la cascade v4574 (m13, 9 benchs 32/32 0-invalide) :
            //   calibration (soc=401) : Q=581535 BIT-EXACT sur les 3 modes | CTRL {6,0·5,5·5,5}
            //                           mode1 HOIST 6,0 (= CTRL, LLVM hisse deja) | mode2 {5,0·5,0·5,0}
            //   point prod (soc=101)  : Q=586194 BIT-EXACT @4,0 s
            //   etalon use_sdp=false  : 4,5 s / Q=564201 (-2,98 %) = plafond de la famille
            // -9,1 % a la calibration, distributions DISJOINTES 3v3. La loi n est PAS touchee :
            // memes noeuds, memes poids, sdp_k=3 conserve -> Q bit-exact, pas "proche".
            sdp_expect_mode: 2,
            // i10 — defaut = 1 (INLINE, zero allocation heap dans la boucle `t`).
            // Le collapse i9 a retire la boucle `k` ; il restait 3 `Vec<f64>` heap par
            // (pas de temps x batterie) dont on n'extrait que 2 scalaires (W, Pbar).
            // Mode 1 = memes noeuds, memes poids, MEME ORDRE de sommation, memes gardes,
            // sur des buffers PILE [f64; 3] -> Q bit-exact par construction.
            sdp_alloc_mode: 1,
            // i11 — 3 = RC-GATE + SAT-SHORTCIRCUIT. Cascade benchée intra-binaire :
            // 0 = CTRL (contrôle de détecteur, doit rendre 4,0 s / Q=586194), 1, 2, 3.
            lns_cg_stop_mode: 3,
            // i12 — défauts NEUTRES : sonde OFF et gate OFF ⇒ le binaire par défaut doit
            // rendre EXACTEMENT la baseline (586194 @ 4,0 s) = contrôle de détecteur.
            lns_probe: 0,
            lns_gate_mode: 0,
            lns_gate_headroom_tol: 0.0,
            lns_gate_streak_k: 3,
            lns_gate_probe_period: 4,
            // i13 — défauts NEUTRES : sonde OFF et mécanisme OFF ⇒ `mat_on == false` ⇒
            // ZÉRO calcul supplémentaire dans la boucle CG ⇒ le binaire par défaut doit
            // rendre EXACTEMENT la baseline (586194 @ 4,0 s) = contrôle de détecteur.
            lns_cg_probe: 0,
            lns_cg_mat_mode: 0,
            lns_cg_mat_delta: 0.0,
            // i14 — défauts NEUTRES : `cg_obj_active(hp) == false` ⇒ le wrapper délègue
            // à `eval_profit_with_price` SANS toucher au calcul ⇒ le binaire par défaut
            // doit rendre EXACTEMENT la baseline (586194 @ 4,0 s) = contrôle de détecteur.
            cg_cont_scale: 1.0,
            cg_prox_rho: 0.0,
            cg_cong_haircut: 0.0,
            use_ldd_proximal: true,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: true,
            dp_sigma: 0.15,
            dp_rho_jump: 0.02,
            dp_alpha: 3.5,
            lns_dual_smooth_alpha: 0.0,
            use_primal_refine: true,
            use_lmp_premiums_kkt: true,
            use_prime_admm: false,
            dw_mu_damping_alpha: 0.0,
            dw_boxstep_delta: 0.0,
            dw_wentges_alpha_min: 0.0,
            dw_wentges_alpha_max: 0.0,
            use_adaptive_lines: false,
            use_binary_congestion_premium: false,
            congestion_quantize_levels: 0,
            use_action_aware_premium: false,
            premium_shape_gamma: 2.0,
            premium_impact_delta: 1.0,
            use_learned_line_weights: false,
            line_weight_w_min: 0.3,
            line_weight_w_max: 3.0,
            use_cap_norm_weights: false,
            use_sqrt_cap_weights: false,
            use_cos_weights: false,
            cos_line_weight_scale: 1.0,
            cos_asymmetry_kappa: 0.0,
            use_cos_cs_weights: false,
            cos_alpha_under: 1.0,
            coord_premium_mode: 0,
            coord_premium_scale: 0.0,
            use_slp_degradation: false,
            use_ptdf_constraint_tracking: false,
            ct_step_eta: 0.5,
            ct_ref_kappa: 0.0,
            ct_oc_kappa: 0.0,
            use_ct_adaptive_per_line: false,
            ct_gdd_rho: 0.0,
            ct_gdd_alpha: 0.0,
            lp_obj_mode: 0,
            lp_pwl_segments: 3,
            // Geometric cut lattice deployed HERE and not only in the `mod.rs` default
            // 0x39a3 bytes across i8/i9/i10 while `hp={}` kept measuring the i8 lattice),
            // so the winning placement must ALSO be the struct-level default, which lives
            // in this file's CGU. `r = 3.5` = argmax of the 8-point curve measured on
            // / 2.0 585 286 / 2.6 586 108 / 3.0 585 710 / 3.5 586 194 / 4.2 586 151 /
            // 6.0 583 431 — broad interior max on [2.6, 4.2].
            // A user `{lp_pwl_spacing: 0}` still wins over this default (override beats
            // default in `override_from_map`), so the uniform control stays bit-exact.
            lp_pwl_spacing: 3,
            lp_pwl_geo_ratio: 3.5,
            lp_pwl_spacing_chg: -1,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        
        solve_with_hp(challenge, save_solution, hp)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub soc_levels: Option<usize>,
    pub action_grid: Option<usize>,
    pub asca_iters: Option<usize>,
    pub ternary_iters: Option<usize>,
    pub convergence_tol: Option<f64>,
    pub anticipate_lmp: Option<bool>,
    pub lmp_threshold: Option<f64>,
    pub lmp_premium_scale: Option<f64>,
    pub jump_premium: Option<f64>,
    pub prune_ratio: Option<f64>,
    pub deflator_iters: Option<usize>,
    pub flow_margin: Option<f64>,
    pub network_derating: Option<f64>,
    pub ldd_iters: Option<usize>,
    pub ldd_step_size: Option<f64>,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    track_congested::solve(challenge, save_solution, hyperparameters)
}