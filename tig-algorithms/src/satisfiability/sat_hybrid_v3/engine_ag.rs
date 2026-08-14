// ══════════════════════════════════════════════════════════════════════════════════════
// engine_ag — t4 / i28 · brick=codegen:empty_list_degenerate_guard · strategy=replace (BAKE)
// ══════════════════════════════════════════════════════════════════════════════════════
// Copie LITTÉRALE d'`engine_af` (t4/i27, ver 4445, sha `0460511e11`) avec `DEFAULT_VARIANT`
// 0 → 2 : **UNE SEULE CONSTANTE CHANGE**. Bake du survivant de la cascade ZL/LM MESURÉE
// (jobs 25656-25659, travail fixe 400 M, M13 ; byte-identité 4/4 `md5(sonde)=90fd5db6…`,
// `bundle_quality` sonde 156 250, 32/32, 0 invalide) :
//   arm 0 `A_ctrl`   = champion i26 VERBATIM ......... 78,51 s   (ancre intra-binaire)
//   arm 1 `V_zlfree` = gardes RMW, classe 2×/flip .... 78,01 s   −0,50 s = 1 pas de grille
//                                                                 ⇒ **NON RÉSOLU**
//   arm 2 `V_lmfree` = garde `sad3 lmin>0`, 1×/flip .. 75,51 s   **−3,00 s = −3,82 %**
//   arm 3 `V_both`   = composition .................... 75,51 s   = arm 2 (Δ **0,00 s**)
// ⇒ le bundle n'est PAS *mesuré* strictement meilleur que le meilleur arm simple ⇒ on bake
// l'arm SIMPLE (arm 2). `ZL` reste `false` en production : deux mesures indépendantes le
// donnent nul (seul : 1 pas de grille, sous le gate de 1,0 s ; en composition sur `LM` :
// Δ = 0,00 s EXACT).
// ⚠️ Le pronostic pré-enregistré d'i27 (arm 1 PORTEUR / arm 2 FAIBLE-À-NUL, fondé sur la
// pondération par la FRÉQUENCE — RMW 2×/flip vs scan 1×/flip) est **RÉFUTÉ PAR LA MESURE,
// exactement à l'envers**. Ni le compte de blocs STATIQUE (déjà réfuté par i25) ni la
// FRÉQUENCE d'exécution seule ne prédisent le gain. Les arms 0/1/3 restent atteignables
// par `sel_variant`.
//
// ── en-tête d'origine (i27 / `engine_af`), conservé ───────────────────────────────────
// COPIE d'`engine_ae` (t4/i26, ver 4443, sha `1c23749875`, CHAMPION PROD 144 020 ms).
// Delta = DEUX const-génériques neufs (`ZL`, `LM`) qui retirent les TROIS gardes de
// dégénérescence « liste d'occurrences vide » du chemin dominant. Corps LITTÉRALEMENT
// partagés par macro (`ru_body!`, `su_body!`, `fused_body!`) ⇒ byte-identité PAR
// CONSTRUCTION entre l'ancre et les variantes.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// CENSUS PONDÉRÉ PAR LA FRÉQUENCE — re-tiré sur le binaire CHAMPION ver 4443
// ══════════════════════════════════════════════════════════════════════════════════════
// Monomorphisation championne identifiée PAR LE BINAIRE (arm `TCS=true, TCR=true` ⇒ ni
// `shr $0x2` ni `shr $0x3` dans sa fenêtre ; les 6 occurrences résiduelles du binaire
// tombent toutes dans les 3 autres arms) = fenêtre `0x152c8b`–`0x154dd0`, back-edge du
// hot flip `0x154dc8 → 0x152c8b`.
//
// **AMAS ⑦ = LES GARDES DE DÉGÉNÉRESCENCE `liste vide`.** Trois sites, tous franchis
// sur le chemin DOMINANT, jamais touchés en 26 iters, hors dead-list :
//   · `0x1545cf` `cmp %rsi,%r9 ; jne 1545f0` puis `mov ; jmp 1548d0` = `if len != 0`
//     de `rmw_inc` — TERMINE le bloc instrumenté `0x154550` et crée le stub de saut ;
//   · `0x154909` `cmp %r14d,0x60(%rsp) ; je 154bc9` = `if len != 0` de `rmw_dec` — le
//     bloc instrumenté `0x1548d3` (fuel **2**) n'existe QUE pour héberger cette garde ;
//   · `0x1531e6` `test %rax,%rax ; je 1535a1` = `if lmin > 0` de `sad3` (précédé de
//     DEUX `cmovb` qui calculent `lmin` et ne servent QU'À ça) — il force le calcul du
//     max `n` à vivre dans un bloc instrumenté SÉPARÉ, `0x153207` (fuel 5, deux `cmova`).
//
// 🔑 FRÉQUENCE (le prédicteur corrigé par i25 — cf `bricks.md`, « le compte de blocs
//    STATIQUE n'est PAS le prédicteur ») : les deux gardes RMW sont franchies **2×/flip**,
//    soit EXACTEMENT la classe de site qui a payé **−4,85 %** à i25 ; la garde `sad3` est
//    à **1×/flip**, la classe qui a rendu **0,00 %**. La cascade sépare donc les deux et
//    l'arm 3 mesure la composition (clause (4) : un arm nul n'est PAS neutre en
//    composition, et son signe n'est pas prédictible — i21 −→ +1,18 %, i24 −→ −1,77 %).
//
// ══════════════════════════════════════════════════════════════════════════════════════
// MÉCANISME — LA SENTINELLE (ce qui rend le retrait LICITE)
// ══════════════════════════════════════════════════════════════════════════════════════
// Les trois gardes existent pour une seule et même raison : quand une liste d'occurrences
// est VIDE, son offset `o_j` (resp. `start`) peut valoir `all_data.len()`, et les voies
// MORTES du déroulage masqué rabattent leur indice de lecture PRÉCISÉMENT sur cet offset
// (`kr = (kk & msz) | (start & !msz)` ⇒ `start` ; `o_j + (t & m_j)` ⇒ `o_j`). L'indice
// maximal atteignable est donc EXACTEMENT `all_data.len()` — jamais au-delà.
// ⇒ **UNE seule case sentinelle** appendue à `all_data` (hors boucle chaude, 1× par nonce)
//   rend ces lectures IN-BOUNDS, et les gardes deviennent RETIRABLES.
// ⭐ La VALEUR lue n'a aucune importance : elle est immédiatement ET-masquée à zéro
//   (`c = (c_raw & msz) | (ncg & !msz)` ⇒ `ncg = nc` = la poubelle `num_good[nc]` déjà
//   établie par `P_rmw_branchless_garbage_slot` ; `sad_hit(..) & m_j` ⇒ 0).
//
// ÉQUIVALENCE D'ÉTAT, cas par cas (⇒ trajectoire BYTE-IDENTIQUE) :
//   · `rmw_*`, `len == 0` : l'ancre ne fait RIEN. La variante exécute UN tour dont les 8
//     voies sont TOUTES mortes (`kk >= stop` pour tout `kk >= start = stop`) ⇒ `c = nc`
//     (poubelle, `wrapping_add`/`wrapping_sub`), `cond = 0` ⇒ `*ulen` INCHANGÉ,
//     `pos = wpos = 0` ⇒ `ubuf[0]` et `cpos[nc32]` (poubelles existantes). `num_good[0..nc]`,
//     `ubuf[1..=ulen]`, `cpos[0..nc]` et `ulen` sont donc byte-identiques. Le latch
//     `base += RU ; base >= stop` sort après CE tour (`start + 8 >= start`).
//   · `sad3`, `lmin == 0` : l'ancre retombe sur la queue « arm 1 » (préfixe entrelacé +
//     3 boucles scalaires) qui calcule les sad EXACTS. La variante reste dans la boucle
//     fusionnée : pour la voie vide `m_j = 0` à tout `t` ⇒ lecture en `o_j` (sentinelle)
//     et contribution 0 ⇒ `s_j = 0` = le sad exact d'une liste vide. Pour les voies non
//     vides le masque est celui déjà prouvé par i9/i10. Si `n == 0` (les TROIS vides) le
//     do-while fait un tour entièrement masqué et rend `(0,0,0)`, comme l'ancre.
//   · Aucun `rng.gen` n'est ajouté, retiré ni déplacé, dans AUCUN des trois cas.
//
// FRÉQUENCE du cas dégénéré (donc du coût des tours morts ajoutés) : `nc*3/nv/2 ≈ 6,4`
// occurrences par (variable, polarité) ⇒ P(liste vide) ≈ e^(−6,4) ≈ **0,17 %**, soit
// ~8 variables sur 5 000 — conforme au commentaire d'i25. Négligeable devant le retrait.
//
// ⛔ ORTHOGONALITÉ AUX 7 ROWS DE LA DEAD-LIST t4 :
//   · `compute_alu_idiv_strength_reduction` (167) : ici on ne remplace AUCUNE arithmétique,
//     on RETIRE des blocs instrumentés (famille `P_basic_block_reduction`, 6 bakes PROD).
//   · `codegen:rmw_constant_block_count` (237) : le nombre de tours reste `ceil(len/RU)`,
//     DONNÉE-DÉPENDANT — sauf dans le cas `len == 0` où il passe de 0 à 1, à 0,17 % de
//     fréquence. Ce n'est PAS un `NB` constant, et le PAS reste `RU = 8` (dead 235 :
//     la loi de l'over-read est monotone en ratio pas/`len`, ici INCHANGÉ sur 99,83 %).
//   · `codegen:branchless_selection_tree` (i19) / `zero_break_path_flatten` (i21) : autre
//     site (l'arbre de sélection de `v_idx`), non touché.
//   · `brick:kick_handler_optim` (230) : le kick est épinglé `RU = 0, TC = false, ZL = false`
//     dans TOUS les arms ⇒ corps VERBATIM, l'iter reste MONO-AXE.
//   · `hp:flip_budget_truncation` (233) : aucun HP touché.
//
// ⭐ LES 3 CLAUSES DE LA LOI DE COÛT DU TRACK :
//   (1) CATÉGORIE ENTIÈRE — l'arm 3 vide les TROIS sites, il ne reste AUCUNE garde de
//       dégénérescence de liste sur le chemin dominant.
//   (2) COMPLÉTUDE, par catégorie ET PAR SITE CHAUD — les arms 1 et 2 sont des retraits
//       PARTIELS (par site chaud : l'arm 1 vide la classe RMW EN ENTIER, l'arm 2 la classe
//       scan EN ENTIER) ⇒ chacun est complet à son échelle, l'arm 3 l'est globalement.
//   (3) GRATUIT EN INSTRUCTIONS — on n'ENLÈVE que : un bloc instrumenté COMPLET
//       (`0x1548d3`), un second (`0x153207`) par fusion dans son prédécesseur, deux
//       `cmovb` (calcul de `lmin`, qui ne sert QU'À la garde), trois `cmp`/`jcc` et un
//       stub `mov`/`jmp`. Rien n'est ajouté dans la boucle : la sentinelle est UN `push`
//       par nonce, hors boucle chaude.
//
// ══════════════════════════════════════════════════════════════════════════════════════
// CASCADE (travail fixe, `probe_max_flips`) — `sel_variant`
// ══════════════════════════════════════════════════════════════════════════════════════
//   0 `A_ctrl`   `ZL=false, LM=false` = `engine_ae` arm 3 VERBATIM = **CHAMPION PROD
//                144 020 ms** ⇒ l'ancre intra-binaire EST le champion, pas une recopie.
//   1 `V_zlfree` `ZL=true , LM=false` = gardes `len != 0` des DEUX RMW retirées (2×/flip).
//   2 `V_lmfree` `ZL=false, LM=true ` = garde `lmin > 0` de `sad3` retirée (1×/flip).
//   3 `V_both`   `ZL=true , LM=true ` = composition — SEUL arm qui vide la catégorie.
//
// PRONOSTIC PRÉ-ENREGISTRÉ (à confronter à la mesure, jamais à lui substituer) : arm 1
// PORTEUR (classe RMW, 2×/flip, 1 bloc entier + 1 stub), arm 2 FAIBLE-à-NUL (classe scan,
// 1×/flip — la classe qui a rendu 0,00 % à i25), arm 3 ≥ arm 1. Règle de bake INCHANGÉE :
// baker le bundle SSI il est MESURÉ strictement meilleur que le meilleur arm simple.
//
// ══════════════════════════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════════════════════════

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use super::engine_e::{preprocess, Prepared};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_reinits: Option<usize>,
    /// 0 = A_ctrl (`engine_r` V_notie VERBATIM) · 1 = V_c3 (chemin `clen == 3` déroulé).
    pub sel_variant: Option<u64>,
    /// Renseigné ⇒ MODE SONDE (remplace `max_flips`). Absent ⇒ PRODUCTION.
    pub probe_max_flips: Option<u64>,
}

/// Arm retenu en PRODUCTION. **BAKÉ À 2 = V_fuse_max** (iter t4/i10).
///
/// Lignage des bakes : i6 `engine_q` `DEFAULT_VARIANT=4` (V_both) → i7 `engine_r`
/// `DEFAULT_VARIANT=1` (V_notie, 290 030 ms) → i8 `engine_t` `DEFAULT_VARIANT=1` (V_c3,
/// 259 530 ms) → **i10 `engine_u` `DEFAULT_VARIANT=2` (V_fuse_max)**.
///
/// VERDICT DU SCREENING i9 (ver 4412, travail fixe 400 M flips, M13, sérialisé sur la même
/// machine, résolution ±0,31 %) — `md5(PER_NONCE_QUALITIES)` =
/// `90fd5db67420a1368380506875d8990d` IDENTIQUE sur les 3 arms, 32/32, 0 invalide :
///   arm 0 `A_ctrl`      140,02 s              (ancre intra-binaire = `engine_t` V_c3 verbatim)
///   arm 1 `V_fuse_min`  131,52 s  (−6,07 %)   fusion du préfixe commun, 0 accès ajouté
///   arm 2 `V_fuse_max`  123,02 s  (−12,14 %)  boucle unique jusqu'au max, over-read masqué
/// ⭐ **arm 2 gagne MALGRÉ ~+35 % de gathers redondants** ⇒ le coût est dominé par le NOMBRE
/// DE BLOCS EXÉCUTÉS, strictement au-dessus du nombre d'accès mémoire. 2ᵉ expérience
/// discriminante du track après i8 ; elle tranche dans le sens de `P_basic_block_reduction`.
///
/// Conséquence : un bench `hp={}` mesure DIRECTEMENT le candidat et valide le bake en
/// même temps ; l'ancre A_ctrl reste atteignable par `sel_variant=0`.
/// **BAKÉ À 2 = `V_rmw8`** (iter t4/i12), survivant de la cascade de screening i11.
///
/// VERDICT DU SCREENING i11 (ver 4415, travail fixe 400 M flips, M13, sérialisé,
/// résolution ±0,31 %) — `md5(PER_NONCE_QUALITIES)` = `90fd5db67420a1368380506875d8990d`
/// IDENTIQUE sur les 3 arms (= la signature de sonde PRÉ-ENREGISTRÉE), 32/32, 0 invalide :
///   arm 0 `A_ctrl`  121,52 s              (ancre intra-binaire = `engine_u` FUSE=2 verbatim)
///   arm 1 `V_rmw4`  109,52 s  (−9,87 %)   pas de 4 ⇒ 2 blocs / RMW, 8 voies
///   arm 2 `V_rmw8`  109,02 s  (−10,29 %)  pas de 8 ⇒ 1 bloc  / RMW, 8 voies
///
/// ⭐ **BORNE DU MODÈLE DE COÛT** — arm1 et arm2 exécutent le MÊME nombre d'accès (8 voies)
/// et ne diffèrent QUE par le compte de blocs (2 vs 1) : l'écart n'est que de **−0,46 %**,
/// alors que la suppression du remainder (arm0 → arm1) vaut **−9,87 %**. ⇒ le gain vient de
/// la **SUPPRESSION DU REMAINDER À COMPTE VARIABLE**, PAS d'une minimisation monotone du
/// nombre de blocs. La chaîne de dépendance SÉRIELLE sur `*ulen` (SACRÉE) borne le
/// déroulage : au-delà de la disparition du remainder, les blocs cessent d'être le mur.
/// `P_basic_block_reduction` est donc **VRAI mais SATURANT**.
///
/// ⭐ VERDICT DU SCREENING i13 (id 4136, ver 4422, travail fixe 400 M, M13, sérialisé,
/// jobs 25559-25562, 32/32 0-invalide, `md5(PER_NONCE)=90fd5db6…` IDENTIQUE 4/4) — il fonde
/// l'acquis `SU=4` baké par i16 et épinglé ici sur les trois arms :
///   arm 0 `A_ctrl` 110,52 s             `engine_v` RU=8 verbatim, remainder LLVM intact
///   arm 1 `V_su4`   98,51 s  (−10,86 %) ⭐ SURVIVANT — ≈ 3 blocs, over-read ≈ +14 %
///   arm 2 `V_su8`   99,01 s  (−10,41 %)  ≈ 2 blocs, over-read ≈ +52 %
///   arm 3 `V_su16` 130,01 s  (+17,63 %)  1 bloc — RÉGRESSION ⇒ dead-list 235
/// ⭐⭐ BORNE : le corollaire i9 « minimiser les accès est SUBORDONNÉ à minimiser les blocs »
/// n'est VRAI QUE TANT QUE L'OVER-READ RESTE PETIT. Le gain en blocs SATURE (i11 : 2→1 bloc
/// à accès égaux = −0,46 %) tandis que le coût en ACCÈS croît LINÉAIREMENT. RÈGLE : prendre
/// le PLUS PETIT pas qui supprime le remainder, JAMAIS surenchérir ; ne pas proposer un pas
/// ≥ 2× le compte de tours moyen du site.
///
/// ⭐ BAKE i16 (id 4142) — le screening i13 s'est TRANSMIS À LA PROD : `SU=4` a donné
/// **180 020 ms** (n=2 : 180 020 / 180 520) contre 199 530 ms pour i12, soit **−9,78 %**
/// pour −10,86 % à travail fixe ⇒ **taux de transmission ≈ 90 %**, 6ᵉ bake consécutif
/// réussi de ce lignage. Q=218 750 EXACT, 32/32, `md5(PER_NONCE)=6817a1ae…` = **7ᵉ moteur
/// consécutif à trajectoire byte-identique** ⇒ gain 100 % TEMPS PUR.
///
/// ⭐ BAKE i20 — le screening i19 a désigné **l'arm 1 `V_nobc`** : cascade 4 arms à travail
/// fixe 400 M (jobs 25607-25610, M13 sérialisés, ver 4433), `md5(PER_NONCE)=90fd5db6…`
/// **IDENTIQUE 4/4**, 32/32, 0 invalide ⇒ gate de byte-identité PASSÉ sur les deux
/// transformations. Mesures : arm 0 `A_ctrl` 91,52 s · **arm 1 `V_nobc` 89,51 s = −2,20 %**
/// · arm 2 `V_selfree` 95,01 s = **+3,81 % REGRESSION** · arm 3 `V_both` 93,01 s = +1,63 %.
/// ⇒ `DEFAULT_VARIANT = 1` : `hp={}` route sur `V_nobc`. **UNE SEULE CONSTANTE CHANGE**
/// entre i19 et i20 (`diff` = 1 ligne de code + commentaires).
// BAKE i22 : routage de `hp={}` sur l'arm 1 (`V_wtfree`), SURVIVANT de la cascade i21
// (85,01 s vs ancre 89,52 s = **−5,04 %** à travail fixe 400 M, `md5(sonde)` identique 4/4).
// ⚠️ On bake l'arm 1 SEUL, PAS l'arm 3 (`V_both` = 86,01 s, soit **+1,18 % PIRE** que l'arm 1 :
// composer le `ZF` null par-dessus le `WT` gagnant COÛTE, cf i21 §9.b).
//
// ⭐ BAKE i24 (id 4154) — DÉPOUILLEMENT DE LA CASCADE i23 (ver 4439, jobs 25630-25633, M13,
// travail fixe 400 M, `md5(PER_NONCE)=90fd5db6…` **IDENTIQUE 4/4**, 32/32, 0 invalide) :
//   arm 0 `A_ctrl`   85,51 s            (= champion i22 `engine_ab` V_wtfree VERBATIM)
//   arm 1 `V_c3div`  84,51 s  (−1,17 %)  diamant de bypass `div r64` + division supprimés
//   arm 2 `V_sgfree` 85,51 s  ( 0,00 %)  **NULL À L'UNITÉ D'HORLOGE** — pronostic pré-enregistré CONFIRMÉ
//   arm 3 `V_both`   83,01 s  (−2,92 %) ⭐ **SURVIVANT** — baké ici
// ⭐⭐ FAIT NEUF : la composition est **SUPER-ADDITIVE**. arm3 (−2,92 %) est **−1,77 % MEILLEUR**
// que arm1 seul (−1,17 %) alors que arm2 seul est NUL à l'unité d'horloge. La clause (4) d'i21
// (« un arm nul n'est pas neutre en composition ») est donc CONFIRMÉE DANS SON ÉNONCÉ mais
// **INVERSÉE DANS SON SIGNE** : en i21 le nul COÛTAIT (+1,18 %), ici il RAPPORTE (−1,77 %).
// ⇒ un arm nul n'est ni neutre ni systématiquement nuisible : **il faut le MESURER en
// composition, jamais extrapoler le signe d'un précédent.** Le bake du BUNDLE est licite ici
// précisément parce que la règle d'i21 est SATISFAITE : le bundle est mesuré STRICTEMENT
// meilleur (1,5 s = 3 pas de grille d'horloge) que le meilleur arm simple.
// ⇒ `DEFAULT_VARIANT = 3` : `hp={}` route sur `V_both`. **UNE SEULE CONSTANTE CHANGE**
// entre i23 et i24 (`diff` = 1 ligne de code + commentaires).
// ═══ i25 (`engine_ad`) — CASCADE `TC`, ancre = CHAMPION i24 ═══════════════════════════
// `DEFAULT_VARIANT = 0` ⇒ `hp={}` route sur `A_ctrl` = `engine_ac` `V_both` VERBATIM
// (`CD=true, SG=true, TCS=false, TCR=false`) = le CHAMPION PROD 152 020 ms (i24, id 4154,
// ver 4440). L'ancre intra-binaire EST le champion, pas une recopie.
// ═══ i26 (`engine_ae`) — BAKE DU SURVIVANT ════════════════════════════════════════════
// `DEFAULT_VARIANT = 3` ⇒ `hp={}` route sur `V_tc_both` (`TCS=true, TCR=true`), mesuré
// **78,01 s vs 82,51 s** pour l'ancre au screening travail-fixe 400 M (−5,45 %, md5 sonde
// identique). L'ancre i24 reste atteignable par `sel_variant=0`.
// ⛔ C'EST LA SEULE LIGNE QUI DIFFÈRE D'`engine_ad` (hors commentaires).
// ═══ i27 (`engine_af`) — CASCADE `ZL`/`LM`, ancre = CHAMPION i26 ══════════════════════
// `DEFAULT_VARIANT = 0` ⇒ `hp={}` routait sur `A_ctrl` = `engine_ae` arm 3 VERBATIM = le
// CHAMPION PROD 144 020 ms (i26, id 4158, ver 4443). Screening 25656-25659 : cf en-tête.
// ═══ i28 (`engine_ag`) — BAKE DU SURVIVANT ════════════════════════════════════════════
// `DEFAULT_VARIANT = 2` ⇒ `hp={}` route sur `V_lmfree` (`ZL=false, LM=true`), mesuré
// **75,51 s vs 78,51 s** pour l'ancre au screening travail-fixe 400 M (**−3,82 %**, md5
// sonde identique 4/4). Arm SIMPLE et non bundle : `V_both` a mesuré 75,51 s = Δ 0,00 s
// vs `V_lmfree` ⇒ la règle « baker le bundle SSI MESURÉ strictement meilleur » l'exclut.
// L'ancre i26 reste atteignable par `sel_variant=0`.
// ⛔ C'EST LA SEULE LIGNE QUI DIFFÈRE D'`engine_af` (hors commentaires).
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

    match hp.sel_variant.unwrap_or(DEFAULT_VARIANT) {
        // 0 = A_ctrl : `engine_ae` arm 3 (`TCS=true, TCR=true`) VERBATIM = **CHAMPION
        //     PROD 144 020 ms** (i26, id 4158, ver 4443). `ZL=false, LM=false` ⇒ tokens
        //     IDENTIQUES à l'original : l'ancre intra-binaire EST le champion.
        0 => run_buf::<true, true, true, true, 2, 8, 1, 4, true, false, true, false, true, true, true, true, false, false>(challenge, save_solution, &hp),
        // 1 = V_zlfree : les DEUX gardes `len != 0` des boucles déroulées masquées de
        //     `rmw_inc` (`0x1545cf`) et `rmw_dec` (`0x154909`) disparaissent. Le bloc
        //     instrumenté `0x1548d3` — qui n'existe QUE pour héberger la seconde —
        //     s'évapore, et le stub `mov`/`jmp 1548d0` de la première aussi.
        //     ⭐ CLASSE À 2×/FLIP = celle qui a payé −4,85 % à i25.
        1 => run_buf::<true, true, true, true, 2, 8, 1, 4, true, false, true, false, true, true, true, true, true, false>(challenge, save_solution, &hp),
        // 2 = V_lmfree : la garde `lmin > 0` de `sad3` (`0x1531e6`) disparaît AVEC les
        //     deux `cmovb` qui calculaient `lmin` (ils ne servaient QU'À elle), et le
        //     calcul du max `n` fusionne dans son prédécesseur ⇒ le bloc instrumenté
        //     `0x153207` (fuel 5) s'évapore. La queue « arm 1 » de `sad3` devient du
        //     code mort pour cette monomorphisation.
        //     ⚠️ CLASSE À 1×/FLIP = celle qui a rendu 0,00 % à i25 ⇒ NULL plausible.
        2 => run_buf::<true, true, true, true, 2, 8, 1, 4, true, false, true, false, true, true, true, true, false, true>(challenge, save_solution, &hp),
        // 3 = V_both : composition — SEUL arm qui VIDE LA CATÉGORIE EN ENTIER (les trois
        //     gardes de dégénérescence du chemin dominant).
        //     ⛔ L'additivité n'est PAS présumée, elle est MESURÉE : clause (4) de la loi
        //     de coût — un arm nul n'est ni neutre ni de signe prédictible (i21 : le nul
        //     COÛTAIT +1,18 % ; i24 : le nul RAPPORTAIT −1,77 %). Sans cet arm, i24
        //     aurait baké la moitié du gain.
        _ => run_buf::<true, true, true, true, 2, 8, 1, 4, true, false, true, false, true, true, true, true, true, true>(challenge, save_solution, &hp),
    }
}

/// Décodage d'un littéral, const-générique sur l'arm.
///
/// ÉQUIVALENCE : `lp[i] = ((cl[i].abs()-1) as u32) << 1 | ((cl[i] > 0) as u32)` est une
/// BIJECTION sur `(v, pol)` (v < nv = 5000 < 2^31 ⇒ aucun débordement). `lp` est construit
/// dans le MÊME ordre que `cl` et subit EXACTEMENT les mêmes `swap` ⇒ à tout instant
/// `dv(lp,i) == cl[i].abs()-1` et `dpol(lp,i) == (cl[i] > 0)`.
#[inline(always)]
unsafe fn dv<const LP: bool>(cl: &[i32], lp: &[u32], i: usize) -> usize {
    if LP {
        (*lp.get_unchecked(i) >> 1) as usize
    } else {
        ((*cl.get_unchecked(i)).abs() - 1) as usize
    }
}

#[inline(always)]
unsafe fn dpol<const LP: bool>(cl: &[i32], lp: &[u32], i: usize) -> bool {
    if LP {
        (*lp.get_unchecked(i) & 1) == 1
    } else {
        *cl.get_unchecked(i) > 0
    }
}

/// Budget de flips — VERBATIM `engine_e`, sauf `probe_max_flips` qui le remplace.
#[inline(always)]
fn flips_budget(hp: &Hparams, nv: usize, density: f64, cl_len: usize, nc: usize) -> usize {
    if let Some(n) = hp.probe_max_flips {
        return n as usize;
    }
    let max_fuel = hp.max_fuel_high.unwrap_or(160_000_000_000.0);
    let avg_clause_size = cl_len as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 }
}

/// `check_interval` — VERBATIM `engine_e`.
#[inline(always)]
fn check_interval_of(hp: &Hparams, nv: usize, density: f64) -> usize {
    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    hp.check_interval.unwrap_or(
        (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0)))
            .max(min_interval) as usize,
    )
}

const PROBS_BREAK: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];
const REINIT_STAGNATION: usize = 2_000_000;
const REINIT_MIN_UNSAT: usize = 10;
const N_BON_RESTARTS: usize = 5;
const NAD: f64 = 1.0;
/// Majorant de `clen` (3-SAT ⇒ clen ≤ 3) — borne les 3 tampons du scan fusionné.
const BUF: usize = 8;

// ══════════════════════════════════════════════════════════════════════════════════════
// ARMS 0 et 3 — CONTENEUR `Vec` D'ORIGINE (`engine_e` verbatim ; LP = littéraux prédécodés)
// ══════════════════════════════════════════════════════════════════════════════════════
#[allow(dead_code)]
fn run_vec<const LP: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    // Construit une fois par nonce, HORS boucle chaude. Même longueur, même largeur que
    // `cl` ⇒ empreinte inchangée (la famille layout n'est pas convoquée).
    let mut lp: Vec<u32> = if LP {
        cl.iter().map(|&l| (((l.abs() - 1) as u32) << 1) | ((l > 0) as u32)).collect()
    } else {
        Vec::new()
    };

    let max_flips = flips_budget(hp, nv, density, cl.len(), nc);
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 { vars[v] = true; continue; }
        if np == 0 && nn > 0 { continue; }
        let vad = if nn > 0 { np as f64 / nn as f64 } else { NAD + 1.0 };
        if vad <= NAD {
            vars[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            vars[v] = rng.gen_bool(prob);
        }
    }

    let mut num_good = vec![0u8; nc];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];

    unsafe {
        for c in 0..nc {
            let s = co[c] as usize;
            let e = co[c + 1] as usize;
            let mut g = 0u8;
            for j in s..e {
                let v = dv::<LP>(&cl, &lp, j);
                if dpol::<LP>(&cl, &lp, j) == *vars.get_unchecked(v) { g += 1; }
            }
            num_good[c] = g;
            if g == 0 {
                unsat_pos[c] = unsat_list.len() as u32;
                unsat_list.push(c as u32);
            }
        }
    }

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let w = vec![1u8; nc];
    let check_interval = check_interval_of(hp, nv, density);
    let mut last_check_residual = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.stagnation_limit.unwrap_or(3);
    let max_reinits = hp.max_reinits.unwrap_or(15);

    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;
    let mut bon_candidate = vec![false; nv];
    let mut bon_num_good = vec![0u8; nc];

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if unsat_list.is_empty() { break; }

            if stagnation_count >= REINIT_STAGNATION && best_unsat >= REINIT_MIN_UNSAT && reinit_count < max_reinits {
                reinit_count += 1;
                let mut best_cand_unsat = usize::MAX;
                for _ in 0..N_BON_RESTARTS {
                    for v in 0..nv { bon_candidate[v] = false; }
                    for v in 0..nv {
                        let np = p_cnt[v] as usize;
                        let nn = n_cnt[v] as usize;
                        if nn == 0 && np > 0 { bon_candidate[v] = true; continue; }
                        if np == 0 && nn > 0 { continue; }
                        let vad = if nn > 0 { np as f64 / nn as f64 } else { NAD + 1.0 };
                        if vad <= NAD {
                            bon_candidate[v] = rng.gen_bool(random_threshold);
                        } else {
                            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                            bon_candidate[v] = rng.gen_bool(prob);
                        }
                    }
                    bon_num_good.fill(0);
                    for c in 0..nc {
                        let s = *co.get_unchecked(c) as usize;
                        let e = *co.get_unchecked(c + 1) as usize;
                        let mut g = 0u8;
                        for j in s..e {
                            let v = dv::<LP>(&cl, &lp, j);
                            if dpol::<LP>(&cl, &lp, j) == *bon_candidate.get_unchecked(v) { g += 1; }
                        }
                        bon_num_good[c] = g;
                    }
                    let cand_unsat = bon_num_good.iter().filter(|&&x| x == 0).count();
                    if cand_unsat < best_cand_unsat {
                        best_cand_unsat = cand_unsat;
                        vars.copy_from_slice(&bon_candidate);
                    }
                }

                num_good.fill(0);
                for c in 0..nc {
                    let s = *co.get_unchecked(c) as usize;
                    let e = *co.get_unchecked(c + 1) as usize;
                    let mut g = 0u8;
                    for j in s..e {
                        let v = dv::<LP>(&cl, &lp, j);
                        if dpol::<LP>(&cl, &lp, j) == *vars.get_unchecked(v) { g += 1; }
                    }
                    num_good[c] = g;
                }

                unsat_list.clear();
                unsat_pos.fill(u32::MAX);
                for c in 0..nc {
                    if num_good[c] == 0 {
                        unsat_pos[c] = unsat_list.len() as u32;
                        unsat_list.push(c as u32);
                    }
                }

                best_unsat = unsat_list.len();
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - unsat_list.len() as i64;
                if progress <= 0 {
                    stagnation += 1;
                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() { break; }
                            let rid = rng.gen::<usize>() % unsat_list.len();
                            let pcid = *unsat_list.get_unchecked(rid) as usize;
                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce { continue; }
                            let v = dv::<LP>(&cl, &lp, pcs + rng.gen::<usize>() % (pce - pcs));

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            } else {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            };
                            let (ds, de) = if was_true {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            } else {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            };

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = *num_good.get_unchecked(c);
                                if ng == 0 {
                                    let pos = *unsat_pos.get_unchecked(c) as usize;
                                    let last_idx = unsat_list.len() - 1;
                                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                    unsat_list.pop();
                                }
                                *num_good.get_unchecked_mut(c) = ng + 1;
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = *num_good.get_unchecked(c);
                                *num_good.get_unchecked_mut(c) = ng - 1;
                                if ng == 1 {
                                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                                    unsat_list.push(c as u32);
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }
                last_check_residual = unsat_list.len();
            }

            if unsat_list.is_empty() { break; }

            let rand_val = rng.gen::<usize>();
            let cid = {
                let uc = unsat_list.len();
                let i1 = (rand_val as u32 as usize) % uc;
                let i2 = (rand_val >> 32) % uc;
                let c1 = *unsat_list.get_unchecked(i1) as usize;
                let c2 = *unsat_list.get_unchecked(i2) as usize;
                if *w.get_unchecked(c1) >= *w.get_unchecked(c2) { c1 } else { c2 }
            };

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                // La permutation est appliquée au tableau RÉELLEMENT LU par l'arm ; `lp`
                // étant une bijection de `cl` construite dans le même ordre, l'ordre
                // résultant est identique ⇒ trajectoire inchangée.
                if LP { lp.swap(cs, cs + ri); } else { cl.swap(cs, cs + ri); }
            }

            // ── SCAN-FUSION (P_fusion_single_pass, i2 KEPT) — INCHANGÉ ────────────────
            let mut zero_buf: [usize; BUF] = [0; BUF];
            let mut zero_cnt: usize = 0;
            let mut pw_weights: [u32; BUF] = [0; BUF];
            let mut pw_vars: [usize; BUF] = [0; BUF];
            let mut pw_cnt: usize = 0;
            let mut total_pw: u32 = 0;

            for j in cs..ce.min(cs + BUF) {
                let abs_l = dv::<LP>(&cl, &lp, j);
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                } else {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                };

                let mut sad = 0usize;
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    sad += (*num_good.get_unchecked(c) == 1) as usize;
                }

                if sad == 0 {
                    *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                    zero_cnt += 1;
                }

                let pw = *PROBS_BREAK.get_unchecked(sad.min(15));
                *pw_weights.get_unchecked_mut(pw_cnt) = pw;
                *pw_vars.get_unchecked_mut(pw_cnt) = abs_l;
                total_pw += pw;
                pw_cnt += 1;
            }

            let v_idx = if zero_cnt > 0 {
                if zero_cnt == 1 {
                    *zero_buf.get_unchecked(0)
                } else {
                    *zero_buf.get_unchecked(rand_val % zero_cnt)
                }
            } else {
                let mut r = (rand_val as u32) % total_pw.max(1);
                let mut chosen = *pw_vars.get_unchecked(0);
                for i in 0..pw_cnt {
                    let pw = *pw_weights.get_unchecked(i);
                    if r < pw { chosen = *pw_vars.get_unchecked(i); break; }
                    r -= pw;
                }
                chosen
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            } else {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            };
            let (ds, de) = if was_true {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            } else {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = *num_good.get_unchecked(c);
                if ng == 0 {
                    let pos = *unsat_pos.get_unchecked(c) as usize;
                    let last_idx = unsat_list.len() - 1;
                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                    unsat_list.pop();
                }
                *num_good.get_unchecked_mut(c) = ng + 1;
            }

            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = *num_good.get_unchecked(c);
                *num_good.get_unchecked_mut(c) = ng - 1;
                if ng == 1 {
                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                    unsat_list.push(c as u32);
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            rounds += 1;

            let cur = unsat_list.len();
            if cur < best_unsat {
                best_unsat = cur;
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
        }
    }

    // RÈGLE DURE : on retourne le MEILLEUR état suivi, jamais le courant.
    let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
    let _ = save_solution(&Solution { variables: final_vars });
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════════════
// ARMS 1, 2 et 4 — TAMPON PLAT 1-BASED (`ubuf[0]` = POUBELLE, `cpos[nc]` = POUBELLE)
//   BL=false ⇒ branches conservées (famille CONTENEUR seule)
//   BL=true  ⇒ masque plein + écritures inconditionnelles (famille BRANCHLESS)
// ══════════════════════════════════════════════════════════════════════════════════════
/// `sad(v)` = nombre de clauses CRITIQUES (`num_good == 1`) parmi les occurrences de `v`
/// dans sa polarité SATISFAISANTE. Extrait **VERBATIM** du chemin générique de `run_buf`
/// (mêmes bornes, même ordre de parcours, mêmes accès) et `#[inline(always)]` pour que la
/// spécialisation `clen == 3` reste straight-line.
#[inline(always)]
unsafe fn sad_of(
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    num_good: &[u8],
    vars: &[bool],
    abs_l: usize,
) -> usize {
    let (os, oe) = if *vars.get_unchecked(abs_l) {
        (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
    } else {
        (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
    };
    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        sad += (*num_good.get_unchecked(c) == 1) as usize;
    }
    sad
}

/// Bornes de la liste d'occurrences SATISFAISANTES de `abs_l` — extrait VERBATIM de la
/// tête de `sad_of` (même test, mêmes tableaux, même ordre) ⇒ aucune sémantique nouvelle.
#[inline(always)]
unsafe fn occ(all_off: &[u32], p_bound: &[u32], vars: &[bool], abs_l: usize) -> (usize, usize) {
    if *vars.get_unchecked(abs_l) {
        (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
    } else {
        (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
    }
}

/// Contribution d'une occurrence au compte `sad` — le corps EXACT de `sad_of`.
#[inline(always)]
unsafe fn sad_hit(all_data: &[u32], num_good: &[u8], k: usize) -> usize {
    let c = *all_data.get_unchecked(k) as usize;
    (*num_good.get_unchecked(c) == 1) as usize
}

/// Les 3 scans `sad` du chemin `clen == 3`, const-génériques sur l'arm de fusion.
///
/// ÉQUIVALENCE (les 3 arms) : chaque somme parcourt EXACTEMENT le même multi-ensemble
/// d'occurrences `os..oe` que `sad_of`, et l'addition sur `usize` est
/// associative-commutative EXACTE ⇒ `s0`/`s1`/`s2` sont identiques À L'UNITÉ quel que
/// soit l'ordre. Tout est en LECTURE SEULE : aucune écriture d'état, **aucun `rng.gen`**
/// ⇒ trajectoire byte-identique par construction.
#[inline(always)]
unsafe fn sad3<const FUSE: u8, const SU: usize, const TC: bool, const LM: bool>(
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    num_good: &[u8],
    vars: &[bool],
    a0: usize,
    a1: usize,
    a2: usize,
) -> (usize, usize, usize) {
    // ── arm 0 : A_ctrl — `engine_t` VERBATIM, 3 appels sérialisés ────────────────────
    if FUSE == 0 {
        return (
            sad_of(all_off, p_bound, all_data, num_good, vars, a0),
            sad_of(all_off, p_bound, all_data, num_good, vars, a1),
            sad_of(all_off, p_bound, all_data, num_good, vars, a2),
        );
    }

    let (o0, e0) = occ(all_off, p_bound, vars, a0);
    let (o1, e1) = occ(all_off, p_bound, vars, a1);
    let (o2, e2) = occ(all_off, p_bound, vars, a2);
    let (l0, l1, l2) = (e0 - o0, e1 - o1, e2 - o2);

    let (mut s0, mut s1, mut s2) = (0usize, 0usize, 0usize);

    // ── arm 2 : V_fuse_max — boucle UNIQUE, voies épuisées masquées ──────────────────
    // Une voie `j` épuisée (`t >= lj`) rabat son index sur `oj + 0` — donc TOUJOURS
    // à l'intérieur de sa propre liste d'occurrences, jamais hors de `all_data` — et
    // ET-masque sa contribution à 0. La garde `lmin > 0` exclut le cas dégénéré d'une
    // liste vide (où `oj` pourrait valoir `all_data.len()`), qui retombe sur l'arm 1.
    if FUSE == 2 {
        // ── CORPS FUSIONNÉ, DÉFINI UNE SEULE FOIS ────────────────────────────────────
        // `su_body!` puis `fused_body!` sont des macros : la voie GARDÉE (`LM = false`)
        // et la voie NON GARDÉE (`LM = true`) expansent LITTÉRALEMENT les mêmes tokens.
        // Le SEUL delta entre les deux arms est donc la PRÉSENCE de la garde `lmin > 0`
        // — aucune dérive de corps n'est possible, la byte-identité est structurelle.
        macro_rules! su_body { ($base:expr) => {{
            // `SU` est une constante ⇒ boucle intégralement déroulée, zéro latch.
            for u in 0..SU {
                let t = $base + u;
                let m0 = ((t < l0) as usize).wrapping_neg();
                let m1 = ((t < l1) as usize).wrapping_neg();
                let m2 = ((t < l2) as usize).wrapping_neg();
                s0 += sad_hit(all_data, num_good, o0 + (t & m0)) & m0;
                s1 += sad_hit(all_data, num_good, o1 + (t & m1)) & m1;
                s2 += sad_hit(all_data, num_good, o2 + (t & m2)) & m2;
            }
        }}; }

        macro_rules! fused_body { () => {{
            let n = l0.max(l1).max(l2);
            if SU > 0 {
                // ═══ ACQUIS i25-a (`TC`) — do-while, préambule de compte de tours ABSENT ═══
                if TC {
                    let mut base = 0usize;
                    loop {
                        su_body!(base);
                        base += SU;
                        if base >= n { break; }
                    }
                    return (s0, s1, s2);
                }
                let nb = (n + SU - 1) / SU;
                let mut base = 0usize;
                for _b in 0..nb {
                    su_body!(base);
                    base += SU;
                }
                return (s0, s1, s2);
            }
            for t in 0..n {
                let m0 = ((t < l0) as usize).wrapping_neg();
                let m1 = ((t < l1) as usize).wrapping_neg();
                let m2 = ((t < l2) as usize).wrapping_neg();
                s0 += sad_hit(all_data, num_good, o0 + (t & m0)) & m0;
                s1 += sad_hit(all_data, num_good, o1 + (t & m1)) & m1;
                s2 += sad_hit(all_data, num_good, o2 + (t & m2)) & m2;
            }
            return (s0, s1, s2);
        }}; }

        // ═══ AXE i27-b — `LM` : SUPPRESSION DE LA GARDE DE DÉGÉNÉRESCENCE `lmin > 0` ═══
        // Census ver 4443 (monomorphisation championne) : la garde vit en `0x1531e6`
        // (`test %rax,%rax ; je 1535a1`) et elle est PRÉCÉDÉE de deux `cmovb` (`0x1531d3`,
        // `0x1531e2`) qui calculent `lmin` et NE SERVENT QU'À ELLE. En scindant le CFG
        // elle force en outre le calcul du max `n` à vivre dans un bloc instrumenté
        // SÉPARÉ — `0x153207`, fuel 5, deux `cmova` — qui fusionne dans son prédécesseur
        // dès que la garde tombe.
        // ⭐ CE QUI REND LE RETRAIT LICITE : la sentinelle appendue à `all_data` (cf
        //    en-tête). L'indice maximal atteignable par une voie MORTE est EXACTEMENT
        //    `o_j` (car `m_j = 0` ⇒ `o_j + (t & m_j) = o_j`), donc au pire
        //    `all_data.len()` — une seule case suffit.
        // ⭐ ÉQUIVALENCE : cf en-tête, cas `sad3, lmin == 0`. Les `s_j` produits sont les
        //    sad EXACTS dans les deux formes ; la queue « arm 1 » ci-dessous les calculait
        //    par un autre chemin, avec les MÊMES valeurs en sortie. Purement LECTEUR,
        //    aucun `rng.gen` ⇒ trajectoire BYTE-IDENTIQUE.
        // ⛔ ORTHOGONAL à dead 235 (`masked_overread_large_stride`) : le PAS reste `SU = 4`
        //    et le nombre de tours reste `ceil(n / SU)` — seul le cas `n == 0` passe de
        //    0 à 1 tour, entièrement masqué, à ~0,17 % de fréquence.
        if LM {
            fused_body!();
        }

        let lmin = l0.min(l1).min(l2);
        if lmin > 0 {
            fused_body!();
        }
    }

    // ── arm 1 : V_fuse_min — préfixe commun entrelacé + 3 queues scalaires ───────────
    // ZÉRO accès mémoire ajouté ni retiré par rapport à l'arm 0 : seul leur
    // ORDONNANCEMENT change (3 chaînes de dépendance en vol au lieu d'une).
    let n = l0.min(l1).min(l2);
    for t in 0..n {
        s0 += sad_hit(all_data, num_good, o0 + t);
        s1 += sad_hit(all_data, num_good, o1 + t);
        s2 += sad_hit(all_data, num_good, o2 + t);
    }
    for k in o0 + n..e0 {
        s0 += sad_hit(all_data, num_good, k);
    }
    for k in o1 + n..e1 {
        s1 += sad_hit(all_data, num_good, k);
    }
    for k in o2 + n..e2 {
        s2 += sad_hit(all_data, num_good, k);
    }
    (s0, s1, s2)
}

fn run_buf<
    const BL: bool,
    const LP: bool,
    const NOTIE: bool,
    const C3: bool,
    const FUSE: u8,
    const RU: usize,
    // AXE DE L'ITER i17 — hoisting des gardes superviseures hors du hot flip.
    // `0` = A_ctrl (boucle plate VERBATIM) · `1` = V_hoist_all (A+C+D hoistées
    // et `ulen == 0` dé-dupliquée) · `2` = V_hoist_ci (D seule hoistée, −1 bloc EXACT).
    const HOIST: u8,
    // ACQUIS i13/i16, ÉPINGLÉ À 4 SUR TOUS LES ARMS (hors périmètre, comme `RU = 8`) :
    // pas de déroulage masqué de la boucle FUSIONNÉE de `sad3` (0 = verbatim `engine_v`).
    const SU: usize,
    // AXE i19-a — `true` = les deux `panic_bounds_check` du `cl.swap`/`lp.swap` du hot
    // flip sont supprimés (indices prouvés dans les bornes, cf en-tête). `false` = tokens
    // VERBATIM `engine_z`.
    const NOBC: bool,
    // AXE i19-b — `true` = la sélection de `v_idx` sur le chemin `clen == 3` est calculée
    // SANS BRANCHE (mêmes valeurs, même décision, CFG plat). `false` = VERBATIM.
    // ⛔ DEAD (`codegen:branchless_selection_tree`, +3,81 %) : conservé pour garder les
    // tokens du corps IDENTIQUES à `engine_aa`, mais JAMAIS instancié à `true` par i21.
    const SEL: bool,
    // AXE i21-a — `true` = le dispatch `was_true` des bornes `inc`/`dec` est aplati par
    // HISSAGE des 3 chargements (`all_off[v]`, `p_bound[v]`, `all_off[v+1]`, que les DEUX
    // branches lisent déjà toutes les trois) + 4 `cmov`. GRATUIT en accès et en
    // instructions. `false` = tokens VERBATIM `engine_aa`.
    const WT: bool,
    // AXE i21-b — `true` = la voie zéro-break SEULE est aplatie ; la marche pondérée reste
    // dans son `else` (⛔ NE PAS confondre avec `SEL`, qui la rendait inconditionnelle).
    // ⛔ DEAD (`codegen:zero_break_path_flatten`, i21 arm 2 = NULL) : conservé pour garder
    // les tokens du corps IDENTIQUES à `engine_ab`, mais JAMAIS instancié à `true` par i23.
    const ZF: bool,
    // AXE i23-a — `true` = le test `clen == 3` est HISSÉ AU-DESSUS du swap, de sorte que
    // le chemin dominant calcule `rand_val % 3` (DIVISEUR CONSTANT ⇒ `mulhi`+`shr`, aucune
    // division, aucun diamant de bypass). `false` = tokens VERBATIM `engine_ab`.
    const CD: bool,
    // AXE i23-b — `true` = `stagnation_count` est mis à jour SANS BRANCHE (`cmov`) ⇒ le
    // bras `else` du diamant `best_unsat` disparaît. `false` = tokens VERBATIM `engine_ab`.
    const SG: bool,
    // AXE i25-a — `true` = le préambule de compte de tours de la boucle FUSIONNÉE de
    // `sad3` (blocs 0x153230 + 0x1533c5 du census ver 4440) est supprimé par conversion
    // en do-while. `false` = tokens VERBATIM `engine_ac`.
    const TCS: bool,
    // AXE i25-b — `true` = idem pour les DEUX boucles `rmw_inc`/`rmw_dec` (blocs 0x154665
    // et 0x1549de). `false` = tokens VERBATIM `engine_ac`.
    const TCR: bool,
    // AXE i27-a — `true` = les DEUX gardes de dégénérescence `len != 0` des boucles
    // déroulées masquées de `rmw_inc` (`0x1545cf`) et `rmw_dec` (`0x154909`) sont
    // retirées, la lecture des voies mortes étant rendue in-bounds par la sentinelle.
    // Le bloc instrumenté `0x1548d3` (fuel 2), qui n'existe QUE pour héberger la
    // seconde, disparaît. `false` = tokens VERBATIM `engine_ae`. **2×/flip.**
    const ZL: bool,
    // AXE i27-b — `true` = la garde de dégénérescence `lmin > 0` de `sad3` est retirée
    // (avec les deux `cmovb` qui calculaient `lmin`), et le bloc `0x153207` fusionne
    // dans son prédécesseur. `false` = tokens VERBATIM `engine_ae`. **1×/flip.**
    const LM: bool,
>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let Prepared {
        mut rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, mut all_data, mut cl, co,
    } = preprocess(challenge, save_solution);

    // ═══ AXE i27 — SENTINELLE : UNE case, HORS boucle chaude, 1× par nonce ═══════════
    // C'est CE `push` qui rend licite le retrait des trois gardes de dégénérescence.
    // Les voies MORTES du déroulage masqué rabattent leur indice de lecture sur `start`
    // (RMW) ou `o_j` (scan) — bornes qui valent AU PLUS `all_data.len()` quand la liste
    // d'occurrences correspondante est vide. Une seule case supplémentaire suffit donc à
    // rendre TOUTES ces lectures in-bounds. La VALEUR n'a aucune importance : elle est
    // immédiatement ET-masquée à zéro (`c = ncg = nc` = la poubelle `num_good[nc]`).
    // ⚠️ Gardé sous const-générique ⇒ dans l'arm 0 (`ZL=false, LM=false`) ce `push`
    //    n'existe PAS dans le code émis : l'ancre reste `engine_ae` arm 3 VERBATIM.
    if ZL || LM {
        all_data.push(0);
    }

    let mut lp: Vec<u32> = if LP {
        cl.iter().map(|&l| (((l.abs() - 1) as u32) << 1) | ((l > 0) as u32)).collect()
    } else {
        Vec::new()
    };

    let max_flips = flips_budget(hp, nv, density, cl.len(), nc);
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 { vars[v] = true; continue; }
        if np == 0 && nn > 0 { continue; }
        let vad = if nn > 0 { np as f64 / nn as f64 } else { NAD + 1.0 };
        if vad <= NAD {
            vars[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            vars[v] = rng.gen_bool(prob);
        }
    }

    let nc32 = nc as u32;
    // +1 : l'entrée `nc` est la POUBELLE des voies MORTES du déroulage masqué (`RU > 0`).
    // Elle n'est JAMAIS lue : les scans `sad` n'indexent `num_good` que par
    // `all_data[k] < nc`, et la reconstruction post-reinit n'écrit que `0..nc`.
    // Allouée pour TOUS les arms (y compris l'ancre `RU = 0`) ⇒ layout mémoire et
    // empreinte cache IDENTIQUES entre arms : la cascade ne mesure QUE les blocs.
    let mut num_good = vec![0u8; nc + 1];
    // +2 : indice 0 = POUBELLE, éléments en 1..=ulen. Tout est INITIALISÉ ⇒ la lecture
    // spéculative `ubuf[ulen]` (y compris ulen==0) est définie, jamais de l'UB.
    let mut ubuf: Vec<u32> = vec![0u32; nc + 2];
    let mut ulen: usize = 0;
    // +1 : l'entrée `nc` est le PUITS des écritures annulées par le masque.
    let mut cpos = vec![u32::MAX; nc + 1];

    unsafe {
        for c in 0..nc {
            let s = co[c] as usize;
            let e = co[c + 1] as usize;
            let mut g = 0u8;
            for j in s..e {
                let v = dv::<LP>(&cl, &lp, j);
                if dpol::<LP>(&cl, &lp, j) == *vars.get_unchecked(v) { g += 1; }
            }
            num_good[c] = g;
            if g == 0 {
                ulen += 1;
                ubuf[ulen] = c as u32;
                cpos[c] = ulen as u32;
            }
        }
    }

    if ulen == 0 {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let w = vec![1u8; nc];
    let check_interval = check_interval_of(hp, nv, density);
    let mut last_check_residual = ulen;
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.stagnation_limit.unwrap_or(3);
    let max_reinits = hp.max_reinits.unwrap_or(15);

    let mut best_unsat = ulen;
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;
    let mut bon_candidate = vec![false; nv];
    let mut bon_num_good = vec![0u8; nc];

    // ══════════════════════════════════════════════════════════════════════════════════
    // `hot_flip!` — LE CORPS DU FLIP, SOURCE UNIQUE PARTAGÉE PAR LES DEUX SQUELETTES
    // ══════════════════════════════════════════════════════════════════════════════════
    // Une macro (et non deux copies) : l'arm 0 et les arms 1/2 exécutent LITTÉRALEMENT
    // les mêmes tokens ⇒ aucune dérive possible entre l'ancre et les variantes. Le corps
    // est repris VERBATIM d'`engine_v` (champion i12) : ni `rng.gen`, ni écriture d'état,
    // ni ordre d'accès n'y sont touchés par l'iter i15 — seul son CONTEXTE DE CONTRÔLE
    // change. Il ne contient AUCUN `break`/`continue` ⇒ il est neutre vis-à-vis de la
    // boucle qui l'héberge (propriété vérifiée : le dernier `break` du corps d'origine
    // est la garde (E), qui reste HORS macro).
    macro_rules! hot_flip { () => {{
                let rand_val = rng.gen::<usize>();
                let cid = {
                    let uc = ulen;
                    let i1 = (rand_val as u32 as usize) % uc;
                    let c1 = *ubuf.get_unchecked(i1 + 1) as usize;
                    if NOTIE {
                        // `w` est constant à 1 ⇒ `w[c1] >= w[c2]` est une TAUTOLOGIE ⇒ cid ≡ c1.
                        // On supprime donc `i2` (1 idiv), `c2` (1 gather ubuf) et les 2 gathers
                        // `w[]` : aucun n'influence la valeur produite. Byte-identique.
                        c1
                    } else {
                        let i2 = (rand_val >> 32) % uc;
                        let c2 = *ubuf.get_unchecked(i2 + 1) as usize;
                        if *w.get_unchecked(c1) >= *w.get_unchecked(c2) { c1 } else { c2 }
                    }
                };

                let cs = *co.get_unchecked(cid) as usize;
                let ce = *co.get_unchecked(cid + 1) as usize;
                let clen = ce - cs;

                // ═══ AXE i23-a — `CD` : HISSAGE du test `clen == 3` AU-DESSUS du swap ═══
                // Le champion teste `clen > 1` (swap), PUIS `clen == 3` (chemin C3). Sur le
                // chemin dominant les DEUX tests s'exécutent, et `rand_val % clen` a un
                // diviseur VARIABLE ⇒ LLVM y greffe sa passe *BypassSlowDivision* : un bloc
                // instrumenté entier (0x161f03) hébergeant un test `(rand_val|clen) >> 32
                // == 0` qui est FAUX à chaque flip (`rand_val` est un hash 64 bits plein),
                // suivi d'un `div r64` et d'un `div r32` MORT à l'exécution.
                //
                // ⭐ ÉQUIVALENCE EXACTE : sur la branche `clen == 3` on a `rand_val % clen
                //    ≡ rand_val % 3` — MÊME valeur, MÊME `rand_val` consommé, MÊME swap,
                //    MÊME ordre. La branche générique conserve les tokens VERBATIM.
                //    ⇒ trajectoire BYTE-IDENTIQUE PAR CONSTRUCTION.
                // ⭐ GRATUITÉ (la clause qui a tué `V_selfree`) : le test `clen == 3` n'est
                //    pas AJOUTÉ, il est DÉPLACÉ — celui qui suivait le swap devient une
                //    tautologie sur cette branche et une contradiction sur l'autre ⇒ LLVM
                //    le replie (GVN / jump-threading). Le chemin dominant passe même de
                //    DEUX tests (`clen > 1` puis `clen == 3`) à UN SEUL. Aucune instruction
                //    ni aucun accès mémoire ajouté.
                // ⛔ ORTHOGONAL à la dead row 167 (`compute_alu_idiv_strength_reduction`) :
                //    i4 remplaçait l'ARITHMÉTIQUE de la division « à CFG constant, zéro
                //    bloc retiré » (evidence : « i4 ne touche QUE de l'ALU pur ») et rendait
                //    NOISE_BOUND sur un dispositif à ±2 %. Ici le mécanisme est le RETRAIT
                //    D'UN BLOC INSTRUMENTÉ par spécialisation straight-line — la famille
                //    `P_basic_block_reduction` (i8/i17/i19/i20/i22, 5 bakes PROD), qui
                //    n'existait pas encore quand i4 a été tirée. Cf strategy.md §3.
                if CD && clen == 3 {
                    let ri = rand_val % 3;
                    if NOBC {
                        if LP {
                            let p = lp.as_mut_ptr();
                            let (pa, pb) = (p.add(cs), p.add(cs + ri));
                            let (va, vb) = (*pa, *pb);
                            *pa = vb;
                            *pb = va;
                        } else {
                            let p = cl.as_mut_ptr();
                            let (pa, pb) = (p.add(cs), p.add(cs + ri));
                            let (va, vb) = (*pa, *pb);
                            *pa = vb;
                            *pb = va;
                        }
                    } else if LP {
                        lp.swap(cs, cs + ri);
                    } else {
                        cl.swap(cs, cs + ri);
                    }
                } else if clen > 1 {
                    let ri = rand_val % clen;
                    // ═══ AXE i19-a — `NOBC` : suppression des DEUX bounds-checks ═══════
                    // `slice::swap(a, b)` = deux `panic_bounds_check` PUIS `ptr::swap`.
                    // Le census objdump de ver 4430 montre que ces deux tests sont les
                    // SEULES arêtes de panique de tout le hot flip (0x14c670 → 0x15dfb8
                    // et 0x14c6ad → 0x15dfb3) et qu'ils scindent le CFG en 3 blocs dont
                    // un instrumenté (0x14c676).
                    // DOMAINE : `ri < clen` et `cs + clen = ce = co[cid+1] <= cl.len()`
                    // ⇒ `cs` et `cs + ri` sont TOUJOURS < len ⇒ les tests sont des
                    // TAUTOLOGIES. On lit/écrit donc directement (⛔ PAS `ptr::swap`, qui
                    // exige des régions disjointes : `ri == 0` arrive 1 fois sur 3).
                    // Pour un `Copy` de 4 octets, read×2 puis write×2 est EXACT même
                    // quand les deux indices coïncident. Valeurs INCHANGÉES ⇒ trajectoire
                    // byte-identique.
                    if NOBC {
                        if LP {
                            let p = lp.as_mut_ptr();
                            let (pa, pb) = (p.add(cs), p.add(cs + ri));
                            let (va, vb) = (*pa, *pb);
                            *pa = vb;
                            *pb = va;
                        } else {
                            let p = cl.as_mut_ptr();
                            let (pa, pb) = (p.add(cs), p.add(cs + ri));
                            let (va, vb) = (*pa, *pb);
                            *pa = vb;
                            *pb = va;
                        }
                    } else if LP {
                        lp.swap(cs, cs + ri);
                    } else {
                        cl.swap(cs, cs + ri);
                    }
                }

                // ═══ V_c3 — chemin STRAIGHT-LINE pour `clen == 3` (100 % des flips en 3-SAT) ═══
                // Équivalent EXACT du chemin générique (conservé en `else`), déroulé à la main
                // pour supprimer des BASIC BLOCKS — chacun coûte 2 appels PLT `__tls_get_addr`
                // + 1 `add` fuel + 1 `xor` signature (cf en-tête du fichier) :
                //   · en-tête + latch de la boucle externe sur les 3 littéraux → supprimés ;
                //   · `zero_buf`/`pw_weights`/`pw_vars` (3 tableaux de pile, 160 o de
                //     zéro-init PAR FLIP, indexation dynamique) → scalaires ;
                //   · `pw_vars` était redondant avec `dv(cs+i)` → n'existe plus ;
                //   · boucle de sélection pondérée → chaîne de 3 comparaisons.
                // ⚠️ `cs + 2 < cs + BUF` (BUF = 8) ⇒ `ce.min(cs + BUF) == ce` : le chemin
                // générique scanne EXACTEMENT ces 3 littéraux, dans cet ordre. Aucun `rng.gen`
                // ajouté, retiré ni déplacé ⇒ trajectoire byte-identique.
                let v_idx = if C3 && clen == 3 {
                    let a0 = dv::<LP>(&cl, &lp, cs);
                    let a1 = dv::<LP>(&cl, &lp, cs + 1);
                    let a2 = dv::<LP>(&cl, &lp, cs + 2);
                    let (s0, s1, s2) =
                        sad3::<FUSE, SU, TCS, LM>(&all_off, &p_bound, &all_data, &num_good, &vars, a0, a1, a2);

                    let z0 = s0 == 0;
                    let z1 = s1 == 0;
                    let z2 = s2 == 0;
                    let zero_cnt = z0 as usize + z1 as usize + z2 as usize;

                    // ═══ AXE i19-b — `SEL` : sélection SANS BRANCHE ════════════════════
                    // MÊME fonction, MÊME décision, MÊME `rand_val` — seule la forme du
                    // CFG change. Les 4 blocs instrumentés de l'arbre (0x14e260 /
                    // 0x14e2d2 / 0x14e497 / 0x14e4d9 dans ver 4430) fondent en un seul.
                    // Démonstration d'équivalence : cf en-tête du fichier, §ARM 2.
                    if SEL {
                        // ── (a) voie « au moins un break-0 » : le k-ième zéro DANS
                        //    L'ORDRE DU SCAN, k = rand_val % zero_cnt. Les trois valeurs
                        //    de k sont calculées à DIVISEUR CONSTANT (0 / &1 / %3) puis
                        //    sélectionnées : aucun `div r64` ni diamant de bypass n'est
                        //    introduit (la famille div, dead 167, reste fermée).
                        let k = if zero_cnt == 3 {
                            rand_val % 3
                        } else if zero_cnt == 2 {
                            rand_val % 2
                        } else {
                            0
                        };
                        // 1er zéro (ordre du scan) ; 2e zéro = `a1` ssi `z0 && z1`, sinon
                        // `a2` (si `!z0` alors `z1 && z2` ⇒ le 2e est bien `a2`) ; 3e = a2.
                        let f0 = if z0 { a0 } else if z1 { a1 } else { a2 };
                        let f1 = if z0 && z1 { a1 } else { a2 };
                        let v_zero = if k == 0 { f0 } else if k == 1 { f1 } else { a2 };

                        // ── (b) marche pondérée, évaluée INCONDITIONNELLEMENT (pure :
                        //    3 lectures d'une table constante de 64 o + un `div r32` dont
                        //    le diviseur est >= 21 ⇒ jamais de division par zéro).
                        //    Seuils CUMULÉS : `r < p0` / `r < p0+p1` / `r < p0+p1+p2` est
                        //    l'exact équivalent de `r < p0` / `r-p0 < p1` / `r-p0-p1 < p2`
                        //    (u32, somme <= 3*2535 = 7605 ⇒ aucun débordement). Le `else`
                        //    terminal `a0` (inatteignable) est CONSERVÉ à l'identique.
                        let p0 = *PROBS_BREAK.get_unchecked(s0.min(15));
                        let p1 = *PROBS_BREAK.get_unchecked(s1.min(15));
                        let p2 = *PROBS_BREAK.get_unchecked(s2.min(15));
                        let tot = (p0 + p1 + p2).max(1);
                        let r = (rand_val as u32) % tot;
                        let t1 = p0 + p1;
                        let t2 = t1 + p2;
                        let v_walk = if r < p0 {
                            a0
                        } else if r < t1 {
                            a1
                        } else if r < t2 {
                            a2
                        } else {
                            a0
                        };

                        if zero_cnt > 0 { v_zero } else { v_walk }
                    } else if zero_cnt > 0 {
                        // ═══ AXE i21-b — `ZF` : voie zéro-break SEULE aplatie ══════════
                        // ⚠️ La marche pondérée reste dans son `else` ci-dessous : on ne
                        // l'évalue PAS spéculativement (c'est ce qui a coûté +3,81 % à
                        // `V_selfree`). Équivalence : « le k-ième `sad == 0` DANS L'ORDRE
                        // DU SCAN, k = rand_val % zero_cnt » — reprise LITTÉRALE de la
                        // partie (a) d'`engine_aa` arm 2, déjà validée byte-identique
                        // 4/4 par le gate `md5` d'i19.
                        if ZF {
                            // Les trois `k` sont à diviseur CONSTANT (0 / %2 / %3) : aucun
                            // `div r64` ni diamant de bypass ⇒ DEAD 167 non rouverte.
                            let k = if zero_cnt == 3 {
                                rand_val % 3
                            } else if zero_cnt == 2 {
                                rand_val % 2
                            } else {
                                0
                            };
                            // 1er zéro (ordre du scan) ; 2e zéro = `a1` ssi `z0 && z1`,
                            // sinon `a2` (si `!z0` alors `z1 && z2` ⇒ le 2e est `a2`) ;
                            // 3e zéro = `a2`.
                            let f0 = if z0 { a0 } else if z1 { a1 } else { a2 };
                            let f1 = if z0 && z1 { a1 } else { a2 };
                            if k == 0 { f0 } else if k == 1 { f1 } else { a2 }
                        } else if zero_cnt == 1 {
                            // `zero_buf[0]` = l'unique littéral à `sad == 0`.
                            if z0 { a0 } else if z1 { a1 } else { a2 }
                        } else if zero_cnt == 3 {
                            // `zero_buf == [a0, a1, a2]` (compacté dans l'ordre du scan).
                            let idx = rand_val % 3;
                            if idx == 0 { a0 } else if idx == 1 { a1 } else { a2 }
                        } else {
                            // `zero_cnt == 2` : exactement un des trois est non-nul ⇒
                            // `zero_buf == [b0, b1]` dans l'ordre du scan.
                            let (b0, b1) = if !z2 { (a0, a1) } else if !z1 { (a0, a2) } else { (a1, a2) };
                            if rand_val % 2 == 0 { b0 } else { b1 }
                        }
                    } else {
                        let p0 = *PROBS_BREAK.get_unchecked(s0.min(15));
                        let p1 = *PROBS_BREAK.get_unchecked(s1.min(15));
                        let p2 = *PROBS_BREAK.get_unchecked(s2.min(15));
                        // `PROBS_BREAK` ≥ 7 partout ⇒ la somme est ≥ 21 ; le `.max(1)` est
                        // conservé pour rester LITTÉRALEMENT le même calcul que le générique.
                        let mut r = (rand_val as u32) % (p0 + p1 + p2).max(1);
                        // Marche pondérée identique ; le défaut hors-boucle du chemin
                        // générique était `pw_vars[0]`, c.-à-d. `a0`.
                        if r < p0 {
                            a0
                        } else {
                            r -= p0;
                            if r < p1 {
                                a1
                            } else {
                                r -= p1;
                                if r < p2 { a2 } else { a0 }
                            }
                        }
                    }
                } else {
                    // ═══ chemin GÉNÉRIQUE — VERBATIM `engine_r` arm 1 (= A_ctrl) ═══
                    // Conservé intact : il traite tout `clen != 3`, y compris le cas
                    // `clen == 0` où `pw_vars[0]` lit sa zéro-initialisation — SEUL point du
                    // moteur où cette valeur initiale est observable (garde exigée par
                    // `iters/t4/i7/screening.md`).
                    let mut zero_buf: [usize; BUF] = [0; BUF];
                    let mut zero_cnt: usize = 0;
                    let mut pw_weights: [u32; BUF] = [0; BUF];
                    let mut pw_vars: [usize; BUF] = [0; BUF];
                    let mut pw_cnt: usize = 0;
                    let mut total_pw: u32 = 0;

                    for j in cs..ce.min(cs + BUF) {
                        let abs_l = dv::<LP>(&cl, &lp, j);
                        let (os, oe) = if *vars.get_unchecked(abs_l) {
                            (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                        } else {
                            (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                        };

                        let mut sad = 0usize;
                        for k in os..oe {
                            let c = *all_data.get_unchecked(k) as usize;
                            sad += (*num_good.get_unchecked(c) == 1) as usize;
                        }

                        if sad == 0 {
                            *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                            zero_cnt += 1;
                        }

                        let pw = *PROBS_BREAK.get_unchecked(sad.min(15));
                        *pw_weights.get_unchecked_mut(pw_cnt) = pw;
                        *pw_vars.get_unchecked_mut(pw_cnt) = abs_l;
                        total_pw += pw;
                        pw_cnt += 1;
                    }

                    if zero_cnt > 0 {
                        if zero_cnt == 1 {
                            *zero_buf.get_unchecked(0)
                        } else {
                            *zero_buf.get_unchecked(rand_val % zero_cnt)
                        }
                    } else {
                        let mut r = (rand_val as u32) % total_pw.max(1);
                        let mut chosen = *pw_vars.get_unchecked(0);
                        for i in 0..pw_cnt {
                            let pw = *pw_weights.get_unchecked(i);
                            if r < pw { chosen = *pw_vars.get_unchecked(i); break; }
                            r -= pw;
                        }
                        chosen
                    }
                };

                let was_true = *vars.get_unchecked(v_idx);
                // ═══ AXE i21-a — `WT` : dispatch `was_true` SANS BRANCHE ═══════════════
                // Les DEUX branches d'origine lisent le MÊME ensemble {a, p, b} (cf la
                // démonstration de gratuité en en-tête) : hisser les 3 chargements
                // n'ajoute AUCUN accès mémoire ni AUCUNE instruction — il ne fait que
                // rendre les `load` inconditionnels, ce qui autorise LLVM à replier les
                // deux tests en `cmov`. Ce qui reste est littéralement deux swaps
                // conditionnels : (is,ds) = swap(p,a) et (ie,de) = swap(b,p).
                // Valeurs, ordre `inc` PUIS `dec`, et `rng` INCHANGÉS ⇒ byte-identique.
                let (is, ie, ds, de) = if WT {
                    let a = *all_off.get_unchecked(v_idx) as usize;
                    let p = *p_bound.get_unchecked(v_idx) as usize;
                    let b = *all_off.get_unchecked(v_idx + 1) as usize;
                    let is = if was_true { p } else { a };
                    let ds = if was_true { a } else { p };
                    let ie = if was_true { b } else { p };
                    let de = if was_true { p } else { b };
                    (is, ie, ds, de)
                } else {
                    let (is, ie) = if was_true {
                        (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
                    } else {
                        (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
                    };
                    let (ds, de) = if was_true {
                        (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
                    } else {
                        (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
                    };
                    (is, ie, ds, de)
                };

                // ⚠️ L'ORDRE inc-PUIS-dec est SACRÉ : il pilote l'ordre des push/pop, donc la
                // PERMUTATION de la liste, donc la clause tirée au flip suivant ⇒ la trajectoire.
                // HOT FLIP — SEUL site portant l'axe de l'iter i11 (`RU`).
                rmw_inc::<BL, RU, TCR, ZL>(&all_data, &mut num_good, &mut ubuf, &mut ulen, &mut cpos, nc32, nc, is, ie);
                rmw_dec::<BL, RU, TCR, ZL>(&all_data, &mut num_good, &mut ubuf, &mut ulen, &mut cpos, nc32, nc, ds, de);

                *vars.get_unchecked_mut(v_idx) = !was_true;
                rounds += 1;

                let cur = ulen;
                // ═══ AXE i23-b — `SG` : maj de `stagnation_count` SANS BRANCHE ══════════
                // Le bras `else` du diamant est un bloc instrumenté COMPLET (0x163320,
                // 2 appels PLT + fuel + signature) pour UNE SEULE instruction utile
                // (`incq`), exécuté sur ~100 % des flips. On le calcule inconditionnellement
                // (1 `add` + 1 `cmov`) ⇒ le bras disparaît. Le `memcpy` de `best_vars`
                // (5 000 o) RESTE conditionnel — le rendre inconditionnel serait
                // catastrophique. MÊMES valeurs ⇒ trajectoire byte-identique.
                // ⚠️ PRONOSTIC PRÉ-ENREGISTRÉ : retrait PARTIEL (le test de tête et le bras
                //    `memcpy` subsistent) ⇒ NULL attendu par la clause (1) de la loi i17.
                if SG {
                    let improved = cur < best_unsat;
                    stagnation_count = if improved { 0 } else { stagnation_count + 1 };
                    if improved {
                        best_unsat = cur;
                        best_vars.copy_from_slice(&vars);
                    }
                } else if cur < best_unsat {
                    best_unsat = cur;
                    best_vars.copy_from_slice(&vars);
                    stagnation_count = 0;
                } else {
                    stagnation_count += 1;
                }
    }}; }

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if ulen == 0 { break; }

            if stagnation_count >= REINIT_STAGNATION && best_unsat >= REINIT_MIN_UNSAT && reinit_count < max_reinits {
                reinit_count += 1;
                let mut best_cand_unsat = usize::MAX;
                for _ in 0..N_BON_RESTARTS {
                    for v in 0..nv { bon_candidate[v] = false; }
                    for v in 0..nv {
                        let np = p_cnt[v] as usize;
                        let nn = n_cnt[v] as usize;
                        if nn == 0 && np > 0 { bon_candidate[v] = true; continue; }
                        if np == 0 && nn > 0 { continue; }
                        let vad = if nn > 0 { np as f64 / nn as f64 } else { NAD + 1.0 };
                        if vad <= NAD {
                            bon_candidate[v] = rng.gen_bool(random_threshold);
                        } else {
                            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                            bon_candidate[v] = rng.gen_bool(prob);
                        }
                    }
                    bon_num_good.fill(0);
                    for c in 0..nc {
                        let s = *co.get_unchecked(c) as usize;
                        let e = *co.get_unchecked(c + 1) as usize;
                        let mut g = 0u8;
                        for j in s..e {
                            let v = dv::<LP>(&cl, &lp, j);
                            if dpol::<LP>(&cl, &lp, j) == *bon_candidate.get_unchecked(v) { g += 1; }
                        }
                        bon_num_good[c] = g;
                    }
                    let cand_unsat = bon_num_good.iter().filter(|&&x| x == 0).count();
                    if cand_unsat < best_cand_unsat {
                        best_cand_unsat = cand_unsat;
                        vars.copy_from_slice(&bon_candidate);
                    }
                }

                num_good.fill(0);
                for c in 0..nc {
                    let s = *co.get_unchecked(c) as usize;
                    let e = *co.get_unchecked(c + 1) as usize;
                    let mut g = 0u8;
                    for j in s..e {
                        let v = dv::<LP>(&cl, &lp, j);
                        if dpol::<LP>(&cl, &lp, j) == *vars.get_unchecked(v) { g += 1; }
                    }
                    num_good[c] = g;
                }

                ulen = 0;
                cpos.fill(u32::MAX);
                for c in 0..nc {
                    if num_good[c] == 0 {
                        ulen += 1;
                        *ubuf.get_unchecked_mut(ulen) = c as u32;
                        *cpos.get_unchecked_mut(c) = ulen as u32;
                    }
                }

                best_unsat = ulen;
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - ulen as i64;
                if progress <= 0 {
                    stagnation += 1;
                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if ulen == 0 { break; }
                            let rid = rng.gen::<usize>() % ulen;
                            let pcid = *ubuf.get_unchecked(rid + 1) as usize;
                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce { continue; }
                            let v = dv::<LP>(&cl, &lp, pcs + rng.gen::<usize>() % (pce - pcs));

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            } else {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            };
                            let (ds, de) = if was_true {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            } else {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            };

                            // ⚠️ KICK — épinglé à `RU = 0` (corps VERBATIM) DANS TOUS LES ARMS.
                            // Le kick pèse < 0,5 % du temps total (mesuré : la fusion de son
                            // double-scan avait donné +0,18 %, sous le bruit ⇒ révoquée). Le
                            // laisser intact garde l'iter MONO-AXE : le seul delta entre arms
                            // est le RMW du HOT FLIP.
                            rmw_inc::<BL, 0, false, false>(&all_data, &mut num_good, &mut ubuf, &mut ulen, &mut cpos, nc32, nc, is, ie);
                            rmw_dec::<BL, 0, false, false>(&all_data, &mut num_good, &mut ubuf, &mut ulen, &mut cpos, nc32, nc, ds, de);
                            *vars.get_unchecked_mut(v) = !was_true;
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }
                last_check_residual = ulen;
            }

            if ulen == 0 { break; }

            // ══════════════════════════════════════════════════════════════════════════
            // AXE i15 — HOISTING DES GARDES SUPERVISEURES (`HOIST`)
            // ══════════════════════════════════════════════════════════════════════════
            // `HOIST` est une CONSTANTE GÉNÉRIQUE ⇒ une seule des deux branches survit à
            // la monomorphisation. Pour `HOIST == 0` le code émis est EXACTEMENT la
            // boucle plate d'`engine_v` (aucune boucle interne, aucun bloc ajouté) :
            // l'ancre est verbatim PAR CONSTRUCTION, pas par recopie.
            if HOIST == 0 {
                hot_flip!();
            } else {
                // ── Longueur du bloc de flips pendant lequel AUCUNE garde hoistée ne
                //    peut tirer. Bornes démontrées dans l'en-tête du fichier.
                //
                // (D) — prochain multiple de `check_interval` STRICTEMENT après `rounds`.
                //       `rounds % ci == 0` ⇒ rend `ci` (le test vient d'être évalué
                //       ci-dessus, le suivant est `ci` flips plus loin). EXACTE.
                let n_chk = check_interval - rounds % check_interval;
                let nblk = if HOIST == 1 {
                    // (A) — `rounds` croît de EXACTEMENT 1 par flip. EXACTE.
                    let n_max = max_flips - rounds;
                    // (C) — `stagnation_count` croît de AU PLUS 1 par flip. Si le seuil
                    //       est déjà franchi sans que (C) ait tiré, c'est qu'une des deux
                    //       autres conjonctions est fausse ; or `best_unsat` ne fait que
                    //       DÉCROÎTRE et `reinit_count` que CROÎTRE ⇒ elles ne peuvent
                    //       pas redevenir vraies ⇒ (C) est morte pour de bon.
                    let n_rei = if stagnation_count < REINIT_STAGNATION {
                        REINIT_STAGNATION - stagnation_count
                    } else {
                        usize::MAX
                    };
                    n_chk.min(n_max).min(n_rei)
                } else {
                    // `HOIST == 2` : SEULE (D) est hoistée ⇒ seule sa borne s'applique.
                    n_chk
                };
                // `nblk >= 1` : `n_chk ∈ [1, ci]`, `n_max >= 1` (on vient de tester
                // `rounds >= max_flips`), `n_rei >= 1` (sinon (C) aurait tiré ou est
                // morte ⇒ `usize::MAX`).
                for _ in 0..nblk {
                    if HOIST == 2 {
                        // Arm 2 = contrôle APPARIÉ d'i4 : (A)(B)(C)(E) restent DANS le
                        // corps, à l'identique et dans le MÊME ORDRE que la boucle plate.
                        // Seule (D) manque ⇒ le delta arm0→arm2 est **UN BLOC**, point.
                        // Sur (A) ou (C) on ressort vers la boucle externe, qui ré-évalue
                        // les cinq gardes dans l'ordre d'origine et traite le cas.
                        if rounds >= max_flips { break; }
                        if ulen == 0 { break; }
                        if stagnation_count >= REINIT_STAGNATION && best_unsat >= REINIT_MIN_UNSAT && reinit_count < max_reinits { break; }
                    }
                    // (B)/(E) fusionnées : `ulen == 0` est la SEULE garde non bornable
                    // (le flip peut satisfaire la formule à tout moment) ⇒ elle reste
                    // par-flip. Dans la boucle plate elle était évaluée DEUX fois par
                    // flip ((B) puis (E)) avec, entre les deux, uniquement (C) et (D) —
                    // toutes deux garanties non prises ici ⇒ dé-dupliquer est EXACT.
                    if ulen == 0 { break; }
                    hot_flip!();
                }
            }

        }
    }

    let final_vars = if ulen == 0 { vars } else { best_vars };
    let _ = save_solution(&Solution { variables: final_vars });
    Ok(())
}

/// Occurrences dont le littéral DEVIENT vrai : `num_good += 1` ; si `num_good` valait 0 la
/// clause SORT de la liste (swap-pop). `BL=false` = branche d'origine ; `BL=true` = masque.
///
/// `RU` = PAS DE DÉROULAGE MASQUÉ (`P_masked_overread_fusion` appliqué à une boucle
/// ÉCRIVAINE). `RU == 0` ⇒ corps VERBATIM `engine_u` (ancre A_ctrl). `RU > 0` ⇒ la boucle
/// tourne sur `ceil(len / RU)` blocs de `RU` voies STRAIGHT-LINE : le remainder LLVM (dont
/// l'instrumentation dépasse 50 %, cf en-tête) DISPARAÎT, au prix de `RU*nb − len` voies
/// mortes qui sur-lisent.
///
/// ÉQUIVALENCE D'ÉTAT (voie MORTE, `live == 0`) — les 4 écritures sont neutralisées :
///   · `kr = start` (masque `msz`) ⇒ l'indice de LECTURE reste DANS `all_data`, jamais d'UB ;
///   · `c = ncg` (= `nc`) ⇒ `num_good[nc]` est une POUBELLE neuve, jamais lue (les scans
///     `sad` n'indexent `num_good` que par `all_data[k] < nc`) ; `wrapping_add` la laisse
///     déborder sans effet ;
///   · `cond = (ng == 0) & live == 0` ⇒ `*ulen` INCHANGÉ, `pos = 0` ⇒ `ubuf[0]` (poubelle
///     déjà établie par `P_rmw_branchless_garbage_slot`, i6) et `cpos[nc32]` (poubelle
///     existante).
/// ⇒ `num_good[0..nc]`, `ubuf[1..=ulen]`, `cpos[0..nc]` et `ulen` sont BYTE-IDENTIQUES à
/// l'arm 0, donc la PERMUTATION de `ubuf` — donc la trajectoire — est intouchée. Aucun
/// `rng.gen` n'est ajouté, retiré ni déplacé.
/// ⛔ Ce n'est PAS `brick:rmw_loop_fusion` (t5/i6, dead) : les deux plages `inc` et `dec`
/// restent DEUX BOUCLES SÉPARÉES dans l'ordre inc-PUIS-dec (SACRÉ), et aucun test
/// `(k < p_end)` par occurrence n'est introduit. C'est du DÉROULAGE — seul autorisé.
#[inline(always)]
unsafe fn rmw_inc<const BL: bool, const RU: usize, const TC: bool, const ZL: bool>(
    all_data: &[u32],
    num_good: &mut [u8],
    ubuf: &mut [u32],
    ulen: &mut usize,
    cpos: &mut [u32],
    nc32: u32,
    ncg: usize,
    start: usize,
    stop: usize,
) {
    if RU > 0 && BL {
        let len = stop - start;
        macro_rules! ru_body { ($base:expr) => {{
            // `RU` est une constante ⇒ boucle intégralement déroulée, zéro latch.
            for j in 0..RU {
                let kk = $base + j;
                let live = (kk < stop) as usize;
                let msz = live.wrapping_neg();
                let kr = (kk & msz) | (start & !msz);
                let c_raw = *all_data.get_unchecked(kr) as usize;
                let c = (c_raw & msz) | (ncg & !msz);
                let ng = *num_good.get_unchecked(c);
                *num_good.get_unchecked_mut(c) = ng.wrapping_add(1);
                let cond = ((ng == 0) as usize) & live;
                let m = 0u32.wrapping_sub(cond as u32);
                let last = *ubuf.get_unchecked(*ulen);
                let pos = (*cpos.get_unchecked(c) & m) as usize; // 0 = poubelle si !cond
                *ulen -= cond;
                *ubuf.get_unchecked_mut(pos) = last;
                *cpos.get_unchecked_mut(((last & m) | (nc32 & !m)) as usize) = pos as u32;
            }
        }}; }
        // ═══ AXE i25-b — `TC` : même transformation que `sad3`, cf sa démonstration ═══
        // Census ver 4440 : le préambule de `nb` coûte ici UN bloc instrumenté par RMW
        // (0x154665 pour `inc`, 0x1549de pour `dec`) — `shr $0x3` + `cmp $0x1 ; adc $0x0` —
        // en plus de la garde `cmp $0x8 ; jae` en queue du bloc producteur.
        // ⚠️ DIFFÉRENCE AVEC `sad3` : ici `len == 0` est POSSIBLE (littéral pur, ~8 var/5000)
        //    ⇒ la garde zéro-tour n'est PAS une tautologie, on la CONSERVE sous la forme
        //    `len != 0`. Elle ne coûte plus le calcul de `nb`, seulement un test déjà
        //    disponible. ⛔ On ne force donc AUCUN tour supplémentaire : c'est pourquoi
        //    ceci n'est PAS `codegen:rmw_constant_block_count` (dead 237, `NB` constant =
        //    over-read forcé) — le nombre de tours reste `ceil(len/RU)`, DONNÉE-DÉPENDANT.
        // ⭐ ÉQUIVALENCE : `nb = ceil(len/RU)`, l'ancre visite `base = start, …,
        //    start+RU*(nb-1)`. Le do-while visite `base = start, start+RU, …` tant que
        //    `base < stop` ⇒ exactement `ceil(len/RU) = nb` tours, MÊME suite de `base`.
        //    Corps LITTÉRALEMENT partagé par macro ⇒ trajectoire BYTE-IDENTIQUE.
        // ═══ AXE i27-a — `ZL` : SUPPRESSION DE LA GARDE DE DÉGÉNÉRESCENCE `len != 0` ═══
        // Census ver 4443 (monomorphisation championne) : cette garde vit en `0x1545cf`
        // (`cmp %rsi,%r9 ; jne 1545f0` + le stub `mov`/`jmp 1548d0`) pour `rmw_inc`, et en
        // `0x154909` (`cmp %r14d,0x60(%rsp) ; je 154bc9`) pour `rmw_dec` — cette dernière
        // étant le SEUL contenu utile du bloc instrumenté `0x1548d3` (fuel 2), qui
        // s'évapore donc en entier. **Franchies 2× par flip** = la classe de site qui a
        // payé −4,85 % à i25 (cf `bricks.md`, correctif du prédicteur : c'est la FRÉQUENCE
        // D'EXÉCUTION du bloc retiré, pas son cardinal statique).
        // ⭐ CE QUI REND LE RETRAIT LICITE : la sentinelle appendue à `all_data` par
        //    `run_buf`. Quand `len == 0` on a `start == stop`, les 8 voies sont TOUTES
        //    mortes (`kk >= stop`), donc `kr = start ≤ all_data.len()` — au pire la
        //    sentinelle, jamais au-delà.
        // ⭐ ÉQUIVALENCE D'ÉTAT (cas `len == 0`) : les 4 écritures du corps sont
        //    neutralisées EXACTEMENT comme une voie morte ordinaire — `c = ncg = nc`
        //    (poubelle `num_good[nc]`, `wrapping_add`/`wrapping_sub`), `cond = 0` ⇒
        //    `*ulen` INCHANGÉ, `pos = wpos = 0` ⇒ `ubuf[0]` et `cpos[nc32]` (poubelles
        //    déjà établies par `P_rmw_branchless_garbage_slot`). `num_good[0..nc]`,
        //    `ubuf[1..=ulen]`, `cpos[0..nc]` et `ulen` sont donc BYTE-IDENTIQUES, la
        //    permutation de `ubuf` est intouchée, aucun `rng.gen` n'est consommé.
        //    Le latch sort après CE tour : `base = start + RU >= stop = start`.
        // ⚠️ COÛT : un tour entièrement mort quand `len == 0`, fréquence ≈ e^(−6,4)
        //    ≈ 0,17 % (~8 variables sur 5 000) ⇒ négligeable devant le retrait.
        // ⛔ CE N'EST PAS dead 237 (`rmw_constant_block_count`) : le nombre de tours reste
        //    `ceil(len/RU)`, DONNÉE-DÉPENDANT, sur 99,83 % des appels ; le PAS reste
        //    `RU = 8` ⇒ le ratio pas/`len` de la loi de l'over-read (dead 235) est
        //    INCHANGÉ. Corps `ru_body!` partagé LITTÉRALEMENT par macro avec l'ancre.
        if TC {
            if ZL {
                let mut base = start;
                loop {
                    ru_body!(base);
                    base += RU;
                    if base >= stop { break; }
                }
                return;
            }
            if len != 0 {
                let mut base = start;
                loop {
                    ru_body!(base);
                    base += RU;
                    if base >= stop { break; }
                }
            }
            return;
        }
        let nb = (len + RU - 1) / RU;
        let mut base = start;
        for _b in 0..nb {
            ru_body!(base);
            base += RU;
        }
        return;
    }
    for k in start..stop {
        let c = *all_data.get_unchecked(k) as usize;
        let ng = *num_good.get_unchecked(c);
        *num_good.get_unchecked_mut(c) = ng + 1;
        if BL {
            let cond = (ng == 0) as usize;
            let m = 0u32.wrapping_sub(cond as u32);
            let last = *ubuf.get_unchecked(*ulen);
            let pos = (*cpos.get_unchecked(c) & m) as usize; // 0 = poubelle si !cond
            *ulen -= cond;
            *ubuf.get_unchecked_mut(pos) = last;
            *cpos.get_unchecked_mut(((last & m) | (nc32 & !m)) as usize) = pos as u32; // nc = poubelle
        } else if ng == 0 {
            let pos = *cpos.get_unchecked(c) as usize;
            let last = *ubuf.get_unchecked(*ulen);
            *ubuf.get_unchecked_mut(pos) = last;
            *cpos.get_unchecked_mut(last as usize) = pos as u32;
            *cpos.get_unchecked_mut(c) = u32::MAX;
            *ulen -= 1;
        }
    }
}

/// Occurrences dont le littéral DEVIENT faux : `num_good -= 1` ; si `num_good` valait 1 la
/// clause ENTRE dans la liste (push). `BL=false` = branche d'origine ; `BL=true` = masque.
///
/// `RU` : même déroulage masqué que `rmw_inc` — cf sa preuve d'équivalence. Voie MORTE :
/// `c = ncg` ⇒ `num_good[nc]` (poubelle, `wrapping_sub`), `cond = (ng == 1) & live == 0`
/// ⇒ `*ulen` INCHANGÉ, `wpos = 0` ⇒ `ubuf[0]` et `cpos[nc32]` (poubelles existantes).
#[inline(always)]
unsafe fn rmw_dec<const BL: bool, const RU: usize, const TC: bool, const ZL: bool>(
    all_data: &[u32],
    num_good: &mut [u8],
    ubuf: &mut [u32],
    ulen: &mut usize,
    cpos: &mut [u32],
    nc32: u32,
    ncg: usize,
    start: usize,
    stop: usize,
) {
    if RU > 0 && BL {
        let len = stop - start;
        macro_rules! ru_body { ($base:expr) => {{
            for j in 0..RU {
                let kk = $base + j;
                let live = (kk < stop) as usize;
                let msz = live.wrapping_neg();
                let kr = (kk & msz) | (start & !msz);
                let c_raw = *all_data.get_unchecked(kr) as usize;
                let c = (c_raw & msz) | (ncg & !msz);
                let ng = *num_good.get_unchecked(c);
                *num_good.get_unchecked_mut(c) = ng.wrapping_sub(1);
                let cond = ((ng == 1) as usize) & live;
                let m = 0u32.wrapping_sub(cond as u32);
                *ulen += cond;
                let wpos = (*ulen as u32 & m) as usize; // 0 = poubelle si !cond
                *ubuf.get_unchecked_mut(wpos) = c as u32;
                *cpos.get_unchecked_mut((((c as u32) & m) | (nc32 & !m)) as usize) = wpos as u32;
            }
        }}; }
        // ═══ AXE i25-b — `TC` : cf `rmw_inc` (même transformation, même preuve) ═══
        // ═══ AXE i27-a — `ZL` : SUPPRESSION DE LA GARDE DE DÉGÉNÉRESCENCE `len != 0` ═══
        // Census ver 4443 (monomorphisation championne) : cette garde vit en `0x1545cf`
        // (`cmp %rsi,%r9 ; jne 1545f0` + le stub `mov`/`jmp 1548d0`) pour `rmw_inc`, et en
        // `0x154909` (`cmp %r14d,0x60(%rsp) ; je 154bc9`) pour `rmw_dec` — cette dernière
        // étant le SEUL contenu utile du bloc instrumenté `0x1548d3` (fuel 2), qui
        // s'évapore donc en entier. **Franchies 2× par flip** = la classe de site qui a
        // payé −4,85 % à i25 (cf `bricks.md`, correctif du prédicteur : c'est la FRÉQUENCE
        // D'EXÉCUTION du bloc retiré, pas son cardinal statique).
        // ⭐ CE QUI REND LE RETRAIT LICITE : la sentinelle appendue à `all_data` par
        //    `run_buf`. Quand `len == 0` on a `start == stop`, les 8 voies sont TOUTES
        //    mortes (`kk >= stop`), donc `kr = start ≤ all_data.len()` — au pire la
        //    sentinelle, jamais au-delà.
        // ⭐ ÉQUIVALENCE D'ÉTAT (cas `len == 0`) : les 4 écritures du corps sont
        //    neutralisées EXACTEMENT comme une voie morte ordinaire — `c = ncg = nc`
        //    (poubelle `num_good[nc]`, `wrapping_add`/`wrapping_sub`), `cond = 0` ⇒
        //    `*ulen` INCHANGÉ, `pos = wpos = 0` ⇒ `ubuf[0]` et `cpos[nc32]` (poubelles
        //    déjà établies par `P_rmw_branchless_garbage_slot`). `num_good[0..nc]`,
        //    `ubuf[1..=ulen]`, `cpos[0..nc]` et `ulen` sont donc BYTE-IDENTIQUES, la
        //    permutation de `ubuf` est intouchée, aucun `rng.gen` n'est consommé.
        //    Le latch sort après CE tour : `base = start + RU >= stop = start`.
        // ⚠️ COÛT : un tour entièrement mort quand `len == 0`, fréquence ≈ e^(−6,4)
        //    ≈ 0,17 % (~8 variables sur 5 000) ⇒ négligeable devant le retrait.
        // ⛔ CE N'EST PAS dead 237 (`rmw_constant_block_count`) : le nombre de tours reste
        //    `ceil(len/RU)`, DONNÉE-DÉPENDANT, sur 99,83 % des appels ; le PAS reste
        //    `RU = 8` ⇒ le ratio pas/`len` de la loi de l'over-read (dead 235) est
        //    INCHANGÉ. Corps `ru_body!` partagé LITTÉRALEMENT par macro avec l'ancre.
        if TC {
            if ZL {
                let mut base = start;
                loop {
                    ru_body!(base);
                    base += RU;
                    if base >= stop { break; }
                }
                return;
            }
            if len != 0 {
                let mut base = start;
                loop {
                    ru_body!(base);
                    base += RU;
                    if base >= stop { break; }
                }
            }
            return;
        }
        let nb = (len + RU - 1) / RU;
        let mut base = start;
        for _b in 0..nb {
            ru_body!(base);
            base += RU;
        }
        return;
    }
    for k in start..stop {
        let c = *all_data.get_unchecked(k) as usize;
        let ng = *num_good.get_unchecked(c);
        *num_good.get_unchecked_mut(c) = ng - 1;
        if BL {
            let cond = (ng == 1) as usize;
            let m = 0u32.wrapping_sub(cond as u32);
            *ulen += cond;
            let wpos = (*ulen as u32 & m) as usize; // 0 = poubelle si !cond
            *ubuf.get_unchecked_mut(wpos) = c as u32;
            *cpos.get_unchecked_mut((((c as u32) & m) | (nc32 & !m)) as usize) = wpos as u32;
        } else if ng == 1 {
            *ulen += 1;
            *ubuf.get_unchecked_mut(*ulen) = c as u32;
            *cpos.get_unchecked_mut(c) = *ulen as u32;
        }
    }
}
