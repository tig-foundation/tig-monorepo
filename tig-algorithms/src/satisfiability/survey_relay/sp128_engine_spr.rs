// ═══ engine_spr — SURVEY PROPAGATION AVEC RENFORCEMENT, utilise comme PHASE 1 (graine) ═════════
//
// Cellule c001, 20/08 soir, iteration t38/i1 (TEMPS a Q egale). Chavas, Furtlehner, Mezard,
// Zecchina, J. Stat. Mech. (2005) P11016 : SP + champ externe par variable oriente vers son biais
// courant, d'intensite gamma_t croissante, SANS decimation. Mesure hors ligne (sp_offline/spr.py,
// t38 nonce 0, n=100 000) : ~100 clauses fausses atteintes en 300 iterations, puis degradation si
// le renforcement continue => on garde le MEILLEUR point et on s'arrete sur stagnation.
// Sur les tracks AU SEUIL la quasi-solution obtenue est globalement fausse (NO-GO Q, cf
// RESULTAT_SP_renforcement_offline_t4_NO_GO.md) ; ici, sous le seuil (alpha=4,20, 32/32 deja),
// la seule question est le TEMPS du polish SLS depuis ce point contre l'init de production.
// Domaine lineaire (pas de ln/exp par arete) : produits de (1-eta) clampes a 1e-9.

const MAXIT: usize = 1500;
const RATE: f64 = 0.001;
const GMAX: f64 = 0.5;
const LAM: f64 = 0.1;
const DAMP: f64 = 0.3;
const CHECK_EVERY: usize = 10;
const PATIENCE: usize = 150;

/// `cl` encode `(v << 1) | pol`, `co` offsets CSR. Rend (graine, unsat de la graine, it du
/// meilleur point, it total).
pub fn seed_enc(nv: usize, nc: usize, cl: &[i32], co: &[u32], rng: &mut impl rand::Rng, stop_when_seed_good: bool)
    -> (Vec<bool>, usize, usize, usize)
{
    // sat_exact_spfast: same f32 operations, in the same order, as the a123/a127 kernel, so the seed,
    // the RNG stream and the solutions are unchanged. Two implementation changes only:
    // (1) the encoded literal l = cl[e] = (v << 1) | pol indexes p[] directly (p[l], and p[l ^ 1] for
    //     the opposite sign) instead of the separate ev[]/es[] edge arrays (standard MiniSat-style
    //     literal encoding);
    // (2) the three-literal clause specialisation of pass 2 is reused from c001_a128 sat_tailwalk_v7
    //     (zhenglcc), which keeps the left-to-right product order of the generic loop.
    // The SP to local-search hand-off rule is a123's, unchanged (it >= 100 and at most 128 false clauses).
    // t38/i4 : DEUX passes par iteration au lieu de trois (la passe « r par arete » est fusionnee
    // dans la passe clause-major qui calcule eta), et f32 partout (le SLS n'a besoin que du
    // SIGNE de W+ - W-). Meme algorithme, memes constantes ; i3 a mesure la phase a ~60 s/nonce.
    let ne = co[nc] as usize;
    let lit: &[i32] = &cl[..ne];
    let mut eta: Vec<f32> = (0..ne).map(|_| rng.gen::<f32>()).collect();
    let mut p = vec![1f32; 2 * nv];
    let mut hp = vec![0f32; nv];
    let mut hm = vec![0f32; nv];
    let mut wp = vec![0f32; nv];
    let mut wm = vec![0f32; nv];
    let mut best_x = vec![false; nv];
    let mut best_u = usize::MAX;
    let mut best_it = 0usize;
    let mut it = 0usize;
    let damp = DAMP as f32;
    while it < MAXIT {
        it += 1;
        let gamma = ((RATE * it as f64).min(GMAX)) as f32;
        // passe 1 (arete-major, scatter) : produits (1-eta) par (variable, signe) + champ externe
        for v in 0..nv {
            p[2 * v] = 1.0 - hm[v];
            p[2 * v + 1] = 1.0 - hp[v];
        }
        for e in 0..ne {
            let o = (1.0 - eta[e]).max(1e-6);
            p[lit[e] as usize] *= o;
        }
        // passe 2 (clause-major, gather) : r des aretes de la clause puis eta
        let mut rr = [0f32; 3];
        for a in 0..nc {
            let s0 = co[a] as usize; let e0 = co[a + 1] as usize;
            let len = e0 - s0;
            if len == 3 {
                for q in 0..3 {
                    let e = s0 + q;
                    let l = lit[e] as usize;
                    let o = (1.0 - eta[e]).max(1e-6);
                    let av = p[l] / o;
                    let bv = p[l ^ 1];
                    let pu = (1.0 - bv) * av; let ps = (1.0 - av) * bv; let p0 = av * bv;
                    let den = pu + ps + p0;
                    rr[q] = if den > 0.0 { pu / den } else { 0.0 };
                }
                // same left-to-right order as the generic loop: (1 * r_i) * r_j
                let prod0 = (1f32 * rr[1]) * rr[2];
                let prod1 = (1f32 * rr[0]) * rr[2];
                let prod2 = (1f32 * rr[0]) * rr[1];
                eta[s0] = damp * eta[s0] + (1.0 - damp) * prod0.min(1.0);
                eta[s0 + 1] = damp * eta[s0 + 1] + (1.0 - damp) * prod1.min(1.0);
                eta[s0 + 2] = damp * eta[s0 + 2] + (1.0 - damp) * prod2.min(1.0);
            } else {
            for q in 0..len {
                let e = s0 + q;
                let l = lit[e] as usize;
                let o = (1.0 - eta[e]).max(1e-6);
                let av = p[l] / o;
                let bv = p[l ^ 1];
                let pu = (1.0 - bv) * av; let ps = (1.0 - av) * bv; let p0 = av * bv;
                let den = pu + ps + p0;
                rr[q] = if den > 0.0 { pu / den } else { 0.0 };
            }
            for q in 0..len {
                let mut prod = 1f32;
                for k in 0..len { if k != q { prod *= rr[k]; } }
                let e = s0 + q;
                eta[e] = damp * eta[e] + (1.0 - damp) * prod.min(1.0);
            }
            }
        }
        // biais et renforcement (O(nv))
        for v in 0..nv {
            let pm = p[2 * v]; let pp = p[2 * v + 1];
            let pip = (1.0 - pp) * pm; let pim = (1.0 - pm) * pp; let pi0 = pp * pm;
            let den = pip + pim + pi0;
            if den > 0.0 { wp[v] = pip / den; wm[v] = pim / den; } else { wp[v] = 0.0; wm[v] = 0.0; }
            hp[v] = (1.0 - LAM as f32) * hp[v] + (LAM as f32) * gamma * wp[v];
            hm[v] = (1.0 - LAM as f32) * hm[v] + (LAM as f32) * gamma * wm[v];
        }
        if it % CHECK_EVERY == 0 {
            let mut u = 0usize;
            for a in 0..nc {
                let mut ok = false;
                for j in co[a] as usize..co[a + 1] as usize {
                    let l = lit[j];
                    let v = (l >> 1) as usize;
                    if ((l & 1) == 1) == (wp[v] > wm[v]) { ok = true; break; }
                }
                if !ok { u += 1; }
            }
            if u < best_u {
                best_u = u; best_it = it;
                for v in 0..nv { best_x[v] = wp[v] > wm[v]; }
                if u == 0 { break; }
            } else if it - best_it >= PATIENCE {
                break;
            }
            // Fixed research rule: only enabled by the public 100000/420000 route.
            // Retain the best prefix assignment; this is not the full-SP donor floor.
            if stop_when_seed_good && it >= 100 && best_u <= 128 {
                break;
            }
        }
    }
    (best_x, best_u, best_it, it)
}
