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
pub fn seed_enc(nv: usize, nc: usize, cl: &[i32], co: &[u32], rng: &mut impl rand::Rng)
    -> (Vec<bool>, usize, usize, usize)
{
    // t38/i4 : DEUX passes par iteration au lieu de trois (la passe « r par arete » est fusionnee
    // dans la passe clause-major qui calcule eta), et f32 partout (le SLS n'a besoin que du
    // SIGNE de W+ - W-). Meme algorithme, memes constantes ; i3 a mesure la phase a ~60 s/nonce.
    let ne = co[nc] as usize;
    let mut ev = vec![0u32; ne];
    let mut es = vec![0u8; ne];
    for a in 0..nc {
        for j in co[a] as usize..co[a + 1] as usize {
            let l = cl[j];
            ev[j] = (l >> 1) as u32; es[j] = (l & 1) as u8;
        }
    }
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
            p[(ev[e] as usize) * 2 + es[e] as usize] *= o;
        }
        // passe 2 (clause-major, gather) : r des aretes de la clause puis eta
        let mut rr = [0f32; 3];
        for a in 0..nc {
            let s0 = co[a] as usize; let e0 = co[a + 1] as usize;
            let len = e0 - s0;
            for q in 0..len {
                let e = s0 + q;
                let v = ev[e] as usize; let sg = es[e] as usize;
                let o = (1.0 - eta[e]).max(1e-6);
                let av = p[v * 2 + sg] / o;
                let bv = p[v * 2 + (1 - sg)];
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
                    let v = ev[j] as usize;
                    if (es[j] == 1) == (wp[v] > wm[v]) { ok = true; break; }
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
        }
    }
    (best_x, best_u, best_it, it)
}
