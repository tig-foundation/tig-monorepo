
// combine (source_algo=131 the base engine): the whole HFS solve stack ported VERBATIM to replace

// Bundle = v9 {types, preprocess, infra_shared, hybrid_flow_shop} inlined as nested pub mods (dep closure
// verified: hybrid_flow_shop -> {types, infra_shared, crate::{seeded_hasher,HashMap}}; infra_shared/preprocess -> types).
// crate::{seeded_hasher, HashMap} = engine crate root (same as track_t46/t48/jss_engine). No wall-clock, no threading.

pub use solver::{solve_challenge, help};

pub mod types {
pub const INF: u32 = u32::MAX / 4;
pub const NONE_USIZE: usize = usize::MAX;

#[derive(Clone)]
pub struct OpInfo {
    pub machines: Vec<(usize, u32)>,
    pub min_pt: u32,
    pub avg_pt: f64,
    pub flex: usize,
    pub bn_avg: f64,
}

#[derive(Clone, Copy, Default)]
pub struct OpRoute {
    pub best_m: u8,
    pub best_w: u8,
    pub second_m: u8,
    pub second_w: u8,
}

pub type RoutePrefLite = Vec<Vec<OpRoute>>;

#[derive(Clone)]
pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    pub product_suf_min: Vec<Vec<u32>>,
    pub product_suf_avg: Vec<Vec<f64>>,
    pub product_suf_bn: Vec<Vec<f64>>,
    pub product_next_min: Vec<Vec<u32>>,
    pub product_next_flex_inv: Vec<Vec<f64>>,
    pub machine_load0: Vec<f64>,
    pub machine_scarcity: Vec<f64>,
    pub machine_weight: Vec<f64>,
    pub machine_best_pop: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_machine_scarcity: f64,
    pub avg_op_min: f64,
    pub horizon: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_avg_work: f64,
    pub max_job_bn: f64,
    pub flex_avg: f64,
    pub flex_factor: f64,
    pub hi_flex: bool,
    pub high_flex: f64,
    pub flow_like: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
    pub chaotic_like: bool,
    pub flow_route: Option<Vec<usize>>,
    pub flow_pt_by_job: Option<Vec<Vec<u32>>>,
    pub strict_route: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

#[derive(Clone, Copy)]
pub struct RawCand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub base_score: f64,
    pub rigidity: f64,
    pub reg_n: f64,
}

#[derive(Clone, Copy)]
pub enum GreedyRule {
    MostWork,
    MostOps,
    LeastFlex,
    ShortestProc,
    LongestProc,
}

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    pub stack: Vec<usize>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE_USIZE; n],
            machine_succ: vec![NONE_USIZE; n],
            stack: Vec::with_capacity(n),
        }
    }
}

#[derive(Clone, Copy)]
pub struct MoveCand {
    pub kind: u8,
    pub m_from: usize,
    pub from: usize,
    pub m_to: usize,
    pub to: usize,
    pub new_pt: u32,
    pub score: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct EffortConfig {
    pub job_shop_iters: usize,
    pub hybrid_flow_shop_iters: usize,
    pub hybrid_flow_shop_tabu_iters: usize,
    pub hybrid_flow_shop_tabu_seeds: usize,
    pub hybrid_flow_shop_tabu_stagnation: usize,
    pub hybrid_flow_shop_tabu_reassign_every: usize,
    
    // Orthogonal to stagnation (which saturated at ~2000): stagnation = tabu DEPTH per basin,
    // these = tabu BREADTH/perturbation. kicks = #escapes to best (was 5); kick_pct = perturbation
    // cadence as pct of stagnation (0 = sentinel → original *2/3, byte-exact ctrl); kick_swaps =
    // #critical-block swaps per in-basin kick (was 3).
    pub hybrid_flow_shop_tabu_kicks: usize,
    pub hybrid_flow_shop_tabu_kick_pct: usize,
    pub hybrid_flow_shop_tabu_kick_swaps: usize,
    
    // destruction operator used at each in-basin kick — 0 = sentinel (byte-exact original
    // critical-adjacent swap), 1 = ensemble (round-robin A→B→C, stagnation-triggered switch),
    // 2 = bottleneck-machine-block solo (B), 3 = dispersed-non-critical solo (C). All three
    // operators use the proven-safe adjacent-swap primitive; the DOE varies the TYPE of
    // destruction, never its size (kick_swaps stays fixed). doe_switch = #kicks on the current
    // operator without a best-makespan improvement before rotating to the next operator (mode 1).
    pub hybrid_flow_shop_doe_mode: usize,
    pub hybrid_flow_shop_doe_switch: usize,
    
    // (Nowicki-Smutnicki): only the first/last adjacent pair of each critical block is swapped, so
    // an INTERNAL critical operation can never leave its block -> topological ceiling (73900, proven
    
    // relocations of internal critical ops toward/out of their block boundary:
    //   0 = N5 sentinel (byte-exact original, no relocate candidates -> reproduces 73900),
    //   1 = N7 (re-insert internal critical op at the block frontier: just before bstart / just after bend),
    //   2 = N8 (push internal critical op just OUTSIDE the block: bstart-1 / bend+1).
    // Relocations are kept SHORT-DISTANCE (boundary-adjacent) on purpose: the head/tail estimate
    // (estimate_reassign_hfs) reuses pre-move buffers and is only accurate for near neighbors — the
    // documented failure mode of a prior long-distance N8 attempt (engine t47 2026-06-23, rejected:
    // "estimateur optimiste sur buffers pre-relocation"). n8_max caps internal ops considered per block.
    pub hybrid_flow_shop_n8: usize,
    pub hybrid_flow_shop_n8_max: usize,
    // L7 HFS-oriented neighborhood — bottleneck load-balancing reassignment. The existing inter-machine
    // reassignment neighborhood only considers CRITICAL-path nodes (crit[node]); it can never move a
    // NON-critical op off an over-loaded machine to free the machine sequence so the critical block
    
    // bn_reassign: 0 = sentinel (byte-exact 73900, gate identical to crit-only), 1 = also offer non-critical
    // ops on THE bottleneck machine (most critical nodes, Op-B def) to their alt machines, 2 = top-2 loaded.
    pub hybrid_flow_shop_bottleneck_reassign: usize,
    
    
    
    // destroys — the reconstruction has always stayed the same blind adjacent-swap primitive.
    // ig_mode replaces the reconstruction itself with an Iterated-Greedy step (Ruiz-Stützle
    
    // IG expected to perform well on HFS): draw ig_d distinct critical ops, then sequentially
    // FORCE-MOVE each to its exact-eval argmin slot (full eval_disj per candidate position,
    // original slot excluded so the perturbation always moves), keeping the schedule fully
    // valid at every eval (one op relocated at a time; infeasible candidates return None and
    // are skipped; if no candidate is valid the op is restored to its original slot).
    //   0 = sentinel (byte-exact original blind-swap kick -> reproduces 73900),
    //   1 = IG reconstruction, candidate slots on the op's CURRENT machine only,
    //   2 = IG reconstruction, candidate slots on ALL eligible machines of the op
    //       (pt_from_op per machine — the HFS parallel-machine axis, eval-driven, unlike
    
    // ig_d = number of critical ops destructed+reinserted per kick (the new family's own
    // intensity knob, distinct from the dead swap-kick knobs).
    pub hybrid_flow_shop_ig_mode: usize,
    pub hybrid_flow_shop_ig_d: usize,
    
    
    
    // moteur champion (harnais /tmp/jss47, hp_json bench 25455) :
    
    //     sur 4 nonces/4 (d_machine 548-744, d_order 3896-22239) => la cause de mort t45
    //     (5a2b83aa "the converged pool renders it ineffective") ne transfere pas ;
    //   * le chemin ne descend jamais sous l'incumbent (delta_path=0 sur 4/4) : PR fournit une
    //     marche de PLATEAU vers un point de makespan EGAL mais structurellement distinct ;
    //   * c'est le TS relance depuis ce point qui descend. CONTROLE NEGATIF (meme budget TS
    //     depuis l'incumbent, zero PR) : gain IDENTIQUE a l'unite sur les nonces 1/6/8, mais
    //     ZERO sur le nonce 2 = le rang 15 de la mediane, ou PR rend -38 mk (+3305 Q).
    //     => sur la coordonnee qui fait la mediane, l'incumbent est un POINT FIXE du TS et
    //        seul le deplacement lateral structurel en sort.
    // 0 = OFF (sentinelle byte-exacte, doit rendre 74702)
    // 1 = TS depuis le meilleur intermediaire PR
    // 2 = 1 + TS chaine depuis l'incumbent (union des deux graines ; borne sup des deux bras)
    pub hybrid_flow_shop_path_relink: usize,
    pub fjsp_medium_iters: usize,
    pub fjsp_high_iters: usize,
}

impl EffortConfig {
    pub fn default_effort() -> Self {
        Self { job_shop_iters: 25000, hybrid_flow_shop_iters: 3500, hybrid_flow_shop_tabu_iters: 1800, hybrid_flow_shop_tabu_seeds: 5, hybrid_flow_shop_tabu_stagnation: 400, hybrid_flow_shop_tabu_reassign_every: 1, hybrid_flow_shop_tabu_kicks: 5, hybrid_flow_shop_tabu_kick_pct: 0, hybrid_flow_shop_tabu_kick_swaps: 3, hybrid_flow_shop_doe_mode: 0, hybrid_flow_shop_doe_switch: 2, hybrid_flow_shop_n8: 0, hybrid_flow_shop_n8_max: 3, hybrid_flow_shop_bottleneck_reassign: 0, hybrid_flow_shop_ig_mode: 0, hybrid_flow_shop_ig_d: 6, hybrid_flow_shop_path_relink: 0, fjsp_medium_iters: 2000, fjsp_high_iters: 2000 }
    }

    pub fn with_job_shop_iters(mut self, v: usize) -> Self {
        self.job_shop_iters = v.clamp(100, 200000);
        self
    }

    pub fn with_hybrid_flow_shop_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_seeds(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_seeds = v.clamp(1, 16);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_stagnation(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_stagnation = v.clamp(20, 10000);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_reassign_every(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_reassign_every = v.clamp(1, 32);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_kicks(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_kicks = v.clamp(1, 64);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_kick_pct(mut self, v: usize) -> Self {
        // 0 = sentinel (keep original stagnation*2/3); else pct of stagnation, clamped to a sane band.
        self.hybrid_flow_shop_tabu_kick_pct = if v == 0 { 0 } else { v.clamp(10, 100) };
        self
    }

    pub fn with_hybrid_flow_shop_tabu_kick_swaps(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_kick_swaps = v.clamp(1, 16);
        self
    }

    pub fn with_hybrid_flow_shop_doe_mode(mut self, v: usize) -> Self {
        // 0 = sentinel (original critical-adjacent), 1 = ensemble, 2 = bottleneck-solo, 3 = dispersed-solo.
        self.hybrid_flow_shop_doe_mode = v.clamp(0, 3);
        self
    }

    pub fn with_hybrid_flow_shop_doe_switch(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_doe_switch = v.clamp(1, 64);
        self
    }

    pub fn with_hybrid_flow_shop_n8(mut self, v: usize) -> Self {
        // 0 = N5 sentinel (byte-exact), 1 = N7 (boundary-insert), 2 = N8 (inner-out).
        self.hybrid_flow_shop_n8 = v.clamp(0, 2);
        self
    }

    pub fn with_hybrid_flow_shop_n8_max(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_n8_max = v.clamp(1, 16);
        self
    }

    pub fn with_hybrid_flow_shop_bottleneck_reassign(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_bottleneck_reassign = v.clamp(0, 2);
        self
    }

    pub fn with_hybrid_flow_shop_ig_mode(mut self, v: usize) -> Self {
        // 0 = sentinel (blind-swap kick, byte-exact), 1 = IG same-machine, 2 = IG cross-machine.
        self.hybrid_flow_shop_ig_mode = v.clamp(0, 2);
        self
    }

    pub fn with_hybrid_flow_shop_path_relink(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_path_relink = v.clamp(0, 2);
        self
    }
    pub fn with_hybrid_flow_shop_ig_d(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_ig_d = v.clamp(1, 32);
        self
    }

    pub fn with_fjsp_medium_iters(mut self, v: usize) -> Self {
        self.fjsp_medium_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_high_iters(mut self, v: usize) -> Self {
        self.fjsp_high_iters = v.clamp(100, 100000);
        self
    }
}
}

pub mod preprocess {
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;

#[inline]
fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() { continue; }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            let v = comp[k].max(comp[k - 1]).saturating_add(row[k]);
            comp[k] = v;
        }
    }
    *comp.last().unwrap_or(&0)
}

pub fn build_pre(challenge: &Challenge) -> Result<Pre> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!("jobs_per_product sum mismatch"));
    }

    let num_products = challenge.product_processing_times.len();

    let mut product_ops: Vec<Vec<OpInfo>> = Vec::with_capacity(num_products);
    let mut best_machine_by_product: Vec<Vec<usize>> = Vec::with_capacity(num_products);

    let mut machine_load0 = vec![0.0f64; num_machines];
    let mut machine_scarcity = vec![0.0f64; num_machines];
    let mut machine_best_cnt = vec![0.0f64; num_machines];

    let mut total_ops: usize = 0;
    let mut total_min_work: f64 = 0.0;
    let mut total_flex_weighted: f64 = 0.0;

    let mut max_ops: usize = 1;
    let mut max_job_avg_work: f64 = 1.0;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        max_ops = max_ops.max(ops.len());

        let mut ops_info: Vec<OpInfo> = Vec::with_capacity(ops.len());
        let mut bests: Vec<usize> = Vec::with_capacity(ops.len());

        let mut sum_min_u64: u64 = 0;
        let mut sum_avg_f: f64 = 0.0;

        for op in ops {
            if op.is_empty() {
                ops_info.push(OpInfo {
                    machines: vec![],
                    min_pt: INF,
                    avg_pt: 0.0,
                    flex: 0,
                    bn_avg: 0.0,
                });
                bests.push(0);
                continue;
            }

            let mut machines: Vec<(usize, u32)> = Vec::with_capacity(op.len());
            let mut min_pt = INF;
            let mut sum = 0u64;

            let mut best_m = 0usize;
            let mut best_pt = INF;

            for (&m, &pt) in op.iter() {
                if m >= num_machines {
                    return Err(anyhow!("machine id out of range"));
                }
                machines.push((m, pt));
                min_pt = min_pt.min(pt);
                sum += pt as u64;

                if pt < best_pt || (pt == best_pt && m < best_m) {
                    best_pt = pt;
                    best_m = m;
                }
            }

            let flex = machines.len().max(1);
            let avg_pt = (sum as f64) / (flex as f64);

            sum_min_u64 += min_pt.min(INF / 2) as u64;
            sum_avg_f += avg_pt;

            machines.sort_unstable_by_key(|x| x.0);

            ops_info.push(OpInfo {
                machines,
                min_pt,
                avg_pt,
                flex,
                bn_avg: 0.0,
            });
            bests.push(best_m);
        }

        max_job_avg_work = max_job_avg_work.max(sum_avg_f);

        let cnt_u = challenge.jobs_per_product[p] as usize;
        let cnt_f = challenge.jobs_per_product[p] as f64;

        total_ops += ops_info.len() * cnt_u;
        total_min_work += (sum_min_u64 as f64) * cnt_f;

        for (oi, &bm) in ops_info.iter().zip(bests.iter()) {
            total_flex_weighted += (oi.flex as f64) * cnt_f;
            if bm < num_machines {
                machine_best_cnt[bm] += cnt_f;
            }

            if oi.min_pt < INF && oi.flex > 0 && !oi.machines.is_empty() {
                let flex_f = (oi.flex as f64).max(1.0);
                let delta = (oi.min_pt as f64) * cnt_f / flex_f;
                let delta_s = (oi.min_pt as f64) * cnt_f / (flex_f * flex_f);
                for &(m, _) in &oi.machines {
                    machine_load0[m] += delta;
                    machine_scarcity[m] += delta_s;
                }
            }
        }

        product_ops.push(ops_info);
        best_machine_by_product.push(bests);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();

    let avg_machine_load = (total_min_work / (num_machines as f64).max(1.0)).max(1.0);
    let horizon = avg_machine_load;

    let avg_op_min = (total_min_work / (total_ops as f64).max(1.0)).max(1.0);
    let flex_avg = (total_flex_weighted / (total_ops as f64).max(1.0)).max(1.0);
    let flex_factor = (3.0 / flex_avg).clamp(0.6, 2.2);
    let hi_flex = flex_avg >= 5.0;
    let high_flex = ((flex_avg - 3.0) / 7.0).clamp(0.0, 1.0);

    let avg_machine_scarcity = {
        let s: f64 = machine_scarcity.iter().sum();
        (s / (num_machines as f64).max(1.0)).max(1e-9)
    };

    let load_cv = {
        let mean = avg_machine_load.max(1e-9);
        let mut var = 0.0f64;
        for &x in &machine_load0 {
            let d = (x / mean) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

    let mut flow_sum = 0.0f64;
    let mut flow_cnt = 0usize;
    let mut counts = vec![0u32; num_machines];
    for op_idx in 0..max_ops {
        counts.fill(0);
        let mut tot = 0u32;

        for p in 0..num_products {
            if op_idx >= best_machine_by_product[p].len() {
                continue;
            }
            let bm = best_machine_by_product[p][op_idx];
            let w_u32 = challenge.jobs_per_product[p] as u32;
            if w_u32 == 0 {
                continue;
            }
            counts[bm] = counts[bm].saturating_add(w_u32);
            tot = tot.saturating_add(w_u32);
        }

        if tot > 0 {
            let mut mx = 0u32;
            for &c in &counts {
                mx = mx.max(c);
            }
            flow_sum += (mx as f64) / (tot as f64);
            flow_cnt += 1;
        }
    }
    let flow_like = if flow_cnt > 0 { (flow_sum / (flow_cnt as f64)).clamp(0.0, 1.0) } else { 0.5 };
    let jobshopness = (1.0 - flow_like).clamp(0.0, 1.0);

    let mut machine_weight = vec![1.0f64; num_machines];
    {
        let mean = avg_machine_load.max(1e-9);
        let exp = (1.10 + 0.35 * load_cv + 0.20 * jobshopness).clamp(1.05, 1.70);
        for m in 0..num_machines {
            let r = (machine_load0[m] / mean).max(0.05);
            machine_weight[m] = r.powf(exp).clamp(0.55, 3.75);
        }
    }

    let machine_best_pop = {
        let tot: f64 = machine_best_cnt.iter().sum();
        let mean = (tot / (num_machines as f64).max(1.0)).max(1e-9);
        let mut pop = vec![0.0f64; num_machines];
        for m in 0..num_machines {
            let r = (machine_best_cnt[m] / mean).clamp(0.0, 10.0);
            pop[m] = (r / (1.0 + r)).clamp(0.0, 1.0);
        }
        pop
    };

    let bn_focus = ((3.0 / flex_avg).clamp(0.7, 2.6) * (1.0 + 0.55 * load_cv) * (0.85 + 0.55 * jobshopness)).clamp(0.6, 3.4);

    let mut product_suf_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_avg: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_bn: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_next_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_next_flex_inv: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());

    let mut max_job_bn: f64 = 1e-9;

    for ops in product_ops.iter_mut() {
        let n = ops.len();
        let mut suf_m = vec![0u32; n + 1];
        let mut suf_a = vec![0.0f64; n + 1];
        let mut suf_bn = vec![0.0f64; n + 1];

        let mut nxt_m = vec![0u32; n + 1];
        let mut nxt_fi = vec![0.0f64; n + 1];

        for i in (0..n).rev() {
            let oi = &mut ops[i];

            if oi.flex == 0 || oi.machines.is_empty() || oi.min_pt >= INF {
                oi.bn_avg = 0.0;
            } else {
                let mut sum = 0.0f64;
                for &(m, pt) in &oi.machines {
                    sum += (pt as f64) * machine_weight[m];
                }
                oi.bn_avg = sum / (oi.flex as f64);
            }

            suf_m[i] = suf_m[i + 1].saturating_add(oi.min_pt.min(INF / 2));
            suf_a[i] = suf_a[i + 1] + oi.avg_pt;
            suf_bn[i] = suf_bn[i + 1] + oi.bn_avg;

            if i + 1 < n {
                let next = &ops[i + 1];
                nxt_m[i] = next.min_pt;
                nxt_fi[i] = if next.flex > 0 { 1.0 / (next.flex as f64) } else { 0.0 };
            }
        }

        max_job_bn = max_job_bn.max(suf_bn[0]);

        product_suf_min.push(suf_m);
        product_suf_avg.push(suf_a);
        product_suf_bn.push(suf_bn);
        product_next_min.push(nxt_m);
        product_next_flex_inv.push(nxt_fi);
    }

    let time_scale = (horizon * (2.65 + 0.15 * load_cv + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);

    let mut job_flow_pref = vec![0.0f64; num_jobs];
    let use_flow_pref = flow_like > 0.82 && jobshopness < 0.38 && max_ops >= 2;

    if use_flow_pref {
        let m = max_ops.max(1);
        let mut job_pt: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let p = job_products[j];
            let ops = &product_ops[p];
            let mut v = vec![0u32; m];
            for s in 0..m.min(ops.len()) {
                v[s] = ops[s].min_pt.min(INF / 2);
            }
            job_pt.push(v);
        }

        let mut jobs2: Vec<usize> = (0..num_jobs).collect();
        jobs2.sort_unstable_by(|&a, &b| {
            let sa: u32 = job_pt[a].iter().copied().sum();
            let sb: u32 = job_pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });

        let mut perm: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut comp = vec![0u32; m];
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);

        for &j in &jobs2 {
            if perm.is_empty() {
                perm.push(j);
                continue;
            }
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=perm.len() {
                tmp.clear();
                tmp.extend_from_slice(&perm[..pos]);
                tmp.push(j);
                tmp.extend_from_slice(&perm[pos..]);
                let mk = flow_makespan(&tmp, &job_pt, &mut comp);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
            }
            perm.insert(best_pos, j);
        }

        let n1 = (num_jobs.saturating_sub(1)) as f64;
        for (pos, &j) in perm.iter().enumerate() {
            job_flow_pref[j] = if n1 > 0.0 { 1.0 - (pos as f64) / n1 } else { 1.0 };
        }
    }

    let flow_w = if use_flow_pref {
        let t = ((flow_like - 0.82) / 0.18).clamp(0.0, 1.0);
        let base = (0.10 + 0.26 * t).clamp(0.10, 0.36);
        let flex_adj = (1.0 - 0.45 * high_flex).clamp(0.55, 1.0);
        base * flex_adj
    } else {
        0.0
    };

    let slack_base = (0.04 + 0.14 * jobshopness + 0.11 * high_flex).clamp(0.03, 0.22);

    let mut flow_route: Option<Vec<usize>> = None;
    let mut flow_pt_by_job: Option<Vec<Vec<u32>>> = None;
    let mut strict_route: Option<Vec<usize>> = None;
    if !product_ops.is_empty() {
        let common_len = product_ops[0].len();
        let mut ok = common_len > 0;

        for ops in &product_ops {
            if ops.len() != common_len {
                ok = false;
                break;
            }
        }

        if ok && flex_avg <= 1.25 {
            let mut route: Vec<usize> = Vec::with_capacity(common_len);
            for i in 0..common_len {
                let mut m0: Option<usize> = None;
                for p in 0..num_products {
                    let op = &product_ops[p][i];
                    if op.flex != 1 || op.machines.len() != 1 {
                        ok = false;
                        break;
                    }
                    let mid = op.machines[0].0;
                    if let Some(mm) = m0 {
                        if mm != mid {
                            ok = false;
                            break;
                        }
                    } else {
                        m0 = Some(mid);
                    }
                }
                if !ok {
                    break;
                }
                route.push(m0.unwrap());
            }

            if ok {
                strict_route = Some(route.clone());

                let mut pt_by_job: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
                for j in 0..num_jobs {
                    let prod = job_products[j];
                    let mut row = Vec::with_capacity(common_len);
                    for i in 0..common_len {
                        row.push(product_ops[prod][i].machines[0].1);
                    }
                    pt_by_job.push(row);
                }
                flow_route = Some(route);
                flow_pt_by_job = Some(pt_by_job);
            }
        }
    }

    let chaotic_like = high_flex > 0.85 && jobshopness > 0.75;

    Ok(Pre {
        job_products,
        job_ops_len,
        product_ops,
        product_suf_min,
        product_suf_avg,
        product_suf_bn,
        product_next_min,
        product_next_flex_inv,
        machine_load0,
        machine_scarcity,
        machine_weight,
        machine_best_pop,
        avg_machine_load,
        avg_machine_scarcity,
        avg_op_min,
        horizon,
        time_scale,
        max_ops: max_ops.max(1),
        max_job_avg_work: max_job_avg_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        flex_avg,
        flex_factor,
        hi_flex,
        high_flex,
        flow_like,
        flow_w,
        job_flow_pref,
        jobshopness,
        bn_focus,
        load_cv,
        slack_base,
        total_ops,
        chaotic_like,
        flow_route,
        flow_pt_by_job,
        strict_route,
    })
}
}

pub mod infra_shared {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>(); let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_simple_greedy_baseline_weighted(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    let base_weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for w in base_weights {
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>();
        let w = [
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(-1.0..0.5),
            rng.gen_range(-1.0..1.0),
        ];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_greedy_weighted(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    weights: [f64; 4], mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    let cand = GCandidate {
                        job: j,
                        priority,
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn run_greedy_rule(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let priority = match rule {
                        GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                        GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                        GreedyRule::LongestProc => pt as f64,
                    };
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let cand = GCandidate {
                        job: j,
                        priority: match rule {
                            GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                            GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                            GreedyRule::LongestProc => pt as f64,
                        },
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let n = job_offsets[num_jobs];
    if n == 0 { return Err(anyhow!("No operations")); }
    let mut node_machine = vec![0usize; n]; let mut node_pt = vec![0u32; n]; let mut node_job = vec![0usize; n]; let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution: job {} ops len mismatch", job)); }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx; let (m, st) = sol.job_schedule[job][op_idx];
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
            if m >= num_machines { return Err(anyhow!("Invalid solution: machine out of range")); }
            node_machine[id] = m; node_pt[id] = pt; node_job[id] = job; node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        machine_seq.push(per_machine[m].iter().map(|&(_, id)| id).collect());
    }
    let mut job_succ = vec![NONE_USIZE; n]; let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job]; let base = job_offsets[job];
        for k in 0..len { let id = base + k; if k + 1 < len { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
    }
    Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, node_machine, node_pt, node_job, node_op, machine_seq })
}

#[inline]
pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
    for &(m, pt) in &op.machines { if m == machine { return Some(pt); } }
    None
}

pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.indeg.clone_from_slice(&ds.indeg_job);
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.stack.clear();
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            buf.indeg[v] = buf.indeg[v].saturating_add(1);
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
    for i in 0..n {
        if buf.indeg[i] == 0 { buf.stack.push(i); }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while let Some(u) = buf.stack.pop() {
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] = buf.indeg[js].saturating_sub(1);
            if buf.indeg[js] == 0 { buf.stack.push(js); }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
            if buf.indeg[ms] == 0 { buf.stack.push(ms); }
        }
    }
    if processed != n { return None; }
    Some((mk, mk_node))
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j]; let mut v = Vec::with_capacity(len); let base = ds.job_offsets[j];
        for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
        job_schedule.push(v);
    }
    Ok(Solution { job_schedule })
}

pub fn critical_block_move_local_search_ex(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(swap_pos, swap_pos + 1); }
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

pub fn critical_block_move_local_search_ex_fjsp(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        let mut extracted = Vec::with_capacity(3);
        let num_extract = 3.min(blocks.len() * 2);
        for _ in 0..num_extract {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let ext_pos = bstart + ((pseed as usize) % block_len);
            if ext_pos < ds.machine_seq[m].len() {
                extracted.push((m, ds.machine_seq[m].remove(ext_pos)));
            }
        }
        for (m, node) in extracted {
            let desired = buf.start[node];
            let mut pos = find_insert_pos_by_start(&ds.machine_seq[m], &buf.start, desired);
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let offset = (pseed as usize) % 3;
            if (pseed >> 3) & 1 == 0 { pos = pos.saturating_add(offset); } else { pos = pos.saturating_sub(offset); }
            let m_len = ds.machine_seq[m].len();
            ds.machine_seq[m].insert(pos.min(m_len), node);
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

fn descent_phase(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    let max_shift = bend - bstart;
                    let mut shifts: [usize; 3] = [1, 2, max_shift];
                    for sh in shifts.iter_mut() { if *sh > max_shift { *sh = 0; } }
                    for &sh in &shifts {
                        if sh == 0 { continue; }
                        { let from = bstart; let to_after = bstart + sh; if from < seq.len() && to_after <= seq.len() { let tgt_idx = (bstart+sh).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); } }
                        { let from = bend; let to_after = bend - sh; push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let best2 = best_two_by_pt(op);
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let desired = buf.start[node];
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

fn descent_phase_fjsp(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut m_load = vec![0u32; ds.num_machines];
        for m in 0..ds.num_machines {
            if let Some(&last) = ds.machine_seq[m].last() {
                m_load[m] = buf.start[last].saturating_add(ds.node_pt[last]);
            }
        }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    let max_shift = bend - bstart;
                    if max_shift > 1 {
                        let to_after = bstart + max_shift;
                        if to_after <= seq.len() { let tgt_idx = (bstart+max_shift).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bstart, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); }
                        let to_after_bend = bend - max_shift;
                        push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bend, m_to: m, to: to_after_bend, new_pt: 0, score: buf.start[seq[bend]] }, top_cands);
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let desired = buf.start[node];
                        let mut best = (NONE_USIZE, INF, u64::MAX);
                        let mut second = (NONE_USIZE, INF, u64::MAX);
                        for &(m_to, pt) in &op.machines {
                            let end_time = m_load.get(m_to).copied().unwrap_or(INF).max(desired).saturating_add(pt);
                            let cost = (end_time as u64).saturating_mul(10000).saturating_add(pt as u64);
                            if cost < best.2 { second = best; best = (m_to, pt, cost); }
                            else if cost < second.2 { second = (m_to, pt, cost); }
                        }
                        let best2 = [(best.0, best.1), (second.0, second.1)];
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

#[inline]
pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len { return from.min(len.saturating_sub(1)); }
    let t = to_after_removal.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() { return false; } seq.swap(i, i + 1); true
}

#[inline]
pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = seq.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if start[seq[mid]] < desired_start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
pub fn apply_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, idx_to: usize, new_pt: u32) -> Option<(usize, u32, usize)> {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || idx_from >= ds.machine_seq[m_from].len() { return None; }
    let node = ds.machine_seq[m_from].remove(idx_from); let old_pt = ds.node_pt[node];
    ds.node_machine[node] = m_to; ds.node_pt[node] = new_pt;
    let ins = idx_to.min(ds.machine_seq[m_to].len()); ds.machine_seq[m_to].insert(ins, node);
    Some((node, old_pt, ins))
}

#[inline]
pub fn undo_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, ins_idx: usize, node: usize, old_pt: u32) -> bool {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || ins_idx >= ds.machine_seq[m_to].len() { return false; }
    let x = ds.machine_seq[m_to].remove(ins_idx);
    if x != node { let len_now = ds.machine_seq[m_to].len(); ds.machine_seq[m_to].insert(ins_idx.min(len_now), x); return false; }
    let ins_from = idx_from.min(ds.machine_seq[m_from].len());
    ds.machine_seq[m_from].insert(ins_from, node);
    ds.node_machine[node] = m_from; ds.node_pt[node] = old_pt; true
}

#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut r = [(NONE_USIZE, INF); 2];
    for &(m, pt) in &op.machines {
        if pt < r[0].1 { r[1] = r[0]; r[0] = (m, pt); }
        else if pt < r[1].1 { r[1] = (m, pt); }
    }
    r
}

pub fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in bias calc"))?;
            end_j = end_j.max(st.saturating_add(pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }

    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 1.2 * pre.high_flex + 0.6 * pre.jobshopness;
    Ok(completion.into_iter().map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0)).collect())
}

pub fn machine_penalty_from_solution(pre: &Pre, sol: &Solution, num_machines: usize) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in machine penalty"))?;
            let end = st.saturating_add(pt);
            if end > m_end[m] { m_end[m] = end; }
            m_sum[m] = m_sum[m].saturating_add(pt as u64);
            makespan = makespan.max(end);
        }
    }

    let mk = (makespan as f64).max(1.0);
    let total: u64 = m_sum.iter().copied().sum();
    let avg = ((total as f64) / (num_machines as f64).max(1.0)).max(1.0);

    let use_load = pre.high_flex > 0.35 || pre.jobshopness > 0.45;
    let w_load = if use_load {
        (0.20 + 0.30 * pre.high_flex + 0.12 * pre.jobshopness).clamp(0.18, 0.58)
    } else {
        0.0
    };
    let w_end = 1.0 - w_load;
    let exp = 2.0 + 1.2 * pre.high_flex + 0.55 * pre.jobshopness;

    let mut mp = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        let endn = (m_end[m] as f64 / mk).clamp(0.0, 1.0);
        let loadr = ((m_sum[m] as f64) / avg).max(0.0);
        let loadn = (loadr / (loadr + 1.0)).clamp(0.0, 1.0);
        let mix = (w_end * endn + w_load * loadn).clamp(0.0, 1.0);
        mp[m] = mix.powf(exp).clamp(0.0, 1.0);
    }
    Ok(mp)
}

pub fn route_pref_from_solution_lite(pre: &Pre, sol: &Solution, challenge: &Challenge) -> Result<RoutePrefLite> {
    let nm = challenge.num_machines;
    let np = challenge.product_processing_times.len();

    let mut counts: Vec<Vec<u16>> = Vec::with_capacity(np);
    let mut ops_len: Vec<usize> = Vec::with_capacity(np);
    for p in 0..np {
        let ol = challenge.product_processing_times[p].len();
        ops_len.push(ol);
        counts.push(vec![0u16; ol.saturating_mul(nm)]);
    }

    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        let ol = ops_len[product];
        for (op_idx, &(m, _st)) in sol.job_schedule[job].iter().enumerate() {
            if op_idx >= ol || m >= nm { continue; }
            let idx = op_idx * nm + m;
            counts[product][idx] = counts[product][idx].saturating_add(1);
        }
    }

    let mut rp: RoutePrefLite = Vec::with_capacity(np);
    for p in 0..np {
        let ol = ops_len[p];
        let denom_u32 = (challenge.jobs_per_product[p].max(1) as u32).max(1);
        let mut v: Vec<OpRoute> = Vec::with_capacity(ol);

        for op_idx in 0..ol {
            let base = op_idx * nm;
            let mut best_m = 0usize;
            let mut best_c = 0u16;
            let mut second_m = 0usize;
            let mut second_c = 0u16;

            for m in 0..nm {
                let c = counts[p][base + m];
                if c > best_c {
                    second_c = best_c; second_m = best_m;
                    best_c = c; best_m = m;
                } else if c > second_c && m != best_m {
                    second_c = c; second_m = m;
                }
            }

            let best_w = (((best_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;
            let second_w = (((second_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;

            v.push(OpRoute { best_m: best_m.min(255) as u8, best_w, second_m: second_m.min(255) as u8, second_w });
        }

        rp.push(v);
    }

    Ok(rp)
}

pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
}

pub fn neh_reentrant_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let ops = route.len(); if ops == 0 || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
    let mut jobs: Vec<usize> = (0..num_jobs).collect();
    jobs.sort_unstable_by(|&a, &b| { let sa: u32 = pt[a].iter().copied().sum(); let sb: u32 = pt[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) });
    let mut seq: Vec<usize> = Vec::with_capacity(num_jobs); let mut mready = vec![0u32; num_machines];
    for &j in &jobs {
        if seq.is_empty() { seq.push(j); continue; }
        let mut best_mk = u32::MAX; let mut best_pos = 0usize; let mut tmp = seq.clone();
        for pos in 0..=seq.len() {
            tmp.clear(); tmp.extend_from_slice(&seq[..pos]); tmp.push(j); tmp.extend_from_slice(&seq[pos..]);
            let mk = reentrant_makespan_local(&tmp, route, pt, &mut mready);
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        seq.insert(best_pos, j);
    }
    let mk = reentrant_makespan_local(&seq, route, pt, &mut mready);
    let sol = build_perm_solution_from_seq_local(&seq, route, pt, num_jobs, num_machines);
    Ok((sol, mk))
}

fn reentrant_makespan_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0); let mut mk = 0u32;
    for &j in seq { let row = &pt[j]; let mut prev = 0u32; for (op_idx, &m) in route.iter().enumerate() { let p = row[op_idx]; let st = prev.max(mready[m]); let end = st.saturating_add(p); mready[m] = end; prev = end; } if prev > mk { mk = prev; } }
    mk
}

fn build_perm_solution_from_seq_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], num_jobs: usize, num_machines: usize) -> Solution {
    let ops = route.len(); let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs]; let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs { continue; } let row = &pt[j]; let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines { break; }
            let p = row[op_idx]; let st = prev_end.max(machine_ready[m]); job_schedule[j].push((m, st)); let end = st.saturating_add(p); machine_ready[m] = end; prev_end = end;
        }
    }
    Solution { job_schedule }
}

#[inline]
pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
    let mut best = INF; let mut second = INF; let mut cnt_best = 0usize; let mut cnt_best_idle = 0usize;
    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best { second = best; best = end; cnt_best = 1; cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 }; }
        else if end == best { cnt_best += 1; if machine_avail[m] <= time { cnt_best_idle += 1; } }
        else if end < second { second = end; }
    }
    if cnt_best > 1 { second = best; }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].base_score < c.base_score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let min_s = top.last().unwrap().score; let n = top.len().min(8);
    let mut w = [0.0f64; 8]; let mut sum = 0.0f64;
    for i in 0..n { let d = (top[i].score - min_s) + 1e-9; let wi = d * d; w[i] = wi; sum += wi; }
    if !(sum > 0.0) { return top[rng.gen_range(0..top.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n { r -= w[i]; if r <= 0.0 { return top[i]; } }
    top[n - 1]
}}

pub mod hybrid_flow_shop {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;
use crate::{seeded_hasher, HashMap};

type DetCache = HashMap<(u64, usize, usize, usize, u8), Option<(Solution, u32)>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
}

#[allow(dead_code)]
#[inline]
fn slack_urgency_hfs(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline]
fn route_pref_bonus_hfs(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(rp) = rp else { return 0.0 };
    if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
    let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
    if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
}


#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
}

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);

    let mut sum = 0.0;
    for &r in rules.iter() {
        let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        sum += ww.max(0.0);
    }

    if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }

    let mut r=rng.gen::<f64>()*sum;
    for &rule in rules.iter() {
        let mk=rule_best[rule_idx(rule)]; let t=rule_tries[rule_idx(rule)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        r-=ww.max(0.0); if r<=0.0 { return rule; }
    }
    rules[rules.len()-1]
}

fn construct_solution_conflict_mode<const HAS_TARGET: bool, const HAS_JOB_BIAS: bool, const HAS_MACHINE_PENALTY: bool, const USE_ROUTE_PREF: bool>(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: u32,
    rng: &mut SmallRng, job_bias: &[f64], machine_penalty: &[f64],
    route_pref: Option<&RoutePrefLite>, horizon: f64, time_scale: f64,
) -> Result<(Solution, u32)> {
    let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
    let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
    let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
    let mut remaining_ops=pre.total_ops; let mut time=0u32;

    let avg_op_min_scale=pre.avg_op_min.max(1.0);
    let horizon_scale=horizon.max(1.0);
    let time_scale_scale=time_scale.max(1.0);
    let max_job_avg_work_scale=pre.max_job_avg_work.max(1e-9);
    let max_job_bn_scale=pre.max_job_bn.max(1e-9);
    let avg_machine_load_scale=pre.avg_machine_load.max(1e-9);
    let flex_factor_nonneg=pre.flex_factor.max(0.0);
    let bn_focus_u=if pre.bn_focus<=0.0{0.0}else{pre.bn_focus/(1.0+pre.bn_focus)};
    let slack_scale=(0.70 * pre.avg_op_min).max(1.0);
    let flex_regime=(pre.high_flex+pre.jobshopness).clamp(0.0,1.5);
    let job_product=&pre.job_products;

    let mut demand: Vec<u16>=vec![0u16;num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(12)).collect();
    let chaotic_like=pre.chaotic_like;
    let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
    let mut sum_work: u64=0;

    let mut job_ops_rem=vec![0usize;num_jobs];
    let mut job_op_ptr: Vec<*const OpInfo>=vec![std::ptr::null();num_jobs];
    let mut job_op_flex=vec![0usize;num_jobs];
    let mut job_op_has_machines=vec![false;num_jobs];
    let mut job_op_min_pt=vec![INF;num_jobs];
    let mut job_rem_min_raw=vec![0u64;num_jobs];
    let mut job_rem_min_u=vec![0.0;num_jobs];
    let mut job_rem_avg_u=vec![0.0;num_jobs];
    let mut job_bn_u=vec![0.0;num_jobs];
    let mut job_dens_u=vec![0.0;num_jobs];
    let mut job_next_u=vec![0.0;num_jobs];
    let mut job_flex_inv=vec![0.0;num_jobs];
    let mut job_flex_u=vec![0.0;num_jobs];

    let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut ready_pos: Vec<usize> = vec![usize::MAX; num_jobs];
    let mut in_ready: Vec<bool> = vec![false; num_jobs];
    let mut ready_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> =
        std::collections::BinaryHeap::new();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = vec![0usize; num_machines];
    for (i, p) in idle_pos.iter_mut().enumerate() { *p = i; }
    let mut machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>> =
        std::collections::BinaryHeap::new();
    let mut machine_gen: Vec<u32> = vec![0u32; num_machines];
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut touched_gen: Vec<u32> = vec![0u32; num_machines];
    let mut cur_gen: u32 = 0;
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

    for j in 0..num_jobs {
        let job_len=pre.job_ops_len[j];
        if job_len == 0 { continue; }
        let product=job_product[j];
        let op=&pre.product_ops[product][0];
        let rem_min_raw=pre.product_suf_min[product][0] as u64;
        let rem_min=rem_min_raw as f64;
        let rem_min_n=rem_min/horizon_scale;
        let rem_avg_n=pre.product_suf_avg[product][0]/max_job_avg_work_scale;
        let bn_n=pre.product_suf_bn[product][0]/max_job_bn_scale;
        let density_n=((rem_min/(job_len as f64).max(1.0))/avg_op_min_scale).clamp(0.0,4.0);
        let next_min_n=(pre.product_next_min[product][0] as f64)/horizon_scale;
        let next_term_raw=(0.55*next_min_n+0.45*pre.product_next_flex_inv[product][0])*(1.0+0.30*density_n*pre.high_flex);
        let flex_inv=1.0/(op.flex as f64).max(1.0);
        let flex_term=flex_inv*flex_factor_nonneg;

        job_ops_rem[j]=job_len;
        job_op_ptr[j]=op as *const OpInfo;
        job_op_flex[j]=op.flex as usize;
        job_op_has_machines[j]=!op.machines.is_empty();
        job_op_min_pt[j]=op.min_pt;
        job_rem_min_raw[j]=rem_min_raw;
        job_rem_min_u[j]=if rem_min_n<=0.0{0.0}else{rem_min_n/(1.0+rem_min_n)};
        job_rem_avg_u[j]=if rem_avg_n<=0.0{0.0}else{rem_avg_n/(1.0+rem_avg_n)};
        job_bn_u[j]=if bn_n<=0.0{0.0}else{bn_n/(1.0+bn_n)};
        job_dens_u[j]=if density_n<=0.0{0.0}else{density_n/(1.0+density_n)};
        job_next_u[j]=if next_term_raw<=0.0{0.0}else{next_term_raw/(1.0+next_term_raw)};
        job_flex_inv[j]=flex_inv;
        job_flex_u[j]=if flex_term<=0.0{0.0}else{flex_term/(1.0+flex_term)};

        in_ready[j] = true;
        ready_pos[j] = ready_jobs.len();
        ready_jobs.push(j);
    }

    let advance_frontier = |time: &mut u32,
                                    ready_jobs: &mut Vec<usize>,
                                    ready_pos: &mut [usize],
                                    in_ready: &mut [bool],
                                    ready_heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>>,
                                    job_next_op: &[usize],
                                    job_ready_time: &[u32],
                                    idle_machines: &mut Vec<usize>,
                                    idle_pos: &mut [usize],
                                    machine_heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>>,
                                    machine_avail: &[u32],
                                    machine_gen: &[u32]| -> Option<u32> {
        while let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() {
            if t > *time { break; }
            machine_heap.pop();
            if m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX { continue; }
            idle_pos[m] = idle_machines.len();
            idle_machines.push(m);
        }

        while let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() {
            if t > *time { break; }
            ready_heap.pop();
            if j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
            in_ready[j] = true;
            ready_pos[j] = ready_jobs.len();
            ready_jobs.push(j);
        }

        let next_machine_time = loop {
            let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() else { break None; };
            if t <= *time || m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX {
                machine_heap.pop();
                continue;
            }
            break Some(t);
        };

        let next_ready_time = loop {
            let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() else { break None; };
            if t <= *time || j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t {
                ready_heap.pop();
                continue;
            }
            break Some(t);
        };

        let nt = match (next_machine_time, next_ready_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return None,
        };
        *time = nt;

        while let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() {
            if t > nt { break; }
            machine_heap.pop();
            if m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX { continue; }
            idle_pos[m] = idle_machines.len();
            idle_machines.push(m);
        }

        while let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() {
            if t > nt { break; }
            ready_heap.pop();
            if j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
            in_ready[j] = true;
            ready_pos[j] = ready_jobs.len();
            ready_jobs.push(j);
        }

        Some(nt)
    };

    while remaining_ops > 0 {
        if idle_machines.is_empty() {
            advance_frontier(
                &mut time,
                &mut ready_jobs,
                &mut ready_pos,
                &mut in_ready,
                &mut ready_heap,
                &job_next_op,
                &job_ready_time,
                &mut idle_machines,
                &mut idle_pos,
                &mut machine_heap,
                &machine_avail,
                &machine_gen,
            ).ok_or_else(||anyhow!("Stalled"))?;
            continue;
        }

        touched_machines.clear();
        cur_gen=cur_gen.wrapping_add(1);
        if cur_gen==0{
            touched_gen.fill(0);
            cur_gen=1;
        }
        let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
        let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};

        for &job in &ready_jobs {
            let op_ptr=job_op_ptr[job];
            if op_ptr.is_null(){continue;}
            let op=unsafe{&*op_ptr};
            let op_flex=job_op_flex[job];
            if op_flex==0||!job_op_has_machines[job]||job_op_min_pt[job]>=INF{continue;}
            let (best_end,second_end,best_cnt_total,best_cnt_idle)=best_second_and_counts(time,&machine_avail,op);
            if best_end>=INF||best_cnt_idle==0{continue;}

            let _op_idx=job_next_op[job];
            let _ops_rem=job_ops_rem[job]; let jb=if HAS_JOB_BIAS { job_bias[job] } else { 0.0 };
            let flex_inv=job_flex_inv[job]; let scarcity_urg=1.0/(best_cnt_total as f64).max(1.0);
            let regret=if second_end>=INF{pre.avg_op_min*2.6}else{(second_end-best_end) as f64};
            let regn=(regret/avg_op_min_scale).clamp(0.0,6.0); let rigidity=(0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
            let exact_only=op_flex<2||((flex_regime<0.34)&&!pre.chaotic_like)||(k==0&&progress<0.10&&best_cnt_total>1);
            let near_band=if exact_only{
                0u32
            }else{
                let base=((pre.avg_op_min*(0.35+0.30*pre.high_flex+0.28*pre.jobshopness+0.16*progress)).round() as u32).max(1);
                let regret_cap=if second_end>=INF{base.max(op.min_pt/2)}else{second_end.saturating_sub(best_end).min(base.max(1))};
                regret_cap.max(1)
            };
            let allow_end=best_end.saturating_add(near_band);
            let detour_mult=if best_cnt_total<=1{0.72}else if best_cnt_total==2{0.86}else{1.0};

            let flow_term=pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
            let slack_u=if HAS_TARGET {
                let lb=(time as u64).saturating_add(job_rem_min_raw[job]);
                let slack=(target_mk as i64) - (lb as i64);
                let pos=(slack.max(0) as f64) / slack_scale; let neg=((-slack).max(0) as f64) / slack_scale;
                (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
            } else { 0.0 };

            let rem_min_u=job_rem_min_u[job];
            let rem_avg_u=job_rem_avg_u[job];
            let bn_u=job_bn_u[job];
            let reg_u=if regn<=0.0{0.0}else{regn/(1.0+regn)};
            let dens_u=job_dens_u[job];
            let next_u=job_next_u[job];
            let end_n=(best_end as f64)/time_scale_scale;
            let end_u=if end_n<=0.0{0.0}else{end_n/(1.0+end_n)};
            let flex_u=job_flex_u[job];
            let sat_scarcity=if scarcity_urg<=0.0{0.0}else{scarcity_urg/(1.0+scarcity_urg)};
            let scarce_slack=scarcity_urg*slack_u;
            let scarce_reg=scarcity_urg*reg_u;
            let prog_gate=if progress<=0.0{0.0}else{progress/(1.0+progress)};
            let base_bias0=jb+flow_term;

            for &(m,pt) in &op.machines {
                if idle_pos[m]==usize::MAX{continue;}
                let end=time.saturating_add(pt); if end>allow_end{continue;}
                if touched_gen[m]!=cur_gen{
                    touched_gen[m]=cur_gen;
                    touched_machines.push(m);
                    demand[m]=0;
                    raw_by_machine[m].clear();
                }
                demand[m]=demand[m].saturating_add(1);
                let mp=if HAS_MACHINE_PENALTY { machine_penalty[m] } else { 0.0 }; let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                let load_n=machine_load[m]/avg_machine_load_scale;
                let proc_n=(pt as f64)/avg_op_min_scale;
                let mpen=mp.clamp(0.0,1.0);
                let pop=pre.machine_best_pop[m].clamp(0.0,1.2);
                let pop_pen=if pre.chaotic_like&&op_flex>=2{
                    (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor
                }else{
                    0.0
                };
                let load_u=if load_n<=0.0{0.0}else{load_n/(1.0+load_n)};
                let proc_u=if proc_n<=0.0{0.0}else{proc_n/(1.0+proc_n)};
                let mpen_u=if mpen<=0.0{0.0}else{mpen/(1.0+mpen)};
                let end_gap=end.saturating_sub(best_end);
                let end_gap_n=(end_gap as f64)/avg_op_min_scale;
                let end_gap_u=if end_gap_n<=0.0{0.0}else{end_gap_n/(1.0+end_gap_n)};
                let preserve_pen=if op_flex>=2&&flex_regime>0.35{
                    (0.06+0.12*flex_regime.min(1.0)+0.08*progress)*pop*(1.0-flex_inv)
                }else{
                    0.0
                };
                let detour_pen=if end_gap>0{
                    end_gap_u*(0.55+0.45*rigidity+0.18*sat_scarcity)*detour_mult
                }else{
                    0.0
                };
                let detour_credit=if end_gap>0{
                    (0.08+0.10*pre.jobshopness+0.08*pre.high_flex)*(1.0-load_u)*(1.0-pop.min(1.0))
                }else{
                    0.0
                };
                let scarce_machine_bonus=if op_flex>=2&&flex_regime>0.35{
                    (0.03+0.06*progress)*(1.0-pop.min(1.0))*sat_scarcity
                }else{
                    0.0
                };
                let base_bias=base_bias0+jitter;
                let base0=match rule {
                    Rule::CriticalPath => {
                        let chain=rem_min_u*(1.0+next_u);
                        let urgent=scarce_slack*(1.0+scarce_reg*prog_gate);
                        chain+urgent+base_bias-end_u-pop_pen
                    }
                    Rule::MostWork => {
                        let work=rem_avg_u*(1.0+dens_u);
                        let smooth=work*(1.0+load_u);
                        smooth+base_bias-end_u-pop_pen
                    }
                    Rule::LeastFlex => {
                        let rigid=flex_u*(1.0+sat_scarcity);
                        rigid+rem_min_u+next_u+base_bias-end_u-pop_pen
                    }
                    Rule::ShortestProc => {
                        let short=0.0-proc_u;
                        short+rem_min_u*(1.0+next_u)+sat_scarcity+base_bias-end_u-pop_pen
                    }
                    Rule::Regret => {
                        let regret_focus=reg_u*(1.0+sat_scarcity)*(1.0+prog_gate);
                        regret_focus+rem_min_u+next_u+base_bias-end_u-pop_pen
                    }
                    Rule::EndTight => {
                        let tight=scarce_slack*(1.0+scarce_reg);
                        let chain=rem_min_u*(1.0+prog_gate)*(1.0+next_u);
                        let penal=end_u*(1.0+prog_gate)+proc_u+mpen_u*pre.flex_factor;
                        chain+tight+base_bias-penal-pop_pen
                    }
                    Rule::BnHeavy => {
                        let bn_focus=bn_u*(1.0+dens_u)*(1.0+bn_focus_u);
                        let chain=rem_min_u*(1.0+next_u);
                        let penal=end_u+proc_u+load_u*pre.flex_factor+mpen_u*pre.flex_factor;
                        bn_focus+chain+scarce_slack+reg_u+flex_u+base_bias-penal-pop_pen
                    }
                    Rule::Adaptive => {
                        let js=pre.jobshopness;
                        let fl=1.0-js;
                        if js>=fl{
                            let hard=reg_u*(1.0+scarce_reg)+flex_u+rem_min_u*(1.0+next_u);
                            hard+base_bias-(end_u+mpen_u*pre.flex_factor)-pop_pen
                        }else{
                            let flow=rem_avg_u*(1.0+dens_u)+(0.0-proc_u)+slack_u;
                            flow+base_bias-(end_u+load_u*pre.flex_factor)-pop_pen
                        }
                    }
                    Rule::FlexBalance => {
                        let flexible=flex_u*(1.0+sat_scarcity);
                        let chain=(rem_avg_u+rem_min_u)*(1.0+next_u);
                        let penal=end_u+load_u*pre.flex_factor+mpen_u*(1.0+pre.flex_factor);
                        flexible+chain+base_bias-penal-pop_pen
                    }
                };
                let base=base0+scarce_machine_bonus+detour_credit-detour_pen-preserve_pen;
                push_top_k_raw(&mut raw_by_machine[m],RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn},cap_per_machine);
            }
        }

        let denom=(idle_machines.len() as f64).max(1.0);
        let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
        let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};

        let mut best: Option<Cand>=None; top.clear();
        if touched_machines.len()>1{touched_machines.sort_unstable();}
        for &m in &touched_machines {
            let dem=demand[m] as f64; if dem<=0.0||raw_by_machine[m].is_empty(){continue;}
            let dem_n=((dem-1.0)/denom).clamp(0.0,2.5);
            let bal_pen=if chaotic_like&&bal_w>0.0{let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n}else{0.0};
            let load_factor = (machine_load[m] as f64) / avg_machine_load_scale;
            let cs = conflict_scale * (1.0 - 0.30 * load_factor).clamp(0.40, 1.60);
            for rc in &raw_by_machine[m] {
                let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                let mut boost=conflict_w * cs * dem_n * (1.15*rig+0.85*regc);
                if chaotic_like{boost=boost.max(-0.26);}
                let c=Cand{job:rc.job,machine:rc.machine,pt:rc.pt,score:rc.base_score+boost+bal_pen};
                if k==0{if best.map_or(true,|bb|c.score>bb.score){best=Some(c);}}else{push_top_k(&mut top,c,k);}
            }
        }

        let chosen=if k==0{
            match best{
                Some(c)=>c,
                None=>{
                    advance_frontier(
                        &mut time,
                        &mut ready_jobs,
                        &mut ready_pos,
                        &mut in_ready,
                        &mut ready_heap,
                        &job_next_op,
                        &job_ready_time,
                        &mut idle_machines,
                        &mut idle_pos,
                        &mut machine_heap,
                        &machine_avail,
                        &machine_gen,
                    ).ok_or_else(||anyhow!("Stalled"))?;
                    continue;
                }
            }
        }else{
            if top.is_empty(){
                advance_frontier(
                    &mut time,
                    &mut ready_jobs,
                    &mut ready_pos,
                    &mut in_ready,
                    &mut ready_heap,
                    &job_next_op,
                    &job_ready_time,
                    &mut idle_machines,
                    &mut idle_pos,
                    &mut machine_heap,
                    &machine_avail,
                    &machine_gen,
                ).ok_or_else(||anyhow!("Stalled"))?;
                continue;
            }
            if USE_ROUTE_PREF{
                let rp=route_pref.unwrap();
                let mut best_rb: Option<f64>=None;
                let mut best_idx=0usize;
                let mut keep_cnt=0usize;
                for (i,c) in top.iter().enumerate(){
                    let job=c.job;
                    if job_op_ptr[job].is_null(){continue;}
                    let op_idx=job_next_op[job];
                    let product=job_product[job];
                    let rb=route_pref_bonus_hfs(Some(rp),product,op_idx,c.machine);
                    match best_rb{
                        None=>{best_rb=Some(rb);best_idx=i;keep_cnt=1;}
                        Some(b)=>{
                            if rb>b{best_rb=Some(rb);best_idx=i;keep_cnt=1;}
                            else if rb==b{keep_cnt+=1;}
                        }
                    }
                }
                if keep_cnt==0{
                    choose_from_top_weighted(rng,&top)
                }else if keep_cnt==1{
                    top[best_idx]
                }else{
                    let best_rb=best_rb.unwrap();
                    let mut write=0usize;
                    for i in 0..top.len(){
                        let c=top[i];
                        let job=c.job;
                        if job_op_ptr[job].is_null(){continue;}
                        let op_idx=job_next_op[job];
                        let product=job_product[job];
                        if route_pref_bonus_hfs(Some(rp),product,op_idx,c.machine)==best_rb{
                            top[write]=c;
                            write+=1;
                        }
                    }
                    top.truncate(write);
                    choose_from_top_weighted(rng,&top)
                }
            }else{
                choose_from_top_weighted(rng,&top)
            }
        };

        let job=chosen.job; let machine=chosen.machine; let pt=chosen.pt;
        let product=job_product[job]; let _op_idx=job_next_op[job]; let op=unsafe{&*job_op_ptr[job]};
        let end_time=time.saturating_add(pt);

        in_ready[job] = false;
        let pos = ready_pos[job];
        if pos < ready_jobs.len() && ready_jobs[pos] == job {
            ready_jobs.swap_remove(pos);
            if pos < ready_jobs.len() {
                let moved = ready_jobs[pos];
                ready_pos[moved] = pos;
            }
        }
        ready_pos[job] = usize::MAX;

        let machine_pos = idle_pos[machine];
        if machine_pos < idle_machines.len() {
            idle_machines.swap_remove(machine_pos);
            if machine_pos < idle_machines.len() {
                let moved = idle_machines[machine_pos];
                idle_pos[moved] = machine_pos;
            }
        }
        idle_pos[machine] = usize::MAX;

        job_schedule[job].push((machine,time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;

        if job_next_op[job] < pre.job_ops_len[job] {
            let new_op_idx=job_next_op[job];
            let next_op=&pre.product_ops[product][new_op_idx];
            let rem_min_raw=pre.product_suf_min[product][new_op_idx] as u64;
            let rem_min=rem_min_raw as f64;
            let rem_min_n=rem_min/horizon_scale;
            let rem_avg_n=pre.product_suf_avg[product][new_op_idx]/max_job_avg_work_scale;
            let bn_n=pre.product_suf_bn[product][new_op_idx]/max_job_bn_scale;
            let ops_rem=pre.job_ops_len[job]-new_op_idx;
            let density_n=((rem_min/(ops_rem as f64).max(1.0))/avg_op_min_scale).clamp(0.0,4.0);
            let next_min_n=(pre.product_next_min[product][new_op_idx] as f64)/horizon_scale;
            let next_term_raw=(0.55*next_min_n+0.45*pre.product_next_flex_inv[product][new_op_idx])*(1.0+0.30*density_n*pre.high_flex);
            let flex_inv=1.0/(next_op.flex as f64).max(1.0);
            let flex_term=flex_inv*flex_factor_nonneg;

            job_ops_rem[job]=ops_rem;
            job_op_ptr[job]=next_op as *const OpInfo;
            job_op_flex[job]=next_op.flex as usize;
            job_op_has_machines[job]=!next_op.machines.is_empty();
            job_op_min_pt[job]=next_op.min_pt;
            job_rem_min_raw[job]=rem_min_raw;
            job_rem_min_u[job]=if rem_min_n<=0.0{0.0}else{rem_min_n/(1.0+rem_min_n)};
            job_rem_avg_u[job]=if rem_avg_n<=0.0{0.0}else{rem_avg_n/(1.0+rem_avg_n)};
            job_bn_u[job]=if bn_n<=0.0{0.0}else{bn_n/(1.0+bn_n)};
            job_dens_u[job]=if density_n<=0.0{0.0}else{density_n/(1.0+density_n)};
            job_next_u[job]=if next_term_raw<=0.0{0.0}else{next_term_raw/(1.0+next_term_raw)};
            job_flex_inv[job]=flex_inv;
            job_flex_u[job]=if flex_term<=0.0{0.0}else{flex_term/(1.0+flex_term)};

            if end_time==time{
                in_ready[job]=true;
                ready_pos[job]=ready_jobs.len();
                ready_jobs.push(job);
            }else{
                ready_heap.push(std::cmp::Reverse((end_time, job)));
            }
        } else {
            job_ops_rem[job]=0;
            job_op_ptr[job]=std::ptr::null();
            job_op_flex[job]=0;
            job_op_has_machines[job]=false;
            job_op_min_pt[job]=INF;
            job_rem_min_raw[job]=0;
            job_rem_min_u[job]=0.0;
            job_rem_avg_u[job]=0.0;
            job_bn_u[job]=0.0;
            job_dens_u[job]=0.0;
            job_next_u[job]=0.0;
            job_flex_inv[job]=0.0;
            job_flex_u[job]=0.0;
        }

        machine_gen[machine]=machine_gen[machine].wrapping_add(1);
        if end_time==time{
            idle_pos[machine]=idle_machines.len();
            idle_machines.push(machine);
        }else{
            machine_heap.push(std::cmp::Reverse((end_time, machine, machine_gen[machine])));
        }

        if chaotic_like{machine_work[machine]=machine_work[machine].saturating_add(pt as u64);sum_work=sum_work.saturating_add(pt as u64);}
        if op.min_pt<INF&&op.flex>0&&!op.machines.is_empty(){let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0);if delta>0.0{for &(mm,_) in &op.machines{let v=machine_load[mm]-delta;machine_load[mm]=if v>0.0{v}else{0.0};}}}
    }

    let mk=machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution{job_schedule},mk))
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64, horizon: f64, time_scale: f64,
) -> Result<(Solution, u32)> {
    let empty: &[f64] = &[];
    let routed = if route_w > 0.0 { route_pref } else { None };

    if let Some(tgt) = target_mk {
        if let Some(jb) = job_bias {
            if let Some(mp) = machine_penalty {
                if let Some(rp) = routed {
                    construct_solution_conflict_mode::<true,true,true,true>(challenge,pre,rule,k,tgt,rng,jb,mp,Some(rp),horizon,time_scale)
                } else {
                    construct_solution_conflict_mode::<true,true,true,false>(challenge,pre,rule,k,tgt,rng,jb,mp,None,horizon,time_scale)
                }
            } else if let Some(rp) = routed {
                construct_solution_conflict_mode::<true,true,false,true>(challenge,pre,rule,k,tgt,rng,jb,empty,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<true,true,false,false>(challenge,pre,rule,k,tgt,rng,jb,empty,None,horizon,time_scale)
            }
        } else if let Some(mp) = machine_penalty {
            if let Some(rp) = routed {
                construct_solution_conflict_mode::<true,false,true,true>(challenge,pre,rule,k,tgt,rng,empty,mp,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<true,false,true,false>(challenge,pre,rule,k,tgt,rng,empty,mp,None,horizon,time_scale)
            }
        } else if let Some(rp) = routed {
            construct_solution_conflict_mode::<true,false,false,true>(challenge,pre,rule,k,tgt,rng,empty,empty,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<true,false,false,false>(challenge,pre,rule,k,tgt,rng,empty,empty,None,horizon,time_scale)
        }
    } else if let Some(jb) = job_bias {
        if let Some(mp) = machine_penalty {
            if let Some(rp) = routed {
                construct_solution_conflict_mode::<false,true,true,true>(challenge,pre,rule,k,0,rng,jb,mp,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<false,true,true,false>(challenge,pre,rule,k,0,rng,jb,mp,None,horizon,time_scale)
            }
        } else if let Some(rp) = routed {
            construct_solution_conflict_mode::<false,true,false,true>(challenge,pre,rule,k,0,rng,jb,empty,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<false,true,false,false>(challenge,pre,rule,k,0,rng,jb,empty,None,horizon,time_scale)
        }
    } else if let Some(mp) = machine_penalty {
        if let Some(rp) = routed {
            construct_solution_conflict_mode::<false,false,true,true>(challenge,pre,rule,k,0,rng,empty,mp,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<false,false,true,false>(challenge,pre,rule,k,0,rng,empty,mp,None,horizon,time_scale)
        }
    } else if let Some(rp) = routed {
        construct_solution_conflict_mode::<false,false,false,true>(challenge,pre,rule,k,0,rng,empty,empty,Some(rp),horizon,time_scale)
    } else {
        construct_solution_conflict_mode::<false,false,false,false>(challenge,pre,rule,k,0,rng,empty,empty,None,horizon,time_scale)
    }
}

#[derive(Clone)]
struct EliteParams {
    jb: Vec<f64>,
    mp: Vec<f64>,
    rp: RoutePrefLite,
    score: u32,
}

fn normalize_elite(elite: &mut Vec<EliteParams>, cap: usize) {
    if elite.is_empty() {
        return;
    }
    if elite.len() <= 1 {
        if elite.len() > cap {
            elite.truncate(cap);
        }
        return;
    }

    #[inline]
    fn elite_sig64(e: &EliteParams) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (i, v) in e.jb.iter().enumerate() {
            (i as u32).hash(&mut hasher);
            v.to_bits().hash(&mut hasher);
        }
        for (i, v) in e.mp.iter().enumerate() {
            (i as u32).hash(&mut hasher);
            v.to_bits().hash(&mut hasher);
        }
        for (p, ops) in e.rp.iter().enumerate() {
            for (o, r) in ops.iter().enumerate() {
                (p as u32).hash(&mut hasher);
                (o as u32).hash(&mut hasher);
                r.best_m.hash(&mut hasher);
                r.second_m.hash(&mut hasher);
                r.best_w.hash(&mut hasher);
                r.second_w.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    #[inline]
    fn dist(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    let mut view: Vec<(u32, u64, usize)> = elite
        .iter()
        .enumerate()
        .map(|(idx, e)| (e.score, elite_sig64(e), idx))
        .collect();
    view.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let best_sig = view[0].1;

    let cand: Vec<(u32, u32, u64, usize)> = view
        .iter()
        .map(|(s, sg, idx)| (*s, dist(*sg, best_sig), *sg, *idx))
        .collect();

    let mut keep = vec![true; cand.len()];
    for i in 0..cand.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..cand.len() {
            if i == j || !keep[i] {
                continue;
            }
            let (si, di, sgi, _) = cand[i];
            let (sj, dj, sgj, _) = cand[j];

            let no_worse = sj <= si && dj >= di;
            let strictly_better = sj < si || dj > di;
            if no_worse && strictly_better {
                if sj == si && dj == di && sgj > sgi {
                    continue;
                }
                keep[i] = false;
            }
        }
    }

    let mut skyline: Vec<(u32, u64, usize)> = Vec::new();
    for (i, k) in keep.iter().copied().enumerate() {
        if k {
            let (s, _d, sg, idx) = cand[i];
            skyline.push((s, sg, idx));
        }
    }
    skyline.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut selected: Vec<(u32, u64, usize)> = Vec::new();
    selected.push(view[0]);

    #[inline]
    fn already_selected(selected: &[(u32, u64, usize)], idx: usize) -> bool {
        selected.iter().any(|x| x.2 == idx)
    }

    while selected.len() < cap && selected.len() < elite.len() {
        let mut best_pick: Option<(u32, u64, usize, u32)> = None;
        for &(s, sg, idx) in &skyline {
            if already_selected(&selected, idx) {
                continue;
            }
            let mut md = u32::MAX;
            for &(_ss, ssg, _ii) in &selected {
                md = md.min(dist(sg, ssg));
            }
            match best_pick {
                None => best_pick = Some((s, sg, idx, md)),
                Some((bs, bsg, _bidx, bmd)) => {
                    if md > bmd || (md == bmd && (s < bs || (s == bs && sg < bsg))) {
                        best_pick = Some((s, sg, idx, md));
                    }
                }
            }
        }
        if let Some((s, sg, idx, _)) = best_pick {
            selected.push((s, sg, idx));
        } else {
            break;
        }
    }

    while selected.len() < cap && selected.len() < elite.len() {
        let mut best_pick: Option<(u32, u64, usize, u32)> = None;
        for &(s, sg, idx) in &view {
            if already_selected(&selected, idx) {
                continue;
            }
            let mut md = u32::MAX;
            for &(_ss, ssg, _ii) in &selected {
                md = md.min(dist(sg, ssg));
            }
            match best_pick {
                None => best_pick = Some((s, sg, idx, md)),
                Some((bs, bsg, _bidx, bmd)) => {
                    if md > bmd || (md == bmd && (s < bs || (s == bs && sg < bsg))) {
                        best_pick = Some((s, sg, idx, md));
                    }
                }
            }
        }
        if let Some((s, sg, idx, _)) = best_pick {
            selected.push((s, sg, idx));
        } else {
            break;
        }
    }

    let mut new_elite: Vec<EliteParams> = selected
        .into_iter()
        .map(|(_s, _sg, idx)| elite[idx].clone())
        .collect();
    new_elite.sort_by_key(|e| e.score);

    if new_elite.len() > cap {
        new_elite.truncate(cap);
    }
    *elite = new_elite;
}

fn pick_elite_idx(rng: &mut SmallRng, elite: &[EliteParams]) -> usize {
    let len = elite.len(); if len <= 1 { return 0; }
    let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
    if elite[a].score <= elite[b].score { a } else { b }
}

fn pick_top_idx(rng: &mut SmallRng, top: &[(Solution, u32)]) -> usize {
    let len = top.len(); if len <= 1 { return 0; }
    let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
    if top[a].1 <= top[b].1 { a } else { b }
}

#[inline]
fn solution_sig64(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, _t)) in ops.iter().enumerate() {
            std::hash::Hash::hash(&(j, o, *m), &mut hasher);
        }
    }
    std::hash::Hasher::finish(&hasher)
}

#[inline]
fn pick_diverse_top_solution(best: &Solution, top: &[(Solution, u32)], scan: usize) -> Option<usize> {
    if top.is_empty() || scan == 0 { return None; }
    let sig_best = solution_sig64(best);
    let lim = scan.min(top.len());
    let mut best_idx: Option<usize> = None;
    let mut best_dist: u32 = 0;
    for i in 0..lim {
        let sig_i = solution_sig64(&top[i].0);
        let dist = (sig_best ^ sig_i).count_ones();
        if best_idx.is_none() || dist > best_dist {
            best_idx = Some(i);
            best_dist = dist;
        }
    }
    best_idx
}

#[inline]
fn exact_solution_sig64(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, t)) in ops.iter().enumerate() {
            std::hash::Hash::hash(&(j, o, *m, *t), &mut hasher);
        }
    }
    std::hash::Hasher::finish(&hasher)
}

#[inline]
fn seed_sig_distance(exact_a: u64, coarse_a: u64, exact_b: u64, coarse_b: u64) -> u32 {
    let d_exact = (exact_a ^ exact_b).count_ones();
    let d_coarse = (coarse_a ^ coarse_b).count_ones();
    d_exact.saturating_add(d_coarse.saturating_mul(2))
}

fn select_seed_portfolio_indices(
    best_sol: Option<&Solution>,
    best_mk: u32,
    top: &[(Solution, u32)],
    scan_limit: usize,
    seed_cap: usize,
    quality_band: u32,
) -> Vec<usize> {
    if top.is_empty() || seed_cap == 0 {
        return Vec::new();
    }
    let scan = scan_limit.min(top.len());
    let best_exact = best_sol.map(exact_solution_sig64);
    let best_coarse = best_sol.map(solution_sig64);
    let mut exact_sigs: Vec<u64> = Vec::with_capacity(scan);
    let mut coarse_sigs: Vec<u64> = Vec::with_capacity(scan);
    for i in 0..scan {
        exact_sigs.push(exact_solution_sig64(&top[i].0));
        coarse_sigs.push(solution_sig64(&top[i].0));
    }

    let build_pool = |limit_mk: Option<u32>| -> Vec<usize> {
        let mut best_by_sig: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(scan.saturating_mul(2).max(1));
        for i in 0..scan {
            let keep = match limit_mk {
                Some(lim) => top[i].1 <= lim || best_exact.map_or(false, |sig| exact_sigs[i] == sig),
                None => true,
            };
            if !keep {
                continue;
            }
            match best_by_sig.get_mut(&exact_sigs[i]) {
                Some(best_i) => {
                    if top[i].1 < top[*best_i].1 {
                        *best_i = i;
                    }
                }
                None => {
                    best_by_sig.insert(exact_sigs[i], i);
                }
            }
        }
        let mut pool: Vec<usize> = best_by_sig.into_values().collect();
        pool.sort_by(|&a, &b| top[a].1.cmp(&top[b].1).then_with(|| a.cmp(&b)));
        pool
    };

    let mut pool = build_pool(Some(best_mk.saturating_add(quality_band.max(1))));
    if pool.len() < seed_cap.min(4) {
        pool = build_pool(None);
    }
    if pool.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<usize> = Vec::with_capacity(seed_cap.min(pool.len()));
    let incumbent_idx = best_exact
        .and_then(|sig| pool.iter().copied().find(|&i| exact_sigs[i] == sig))
        .unwrap_or(pool[0]);
    out.push(incumbent_idx);

    if out.len() < seed_cap {
        if let Some(idx) = pool.iter().copied().find(|&i| i != incumbent_idx) {
            out.push(idx);
        }
    }

    if out.len() < seed_cap {
        let ref_exact = best_exact.unwrap_or(exact_sigs[incumbent_idx]);
        let ref_coarse = best_coarse.unwrap_or(coarse_sigs[incumbent_idx]);
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) {
                continue;
            }
            let dist = seed_sig_distance(ref_exact, ref_coarse, exact_sigs[i], coarse_sigs[i]);
            match best_pick {
                None => best_pick = Some((i, dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if dist > bd || (dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        }
    }

    if out.len() < seed_cap {
        let ref_exact = best_exact.unwrap_or(exact_sigs[incumbent_idx]);
        let ref_coarse = best_coarse.unwrap_or(coarse_sigs[incumbent_idx]);
        let close_limit = best_mk.saturating_add((quality_band / 2).max(1));
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) || top[i].1 > close_limit {
                continue;
            }
            let dist = seed_sig_distance(ref_exact, ref_coarse, exact_sigs[i], coarse_sigs[i]);
            match best_pick {
                None => best_pick = Some((i, dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if dist > bd || (dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        }
    }

    let target_len = seed_cap.min(pool.len());
    while out.len() < target_len {
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) {
                continue;
            }
            let mut min_dist = u32::MAX;
            for &j in &out {
                min_dist = min_dist.min(seed_sig_distance(exact_sigs[i], coarse_sigs[i], exact_sigs[j], coarse_sigs[j]));
            }
            match best_pick {
                None => best_pick = Some((i, min_dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if min_dist > bd || (min_dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, min_dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        } else {
            break;
        }
    }
    out
}

fn maybe_add_elite(elite: &mut Vec<EliteParams>, cand: EliteParams, cap: usize) {
    if elite.is_empty() { elite.push(cand); return; }
    if elite.len() < cap { elite.push(cand); normalize_elite(elite, cap); return; }
    normalize_elite(elite, cap);
    let worst = elite.last().map(|e| e.score).unwrap_or(u32::MAX);
    if cand.score < worst {
        if let Some(last) = elite.last_mut() { *last = cand; } else { elite.push(cand); }
        normalize_elite(elite, cap);
    }
}

fn commit_best(save_solution: &dyn Fn(&Solution) -> Result<()>, best_mk: &mut u32, best_sol: &mut Option<Solution>, sol: &Solution, mk: u32) -> Result<bool> {
    if mk < *best_mk { *best_mk = mk; *best_sol = Some(sol.clone()); save_solution(sol)?; Ok(true) } else { Ok(false) }
}

fn cached_cbm(pre: &Pre, challenge: &Challenge, sol: &Solution, p1: usize, p2: usize, p3: usize, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
    let key = (exact_solution_sig64(sol), p1, p2, p3, 0u8);
    if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
    let res = critical_block_move_local_search_ex(pre, challenge, sol, p1, p2, p3)?;
    if cache.len() >= 1024 { cache.clear(); }
    cache.insert(key, res.clone());
    Ok(res)
}

fn cached_gr(pre: &Pre, challenge: &Challenge, sol: &Solution, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
    let key = (exact_solution_sig64(sol), 0usize, 0usize, 0usize, 1u8);
    if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
    let res = greedy_reassign_pass(pre, challenge, sol)?;
    if cache.len() >= 1024 { cache.clear(); }
    cache.insert(key, res.clone());
    Ok(res)
}

fn maybe_intensify_ls(pre: &Pre, challenge: &Challenge, rng: &mut SmallRng, sol: &Solution, mk: u32, best_mk: u32, target_margin: u32, stuck: usize, late: bool, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
    let flex = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let near_best = mk <= best_mk.saturating_add((target_margin / 3).max(1));
    let very_near_best = mk <= best_mk.saturating_add((target_margin / 6).max(1));
    let do_ls = if mk < best_mk { late || stuck > 20 || flex >= 0.12 || rng.gen::<f64>() < 0.55 }
        else if very_near_best && (late || stuck > 80) { rng.gen::<f64>() < (0.05 + 0.05 * flex).clamp(0.04, 0.11) }
        else if near_best && stuck > 140 { rng.gen::<f64>() < (0.035 + 0.045 * flex).clamp(0.03, 0.085) }
        else { false };
    if !do_ls { return Ok(None); }
    let (p1, p2, p3) = if mk < best_mk { let bump = if flex > 0.60 { 1.0 } else { 0.0 }; (38 + (6.0 * bump) as usize, 60 + (10.0 * bump) as usize, 12) }
        else if stuck > 180 { (30, 48, 10) } else { (24, 36, 8) };

    let chain_on = flex > 0.52 && (mk < best_mk || (very_near_best && (late || stuck > 110)));
    if chain_on {
        let mut best: Option<(Solution, u32)> = None;

        if let Some((s1, m1)) = cached_gr(pre, challenge, sol, cache)? {
            if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                best = Some((s1.clone(), m1));
            }
            if let Some((s2, m2)) = cached_cbm(pre, challenge, &s1, p1, p2, p3, cache)? {
                if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                    best = Some((s2, m2));
                }
            }
        }

        let need_cbm_branch = best.is_none() || flex < 0.85;
        if need_cbm_branch {
            if let Some((s1, m1)) = cached_cbm(pre, challenge, sol, p1, p2, p3, cache)? {
                if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                    best = Some((s1.clone(), m1));
                }
                if m1 <= mk.saturating_add((target_margin / 8).max(1)) {
                    if let Some((s2, m2)) = cached_gr(pre, challenge, &s1, cache)? {
                        if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                            best = Some((s2, m2));
                        }
                    }
                }
            }
        }
        return Ok(best);
    }

    if flex > 0.62 && near_best && late && rng.gen::<f64>() < 0.35 {
        if let Some(res) = cached_gr(pre, challenge, sol, cache)? {
            return Ok(Some(res));
        }
    }

    cached_cbm(pre, challenge, sol, p1, p2, p3, cache)
}

fn maybe_escape_ls(
    pre: &Pre,
    challenge: &Challenge,
    rng: &mut SmallRng,
    top_solutions: &[(Solution, u32)],
    best_sol: &Solution,
    scan: usize,
    stuck: usize,
    flex01: f64,
    cache: &mut DetCache,
) -> Result<Option<(Solution, u32)>> {
    if top_solutions.is_empty() || stuck < 60 {
        return Ok(None);
    }
    let p = (0.040 + 0.060 * flex01 + 0.040 * ((stuck as f64) / 160.0).clamp(0.0, 1.0)).clamp(0.04, 0.14);
    if rng.gen::<f64>() >= p {
        return Ok(None);
    }

    let idx = pick_diverse_top_solution(best_sol, top_solutions, scan)
        .unwrap_or_else(|| pick_top_idx(rng, top_solutions));
    let base = &top_solutions[idx].0;

    let bump = if flex01 > 0.55 { 1.0 } else { 0.0 };
    let p1 = (34.0 + 8.0 * bump) as usize;
    let p2 = (56.0 + 10.0 * bump) as usize;
    let p3 = (10.0 + 2.0 * bump) as usize;
    cached_cbm(pre, challenge, base, p1, p2, p3, cache)
}

fn greedy_reassign_try_node(
    pre: &Pre,
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    job_pred_node: &[usize],
    cur_starts: &mut Vec<u32>,
    current_mk: &mut u32,
    node: usize,
) -> Result<bool> {
    let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
    let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{return Ok(false);}
    let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
    let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>return Ok(false)};
    let cur_start=cur_starts[node];
    let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=*current_mk; let mut best_ins_pos=0usize;

    {
        let seq=&mut ds.machine_seq[cur_machine];
        seq[old_pos..].rotate_left(1);
        seq.pop();
    }

    for &(new_m,new_pt) in &op_info.machines {
        if new_m==cur_machine{continue;}
        ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
        let target_len=ds.machine_seq[new_m].len();
        let mut positions=find_insert_positions_hfs(ds,cur_starts.as_slice(),node,new_m,job_pred_node);
        let mut sorted_pos=target_len;
        for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if cur_starts[nd]>=cur_start{sorted_pos=k;break;}}
        for p in [sorted_pos, sorted_pos.saturating_sub(1), target_len] {
            if p<=target_len&&!positions.contains(&p){positions.push(p);}
        }
        if target_len>2 {
            let mid=(sorted_pos+target_len)/2;
            if mid<=target_len&&!positions.contains(&mid){positions.push(mid);}
        }
        if positions.len()>7{positions.truncate(7);}

        for insert_pos in positions {
            let ins=insert_pos.min(ds.machine_seq[new_m].len());
            {
                let seq=&mut ds.machine_seq[new_m];
                seq.insert(ins,node);
            }
            if let Some((test_mk,_))=eval_disj(ds,buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=ins;}}
            {
                let seq=&mut ds.machine_seq[new_m];
                seq.remove(ins);
            }
        }
    }

    if best_m!=cur_machine {
        let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
        {
            let seq=&mut ds.machine_seq[best_m];
            seq.insert(ins,node);
        }
        ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
        let Some((mk_now,_))=eval_disj(ds,buf) else{return Err(anyhow!("Stalled greedy reassign"))};
        *current_mk=mk_now; cur_starts.clone_from(&buf.start);
        Ok(true)
    } else {
        let seq=&mut ds.machine_seq[cur_machine];
        seq.push(node);
        seq[old_pos..].rotate_right(1);
        ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
        Ok(false)
    }
}

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
    let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let initial_mk=current_mk;
    let mut job_pred_node=vec![NONE_USIZE;n];
    for j in 0..ds.num_jobs {
        let base=ds.job_offsets[j];
        let end=ds.job_offsets[j+1];
        for k in (base+1)..end { job_pred_node[k]=k-1; }
    }
    let mut cur_starts=buf.start.clone();
    let flex01=(pre.high_flex+pre.jobshopness).clamp(0.0,1.5);
    let focus_base=((pre.avg_op_min*(1.8+1.2*flex01)).max(1.0)) as u32;
    let max_rounds=if flex01>0.60{12}else{10};
    let mut rounds=0usize;
    let mut improved_any=false;

    while rounds<max_rounds {
        rounds+=1;
        let Some((mk_now,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        current_mk=mk_now; cur_starts.clone_from(&buf.start);
        let tails=compute_tails_hfs(&ds,&buf);

        let build_order = |focused: bool, ds_ref: &DisjSchedule, cur_starts_ref: &[u32], current_mk_val: u32, tails_ref: &[u32]| -> Vec<usize> {
            let mut order: Vec<(usize,u32,u16,usize,usize)> = Vec::with_capacity(n);
            for node in 0..n {
                let job=ds_ref.node_job[node]; let op_idx=ds_ref.node_op[node]; let product=pre.job_products[job];
                let op_info=&pre.product_ops[product][op_idx];
                let flex=op_info.machines.len();
                if flex<=1{continue;}
                let cur_machine=ds_ref.node_machine[node];
                let path=cur_starts_ref[node].saturating_add(ds_ref.node_pt[node]).saturating_add(tails_ref[node]);
                let slack=current_mk_val.saturating_sub(path);
                if focused && flex01>0.55 {
                    let extra=(flex.saturating_sub(2).min(3) as u32).saturating_mul((pre.avg_op_min.max(1.0) as u32).max(1));
                    if slack>focus_base.saturating_add(extra){continue;}
                }
                let pop=(pre.machine_best_pop[cur_machine].clamp(0.0,1.0)*1000.0) as u16;
                let mach_len=ds_ref.machine_seq[cur_machine].len();
                order.push((node,slack,pop,flex,mach_len));
            }
            order.sort_by(|a,b|a.1.cmp(&b.1).then_with(||b.2.cmp(&a.2)).then_with(||b.3.cmp(&a.3)).then_with(||b.4.cmp(&a.4)).then_with(||a.0.cmp(&b.0)));
            order.into_iter().map(|x|x.0).collect()
        };

        let mut moved=false;

        if flex01>0.55 {
            let mut order=build_order(true, &ds, cur_starts.as_slice(), current_mk, &tails);
            if !order.is_empty() {
                let cap=((n/3).max(24)).min(order.len());
                order.truncate(cap);
                for node in order {
                    if greedy_reassign_try_node(pre,&mut ds,&mut buf,&job_pred_node,&mut cur_starts,&mut current_mk,node)? {
                        improved_any=true; moved=true; break;
                    }
                }
            }
        }

        if !moved {
            for node in build_order(false, &ds, cur_starts.as_slice(), current_mk, &tails) {
                if greedy_reassign_try_node(pre,&mut ds,&mut buf,&job_pred_node,&mut cur_starts,&mut current_mk,node)? {
                    improved_any=true; moved=true; break;
                }
            }
        }

        if !moved { break; }
    }

    if !improved_any||current_mk>=initial_mk{return Ok(None);}
    let Some((mk_now,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    current_mk=mk_now;
    if current_mk>=initial_mk{return Ok(None);}
    let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
}

fn compute_tails_hfs(ds: &DisjSchedule, buf: &EvalBuf) -> Vec<u32> {
    let mut order: Vec<usize> = (0..ds.n).collect();
    order.sort_unstable_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
    let mut tails = vec![0u32; ds.n];
    for &nd in &order {
        let mut after = 0u32;
        let js = ds.job_succ[nd];
        if js != NONE_USIZE {
            after = after.max(ds.node_pt[js].saturating_add(tails[js]));
        }
        let ms = buf.machine_succ[nd];
        if ms != NONE_USIZE {
            after = after.max(ds.node_pt[ms].saturating_add(tails[ms]));
        }
        tails[nd] = after;
    }
    tails
}

#[inline]
fn estimate_swap_hfs(
    u: usize,
    v: usize,
    heads: &[u32],
    tails: &[u32],
    pt: &[u32],
    job_pred: &[usize],
    job_succ: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);
    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);
    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));
    new_r_v.saturating_add(pt[v]).saturating_add(new_q_v).max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
}

#[inline]
fn estimate_reassign_hfs(
    ds: &DisjSchedule,
    heads: &[u32],
    tails: &[u32],
    node: usize,
    new_machine: usize,
    new_pt: u32,
    insert_pos: usize,
    job_pred: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let jp = job_pred[node];
    let js = ds.job_succ[node];
    let old_mp = machine_pred[node];
    let old_ms = machine_succ[node];
    let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let new_seq = &ds.machine_seq[new_machine];
    let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
        let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
        heads[pred].saturating_add(ds.node_pt[pred])
    } else {
        0
    };
    let new_end = jp_end.max(new_mp_end).saturating_add(new_pt);
    let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
    let new_ms_tail = if insert_pos < new_seq.len() {
        let succ = new_seq[insert_pos];
        ds.node_pt[succ].saturating_add(tails[succ])
    } else {
        0
    };
    let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));
    let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
        let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
        old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else {
        0
    };
    node_path.max(old_reconnect)
}

fn find_insert_positions_hfs(ds: &DisjSchedule, starts: &[u32], node: usize, new_machine: usize, job_pred: &[usize]) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 {
        return vec![0];
    }
    let jp = job_pred[node];
    let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let cur_start = starts[node];
    let mut pos_after_jp = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] > job_pred_end {
            pos_after_jp = i;
            break;
        }
    }
    let mut pos_by_cur = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] >= cur_start {
            pos_by_cur = i;
            break;
        }
    }
    let mut out: Vec<usize> = Vec::with_capacity(6);
    for p in [pos_after_jp, pos_after_jp.saturating_sub(1), pos_by_cur, pos_by_cur.saturating_sub(1), 0, len] {
        if p <= len && !out.contains(&p) {
            out.push(p);
        }
    }
    if out.is_empty() {
        out.push(len);
    }
    if out.len() > 6 {
        out.truncate(6);
    }
    out
}

enum MoveHfs {
    Swap { machine: usize, pos: usize },
    Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
    
    // from_pos = current index in machine_seq[machine]; to_pos = target index (apply_insert).
    Relocate { machine: usize, from_pos: usize, to_pos: usize },
}

fn bottleneck_bridge_repair(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    _seed_mk: u32,
    max_machines: usize,
    max_rounds: usize,
) -> Result<Option<(Solution, u32)>> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, base_sol) else {
        return Ok(None);
    };
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let initial_mk = current_mk;
    let n = ds.n;
    if n == 0 || ds.num_machines == 0 {
        return Ok(None);
    }

    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }

    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let near_slack = ((pre.avg_op_min * (0.80 + 0.75 * flex01)).round() as u32).max(1);
    let swap_limit_per_machine = if flex01 < 0.40 { 8usize } else { 5usize };
    let reassign_node_cap = if flex01 > 0.70 { 8usize } else if flex01 > 0.35 { 6usize } else { 3usize };
    let alt_machine_cap = if flex01 > 0.70 { 3usize } else { 2usize };
    let eval_cap = if flex01 > 0.60 { 84usize } else { 60usize };
    let rounds = max_rounds.max(1);
    let mut improved_any = false;
    let mut crit = vec![false; n];
    let mut near = vec![false; n];
    let mut selected_mask = vec![false; ds.num_machines];

    for _round in 0..rounds {
        let Some((mk_now, mk_node)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        current_mk = mk_now;
        let cur_starts = buf.start.clone();
        let tails = compute_tails_hfs(&ds, &buf);

        crit.fill(false);
        near.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }
        for node in 0..n {
            let path = cur_starts[node].saturating_add(ds.node_pt[node]).saturating_add(tails[node]);
            if current_mk.saturating_sub(path) <= near_slack {
                near[node] = true;
            }
        }

        let mut machine_scores: Vec<(u32, u32, u32, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.is_empty() {
                continue;
            }
            let mut crit_cnt = 0u32;
            let mut near_cnt = 0u32;
            let mut adj_cnt = 0u32;
            for i in 0..seq.len() {
                let nd = seq[i];
                if crit[nd] { crit_cnt += 1; }
                if near[nd] { near_cnt += 1; }
                if i + 1 < seq.len() {
                    let nx = seq[i + 1];
                    if (crit[nd] || near[nd]) && (crit[nx] || near[nx]) {
                        adj_cnt += 1;
                    }
                }
            }
            let score = crit_cnt
                .saturating_mul(10)
                .saturating_add(near_cnt.saturating_mul(4))
                .saturating_add(adj_cnt.saturating_mul(5))
                .saturating_add(seq.len().min(6) as u32);
            if score > 0 {
                machine_scores.push((score, crit_cnt, near_cnt, m));
            }
        }
        machine_scores.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.3.cmp(&b.3)));
        if machine_scores.is_empty() {
            break;
        }

        selected_mask.fill(false);
        let top_score = machine_scores[0].0;
        let mut selected_machines: Vec<usize> = Vec::with_capacity(max_machines.min(ds.num_machines));
        for &(score, crit_cnt, near_cnt, m) in &machine_scores {
            if selected_machines.len() >= max_machines.max(1) {
                break;
            }
            if !selected_machines.is_empty() && score.saturating_mul(3).saturating_add(1) < top_score.saturating_mul(2) {
                break;
            }
            if crit_cnt == 0 && near_cnt < 2 {
                continue;
            }
            selected_mask[m] = true;
            selected_machines.push(m);
        }
        if selected_machines.is_empty() {
            break;
        }

        let mut best_move: Option<MoveHfs> = None;
        let mut best_move_mk = current_mk;
        let mut evals = 0usize;

        for &m in &selected_machines {
            if ds.machine_seq[m].len() <= 1 {
                continue;
            }
            let mut ranked_pos: Vec<(u32, usize)> = Vec::new();
            for pos in 0..(ds.machine_seq[m].len() - 1) {
                let a = ds.machine_seq[m][pos];
                let b = ds.machine_seq[m][pos + 1];
                let mut rank = 0u32;
                if crit[a] { rank += 6; }
                if crit[b] { rank += 6; }
                if near[a] { rank += 3; }
                if near[b] { rank += 3; }
                if rank == 0 {
                    continue;
                }
                if pos > 0 {
                    let p = ds.machine_seq[m][pos - 1];
                    if crit[p] || near[p] { rank += 1; }
                }
                if pos + 2 < ds.machine_seq[m].len() {
                    let s = ds.machine_seq[m][pos + 2];
                    if crit[s] || near[s] { rank += 1; }
                }
                ranked_pos.push((rank, pos));
            }
            ranked_pos.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let lim = ranked_pos.len().min(swap_limit_per_machine);
            for &(_, pos) in ranked_pos.iter().take(lim) {
                ds.machine_seq[m].swap(pos, pos + 1);
                evals += 1;
                if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                    if test_mk < best_move_mk {
                        best_move_mk = test_mk;
                        best_move = Some(MoveHfs::Swap { machine: m, pos });
                    }
                }
                ds.machine_seq[m].swap(pos, pos + 1);
                if evals >= eval_cap {
                    break;
                }
            }
            if evals >= eval_cap {
                break;
            }
        }

        if evals < eval_cap && reassign_node_cap > 0 {
            let mut ranked_nodes: Vec<(u32, usize, usize)> = Vec::new();
            for &m in &selected_machines {
                for &node in &ds.machine_seq[m] {
                    let job = ds.node_job[node];
                    let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    let flex = op_info.machines.len();
                    if flex <= 1 {
                        continue;
                    }
                    let path = cur_starts[node].saturating_add(ds.node_pt[node]).saturating_add(tails[node]);
                    let slack = current_mk.saturating_sub(path);
                    let pri = if crit[node] { 0 } else if near[node] { slack } else { slack.saturating_add(near_slack) };
                    ranked_nodes.push((pri, flex, node));
                }
            }
            ranked_nodes.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));

            let node_lim = ranked_nodes.len().min(reassign_node_cap);
            for &(_, _, node) in ranked_nodes.iter().take(node_lim) {
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                let cur_machine = ds.node_machine[node];
                let cur_pt = ds.node_pt[node];
                let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) {
                    Some(p) => p,
                    None => continue,
                };

                {
                    let seq = &mut ds.machine_seq[cur_machine];
                    seq[old_pos..].rotate_left(1);
                    seq.pop();
                }

                let mut alt_machines: Vec<(usize, u32)> = op_info.machines.iter().copied().filter(|(mm, _)| *mm != cur_machine).collect();
                alt_machines.sort_by(|a, b| {
                    let ka = (if selected_mask[a.0] { 1u8 } else { 0u8 }, ds.machine_seq[a.0].len(), a.1, a.0);
                    let kb = (if selected_mask[b.0] { 1u8 } else { 0u8 }, ds.machine_seq[b.0].len(), b.1, b.0);
                    ka.cmp(&kb)
                });

                for &(new_m, new_pt) in alt_machines.iter().take(alt_machine_cap) {
                    ds.node_machine[node] = new_m;
                    ds.node_pt[node] = new_pt;
                    let target_len = ds.machine_seq[new_m].len();
                    let mut positions = find_insert_positions_hfs(&ds, &cur_starts, node, new_m, &job_pred_node);
                    for p in [0usize, target_len / 2, target_len] {
                        if p <= target_len && !positions.contains(&p) {
                            positions.push(p);
                        }
                    }
                    if positions.len() > 6 {
                        positions.truncate(6);
                    }

                    for insert_pos in positions {
                        let ins = insert_pos.min(ds.machine_seq[new_m].len());
                        {
                            let seq = &mut ds.machine_seq[new_m];
                            seq.insert(ins, node);
                        }
                        evals += 1;
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_move_mk {
                                best_move_mk = test_mk;
                                best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos: ins });
                            }
                        }
                        {
                            let seq = &mut ds.machine_seq[new_m];
                            seq.remove(ins);
                        }
                        if evals >= eval_cap {
                            break;
                        }
                    }
                    if evals >= eval_cap {
                        break;
                    }
                }

                {
                    let seq = &mut ds.machine_seq[cur_machine];
                    seq.insert(old_pos.min(seq.len()), node);
                }
                ds.node_machine[node] = cur_machine;
                ds.node_pt[node] = cur_pt;

                if evals >= eval_cap {
                    break;
                }
            }
        }

        if best_move_mk >= current_mk {
            break;
        }

        match best_move {
            Some(MoveHfs::Swap { machine, pos }) => {
                ds.machine_seq[machine].swap(pos, pos + 1);
            }
            Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
            }
            // Relocate is only generated by tabu_ils_hfs (N7/N8); this routine never emits it.
            Some(MoveHfs::Relocate { .. }) => break,
            None => break,
        }

        let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        if mk_after >= current_mk {
            break;
        }
        current_mk = mk_after;
        improved_any = true;
    }

    if !improved_any || current_mk >= initial_mk {
        return Ok(None);
    }
    let Some((mk_now, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    current_mk = mk_now;
    if current_mk >= initial_mk {
        return Ok(None);
    }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, current_mk)))
}


/// `dm` = ops affectees a une machine differente ; `dord` = inversions d'ordre relatif entre
/// paires co-machines. A affectation HFS fixe, l'ordre disjonctif est la seule chose qui varie

fn pr_struct_dist(a: &DisjSchedule, b: &DisjSchedule) -> usize {
    let mut rb = vec![usize::MAX; b.n];
    for seq in b.machine_seq.iter() {
        for (k, &node) in seq.iter().enumerate() { rb[node] = k; }
    }
    let mut d = (0..a.n).filter(|&i| a.node_machine[i] != b.node_machine[i]).count();
    for seq in a.machine_seq.iter() {
        for i in 0..seq.len() {
            for j in (i + 1)..seq.len() {
                let (u, v) = (seq[i], seq[j]);
                if b.node_machine[u] == b.node_machine[v] && rb[u] > rb[v] { d += 1; }
            }
        }
    }
    d
}


/// Retourne `None` si une op se retrouve sur une machine non eligible (pt absent).
fn pr_rebuild(base: &DisjSchedule, pre: &Pre, node_machine: &[usize], machine_seq: &[Vec<usize>]) -> Option<DisjSchedule> {
    let mut node_pt = vec![0u32; base.n];
    for id in 0..base.n {
        let product = pre.job_products[base.node_job[id]];
        let op = &pre.product_ops[product][base.node_op[id]];
        node_pt[id] = pt_from_op(op, node_machine[id])?;
    }
    Some(DisjSchedule {
        n: base.n,
        num_jobs: base.num_jobs,
        num_machines: base.num_machines,
        job_offsets: base.job_offsets.clone(),
        job_succ: base.job_succ.clone(),
        indeg_job: base.indeg_job.clone(),
        node_machine: node_machine.to_vec(),
        node_pt,
        node_job: base.node_job.clone(),
        node_op: base.node_op.clone(),
        machine_seq: machine_seq.to_vec(),
    })
}


///
/// Un pas = deplacer UNE operation encore en desaccord vers sa machine ET sa position
/// relative dans la solution guide, en choisissant a chaque pas le move de makespan minimal
/// parmi tous les desaccords restants. Les moves rendant le graphe cyclique sont rejetes par
/// `eval_disj` (qui renvoie `None`), donc jamais appliques.
///
/// Retourne le MEILLEUR intermediaire strict rencontre (ni le depart, ni l'arrivee).
/// Entierement deterministe : aucun RNG, aucune horloge, ordre de balayage fixe.
///
/// `max_steps` borne le travail (budget en TRAVAIL, jamais en temps). Mesure offline : le
/// chemin s'arrete de lui-meme (`PATH_STUCK`, tous les moves restants cycliques) apres
/// 237-594 pas sur les nonces de bande, bien avant toute borne.
fn pr_forward(pre: &Pre, challenge: &Challenge, from_sol: &Solution, to_sol: &Solution, max_steps: usize) -> Option<(Solution, u32)> {
    let base = build_disj_from_solution(pre, challenge, from_sol).ok()?;
    let tgt = build_disj_from_solution(pre, challenge, to_sol).ok()?;
    let n = base.n;

    let mut r_tgt = vec![usize::MAX; n];
    for seq in tgt.machine_seq.iter() {
        for (k, &node) in seq.iter().enumerate() { r_tgt[node] = k; }
    }
    let m_tgt = &tgt.node_machine;

    let mut cur_machine = base.node_machine.clone();
    let mut cur_seq = base.machine_seq.clone();
    let mut buf = EvalBuf::new(n);

    let mut best: Option<(u32, Vec<usize>, Vec<Vec<usize>>)> = None;
    let mut todo_flag = vec![false; n];
    let mut todo: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..max_steps {
        // ops encore en desaccord avec la guide : machine differente, ou ordre local inverse
        todo.clear();
        for f in todo_flag.iter_mut() { *f = false; }
        for id in 0..n {
            if cur_machine[id] != m_tgt[id] { todo_flag[id] = true; }
        }
        for seq in cur_seq.iter() {
            for w in seq.windows(2) {
                let (u, v) = (w[0], w[1]);
                if m_tgt[u] == m_tgt[v] && r_tgt[u] > r_tgt[v] { todo_flag[u] = true; }
            }
        }
        for id in 0..n { if todo_flag[id] { todo.push(id); } }
        if todo.is_empty() { break; }

        let mut best_move: Option<(u32, usize, usize, usize)> = None; // mk, id, from_m, pos
        for &id in todo.iter() {
            let from_m = cur_machine[id];
            let to_m = m_tgt[id];
            // etat candidat : retirer id de sa machine, l'inserer a sa place relative cible
            let mut nm = cur_machine.clone();
            let mut ns = cur_seq.clone();
            ns[from_m].retain(|&x| x != id);
            nm[id] = to_m;
            let pos = ns[to_m].iter().position(|&x| r_tgt[x] > r_tgt[id]).unwrap_or(ns[to_m].len());
            ns[to_m].insert(pos, id);
            let Some(cand) = pr_rebuild(&base, pre, &nm, &ns) else { continue };
            // `eval_disj` renvoie None si le graphe est cyclique => move infaisable, ignore
            let Some((mk, _)) = eval_disj(&cand, &mut buf) else { continue };
            if best_move.map_or(true, |(b, _, _, _)| mk < b) {
                best_move = Some((mk, id, from_m, pos));
            }
        }
        // tous les moves restants sont cycliques => le chemin est bloque
        let Some((mk, id, from_m, pos)) = best_move else { break };
        cur_seq[from_m].retain(|&x| x != id);
        cur_machine[id] = m_tgt[id];
        let to_m = m_tgt[id];
        let pos = pos.min(cur_seq[to_m].len());
        cur_seq[to_m].insert(pos, id);
        if mk < best.as_ref().map_or(u32::MAX, |(b, _, _)| *b) {
            best = Some((mk, cur_machine.clone(), cur_seq.clone()));
        }
    }

    let (mk, m, sq) = best?;
    let ds = pr_rebuild(&base, pre, &m, &sq)?;
    let mut buf2 = EvalBuf::new(n);
    let (mk2, _) = eval_disj(&ds, &mut buf2)?;
    if mk2 != mk { return None; }
    let sol = disj_to_solution(pre, &ds, &buf2.start).ok()?;
    Some((sol, mk))
}

fn tabu_ils_hfs(
    pre: &Pre,
    challenge: &Challenge,
    seed_sol: &Solution,
    seed_mk: u32,
    max_iters: usize,
    tenure_base: usize,
    stagnation_limit: usize,
    reassign_every: usize,
    max_kicks: usize,
    kick_pct: usize,
    kick_swaps_count: usize,
    doe_mode: usize,
    doe_switch: usize,
    n8_mode: usize,
    n8_max: usize,
    bn_reassign: usize,
    ig_mode: usize,
    ig_d: usize,
) -> Result<(Solution, u32)> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, seed_sol) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;
    let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    let mut best_mk = initial_mk.min(seed_mk);
    let mut best_ds = ds.clone();
    let tenure = tenure_base.max(5);
    let tenure_delta = (tenure / 3).max(2);
    
    let kick_threshold = if kick_pct == 0 {
        (stagnation_limit * 2 / 3).max(50)
    } else {
        (stagnation_limit * kick_pct / 100).max(50)
    };
    let aspiration_margin = ((pre.avg_op_min * (0.45 + 0.35 * pre.high_flex + 0.25 * pre.jobshopness)).round() as u32).max(1);
    let recent_window = (tenure * 2).clamp(8, 64);
    let freq_decay_every = (12usize + tenure).clamp(12, 28);
    let mut tabu_swap: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::with_capacity(tenure * 8);
    let mut tabu_reassign: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::with_capacity(tenure * 4);
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }
    let mut no_improve = 0usize;
    let mut kicks_left = max_kicks;
    
    let mut doe_op: usize = 0;          // active operator index in the ensemble (0=A,1=B,2=C)
    let mut doe_since: usize = 0;       // kicks on current op without a strict best improvement
    let mut doe_last_best: u32 = best_mk;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut crit = vec![false; n];
    let mut job_touch_iter = vec![usize::MAX; ds.num_jobs];
    let mut machine_touch_iter = vec![usize::MAX; ds.num_machines];
    let mut job_freq = vec![0u16; ds.num_jobs];
    let mut machine_freq = vec![0u16; ds.num_machines];

    for iter in 0..max_iters {
        if no_improve >= stagnation_limit {
            if kicks_left == 0 {
                break;
            }
            ds = best_ds.clone();
            if eval_disj(&ds, &mut buf).is_none() {
                break;
            }
            no_improve = 0;
            kicks_left -= 1;
            tabu_swap.clear();
            tabu_reassign.clear();
            continue;
        }

        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false);
            let mut u = kick_mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }
            // ---- Destruction operator selection (IG-DOE) --------------------------------
            // Build the candidate adjacent-swap pool for the active destruction operator.
            // All three operators reuse the proven-safe adjacent-swap primitive (only positions
            // differ), so feasibility risk is identical to the original operator — an invalid
            // (cyclic) eval simply breaks the loop and the tracked best (always valid) is kept.
            //   Op A (0) = critical-adjacent  : pairs where a node lies on the critical path (ORIGINAL).
            //   Op B (1) = bottleneck-machine : all pairs on the single machine carrying the most
            //                                   critical nodes → concentrated block reshuffle.
            //   Op C (2) = dispersed          : all adjacent pairs regardless of criticality →
            //                                   diversify into non-critical regions.
            // doe_mode: 0 = sentinel (byte-exact Op A, no DOE bookkeeping), 1 = ensemble round-robin
            // with stagnation-triggered switch, 2 = Op B solo, 3 = Op C solo.
            let active_op: usize = if doe_mode == 0 {
                0
            } else {
                // stagnation-triggered sequential switching (mode 1); solo modes pin the operator.
                if best_mk < doe_last_best { doe_since = 0; doe_last_best = best_mk; }
                else { doe_since += 1; }
                match doe_mode {
                    2 => 1,
                    3 => 2,
                    _ => {
                        if doe_since >= doe_switch { doe_op = (doe_op + 1) % 3; doe_since = 0; }
                        doe_op
                    }
                }
            };
            let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
            match active_op {
                1 => {
                    // Op B: concentrate destruction on the bottleneck machine (most critical nodes).
                    let mut best_m = 0usize;
                    let mut best_c = 0usize;
                    for m in 0..ds.num_machines {
                        let seq = &ds.machine_seq[m];
                        if seq.len() <= 1 { continue; }
                        let mut c = 0usize;
                        for &node in seq.iter() { if crit[node] { c += 1; } }
                        if c > best_c { best_c = c; best_m = m; }
                    }
                    let seq_len = ds.machine_seq[best_m].len();
                    if seq_len > 1 {
                        for i in 0..(seq_len - 1) { kick_swaps.push((best_m, i)); }
                    }
                }
                2 => {
                    // Op C: dispersed — every adjacent pair across all machines (critical or not).
                    for m in 0..ds.num_machines {
                        let l = ds.machine_seq[m].len();
                        if l <= 1 { continue; }
                        for i in 0..(l - 1) { kick_swaps.push((m, i)); }
                    }
                }
                _ => {
                    // Op A (original): critical-adjacent pairs only.
                    for m in 0..ds.num_machines {
                        if ds.machine_seq[m].len() <= 1 {
                            continue;
                        }
                        for i in 0..(ds.machine_seq[m].len() - 1) {
                            if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i + 1]] {
                                kick_swaps.push((m, i));
                            }
                        }
                    }
                }
            }
            if ig_mode >= 1 {
                
                // Destruction = draw ig_d DISTINCT critical ops (same xorshift discipline as the
                // blind kick). Reconstruction = for each drawn op, sequentially FORCE-MOVE it to
                // the exact-eval argmin slot among candidate positions (full eval_disj per
                // candidate; the original slot is excluded so the perturbation always moves).
                // ig_mode==1: candidate slots on the op's current machine only.
                // ig_mode==2: candidate slots on ALL eligible machines (op_info.machines, with
                //             the machine-specific pt) — the HFS parallel-machine axis.
                // The schedule stays fully scheduled at every eval (one op in flight, always
                // inserted before eval); an infeasible candidate (disjunctive cycle) makes
                // eval_disj return None and is skipped; if no candidate is valid the op is
                // restored to its original slot. Deterministic, no wall-clock, no threading.
                let mut ig_pool: Vec<usize> = (0..n).filter(|&u| crit[u]).collect();
                let ig_moves = ig_d.min(ig_pool.len());
                for _ in 0..ig_moves {
                    if ig_pool.is_empty() { break; }
                    pseed ^= pseed.wrapping_shl(13);
                    pseed ^= pseed.wrapping_shr(7);
                    pseed ^= pseed.wrapping_shl(17);
                    let pick = (pseed as usize) % ig_pool.len();
                    let u = ig_pool.swap_remove(pick);
                    let m0 = ds.node_machine[u];
                    let Some(p0) = ds.machine_seq[m0].iter().position(|&x| x == u) else { continue };
                    let pt0 = ds.node_pt[u];
                    ds.machine_seq[m0].remove(p0);
                    let job = ds.node_job[u];
                    let op_idx = ds.node_op[u];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    let mut best_cand: Option<(usize, usize, u32)> = None; // (machine, pos, pt)
                    let mut best_cand_mk = u32::MAX;
                    for &(m2, pt2) in &op_info.machines {
                        if ig_mode == 1 && m2 != m0 { continue; }
                        let len2 = ds.machine_seq[m2].len();
                        for pos in 0..=len2 {
                            if m2 == m0 && pos == p0 { continue; } // forced move: original slot excluded
                            ds.machine_seq[m2].insert(pos, u);
                            ds.node_machine[u] = m2;
                            ds.node_pt[u] = pt2;
                            if let Some((mk2, _)) = eval_disj(&ds, &mut buf) {
                                if mk2 < best_cand_mk {
                                    best_cand_mk = mk2;
                                    best_cand = Some((m2, pos, pt2));
                                }
                            }
                            ds.machine_seq[m2].remove(pos);
                        }
                    }
                    match best_cand {
                        Some((m2, pos, pt2)) => {
                            ds.machine_seq[m2].insert(pos, u);
                            ds.node_machine[u] = m2;
                            ds.node_pt[u] = pt2;
                        }
                        None => {
                            ds.machine_seq[m0].insert(p0, u);
                            ds.node_machine[u] = m0;
                            ds.node_pt[u] = pt0;
                        }
                    }
                }
            } else if !kick_swaps.is_empty() {
                for _ in 0..kick_swaps_count {
                    pseed ^= pseed.wrapping_shl(13);
                    pseed ^= pseed.wrapping_shr(7);
                    pseed ^= pseed.wrapping_shl(17);
                    let (m, pos) = kick_swaps[(pseed as usize) % kick_swaps.len()];
                    if pos + 1 < ds.machine_seq[m].len() {
                        ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }
            kicks_left -= 1;
            continue;
        }

        if iter > 0 && iter % freq_decay_every == 0 {
            for v in &mut job_freq { *v /= 2; }
            for v in &mut machine_freq { *v /= 2; }
        }

        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        if cur_mk < best_mk {
            best_mk = cur_mk;
            best_ds = ds.clone();
            no_improve = 0;
        } else {
            no_improve += 1;
        }

        let tails = compute_tails_hfs(&ds, &buf);
        machine_pred_node.fill(NONE_USIZE);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() {
                machine_pred_node[seq[i]] = seq[i - 1];
            }
        }

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let age_since = |last: usize| -> usize {
            if last == usize::MAX { recent_window + 1 } else { iter.saturating_sub(last) }
        };

        let mut best_move: Option<MoveHfs> = None;
        let mut best_move_rank = u32::MAX;
        let mut fallback_move: Option<MoveHfs> = None;
        let mut fallback_rank = u32::MAX;

        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 {
                continue;
            }
            let mut i = 0usize;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < ds.machine_seq[m].len() {
                    let x = ds.machine_seq[m][bend];
                    let y = ds.machine_seq[m][bend + 1];
                    if !crit[y] {
                        break;
                    }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    let block_len = bend - bstart + 1;
                    let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                    let num_swaps = if block_len >= 3 { 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos = swap_positions[si];
                        if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() {
                            continue;
                        }
                        let node_u = ds.machine_seq[m][pos];
                        let node_v = ds.machine_seq[m][pos + 1];
                        let job_u = ds.node_job[node_u];
                        let job_v = ds.node_job[node_v];
                        let est_mk = estimate_swap_hfs(node_u, node_v, &buf.start, &tails, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                        let key = (node_u.min(node_v), node_u.max(node_v));
                        let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                        let age_ju = age_since(job_touch_iter[job_u]);
                        let age_jv = age_since(job_touch_iter[job_v]);
                        let age_m = age_since(machine_touch_iter[m]);
                        let recent_hits =
                            (if age_ju < recent_window { 1u32 } else { 0u32 }) +
                            (if age_jv < recent_window { 1u32 } else { 0u32 }) +
                            (if age_m < recent_window { 1u32 } else { 0u32 });
                        let freq_pen = ((machine_freq[m] as u32) + ((job_freq[job_u] as u32 + job_freq[job_v] as u32 + 1) / 2)).min(18);
                        let recent_pen = (2 * recent_hits).min(8);
                        let novelty = recent_hits <= 1 || age_ju + age_jv + age_m > recent_window.saturating_mul(2);
                        let aspiration = est_mk < best_mk || (est_mk <= best_mk.saturating_add(aspiration_margin) && novelty);
                        let tabu_pen = if is_tabu && !aspiration { aspiration_margin.min(12) } else { 0 };
                        let rank = est_mk.saturating_add(freq_pen).saturating_add(recent_pen).saturating_add(tabu_pen);
                        if (!is_tabu || aspiration) && rank < best_move_rank {
                            best_move_rank = rank;
                            best_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                        if rank < fallback_rank {
                            fallback_rank = rank;
                            fallback_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                    }
                    
                    // N5 (above) only swaps the block's first/last adjacent pair -> internal critical
                    // ops are frozen inside the block (73900 ceiling). Here we additionally offer the
                    // best-ranked SHORT boundary relocation of an internal op on the SAME machine:
                    //   n8_mode==1 (N7): to the block frontier (just before bstart / just after bend);
                    //   n8_mode==2 (N8): just OUTSIDE the block (bstart-1 / bend+1).
                    // Kept short-distance so the pre-move head/tail estimate stays accurate (the
                    // documented cause of the earlier long-distance N8 regression). n8_mode==0 emits
                    // nothing -> byte-exact N5 sentinel.
                    if n8_mode >= 1 && block_len >= 3 {
                        let seq_len = ds.machine_seq[m].len();
                        let (to_lo, to_hi) = if n8_mode == 1 {
                            (bstart, bend)
                        } else {
                            (bstart.saturating_sub(1), (bend + 1).min(seq_len.saturating_sub(1)))
                        };
                        let internal_end = (bstart + 1 + n8_max).min(bend); // exclusive; only interior positions
                        for p in (bstart + 1)..internal_end {
                            let node_r = ds.machine_seq[m][p];
                            let job_r = ds.node_job[node_r];
                            for &to_pos in &[to_lo, to_hi] {
                                if to_pos == p || to_pos >= seq_len {
                                    continue;
                                }
                                let est_mk = estimate_reassign_hfs(&ds, &buf.start, &tails, node_r, m, ds.node_pt[node_r], to_pos, &job_pred_node, &machine_pred_node, &buf.machine_succ);
                                let key = (node_r, m);
                                let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                                let age_jr = age_since(job_touch_iter[job_r]);
                                let age_mr = age_since(machine_touch_iter[m]);
                                let recent_hits =
                                    (if age_jr < recent_window { 1u32 } else { 0u32 }) +
                                    (if age_mr < recent_window { 1u32 } else { 0u32 });
                                let freq_pen = ((job_freq[job_r] as u32) + (machine_freq[m] as u32)).min(18);
                                let recent_pen = (2 * recent_hits).min(8);
                                let novelty = recent_hits <= 1 || age_jr + age_mr > recent_window.saturating_mul(2);
                                let aspiration = est_mk < best_mk || (est_mk <= best_mk.saturating_add(aspiration_margin) && novelty);
                                let tabu_pen = if is_tabu && !aspiration { aspiration_margin.min(12) } else { 0 };
                                let rank = est_mk.saturating_add(freq_pen).saturating_add(recent_pen).saturating_add(tabu_pen);
                                if (!is_tabu || aspiration) && rank < best_move_rank {
                                    best_move_rank = rank;
                                    best_move = Some(MoveHfs::Relocate { machine: m, from_pos: p, to_pos });
                                }
                                if rank < fallback_rank {
                                    fallback_rank = rank;
                                    fallback_move = Some(MoveHfs::Relocate { machine: m, from_pos: p, to_pos });
                                }
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }

        if iter % reassign_every == 0 {
            // L7 bottleneck load-balancing: identify the machine(s) carrying the most critical nodes
            // (Op-B bottleneck definition) so their NON-critical ops also become reassignment candidates —
            // a move the crit-only neighborhood structurally cannot make. Offering a non-critical op OFF an
            // over-loaded machine to a lighter parallel machine can free the sequence so the critical block
            
            // bn_reassign==0 → bn_m1/bn_m2 stay NONE_USIZE; node_machine is never NONE_USIZE so the gate is
            // byte-exact crit-only (proven sentinel). 1 = the single bottleneck machine, 2 = the top-2 loaded.
            let (mut bn_m1, mut bn_m2) = (NONE_USIZE, NONE_USIZE);
            if bn_reassign >= 1 {
                let (mut c1, mut c2) = (0usize, 0usize);
                for m in 0..ds.num_machines {
                    if ds.machine_seq[m].len() <= 1 { continue; }
                    let mut c = 0usize;
                    for &nd in ds.machine_seq[m].iter() { if crit[nd] { c += 1; } }
                    if c > c1 { c2 = c1; bn_m2 = bn_m1; c1 = c; bn_m1 = m; }
                    else if c > c2 { c2 = c; bn_m2 = m; }
                }
                if bn_reassign < 2 { bn_m2 = NONE_USIZE; }
            }
            for node in 0..n {
                let on_bottleneck = bn_reassign >= 1
                    && (ds.node_machine[node] == bn_m1
                        || (bn_reassign >= 2 && ds.node_machine[node] == bn_m2));
                if !crit[node] && !on_bottleneck {
                    continue;
                }
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                if op_idx >= pre.product_ops[product].len() {
                    continue;
                }
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 {
                    continue;
                }
                let cur_machine = ds.node_machine[node];
                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine {
                        continue;
                    }
                    let key = (node, new_m);
                    let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                    let positions = find_insert_positions_hfs(&ds, &buf.start, node, new_m, &job_pred_node);
                    for insert_pos in positions {
                        let est_mk = estimate_reassign_hfs(&ds, &buf.start, &tails, node, new_m, new_pt, insert_pos, &job_pred_node, &machine_pred_node, &buf.machine_succ);
                        let age_job = age_since(job_touch_iter[job]);
                        let age_old = age_since(machine_touch_iter[cur_machine]);
                        let age_new = age_since(machine_touch_iter[new_m]);
                        let recent_hits =
                            (if age_job < recent_window { 1u32 } else { 0u32 }) +
                            (if age_old < recent_window { 1u32 } else { 0u32 }) +
                            (if age_new < recent_window { 1u32 } else { 0u32 });
                        let freq_pen = ((job_freq[job] as u32) + (machine_freq[new_m] as u32) + ((machine_freq[cur_machine] as u32 + 1) / 2)).min(20);
                        let recent_pen = (2 * recent_hits).min(8);
                        let novelty = recent_hits <= 1 || age_job + age_old + age_new > recent_window.saturating_mul(2);
                        let aspiration = est_mk < best_mk || (est_mk <= best_mk.saturating_add(aspiration_margin) && novelty);
                        let tabu_pen = if is_tabu && !aspiration { aspiration_margin.min(12) } else { 0 };
                        let rank = est_mk.saturating_add(freq_pen).saturating_add(recent_pen).saturating_add(tabu_pen);
                        if (!is_tabu || aspiration) && rank < best_move_rank {
                            best_move_rank = rank;
                            best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                        if rank < fallback_rank {
                            fallback_rank = rank;
                            fallback_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                    }
                }
            }
        }

        match best_move.or(fallback_move) {
            Some(MoveHfs::Swap { machine, pos }) => {
                let node_a = ds.machine_seq[machine][pos];
                let node_b = ds.machine_seq[machine][pos + 1];
                let job_a = ds.node_job[node_a];
                let job_b = ds.node_job[node_b];
                let age_a = age_since(job_touch_iter[job_a]);
                let age_b = age_since(job_touch_iter[job_b]);
                let age_m = age_since(machine_touch_iter[machine]);
                ds.machine_seq[machine].swap(pos, pos + 1);
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let base_tenure = (tenure + offset).saturating_sub(tenure_delta);
                let recent_bonus =
                    (if age_a < recent_window { 1usize } else { 0usize }) +
                    (if age_b < recent_window { 1usize } else { 0usize }) +
                    (if age_m < recent_window { 1usize } else { 0usize });
                let freq_bonus = ((job_freq[job_a] as usize + job_freq[job_b] as usize + machine_freq[machine] as usize) / 5).min(4);
                let this_tenure = base_tenure + 1 + recent_bonus + freq_bonus;
                tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
                job_touch_iter[job_a] = iter;
                job_touch_iter[job_b] = iter;
                machine_touch_iter[machine] = iter;
                job_freq[job_a] = job_freq[job_a].saturating_add(1);
                job_freq[job_b] = job_freq[job_b].saturating_add(1);
                machine_freq[machine] = machine_freq[machine].saturating_add(1);
            }
            Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                let job = ds.node_job[node];
                let age_job = age_since(job_touch_iter[job]);
                let age_old = age_since(machine_touch_iter[old_machine]);
                let age_new = age_since(machine_touch_iter[new_machine]);
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let base_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                let recent_bonus =
                    (if age_job < recent_window { 1usize } else { 0usize }) +
                    (if age_old < recent_window { 1usize } else { 0usize }) +
                    (if age_new < recent_window { 1usize } else { 0usize });
                let freq_bonus = ((job_freq[job] as usize + machine_freq[old_machine] as usize + machine_freq[new_machine] as usize) / 4).min(5);
                let this_tenure = base_tenure + 2 + recent_bonus + freq_bonus;
                tabu_reassign.insert((node, old_machine), iter + this_tenure);
                job_touch_iter[job] = iter;
                machine_touch_iter[old_machine] = iter;
                machine_touch_iter[new_machine] = iter;
                job_freq[job] = job_freq[job].saturating_add(1);
                machine_freq[old_machine] = machine_freq[old_machine].saturating_add(1);
                machine_freq[new_machine] = machine_freq[new_machine].saturating_add(1);
            }
            Some(MoveHfs::Relocate { machine, from_pos, to_pos }) => {
                
                // machine/node stay the same (node_machine, node_pt unchanged) — only the machine
                // sequence is re-ordered via apply_insert (always a valid permutation; an infeasible
                // disjunctive cycle is caught by eval_disj at the top of the loop, tracked best kept).
                if from_pos < ds.machine_seq[machine].len() {
                    let node = ds.machine_seq[machine][from_pos];
                    let job = ds.node_job[node];
                    let age_job = age_since(job_touch_iter[job]);
                    let age_m = age_since(machine_touch_iter[machine]);
                    apply_insert(&mut ds.machine_seq[machine], from_pos, to_pos);
                    pseed ^= pseed.wrapping_shl(13);
                    pseed ^= pseed.wrapping_shr(7);
                    pseed ^= pseed.wrapping_shl(17);
                    let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                    let base_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                    let recent_bonus =
                        (if age_job < recent_window { 1usize } else { 0usize }) +
                        (if age_m < recent_window { 1usize } else { 0usize });
                    let freq_bonus = ((job_freq[job] as usize + machine_freq[machine] as usize) / 4).min(5);
                    let this_tenure = base_tenure + 2 + recent_bonus + freq_bonus;
                    tabu_reassign.insert((node, machine), iter + this_tenure);
                    job_touch_iter[job] = iter;
                    machine_touch_iter[machine] = iter;
                    job_freq[job] = job_freq[job].saturating_add(1);
                    machine_freq[machine] = machine_freq[machine].saturating_add(1);
                }
            }
            None => break,
        }
    }

    let Some((_, _)) = eval_disj(&best_ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    match disj_to_solution(pre, &best_ds, &buf.start) {
        Ok(s) => Ok((s, best_mk)),
        Err(_) => Ok((seed_sol.clone(), seed_mk)),
    }
}

fn ensure_solution_features(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    feature_cache: &mut std::collections::HashMap<u64, (Vec<f64>, Vec<f64>, RoutePrefLite)>,
) -> Result<u64> {
    let sig = exact_solution_sig64(sol);
    match feature_cache.entry(sig) {
        std::collections::hash_map::Entry::Occupied(_) => {}
        std::collections::hash_map::Entry::Vacant(e) => {
            let jb=job_bias_from_solution(pre,sol)?;
            let mp=machine_penalty_from_solution(pre,sol,challenge.num_machines)?;
            let rp=route_pref_from_solution_lite(pre,sol,challenge)?;
            e.insert((jb,mp,rp));
        }
    }
    Ok(sig)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;
    let mut cache: DetCache = HashMap::with_hasher(seeded_hasher(&challenge.seed));
    let mut rng = SmallRng::from_seed(challenge.seed);
    let horizon = pre.machine_load0.iter().cloned().fold(0.0f64, f64::max).max(pre.horizon);
    let time_scale = (horizon * (2.65 + 0.15 * pre.load_cv + 0.10 * pre.jobshopness + 0.10 * pre.high_flex)).max(1.0);
    let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol.clone());
    let mut top_solutions: Vec<(Solution,u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    let mut feature_cache: std::collections::HashMap<u64, (Vec<f64>, Vec<f64>, RoutePrefLite)> =
        std::collections::HashMap::with_capacity(64);
    let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = (0.040+0.10*pre.high_flex+0.08*pre.jobshopness).clamp(0.04,0.22);

    if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
        if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
            push_top_solutions(&mut top_solutions,&sol,mk,15);
        }
    }

    let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0,horizon,time_scale)?;
        commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
        push_top_solutions(&mut top_solutions,&sol,mk,15); ranked.push((rule,mk,sol));
    }
    ranked.sort_by_key(|x|x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
    let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
    for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);}

    let elite_cap: usize = (6usize + (2.0 * flex01).round() as usize).clamp(6, 8);
    let mut elite: Vec<EliteParams> = Vec::new();
    for i in 0..ranked.len().min(3) {
        let sol=&ranked[i].2; let mk=ranked[i].1;
        let sig=ensure_solution_features(pre,challenge,sol,&mut feature_cache)?;
        let cached=feature_cache.get(&sig).unwrap();
        elite.push(EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk});
    }
    {
        let sig=ensure_solution_features(pre,challenge,&greedy_sol,&mut feature_cache)?;
        let cached=feature_cache.get(&sig).unwrap();
        elite.push(EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:greedy_mk});
    }
    normalize_elite(&mut elite, elite_cap);

    let num_restarts=effort.hybrid_flow_shop_iters;
    let k_hi=6usize;
    let mut stuck: usize=0;
    let mut escape_cooldown: usize=0;

    for r in 0..num_restarts {
        if escape_cooldown > 0 { escape_cooldown -= 1; }

        if escape_cooldown == 0 {
            if let Some((sol2,mk2)) = maybe_escape_ls(pre,challenge,&mut rng,&top_solutions,best_solution.as_ref().unwrap(),top_solutions.len().min(15),stuck,flex01,&mut cache)? {
                escape_cooldown = 14;
                let improved = commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                if improved {
                    stuck = 0;
                    let sig=ensure_solution_features(pre,challenge,&sol2,&mut feature_cache)?;
                    let cached=feature_cache.get(&sig).unwrap();
                    maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk2},elite_cap);
                } else { stuck = stuck.saturating_add(1); }
                push_top_solutions(&mut top_solutions,&sol2,mk2,15);
                continue;
            } else if stuck > 150 { escape_cooldown = 6; }
        }

        let late = r >= (num_restarts*2)/3;
        let (k_min,k_max) = if stuck>170{(4usize,6usize)}else if stuck>90{(3usize,6usize)}else if stuck>35{(2usize,6usize)}else{(2usize,4usize)};

        let rule = if r < 35 {
            let u: f64=rng.gen();
            if u<0.11{Rule::FlexBalance}else if u<0.18{Rule::ShortestProc}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}
        } else {
            choose_rule_bandit(&mut rng,&rules,&rule_best,&rule_tries,best_makespan,target_margin,stuck,false,late)
        };

        let k = if k_max<=k_min{k_min}else if stuck>120&&rng.gen::<f64>()<0.55{k_max}else{rng.gen_range(k_min..=k_max)}.min(k_hi);
        let learn_base = (0.09+0.24*pre.jobshopness+0.20*pre.high_flex).clamp(0.06,0.44);
        let learn_boost = (1.0+0.38*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.38);
        let learn_p = (learn_base*learn_boost).clamp(0.0,0.65);

        if stuck > 80 && !top_solutions.is_empty() && rng.gen::<f64>() < 0.04 {
            let idx=pick_top_idx(&mut rng,&top_solutions); let (sref,mkref)=(&top_solutions[idx].0,top_solutions[idx].1);
            let sig=ensure_solution_features(pre,challenge,sref,&mut feature_cache)?;
            let cached=feature_cache.get(&sig).unwrap();
            maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mkref},elite_cap);
        }

        let use_learn = !elite.is_empty() && rng.gen::<f64>() < learn_p;
        let target = if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};

        let (mut sol, mut mk) = if use_learn {
            let mix_p=(0.055+0.10*pre.high_flex+0.09*pre.jobshopness+0.16*((stuck as f64)/160.0).clamp(0.0,1.0)).clamp(0.05,0.40);
            let base_idx=pick_elite_idx(&mut rng,&elite); let mut mp_idx=base_idx; let mut rp_idx=base_idx;
            if elite.len()>1&&rng.gen::<f64>()<mix_p{mp_idx=pick_elite_idx(&mut rng,&elite);}
            if elite.len()>1&&rng.gen::<f64>()<mix_p{rp_idx=pick_elite_idx(&mut rng,&elite);}
            let drop_mp_p=(0.030+0.060*pre.high_flex).clamp(0.03,0.10); let drop_rp_p=(0.030+0.070*pre.jobshopness).clamp(0.03,0.12);
            let mp_opt=if rng.gen::<f64>()<drop_mp_p{None}else{Some(&elite[mp_idx].mp)};
            let rp_opt=if rng.gen::<f64>()<drop_rp_p{None}else{Some(&elite[rp_idx].rp)};
            let jitter=(0.80+0.70*rng.gen::<f64>()).clamp(0.65,1.55);
            let route_w=if rp_opt.is_some(){(route_w_base*jitter).clamp(route_w_base*0.55,0.45)}else{0.0};
            construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,Some(&elite[base_idx].jb),mp_opt.map(|v|&**v),rp_opt,route_w,horizon,time_scale)?
        } else {
            construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0,horizon,time_scale)?
        };

        if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,stuck,late,&mut cache)?{sol=sol2;mk=mk2;}

        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        let improved=commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;

        if improved {
            stuck=0;
            let sig=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
            let cached=feature_cache.get(&sig).unwrap();
            maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk},elite_cap);
        } else {
            stuck=stuck.saturating_add(1);
            let add_p=(0.075+0.025*flex01).clamp(0.07,0.11);
            if mk<=best_makespan.saturating_add(target_margin/2)&&rng.gen::<f64>()<add_p {
                let sig=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
                let cached=feature_cache.get(&sig).unwrap();
                maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk},elite_cap);
            }
        }
        push_top_solutions(&mut top_solutions,&sol,mk,15);
    }

    let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
    let mut refine_results: Vec<(Solution,u32)>=Vec::new();
    let refine_top_len=top_solutions.len();
    let mut refine_phase_exact_sigs: Vec<u64>=Vec::with_capacity(refine_top_len);
    for (sol,_) in top_solutions.iter() {
        refine_phase_exact_sigs.push(exact_solution_sig64(sol));
    }
    for (idx,(base_sol,_)) in top_solutions.iter().enumerate() {
        let sig=refine_phase_exact_sigs[idx];
        match feature_cache.entry(sig) {
            std::collections::hash_map::Entry::Occupied(_) => {}
            std::collections::hash_map::Entry::Vacant(e) => {
                let jb=job_bias_from_solution(pre,base_sol)?;
                let mp_base=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
                let rp_base=route_pref_from_solution_lite(pre,base_sol,challenge)?;
                e.insert((jb,mp_base,rp_base));
            }
        }
        let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
        let mix_ref_p=(0.045+0.10*pre.high_flex+0.09*pre.jobshopness).clamp(0.04,0.22);
        for attempt in 0..10 {
            let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>Rule::ShortestProc};
            let k=match attempt%6{0=>2,1=>3,2=>4,3=>5,4=>3,_=>2}.min(k_hi);
            let (mut sol,mut mk)={
                let cached=feature_cache.get(&sig).unwrap();
                let jb:&[f64]=&cached.0;
                let mut mp_ref: Option<&Vec<f64>>=Some(&cached.1);
                let mut rp_ref: Option<&RoutePrefLite>=Some(&cached.2);
                if !elite.is_empty()&&rng.gen::<f64>()<mix_ref_p {
                    let eidx=pick_elite_idx(&mut rng,&elite);
                    if rng.gen::<f64>()<0.62{mp_ref=Some(&elite[eidx].mp);}
                    if rng.gen::<f64>()<0.72{rp_ref=Some(&elite[eidx].rp);}
                    if rng.gen::<f64>()<0.055{rp_ref=None;}
                }
                let rw_j=if rp_ref.is_some(){(route_w_ls*(0.86+0.50*rng.gen::<f64>())).clamp(route_w_ls*0.70,0.45)}else{0.0};
                construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(jb),mp_ref.map(|v|&**v),rp_ref,rw_j,horizon,time_scale)
            }?;
            if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,attempt,true,&mut cache)?{sol=sol2;mk=mk2;}
            if commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)? {
                let sig2=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
                let cached2=feature_cache.get(&sig2).unwrap();
                maybe_add_elite(&mut elite,EliteParams{jb:cached2.0.clone(),mp:cached2.1.clone(),rp:cached2.2.clone(),score:mk},elite_cap);
            }
            refine_results.push((sol,mk));
        }
    }
    for (sol,mk) in refine_results { push_top_solutions(&mut top_solutions,&sol,mk,15); }

    let post_band = target_margin.saturating_mul(3).max(((pre.avg_op_min * (1.0 + 0.80 * flex01)).round() as u32).max(2));
    let ls_base_indices = select_seed_portfolio_indices(best_solution.as_ref(), best_makespan, &top_solutions, 15, 15, post_band);
    for &i in &ls_base_indices {
        let base_sol=&top_solutions[i].0;
        if let Some((sol2,mk2))=cached_cbm(pre,challenge,base_sol,40,64,12,&mut cache)?{
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            push_top_solutions(&mut top_solutions,&sol2,mk2,15);
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        if pre.high_flex+pre.jobshopness > 0.55 {
            if let Some((sol2,mk2))=cached_cbm(pre,challenge,sol,50,80,14,&mut cache)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            }
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        let sig_best_exact=exact_solution_sig64(sol);
        let gr_base_indices = select_seed_portfolio_indices(Some(sol), best_makespan, &top_solutions, 15, 4, post_band);

        let mut best_improved: Option<(Solution, u32)> = None;

        if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, sol, &mut cache) {
            if mk2 < best_makespan { best_improved = Some((sol2, mk2)); }
        }

        for &idx in &gr_base_indices {
            let base=&top_solutions[idx].0;
            if exact_solution_sig64(base)==sig_best_exact { continue; }
            if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, base, &mut cache) {
                if mk2 < best_makespan && best_improved.as_ref().map_or(true, |x| mk2 < x.1) {
                    best_improved = Some((sol2, mk2));
                }
            }
        }

        if let Some((sol2, mk2)) = best_improved {
            best_makespan = mk2;
            best_solution = Some(sol2.clone());
            push_top_solutions(&mut top_solutions,&sol2,mk2,15);
            save_solution(&sol2)?;
        }
    }

    {
        let bridge_seed_cap = if flex01 > 0.55 { 4usize } else { 3usize };
        let bridge_rounds = if flex01 > 0.60 { 2usize } else { 1usize };
        let bridge_indices = select_seed_portfolio_indices(
            best_solution.as_ref(),
            best_makespan,
            &top_solutions,
            top_solutions.len().min(15),
            bridge_seed_cap,
            post_band,
        );
        let mut bridge_results: Vec<(Solution,u32)> = Vec::with_capacity(bridge_indices.len());
        let mut bridge_sigs: Vec<u64> = Vec::with_capacity(bridge_indices.len());
        for &idx in &bridge_indices {
            let base_sol=&top_solutions[idx].0;
            let base_mk=top_solutions[idx].1;
            let sig=exact_solution_sig64(base_sol);
            if bridge_sigs.contains(&sig) { continue; }
            bridge_sigs.push(sig);
            if let Some((sol2,mk2))=bottleneck_bridge_repair(pre,challenge,base_sol,base_mk,3,bridge_rounds)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                bridge_results.push((sol2,mk2));
            }
        }
        for (sol,mk) in bridge_results {
            push_top_solutions(&mut top_solutions,&sol,mk,15);
        }
    }

    {
        let tenure=((pre.total_ops as f64).sqrt() as usize).clamp(6,16);
        let tabu_seed_cap=effort.hybrid_flow_shop_tabu_seeds;
        let seed_indices = select_seed_portfolio_indices(best_solution.as_ref(), best_makespan, &top_solutions, top_solutions.len().min(15), tabu_seed_cap, post_band.saturating_mul(2));
        let mut seeds: Vec<(Solution,u32)> = Vec::with_capacity(tabu_seed_cap);
        if let Some(ref sol)=best_solution {
            seeds.push((sol.clone(),best_makespan));
        }
        for &idx in &seed_indices {
            if seeds.len()>=tabu_seed_cap{break;}
            let (sol,mk)=(&top_solutions[idx].0, top_solutions[idx].1);
            let sig=exact_solution_sig64(sol);
            if seeds.iter().any(|(s,_)|exact_solution_sig64(s)==sig){continue;}
            seeds.push((sol.clone(),mk));
        }
        for (sol,mk) in top_solutions.iter().take(8) {
            if seeds.len()>=tabu_seed_cap{break;}
            let sig=exact_solution_sig64(sol);
            if seeds.iter().any(|(s,_)|exact_solution_sig64(s)==sig){continue;}
            seeds.push((sol.clone(),*mk));
        }
        for (seed_sol,seed_mk) in seeds {
            let (sol2,mk2)=tabu_ils_hfs(pre,challenge,&seed_sol,seed_mk,effort.hybrid_flow_shop_tabu_iters,tenure,effort.hybrid_flow_shop_tabu_stagnation,effort.hybrid_flow_shop_tabu_reassign_every,effort.hybrid_flow_shop_tabu_kicks,effort.hybrid_flow_shop_tabu_kick_pct,effort.hybrid_flow_shop_tabu_kick_swaps,effort.hybrid_flow_shop_doe_mode,effort.hybrid_flow_shop_doe_switch,effort.hybrid_flow_shop_n8,effort.hybrid_flow_shop_n8_max,effort.hybrid_flow_shop_bottleneck_reassign,effort.hybrid_flow_shop_ig_mode,effort.hybrid_flow_shop_ig_d)?;
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
        }
    }

    
    // Place APRES la boucle de graines tabou : `best_solution` est l'incumbent final et
    // `top_solutions` porte le pool complet. On ne passe QUE par `commit_best` (qui n'ecrit
    // que sur amelioration STRICTE) => cette phase ne peut PAS degrader Q, au pire elle
    // coute du temps. `path_relink == 0` court-circuite tout : sentinelle byte-exacte.
    if effort.hybrid_flow_shop_path_relink > 0 {
        if let Some(inc) = best_solution.clone() {
            let tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(6, 16);
            let mut seeds: Vec<(Solution, u32)> = Vec::with_capacity(2);

            
            // Cible choisie par distance STRUCTURELLE, pas par makespan : le but est de
            // traverser des bassins, et la mesure offline montre que le pool n'est pas
            
            if let Ok(ds_inc) = build_disj_from_solution(pre, challenge, &inc) {
                let mut tgt: Option<(usize, usize)> = None; // (dist, idx)
                for (i, (s, _)) in top_solutions.iter().enumerate() {
                    let Ok(d) = build_disj_from_solution(pre, challenge, s) else { continue };
                    let dist = pr_struct_dist(&ds_inc, &d);
                    if dist > 0 && tgt.map_or(true, |(bd, _)| dist > bd) { tgt = Some((dist, i)); }
                }
                if let Some((_, idx)) = tgt {
                    if let Some((inter_sol, inter_mk)) =
                        pr_forward(pre, challenge, &inc, &top_solutions[idx].0, pre.total_ops.max(1))
                    {
                        seeds.push((inter_sol, inter_mk));
                    }
                }
            }

            // Graine 2 (mode 2 seulement) — TS chaine depuis l'incumbent lui-meme, SANS PR.
            // C'est exactement le controle negatif de la mesure offline : il rend le meme gain
            // que PR sur les nonces 1/6/8 mais ZERO sur le nonce 2 (rang 15). Le mode 2 est
            // donc l'UNION des deux bras, borne superieure des modes 0 et 1.
            if effort.hybrid_flow_shop_path_relink >= 2 {
                seeds.push((inc.clone(), best_makespan));
            }

            for (seed_sol, seed_mk) in seeds {
                let (sol2, mk2) = tabu_ils_hfs(
                    pre, challenge, &seed_sol, seed_mk,
                    effort.hybrid_flow_shop_tabu_iters, tenure,
                    effort.hybrid_flow_shop_tabu_stagnation,
                    effort.hybrid_flow_shop_tabu_reassign_every,
                    effort.hybrid_flow_shop_tabu_kicks,
                    effort.hybrid_flow_shop_tabu_kick_pct,
                    effort.hybrid_flow_shop_tabu_kick_swaps,
                    effort.hybrid_flow_shop_doe_mode,
                    effort.hybrid_flow_shop_doe_switch,
                    effort.hybrid_flow_shop_n8,
                    effort.hybrid_flow_shop_n8_max,
                    effort.hybrid_flow_shop_bottleneck_reassign,
                    effort.hybrid_flow_shop_ig_mode,
                    effort.hybrid_flow_shop_ig_d,
                )?;
                commit_best(save_solution, &mut best_makespan, &mut best_solution, &sol2, mk2)?;
            }
        }
    }

    if let Some(sol)=best_solution { save_solution(&sol)?; }

    Ok(())
}}

pub mod solver {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::job_scheduling::*;
    use super::types::EffortConfig;
    use super::preprocess::build_pre;
    use super::hybrid_flow_shop;

    fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
        let mut cfg = EffortConfig::default_effort();
        if let Some(map) = hyperparameters {
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_iters(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_iters") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_iters(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_seeds") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_seeds(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_stagnation") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_stagnation(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_reassign_every") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_reassign_every(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_kicks") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_kicks(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_kick_pct") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_kick_pct(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_kick_swaps") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_tabu_kick_swaps(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_doe_mode") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_doe_mode(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_doe_switch") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_doe_switch(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_n8") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_n8(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_n8_max") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_n8_max(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_bottleneck_reassign") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_bottleneck_reassign(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_ig_mode") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_ig_mode(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_ig_d") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_ig_d(v as usize); }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_path_relink") {
                if let Some(v) = n.as_u64() { cfg = cfg.with_hybrid_flow_shop_path_relink(v as usize); }
            }
        }
        cfg
    }

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let pre = build_pre(challenge)?;
        let effort = parse_effort(hyperparameters);
        hybrid_flow_shop::solve(challenge, save_solution, &pre, &effort)
    }

    pub fn help() {
        println!("engine t47 — the base engine hybrid_flow_shop engine (verbatim port, SOTA 63016@45s)");
    }
}
