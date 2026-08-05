// Operation-level model of a job_scheduling instance.
//
// Structural note (verified against tig-challenges/src/job_scheduling/mod.rs
// and scenarios.rs): `flow_shop` and `hybrid_flow_shop` both set
// `flow_structure = 0.0`, so the generator builds exactly one route
// (mod.rs:62), maps every product onto it (mod.rs:88-94) and fills it with the
// identity operation order (mod.rs:107-113) plus a few reentrant repeats
// (mod.rs:119-126). All 50 jobs therefore share one machine route; only the
// processing times differ per product.
//
// The dominant term in these instances is a single bottleneck machine: its
// total load is typically ~2x the next busiest machine's, and the dispatching
// SOTA baseline lands only 4-25% above that load. So the objective is almost
// entirely "keep the bottleneck busy", which is a sequencing problem on the
// disjunctive graph rather than a permutation-flow-shop problem.

use tig_challenges::job_scheduling::Challenge;

/// Flattened per-operation view of the instance.
pub struct Model {
    pub n_jobs: usize,
    pub n_machines: usize,
    pub n_ops: usize,
    /// canonical job index -> product index
    pub job_prod: Vec<usize>,
    /// product -> route length
    pub prod_len: Vec<usize>,
    /// job -> global index of its first operation
    pub job_base: Vec<u32>,
    /// job -> number of operations
    pub job_len: Vec<u32>,
    /// operation -> owning job
    pub op_job: Vec<u32>,
    /// operation -> position along its route
    pub op_k: Vec<u32>,
    /// operation -> [lo, hi) slice of `opts`
    pub opt_lo: Vec<u32>,
    pub opt_hi: Vec<u32>,
    /// flattened eligible (machine, processing time) pairs, sorted by machine
    pub opts: Vec<(u32, u32)>,
}

impl Model {
    pub fn build(challenge: &Challenge) -> Model {
        let n_jobs = challenge.num_jobs;
        let n_machines = challenge.num_machines;

        let mut job_prod = Vec::with_capacity(n_jobs);
        for (product, count) in challenge.jobs_per_product.iter().enumerate() {
            for _ in 0..*count {
                job_prod.push(product);
            }
        }

        let prod_len: Vec<usize> = challenge
            .product_processing_times
            .iter()
            .map(|ops| ops.len())
            .collect();

        // Per product, the sorted machine-option list of each route position.
        // Sorting by machine id makes every downstream tie-break independent of
        // HashMap iteration order, which keeps the run deterministic.
        let mut prod_opts: Vec<Vec<Vec<(u32, u32)>>> = Vec::with_capacity(prod_len.len());
        for ops in challenge.product_processing_times.iter() {
            let mut per_pos = Vec::with_capacity(ops.len());
            for op in ops.iter() {
                let mut e: Vec<(u32, u32)> = op.iter().map(|(&m, &t)| (m as u32, t)).collect();
                e.sort_unstable();
                per_pos.push(e);
            }
            prod_opts.push(per_pos);
        }

        let mut job_base = Vec::with_capacity(n_jobs);
        let mut job_len = Vec::with_capacity(n_jobs);
        let mut op_job = Vec::new();
        let mut op_k = Vec::new();
        let mut opt_lo = Vec::new();
        let mut opt_hi = Vec::new();
        let mut opts: Vec<(u32, u32)> = Vec::new();

        for j in 0..n_jobs {
            let p = job_prod[j];
            job_base.push(op_job.len() as u32);
            job_len.push(prod_len[p] as u32);
            for k in 0..prod_len[p] {
                op_job.push(j as u32);
                op_k.push(k as u32);
                opt_lo.push(opts.len() as u32);
                opts.extend_from_slice(&prod_opts[p][k]);
                opt_hi.push(opts.len() as u32);
            }
        }

        let n_ops = op_job.len();

        Model {
            n_jobs,
            n_machines,
            n_ops,
            job_prod,
            prod_len,
            job_base,
            job_len,
            op_job,
            op_k,
            opt_lo,
            opt_hi,
            opts,
        }
    }

    /// Processing time of operation `o` on machine `m`, if eligible.
    #[inline]
    pub fn dur_on(&self, o: usize, m: u32) -> Option<u32> {
        let lo = self.opt_lo[o] as usize;
        let hi = self.opt_hi[o] as usize;
        for i in lo..hi {
            if self.opts[i].0 == m {
                return Some(self.opts[i].1);
            }
        }
        None
    }

    #[inline]
    pub fn n_opts(&self, o: usize) -> usize {
        (self.opt_hi[o] - self.opt_lo[o]) as usize
    }
}
