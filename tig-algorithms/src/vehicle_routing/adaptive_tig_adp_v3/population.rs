pub mod individual {
    use crate::tig_adaptive::TIGState;
    use crate::delta::DeltaTables;

    /// A lightweight individual wrapping a `TIGState`, cached fitness, and optional cached DeltaTables.
    pub struct Individual {
        pub state: TIGState,
        pub fitness: f64,
        pub delta_cache: Option<DeltaTables>,
    }

    impl Clone for Individual {
        fn clone(&self) -> Self {
            Self { state: self.state.clone(), fitness: self.fitness, delta_cache: None }
        }
    }

    impl Individual {
        pub fn new(state: TIGState, fitness: f64) -> Self {
            Self { state, fitness, delta_cache: None }
        }

        pub fn with_delta(state: TIGState, fitness: f64, delta: DeltaTables) -> Self {
            Self { state, fitness, delta_cache: Some(delta) }
        }

        pub fn from_state(state: TIGState, fitness: f64) -> Self {
            Self::new(state, fitness)
        }

        pub fn set_delta(&mut self, delta: DeltaTables) {
            self.delta_cache = Some(delta);
        }
    }
}

pub mod population {
    use crate::population::individual::Individual;
    use crate::tig_adaptive::TIGState;
    use crate::adp::vfa::VFA;
    use crate::local_search::LocalSearch;
    use crate::utilities::IZS;
    use rand::Rng;

    /// Simple micro-genetic population for route improvement
    pub struct Population<'a, R: Rng> {
        pub individuals: Vec<Individual>,
        pub rng: &'a mut R,
        pub vfa: &'a VFA,
        pub ls: &'a LocalSearch,
        pub izs: &'a IZS,
    }

    impl<'a, R: Rng> Population<'a, R> {
        pub fn new(rng: &'a mut R, vfa: &'a VFA, ls: &'a LocalSearch, izs: &'a IZS) -> Self {
            Self { individuals: Vec::new(), rng, vfa, ls, izs }
        }

        pub fn evaluate_fitness(&mut self, state: &TIGState) -> f64 {
            self.vfa.estimate(state, &mut *self.rng)
        }

        pub fn initialize_from(&mut self, seed: TIGState, pop_size: usize) {
            self.individuals.clear();
            for _ in 0..pop_size {
                let mut s = seed.clone();
                // small perturbation via random relocate
                if s.route.len() > 3 {
                    let i = self.rng.gen_range(1..s.route.len()-1);
                    let j = self.rng.gen_range(1..s.route.len());
                    let node = s.route.remove(i);
                    let insert_pos = if j > i { j - 1 } else { j };
                    s.route.insert(insert_pos, node);
                    s.recompute_times_from(std::cmp::min(i, insert_pos));
                }
                let fitness = self.evaluate_fitness(&s);
                // compute and store DeltaTables cache for this individual
                let dt = crate::delta::DeltaTables::from_state(&s);
                self.individuals.push(Individual::with_delta(s, fitness, dt));
            }
        }

        pub fn best(&self) -> Option<&Individual> {
            self.individuals.iter().max_by(|a,b| a.fitness.partial_cmp(&b.fitness).unwrap())
        }

        pub fn step_evolve(&mut self) {
            use crate::population::crossover::one_point_crossover;
            use crate::population::mutation::{swap_mutation, relocate_mutation};

            if self.individuals.len() < 2 { return; }

            // tournament selection + produce 2 offspring
            let n = self.individuals.len();
            let a = self.rng.gen_range(0..n);
            let b = self.rng.gen_range(0..n);
            let parent_a = &self.individuals[a].state;
            let parent_b = &self.individuals[b].state;

            let (mut c1, mut c2) = one_point_crossover(parent_a, parent_b, &mut *self.rng);

            // quick local search refinement: use a cheaper local search instance to limit time spent per offspring
            let quick_ls = crate::local_search::LocalSearch::new((self.ls.time_limit_ms / 2).max(1), self.ls.neighborhood_size);
            
            // Sequential offspring refinement (replaced parallel thread execution)
            // Refine first child
            {
                let mut ss = vec![c1];
                quick_ls.optimize_multi_with_config(&mut ss, self.izs);
                c1 = ss.remove(0);
            }

            // Refine second child
            {
                let mut ss = vec![c2];
                quick_ls.optimize_multi_with_config(&mut ss, self.izs);
                c2 = ss.remove(0);
            }

            // mutations
            if self.rng.gen_bool(0.2) { swap_mutation(&mut c1, &mut *self.rng); }
            if self.rng.gen_bool(0.2) { relocate_mutation(&mut c2, &mut *self.rng); }

            let f1 = self.evaluate_fitness(&c1);
            let f2 = self.evaluate_fitness(&c2);

            // compute DeltaTables for offspring and reuse later
            let dt1 = crate::delta::DeltaTables::from_state(&c1);
            let dt2 = crate::delta::DeltaTables::from_state(&c2);

            // replace worst two in population
            self.individuals.sort_by(|a,b| a.fitness.partial_cmp(&b.fitness).unwrap());
            let len = self.individuals.len();
            if len >= 2 {
                self.individuals[0] = Individual::with_delta(c1, f1, dt1);
                self.individuals[1] = Individual::with_delta(c2, f2, dt2);
            }
        }
    }
}

pub mod crossover {
    use crate::tig_adaptive::TIGState;
    use rand::Rng;

    /// One-point crossover for routes: pick cut points and exchange tails.
    pub fn one_point_crossover<R: Rng>(a: &TIGState, b: &TIGState, rng: &mut R) -> (TIGState, TIGState) {
        let na = a.route.len();
        let nb = b.route.len();
        if na < 3 || nb < 3 {
            return (a.clone(), b.clone());
        }

        let cut_a = rng.gen_range(1..na-1);
        let cut_b = rng.gen_range(1..nb-1);

        let mut r1: Vec<usize> = Vec::with_capacity(cut_a + (nb - cut_b));
        r1.extend_from_slice(&a.route.nodes[0..cut_a]);
        r1.extend_from_slice(&b.route.nodes[cut_b..]);

        let mut r2: Vec<usize> = Vec::with_capacity(cut_b + (na - cut_a));
        r2.extend_from_slice(&b.route.nodes[0..cut_b]);
        r2.extend_from_slice(&a.route.nodes[cut_a..]);

        let s1 = TIGState::new(
            r1,
            a.time,
            a.max_capacity,
            a.tw_start.as_ref().to_vec(),
            a.tw_end.as_ref().to_vec(),
            a.service.as_ref().to_vec(),
            a.distance_matrix_nested(),
            a.demands.as_ref().to_vec(),
        );
        let s2 = TIGState::new(
            r2,
            b.time,
            b.max_capacity,
            b.tw_start.as_ref().to_vec(),
            b.tw_end.as_ref().to_vec(),
            b.service.as_ref().to_vec(),
            b.distance_matrix_nested(),
            b.demands.as_ref().to_vec(),
        );

        (s1, s2)
    }
}

pub mod mutation {
    use crate::tig_adaptive::TIGState;
    use rand::Rng;

    /// Simple mutation operators: swap two nodes, or relocate one node.
    pub fn swap_mutation<R: Rng>(state: &mut TIGState, rng: &mut R) {
        let n = state.route.len();
        if n < 4 { return; }
        let i = rng.gen_range(1..n-1);
        let j = rng.gen_range(1..n-1);
        state.route.swap(i, j);
        state.recompute_times_from(std::cmp::min(i, j));
    }

    pub fn relocate_mutation<R: Rng>(state: &mut TIGState, rng: &mut R) {
        let n = state.route.len();
        if n < 4 { return; }
        let from = rng.gen_range(1..n-1);
        let to = rng.gen_range(1..n);
        let node = state.route.remove(from);
        let insert_pos = if to > from { to - 1 } else { to };
        state.route.insert(insert_pos, node);
        state.recompute_times_from(std::cmp::min(from, insert_pos));
    }
}

pub use individual::Individual;
pub use population::Population;
