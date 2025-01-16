/*!
Copyright 2024 MateusMelo

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{Rng, RngCore, SeedableRng, Error};

use std::collections::HashMap;
use tig_challenges::satisfiability::*;
use std::num::Wrapping;
use std::fmt;

const NN: usize = 312;
const MM: usize = 156;
const ONE: Wrapping<u64> = Wrapping(1);
const MATRIX_A: Wrapping<u64> = Wrapping(0xb502_6f5a_a966_19e9);
const UM: Wrapping<u64> = Wrapping(0xffff_ffff_8000_0000); // Most significant 33 bits
const LM: Wrapping<u64> = Wrapping(0x7fff_ffff); // Least significant 31 bits

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mt19937_64 {
    idx: usize,
    state: [Wrapping<u64>; NN],
}


impl Default for Mt19937_64 {
    #[inline]
    fn default() -> Self {
        Self::new_unseeded()
    }
}

impl From<[u8; 8]> for Mt19937_64 {
    #[inline]
    fn from(seed: [u8; 8]) -> Self {
        Self::new(u64::from_le_bytes(seed))
    }
}

impl From<u64> for Mt19937_64 {
    
    #[inline]
    fn from(seed: u64) -> Self {
        Self::new(seed)
    }
}

impl From<[u64; NN]> for Mt19937_64 {
    #[inline]
    fn from(key: [u64; NN]) -> Self {
        let mut mt = Self {
            idx: NN,
            state: [Wrapping(0); NN],
        };
        for (sample, out) in key.iter().copied().zip(mt.state.iter_mut()) {
            *out = Wrapping(untemper(sample));
        }
        mt
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RecoverRngError {
    TooFewSamples(usize),

    TooManySamples(usize),
}


impl fmt::Display for RecoverRngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSamples(expected) => {
                write!(f, "Too few samples given to recover: expected {}", expected)
            }
            Self::TooManySamples(expected) => write!(
                f,
                "Too many samples given to recover: expected {}",
                expected
            ),
        }
    }
}

impl std::error::Error for RecoverRngError {}

impl TryFrom<&[u64]> for Mt19937_64 {
    type Error = RecoverRngError;

    #[inline]
    fn try_from(key: &[u64]) -> Result<Self, Self::Error> {
        Self::recover(key.iter().copied())
    }
}

impl Mt19937_64 {
    pub const DEFAULT_SEED: u64 = 5489_u64;

    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut mt = Self {
            idx: 0,
            state: [Wrapping(0); NN],
        };
        mt.reseed(seed);
        mt
    }
    #[inline]
    #[must_use]
    pub fn new_with_key<I>(key: I) -> Self
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: Clone,
    {
        let mut mt = Self {
            idx: 0,
            state: [Wrapping(0); NN],
        };
        mt.reseed_with_key(key);
        mt
    }

    #[inline]
    #[must_use]
    pub fn new_unseeded() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // Failing this check indicates that, somehow, the structure
        // was not initialized.
        debug_assert!(self.idx != 0);
        if self.idx >= NN {
            fill_next_state(self);
        }
        let Wrapping(x) = self.state[self.idx];
        self.idx += 1;
        temper(x)
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    #[inline]
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        const CHUNK: usize = size_of::<u64>();
        let mut dest_chunks = dest.chunks_exact_mut(CHUNK);

        for next in &mut dest_chunks {
            let chunk: [u8; CHUNK] = self.next_u64().to_le_bytes();
            next.copy_from_slice(&chunk);
        }

        let remainder = dest_chunks.into_remainder();
        if remainder.is_empty() {
            return;
        }
        remainder
            .iter_mut()
            .zip(self.next_u64().to_le_bytes().iter())
            .for_each(|(cell, &byte)| {
                *cell = byte;
            });
    }

    #[inline]
    pub fn recover<I>(key: I) -> Result<Self, RecoverRngError>
    where
        I: IntoIterator<Item = u64>,
    {
        let mut mt = Self {
            idx: NN,
            state: [Wrapping(0); NN],
        };
        let mut state = mt.state.iter_mut();
        for sample in key {
            let out = state.next().ok_or(RecoverRngError::TooManySamples(NN))?;
            *out = Wrapping(untemper(sample));
        }

        if state.next().is_none() {
            Ok(mt)
        } else {
            Err(RecoverRngError::TooFewSamples(NN))
        }
    }

    #[inline]
    pub fn reseed(&mut self, seed: u64) {
        self.idx = NN;
        self.state[0] = Wrapping(seed);
        for i in 1..NN {
            self.state[i] = Wrapping(6_364_136_223_846_793_005)
                * (self.state[i - 1] ^ (self.state[i - 1] >> 62))
                + Wrapping(i as u64);
        }
    }


    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn reseed_with_key<I>(&mut self, key: I)
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: Clone,
    {
        self.reseed(19_650_218_u64);
        let mut i = 1_usize;
        for (j, piece) in key.into_iter().enumerate().cycle().take(NN) {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 62))
                    * Wrapping(3_935_559_000_370_003_845)))
                + Wrapping(piece)
                + Wrapping(j as u64);
            i += 1;
            if i >= NN {
                self.state[0] = self.state[NN - 1];
                i = 1;
            }
        }
        for _ in 0..NN - 1 {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 62))
                    * Wrapping(2_862_933_555_777_941_757)))
                - Wrapping(i as u64);
            i += 1;
            if i >= NN {
                self.state[0] = self.state[NN - 1];
                i = 1;
            }
        }
        self.state[0] = Wrapping(1 << 63);
    }
}

#[inline]
fn temper(mut x: u64) -> u64 {
    x ^= (x >> 29) & 0x5555_5555_5555_5555;
    x ^= (x << 17) & 0x71d6_7fff_eda6_0000;
    x ^= (x << 37) & 0xfff7_eee0_0000_0000;
    x ^= x >> 43;
    x
}

#[inline]
fn untemper(mut x: u64) -> u64 {
    x ^= x >> 43;

    x ^= (x << 37) & 0xfff7_eee0_0000_0000;


    x ^= (x << 17) & 0x0000_0003_eda6_0000;
    x ^= (x << 17) & 0x0006_7ffc_0000_0000;
    x ^= (x << 17) & 0x71d0_0000_0000_0000;


    x ^= (x >> 29) & 0x0000_0005_5555_5540;
    x ^= (x >> 29) & 0x0000_0000_0000_0015;

    x
}

#[inline]
fn fill_next_state(rng: &mut Mt19937_64) {
    for i in 0..NN - MM {
        let x = (rng.state[i] & UM) | (rng.state[i + 1] & LM);
        rng.state[i] = rng.state[i + MM] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    }
    for i in NN - MM..NN - 1 {
        let x = (rng.state[i] & UM) | (rng.state[i + 1] & LM);
        rng.state[i] = rng.state[i + MM - NN] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    }
    let x = (rng.state[NN - 1] & UM) | (rng.state[0] & LM);
    rng.state[NN - 1] = rng.state[MM - 1] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
    rng.idx = 0;
}

impl SeedableRng for Mt19937_64 {
    type Seed = [u8; 8];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::from(seed)
    }
}

impl RngCore for Mt19937_64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        Self::next_u64(self)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        Self::next_u32(self)
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        Self::fill_bytes(self, dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Self::fill_bytes(self, dest);
        Ok(())
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    //let mut rng = Mt19937::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let mut rng = Mt19937_64::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
  
    let mut p_single = vec![false; challenge.difficulty.num_variables];
    let mut n_single = vec![false; challenge.difficulty.num_variables];

    let mut clauses_ = challenge.clauses.clone();
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses_.len());

    let mut dead = false;

    // let r1 = rng.gen_bool(0.5);
    // let r2 = rng2.gen_bool(0.5);

    // if r1 != r2 {
    //     println!("valuer differ {} {}", r1, r2);
    // }

    while !(dead) {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;
            for (i, l) in c.iter().enumerate() {
                if (p_single[(l.abs() - 1) as usize] && *l > 0)
                    || (n_single[(l.abs() - 1) as usize] && *l < 0)
                    || c[(i + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[(l.abs() - 1) as usize]
                    || n_single[(l.abs() - 1) as usize]
                    || c[(i + 1)..].contains(&l)
                {
                    done = false;
                    continue;
                } else {
                    c_.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            };
            match c_[..] {
                [l] => {
                    done = false;
                    if l > 0 {
                        if n_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            p_single[(l.abs() - 1) as usize] = true;
                        }
                    } else {
                        if p_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            n_single[(l.abs() - 1) as usize] = true;
                        }
                    }
                }
                [] => {
                    dead = true;
                    break;
                }
                _ => {
                    clauses.push(c_);
                }
            }
        }
        if done {
            break;
        } else {
            clauses_ = clauses;
            clauses = Vec::with_capacity(clauses_.len());
        }
    }

    if dead {
        return Ok(None);
    }

    let num_variables = challenge.difficulty.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![vec![]; num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![vec![]; num_variables];

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        if p_single[v] {
            variables[v] = true
        } else if n_single[v] {
            variables[v] = false
        } else {
            variables[v] = rng.gen_bool(0.5)
        }
    }
    let mut num_good_so_far: Vec<usize> = vec![0; num_clauses];

    for (i, &ref c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
                if variables[var] {
                    num_good_so_far[i] += 1
                }
            } else {
                n_clauses[var].push(i);
                if !variables[var] {
                    num_good_so_far[i] += 1
                }
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);
    let mut residual_indices = HashMap::with_capacity(num_clauses);

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices.insert(i, residual_.len() - 1);
        }
    }

    let mut attempts = 0;
    loop {
        if attempts >= num_variables * 25 {
            return Ok(None);
        }
        if !residual_.is_empty() {
            let i = residual_[0];
            let mut min_sad = clauses.len();
            let mut v_min_sad = vec![];
            let c = &clauses[i];
            for &l in c {
                let mut sad = 0 as usize;
                if variables[(l.abs() - 1) as usize] {
                    for &c in &p_clauses[(l.abs() - 1) as usize] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                } else {
                    for &c in &n_clauses[(l.abs() - 1) as usize] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                }

                if sad < min_sad {
                    min_sad = sad;
                    v_min_sad = vec![(l.abs() - 1) as usize];
                } else if sad == min_sad {
                    v_min_sad.push((l.abs() - 1) as usize);
                }
            }
            let v = if min_sad == 0 {
                if v_min_sad.len() == 1 {
                    v_min_sad[0]
                } else {
                    v_min_sad[rng.gen_range(0..v_min_sad.len())]
                }
            } else {
                if rng.gen_bool(0.5) {
                    let l = c[rng.gen_range(0..c.len())];
                    (l.abs() - 1) as usize
                } else {
                    v_min_sad[rng.gen_range(0..v_min_sad.len())]
                }
            };

            if variables[v] {
                for &c in &n_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
                for &c in &p_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }
            } else {
                for &c in &n_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }

                for &c in &p_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
            }

            variables[v] = !variables[v];
        } else {
            break;
        }
        attempts += 1;
    }

    return Ok(Some(Solution { variables }));
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
