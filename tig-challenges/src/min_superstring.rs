use std::collections::HashSet;

use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub size: usize,
    #[cfg(not(feature = "hide_verification"))]
    pub better_than_baseline: u32,
    #[cfg(feature = "hide_verification")]
    better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            size: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }
}

impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.size as i32, self.better_than_baseline as i32]
    }
}

impl_base64_serde! {
     Solution {
        permuted_strings: Vec<String>,
        superstring_idxs: Vec<usize>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            permuted_strings: Vec::new(),
            superstring_idxs: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub strings: Vec<String>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline_length: usize,
    #[cfg(feature = "hide_verification")]
    baseline_length: usize,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(seed.clone());
        let strings = (0..difficulty.size)
            .map(|_| {
                (0..5)
                    .map(|_| (b'a' + (rng.gen_range(0..26) as u8)) as char)
                    .collect::<HashSet<_>>()
            })
            .collect::<Vec<HashSet<_>>>();

        let mut total_overlap = 0;
        let mut count = 0;
        let half_size = difficulty.size / 2;
        let visited = HashSet::<(usize, usize)>::new();
        for i in 0..difficulty.size {
            loop {
                let j = rng.gen_range(0..strings.len());
                if i == j
                    || (i > half_size && visited.contains(&(j, i)))
                    || (i <= half_size && visited.contains(&(i, j)))
                {
                    continue;
                }
                let overlap = strings[i].union(&strings[j]).count();
                total_overlap += overlap;
                count += 1;
            }
        }
        let avg_overlap = total_overlap as f32 / count as f32;
        let total_length: usize = strings.iter().map(|s| s.len()).sum();
        let baseline_length = (total_length as f32
            - (avg_overlap * (difficulty.size as f32 - 1.0)) / 10.0)
            .ceil() as usize;

        let strings = strings
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            strings,
            baseline_length,
        })
    }

    pub fn calc_superstring(&self, solution: &Solution) -> Result<String> {
        calc_superstring(&solution.permuted_strings, &solution.superstring_idxs)
    }

    conditional_pub!(
        fn verify_solution(&self, solution: &Solution) -> Result<()> {
            if self.strings.len() != solution.permuted_strings.len() {
                return Err(anyhow!(
                    "Mismatch number of strings in challenge vs solution: {} vs {}",
                    self.strings.len(),
                    solution.permuted_strings.len()
                ));
            }
            for (s1, s2) in self.strings.iter().zip(solution.permuted_strings.iter()) {
                let mut v1: Vec<char> = s1.chars().collect();
                let mut v2: Vec<char> = s2.chars().collect();
                v1.sort();
                v2.sort();
                if v1 != v2 {
                    return Err(anyhow!(
                        "Must be permuted strings in challenge vs solution: '{}' vs '{}'",
                        s1,
                        s2
                    ));
                }
            }

            let superstring = self.calc_superstring(solution)?;
            let superstring_length = superstring.len();
            let btb = self.difficulty.better_than_baseline as f32 / 1000.0;
            let length_threshold = (self.baseline_length as f32 * (1.0 - btb)).floor() as usize;
            let actual_btb =
                (1.0 - superstring_length as f32 / self.baseline_length as f32) * 100.0;
            if superstring_length > length_threshold {
                Err(anyhow!(
                    "Superstring length ({}) is greater than threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    superstring_length,
                    length_threshold,
                    self.baseline_length,
                    actual_btb
                ))
            } else {
                println!(
                    "Superstring length ({}) is less than or equal to threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    superstring_length,
                    length_threshold,
                    self.baseline_length,
                    actual_btb
                );
                Ok(())
            }
        }
    );
}

fn calc_superstring(
    permuted_strings: &Vec<String>,
    superstring_idxs: &Vec<usize>,
) -> Result<String> {
    if permuted_strings.len() != superstring_idxs.len() {
        return Err(anyhow!(
            "Length of permuted_strings and superstring_idxs must be the same"
        ));
    }
    let max_len = permuted_strings.iter().map(|s| s.len()).sum::<usize>();
    let len = permuted_strings
        .iter()
        .zip(superstring_idxs.iter())
        .map(|(s, &idx)| s.len() + idx)
        .max()
        .unwrap();
    if len > max_len {
        return Err(anyhow!(
            "Invalid superstring_idxs. Exceeds maximum possible length",
        ));
    }
    let mut superstring = vec![' '; len];
    for (s, &idx) in permuted_strings.iter().zip(superstring_idxs.iter()) {
        for (i, c) in s.chars().enumerate() {
            if superstring[idx + i] != ' ' && superstring[idx + i] != c {
                return Err(anyhow!(
                    "Invalid superstring. Conflict at position {}: '{}' vs '{}'",
                    idx + i,
                    superstring[idx + i],
                    c
                ));
            } else {
                superstring[idx + i] = c;
            }
        }
    }
    if superstring.iter().any(|&c| c == ' ') {
        return Err(anyhow!("Invalid superstring. Contains empty positions"));
    }
    Ok(superstring.iter().collect())
}
