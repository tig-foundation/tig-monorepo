use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, iter::repeat, usize};

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
    pub baseline_superstring: String,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline_length: usize,
    #[cfg(feature = "hide_verification")]
    baseline_superstring: String,
    #[cfg(feature = "hide_verification")]
    baseline_length: usize,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        if difficulty.size < 2 {
            return Err(anyhow!("Size must at least 2"));
        }
        let mut rng = SmallRng::from_seed(seed.clone());
        let strings = (0..difficulty.size)
            .map(|_| {
                (0..5)
                    .map(|_| (b'a' + (rng.gen_range(0..26) as u8)) as char)
                    .collect::<String>()
            })
            .collect::<Vec<String>>();

        let mut baseline_superstring = String::new();
        for _ in 0..difficulty.size {
            let mut strings2 = strings.clone();
            strings2.shuffle(&mut rng);
            let s1 = counter(strings2.pop().unwrap().chars());
            let s2 = counter(strings2.pop().unwrap().chars());
            let (left, overlap, right) = overlaps(s1, s2);
            let mut superstring = left
                .iter()
                .flat_map(|(&c, &count)| repeat(c).take(count))
                .chain(
                    overlap
                        .iter()
                        .flat_map(|(&c, &count)| repeat(c).take(count)),
                )
                .chain(right.iter().flat_map(|(&c, &count)| repeat(c).take(count)))
                .collect::<String>();

            while !strings2.is_empty() {
                // for each remaining string, add to left or right based on max overlap
                let mut left = counter(strings2.pop().unwrap().chars());
                let mut right = left.clone();
                remove_overlaps(&mut left, superstring[..5].chars());
                remove_overlaps(
                    &mut right,
                    superstring[superstring.len() - 5..].chars().rev(),
                );
                if left.values().sum::<usize>() > right.values().sum::<usize>() {
                    // append right
                    superstring = format!(
                        "{}{}",
                        superstring,
                        right
                            .iter()
                            .flat_map(|(&c, &count)| repeat(c).take(count))
                            .collect::<String>()
                    );
                } else {
                    // prepend left
                    superstring = format!(
                        "{}{}",
                        left.iter()
                            .flat_map(|(&c, &count)| repeat(c).take(count))
                            .collect::<String>(),
                        superstring
                    );
                }
            }
            if baseline_superstring.len() == 0 || superstring.len() < baseline_superstring.len() {
                baseline_superstring = superstring;
            }
        }

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            strings,
            baseline_length: baseline_superstring.len(),
            baseline_superstring,
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

fn counter(chars: impl Iterator<Item = char>) -> HashMap<char, usize> {
    let mut map = HashMap::new();
    for c in chars {
        *map.entry(c).or_default() += 1;
    }
    map
}

fn remove_overlaps(counter: &mut HashMap<char, usize>, chars_iter: impl Iterator<Item = char>) {
    for c in chars_iter {
        if let Some(count) = counter.get_mut(&c) {
            if *count > 0 {
                *count -= 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    counter.retain(|_, count| *count > 0);
}

fn overlaps(
    mut counter1: HashMap<char, usize>,
    mut counter2: HashMap<char, usize>,
) -> (
    HashMap<char, usize>,
    HashMap<char, usize>,
    HashMap<char, usize>,
) {
    let overlap = counter1
        .iter_mut()
        .filter_map(|(&c, count1)| match counter2.get_mut(&c) {
            Some(count2) => {
                let n = (*count1).min(*count2);
                *count1 -= n;
                *count2 -= n;
                Some((c, n))
            }
            None => None,
        })
        .collect();
    counter1.retain(|_, count| *count > 0);
    counter2.retain(|_, count| *count > 0);
    (counter1, overlap, counter2)
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
