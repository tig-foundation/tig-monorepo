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
pub struct Baseline {
    pub permuted_strings: Vec<String>,
    pub superstring_idxs: Vec<usize>,
    pub superstring: String,
    pub length: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub strings: Vec<String>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline: Baseline,
    #[cfg(feature = "hide_verification")]
    baseline: Baseline,
}

impl Challenge {
    pub const N_CHARS: usize = 5;

    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        if difficulty.size < 2 {
            return Err(anyhow!("Size must at least 2"));
        }
        let mut rng = SmallRng::from_seed(seed.clone());
        let strings = (0..difficulty.size)
            .map(|_| {
                (0..Self::N_CHARS)
                    .map(|_| (b'a' + (rng.gen_range(0..26) as u8)) as char)
                    .collect::<String>()
            })
            .collect::<Vec<String>>();

        let mut baseline = Baseline {
            permuted_strings: Vec::new(),
            superstring_idxs: Vec::new(),
            superstring: String::new(),
            length: 0,
        };
        for _ in 0..difficulty.size {
            let mut superstring_idxs = HashMap::<usize, usize>::new();
            let mut string_idxs = (0..strings.len()).collect::<Vec<usize>>();
            string_idxs.shuffle(&mut rng);
            let s1_idx = string_idxs.pop().unwrap();
            let s2_idx = string_idxs.pop().unwrap();
            let s1 = counter(strings[s1_idx].chars());
            let s2 = counter(strings[s2_idx].chars());
            let (mut left, mut overlap, mut right) = overlaps(s1, s2);
            left.sort_by_key(|x| x.0);
            overlap.sort_by_key(|x| x.0);
            right.sort_by_key(|x| x.0);
            let mut superstring = left
                .into_iter()
                .flat_map(|(c, count)| repeat(c).take(count))
                .chain(
                    overlap
                        .into_iter()
                        .flat_map(|(c, count)| repeat(c).take(count)),
                )
                .chain(
                    right
                        .into_iter()
                        .flat_map(|(c, count)| repeat(c).take(count)),
                )
                .collect::<String>();
            superstring_idxs.insert(s1_idx, 0);
            superstring_idxs.insert(s2_idx, superstring.len() - Self::N_CHARS);

            while !string_idxs.is_empty() {
                // for each remaining string, add to left or right based on max overlap
                let s_idx = string_idxs.pop().unwrap();
                let mut left = counter(strings[s_idx].chars());
                let mut right = left.clone();
                remove_overlaps(&mut left, superstring[..Self::N_CHARS].chars());
                remove_overlaps(
                    &mut right,
                    superstring[superstring.len() - Self::N_CHARS..]
                        .chars()
                        .rev(),
                );
                if left.values().sum::<usize>() > right.values().sum::<usize>() {
                    // append right
                    let mut right = right.into_iter().collect::<Vec<(char, usize)>>();
                    right.sort_by_key(|x| x.0);
                    superstring = format!(
                        "{}{}",
                        superstring,
                        right
                            .into_iter()
                            .flat_map(|(c, count)| repeat(c).take(count))
                            .collect::<String>()
                    );
                    superstring_idxs.insert(s_idx, superstring.len() - Self::N_CHARS);
                } else {
                    // prepend left
                    let offset = left.values().sum::<usize>();
                    let mut left = left.into_iter().collect::<Vec<(char, usize)>>();
                    left.sort_by_key(|x| x.0);
                    superstring = format!(
                        "{}{}",
                        left.into_iter()
                            .flat_map(|(c, count)| repeat(c).take(count))
                            .collect::<String>(),
                        superstring
                    );
                    superstring_idxs.values_mut().for_each(|idx| *idx += offset);
                    superstring_idxs.insert(s_idx, 0);
                }
            }
            if baseline.superstring.len() == 0 || superstring.len() < baseline.superstring.len() {
                baseline.superstring = superstring;
                baseline.superstring_idxs =
                    (0..strings.len()).map(|i| superstring_idxs[&i]).collect();
                baseline.permuted_strings = baseline
                    .superstring_idxs
                    .iter()
                    .map(|&start| baseline.superstring[start..start + Self::N_CHARS].to_string())
                    .collect();
                baseline.length = baseline.superstring.len();
            }
        }

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            strings,
            baseline,
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
            let length_threshold = (self.baseline.length as f32 * (1.0 - btb)).floor() as usize;
            let actual_btb =
                (1.0 - superstring_length as f32 / self.baseline.length as f32) * 100.0;
            if superstring_length > length_threshold {
                Err(anyhow!(
                    "Superstring length ({}) is greater than threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    superstring_length,
                    length_threshold,
                    self.baseline.length,
                    actual_btb
                ))
            } else {
                println!(
                    "Superstring length ({}) is less than or equal to threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    superstring_length,
                    length_threshold,
                    self.baseline.length,
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
) -> (Vec<(char, usize)>, Vec<(char, usize)>, Vec<(char, usize)>) {
    let overlap: HashMap<char, usize> = counter1
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
    (
        counter1.into_iter().collect(),
        overlap.into_iter().collect(),
        counter2.into_iter().collect(),
    )
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
