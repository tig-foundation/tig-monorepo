use std::collections::HashSet;

use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
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
        arrangement: Vec<Vec<usize>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            arrangement: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Baseline {
    pub arrangement: Vec<Vec<usize>>,
    pub variance: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub numbers: Vec<i32>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline: Baseline,
    #[cfg(feature = "hide_verification")]
    baseline: Baseline,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        if difficulty.size == 0 {
            return Err(anyhow!("Size must be greater than 0"));
        }
        let mut rng = SmallRng::from_seed(seed.clone());
        let numbers = (0..(difficulty.size * difficulty.size))
            .map(|_| rng.gen_range(1..=100))
            .collect::<Vec<i32>>();

        // randomly shuffle and find the best variance over `difficulty.size` tries
        let mut baseline = Baseline {
            arrangement: Vec::new(),
            variance: f32::MAX,
        };
        let mut indices = (0..numbers.len()).collect::<Vec<usize>>();
        for _ in 0..difficulty.size {
            indices.shuffle(&mut rng);
            let arrangement = indices
                .chunks(difficulty.size)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<Vec<usize>>>();
            let variance = calc_variance(&numbers, &arrangement)?;
            if variance < baseline.variance {
                baseline.variance = variance;
                baseline.arrangement = arrangement;
            }
        }

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            numbers,
            baseline,
        })
    }

    pub fn calc_variance(&self, solution: &Solution) -> Result<f32> {
        calc_variance(&self.numbers, &solution.arrangement)
    }

    conditional_pub!(
        fn verify_solution(&self, solution: &Solution) -> Result<()> {
            let variance = self.calc_variance(solution)?;
            let btb = self.difficulty.better_than_baseline as f32 / 1000.0;
            let variance_threshold = self.baseline.variance * (1.0 - btb);
            let actual_btb = (1.0 - variance / self.baseline.variance) * 100.0;
            if variance > variance_threshold {
                Err(anyhow!(
                    "Variance ({}) is greater than threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    variance,
                    variance_threshold,
                    self.baseline.variance,
                    actual_btb
                ))
            } else {
                println!(
                    "Variance ({}) is less than or equal to threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    variance,
                    variance_threshold,
                    self.baseline.variance,
                    actual_btb
                );
                Ok(())
            }
        }
    );
}

fn calc_variance(numbers: &Vec<i32>, arrangement: &Vec<Vec<usize>>) -> Result<f32> {
    if (0..numbers.len()).collect::<HashSet<_>>()
        != arrangement
            .iter()
            .flatten()
            .cloned()
            .collect::<HashSet<_>>()
    {
        return Err(anyhow!("Arrangement must use all numbers exactly once",));
    }
    if !arrangement.iter().all(|row| row.len() == arrangement.len()) {
        return Err(anyhow!("Arrangement must be a square",));
    }
    let size = arrangement.len();
    let sums = (0..arrangement.len())
        .flat_map(|i| {
            let h_sum = (0..size).map(|j| numbers[arrangement[i][j]]).sum::<i32>();
            let v_sum = (0..size).map(|j| numbers[arrangement[j][i]]).sum::<i32>();
            vec![h_sum, v_sum]
        })
        .chain(vec![
            (0..size).map(|i| numbers[arrangement[i][i]]).sum::<i32>(),
            (0..size)
                .map(|i| numbers[arrangement[i][size - 1 - i]])
                .sum::<i32>(),
        ])
        .collect::<Vec<i32>>();
    let mean = sums.iter().sum::<i32>() as f32 / sums.len() as f32;
    let variance = sums.iter().map(|&x| (x as f32 - mean).powi(2)).sum::<f32>() / sums.len() as f32;
    Ok(variance)
}
