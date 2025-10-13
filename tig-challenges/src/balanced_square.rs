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
        square: Vec<Vec<i32>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self { square: Vec::new() }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub numbers: Vec<i32>,
    #[cfg(not(feature = "hide_verification"))]
    pub baseline_variance: f32,
    #[cfg(feature = "hide_verification")]
    baseline_variance: f32,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(seed.clone());
        let mut numbers = (0..(difficulty.size * difficulty.size))
            .map(|_| rng.gen_range(1..=100))
            .collect::<Vec<i32>>();

        // randomly shuffle and find the best variance over `difficulty.size` tries
        let mut best_variance = f32::MAX;
        for _ in 0..difficulty.size {
            let baseline_square = (0..difficulty.size)
                .map(|i| {
                    (0..difficulty.size)
                        .map(|j| numbers[(i * difficulty.size + j) % numbers.len()])
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>();
            let baseline_variance = calc_variance(difficulty.size, &baseline_square)?;
            if baseline_variance < best_variance {
                best_variance = baseline_variance;
            }
            numbers.shuffle(&mut rng);
        }

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            numbers,
            baseline_variance: best_variance,
        })
    }

    pub fn calc_variance(&self, solution: &Solution) -> Result<f32> {
        calc_variance(self.difficulty.size, &solution.square)
    }

    conditional_pub!(
        fn verify_solution(&self, solution: &Solution) -> Result<()> {
            let variance = self.calc_variance(solution)?;
            let btb = self.difficulty.better_than_baseline as f32 / 1000.0;
            let variance_threshold = self.baseline_variance * (1.0 - btb);
            if variance > variance_threshold {
                Err(anyhow!(
                    "Variance ({}) is greater than threshold ({}) (baseline: {}, better_than_baseline: {}%)",
                    variance,
                    variance_threshold,
                    self.baseline_variance,
                    btb * 100.0
                ))
            } else {
                Ok(())
            }
        }
    );
}

fn calc_variance(size: usize, square: &Vec<Vec<i32>>) -> Result<f32> {
    if square.len() != size || square.iter().any(|row| row.len() != size) {
        return Err(anyhow!("Square size must be exactly {}x{}", size, size));
    }
    let sums = (0..size)
        .flat_map(|i| {
            let h_sum = (0..size).map(|j| square[i][j]).sum::<i32>();
            let v_sum = (0..size).map(|j| square[j][i]).sum::<i32>();
            let d1_sum = (0..size).map(|j| square[(i + j) % size][j]).sum::<i32>();
            let d2_sum = (0..size)
                .map(|j| square[(i + size - j) % size][j])
                .sum::<i32>();
            vec![h_sum, v_sum, d1_sum, d2_sum]
        })
        .collect::<Vec<i32>>();
    let mean = sums.iter().sum::<i32>() as f32 / sums.len() as f32;
    let variance = sums.iter().map(|&x| (x as f32 - mean).powi(2)).sum::<f32>() / sums.len() as f32;
    Ok(variance)
}
