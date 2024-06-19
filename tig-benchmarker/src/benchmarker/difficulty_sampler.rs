use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
};
use tig_structs::core::*;

const PADDING_FACTOR: f32 = 0.2;
const DECAY: f32 = 0.7;
const INITIAL_SOLUTIONS_WEIGHT: f32 = 500.0;
const SOLUTIONS_MULTIPLIER: f32 = 10.0;

#[derive(Debug, Clone)]
pub struct Weights {
    pub qualifier: f32,
    pub solutions: f32,
    pub within_range: bool,
}

impl Weights {
    pub fn new() -> Self {
        Self {
            qualifier: 1.0,
            solutions: INITIAL_SOLUTIONS_WEIGHT,
            within_range: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DifficultySampler {
    pub min_difficulty: Vec<i32>,
    pub padding: Vec<usize>,
    pub dimensions: Vec<usize>,
    pub weights: Vec<Vec<Weights>>,
    pub distribution: Option<WeightedIndex<f32>>,
}

impl DifficultySampler {
    pub fn new() -> Self {
        Self {
            min_difficulty: Vec::new(),
            padding: Vec::new(),
            dimensions: Vec::new(),
            weights: Vec::new(),
            distribution: None,
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> Vec<i32> {
        // samples an index from the distribution
        let idx = self
            .distribution
            .clone()
            .expect("You must update sampler first")
            .sample(rng);

        // convert index into difficulty
        let num_cols = self.dimensions[1] + self.padding[1];
        let x = idx / num_cols;
        let y = idx % num_cols;
        vec![
            x as i32 + self.min_difficulty[0],
            y as i32 + self.min_difficulty[1],
        ]
    }

    pub fn update_with_block_data(
        &mut self,
        min_difficulty: &Vec<i32>,
        block_data: &ChallengeBlockData,
    ) {
        assert_eq!(
            min_difficulty.len(),
            2,
            "Only difficulty with 2 parameters are supported"
        );

        let left_pad = (0..2)
            .into_iter()
            .map(|i| match self.min_difficulty.get(i) {
                Some(x) => x - min_difficulty[i],
                None => 0,
            })
            .collect();
        self.min_difficulty = min_difficulty.clone();
        self.update_dimensions_and_padding(block_data);
        let size = (0..2)
            .into_iter()
            .map(|i| self.dimensions[i] + self.padding[i])
            .collect();
        self.resize_weights(&left_pad, &size);

        self.update_qualifier_weights(block_data);
        self.update_valid_range(block_data);
        self.update_distributions();
    }

    pub fn update_with_solutions(&mut self, difficulty: &Vec<i32>, num_solutions: u32) {
        let (x, y) = (
            (difficulty[0] - self.min_difficulty[0]) as usize,
            (difficulty[1] - self.min_difficulty[1]) as usize,
        );
        for x_offset in 0..self.padding[0] {
            for y_offset in 0..self.padding[1] {
                let dist = ((x_offset as f32 / self.padding[0] as f32).powf(2.0)
                    + (y_offset as f32 / self.padding[1] as f32).powf(2.0))
                .sqrt();
                if dist > 1.0 {
                    break;
                }
                let decay = dist * (1.0 - DECAY) + DECAY;
                let delta = (1.0 - decay) * num_solutions as f32 * SOLUTIONS_MULTIPLIER;
                self.weights[x + x_offset][y + y_offset].solutions *= decay;
                self.weights[x + x_offset][y + y_offset].solutions += delta;
                if x_offset != 0 && x >= x_offset {
                    self.weights[x - x_offset][y + y_offset].solutions *= decay;
                    self.weights[x - x_offset][y + y_offset].solutions += delta;
                }
                if y_offset != 0 && y >= y_offset {
                    self.weights[x + x_offset][y - y_offset].solutions *= decay;
                    self.weights[x + x_offset][y - y_offset].solutions += delta;
                }
                if x_offset != 0 && y_offset != 0 && x >= x_offset && y >= y_offset {
                    self.weights[x - x_offset][y - y_offset].solutions *= decay;
                    self.weights[x - x_offset][y - y_offset].solutions += delta;
                }
            }
        }
    }

    fn update_valid_range(&mut self, block_data: &ChallengeBlockData) {
        let mut lower_cutoff_points: Vec<Vec<usize>> = block_data
            .base_frontier()
            .iter()
            .map(|x| {
                vec![
                    (x[0] - self.min_difficulty[0]) as usize,
                    (x[1] - self.min_difficulty[1]) as usize,
                ]
            })
            .collect();
        let mut upper_cutoff_points: Vec<Vec<usize>> = block_data
            .scaled_frontier()
            .iter()
            .map(|x| {
                vec![
                    (x[0] - self.min_difficulty[0]) as usize,
                    (x[1] - self.min_difficulty[1]) as usize,
                ]
            })
            .collect();
        lower_cutoff_points.sort_by(|a, b| a[0].cmp(&b[0]));
        upper_cutoff_points.sort_by(|a, b| a[0].cmp(&b[0]));
        if *block_data.scaling_factor() < 1.0 {
            (lower_cutoff_points, upper_cutoff_points) = (upper_cutoff_points, lower_cutoff_points);
        }
        let mut lower_cutoff_idx = 0;
        let mut lower_cutoff = lower_cutoff_points.get(0).unwrap().clone();
        let mut upper_cutoff_idx = 0;
        let mut upper_cutoff1 = upper_cutoff_points.get(0).unwrap().clone();
        let mut upper_cutoff2 = upper_cutoff_points.get(1).unwrap_or(&upper_cutoff1).clone();
        for (i, row) in self.weights.iter_mut().enumerate() {
            if lower_cutoff_idx + 1 < lower_cutoff_points.len()
                && i == lower_cutoff_points[lower_cutoff_idx + 1][0]
            {
                lower_cutoff = lower_cutoff_points[lower_cutoff_idx + 1].clone();
                lower_cutoff_idx += 1;
            }
            if upper_cutoff_idx + 1 < upper_cutoff_points.len()
                && i == upper_cutoff_points[upper_cutoff_idx + 1][0]
            {
                upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx + 1].clone();
                upper_cutoff2 = upper_cutoff_points
                    .get(upper_cutoff_idx + 2)
                    .unwrap_or(&upper_cutoff1)
                    .clone();
                upper_cutoff_idx += 1;
            }
            for (j, w) in row.iter_mut().enumerate() {
                let within_lower =
                    j > lower_cutoff[1] || (j == lower_cutoff[1] && i >= lower_cutoff[0]);
                let within_upper = (j <= upper_cutoff2[1] && i <= upper_cutoff2[0])
                    || (j < upper_cutoff1[1] && i < upper_cutoff2[0])
                    || (j == upper_cutoff1[1] && i == upper_cutoff1[0]);
                w.within_range = within_lower && within_upper;
            }
        }
    }

    fn update_distributions(&mut self) {
        let mut distribution = Vec::<f32>::new();
        for row in self.weights.iter() {
            for w in row.iter() {
                distribution.push(if w.within_range {
                    w.qualifier * w.solutions
                } else {
                    0.0
                });
            }
        }
        self.distribution = Some(WeightedIndex::new(&distribution).unwrap());
    }

    fn update_qualifier_weights(&mut self, block_data: &ChallengeBlockData) {
        let mut cutoff_points: Vec<Vec<usize>> = block_data
            .cutoff_frontier()
            .iter()
            .map(|x| {
                vec![
                    (x[0] - self.min_difficulty[0]) as usize,
                    (x[1] - self.min_difficulty[1]) as usize,
                ]
            })
            .collect();
        cutoff_points.sort_by(|a, b| a[0].cmp(&b[0]));

        let mut cutoff_idx = 0;
        let mut cutoff = cutoff_points.get(0).unwrap_or(&vec![0, 0]).clone(); // every point is a qualifier if there is no cutoff
        for (i, row) in self.weights.iter_mut().enumerate() {
            if cutoff_idx + 1 < cutoff_points.len() && i == cutoff_points[cutoff_idx + 1][0] {
                cutoff = cutoff_points[cutoff_idx + 1].clone();
                cutoff_idx += 1;
            }
            for (j, w) in row.iter_mut().enumerate() {
                w.qualifier *= 0.9;
                if j > cutoff[1] || (j == cutoff[1] && i >= cutoff[0]) {
                    w.qualifier += 0.1;
                }
            }
        }
    }

    fn resize_weights(&mut self, left_pad: &Vec<i32>, size: &Vec<usize>) {
        if left_pad[0] > 0 {
            self.weights
                .splice(0..0, vec![Vec::new(); left_pad[0] as usize]);
        } else if left_pad[0] < 0 {
            self.weights.drain(0..(left_pad[0].abs() as usize));
        }

        if left_pad[1] > 0 {
            let padding_vec = vec![Weights::new(); left_pad[1] as usize];
            for row in self.weights.iter_mut() {
                row.splice(0..0, padding_vec.clone());
            }
        } else if left_pad[1] < 0 {
            for row in self.weights.iter_mut() {
                row.drain(0..(left_pad[1].abs() as usize));
            }
        }

        if self.weights.len() != size[0] {
            self.weights.resize_with(size[0], || Vec::new());
        }
        for row in self.weights.iter_mut() {
            if row.len() != size[1] {
                row.resize(size[1], Weights::new());
            }
        }
    }

    fn update_dimensions_and_padding(&mut self, block_data: &ChallengeBlockData) {
        let hardest_difficulty: Vec<i32> = (0..2)
            .into_iter()
            .map(|i| {
                let v1 = block_data
                    .qualifier_difficulties()
                    .iter()
                    .map(|x| x[i])
                    .max()
                    .unwrap();
                let v2 = block_data
                    .scaled_frontier()
                    .iter()
                    .map(|x| x[i])
                    .max()
                    .unwrap();
                let v3 = block_data
                    .base_frontier()
                    .iter()
                    .map(|x| x[i])
                    .max()
                    .unwrap();
                v1.max(v2).max(v3)
            })
            .collect();
        self.dimensions = (0..2)
            .into_iter()
            .map(|i| (hardest_difficulty[i] - self.min_difficulty[i] + 1) as usize)
            .collect();
        self.padding = self
            .dimensions
            .iter()
            .map(|x| (*x as f32 * PADDING_FACTOR).ceil() as usize)
            .collect();
    }
}
