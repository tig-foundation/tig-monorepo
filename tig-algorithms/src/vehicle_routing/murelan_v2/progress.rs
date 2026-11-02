const G600: [i64; 32] = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 10725, 1000000, 9932, 8354, 1000000, 7741, 6989, 5729, 1000000, 5266, 4350, 1000000, 3974, 3637, 2989, 1000000, 2453, 1670, 1000000, 1059, 812, 610, 1000000, 408, 199, 1000000];
const G700: [i64; 32] = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 9837, 1000000, 10025, 8826, 1000000, 7124, 6193, 1000000, 4629, 3477, 1000000, 3199, 2657, 1000000, 2284, 1870, 1000000, 1270, 932, 1000000, 661, 485, 1000000, 248, 1000000];

pub struct Progress {
    costs: [i64; 3],
    baseline: f64,
    gain: Option<&'static [i64; 32]>,
}

impl Progress {
    pub fn new(nodes: usize, baseline: f64, enabled: bool) -> Self {
        let gain = if enabled {
            match nodes {
                600 => Some(&G600),
                700 => Some(&G700),
                _ => None,
            }
        } else {
            None
        };
        Self { costs: [0; 3], baseline, gain }
    }

    pub fn bound(&mut self, cost: i64, done: usize, total: usize) -> Option<i64> {
        let gain = self.gain?;
        let old = self.costs[(done + 1) % 3];
        self.costs[done % 3] = cost;
        if done < 2 || total == 0 || old < cost || self.baseline <= 0.0
            || (old - cost) as f64 * 1000.0 > self.baseline {
            return None;
        }
        Some(gain[(done * gain.len() / total).min(gain.len() - 1)])
    }
}
