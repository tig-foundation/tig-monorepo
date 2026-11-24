use super::baseline::baseline_distance;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::Challenge;

const SCALE: f64 = 1.0;
const BELOW: f64 = 0.004;
const ABOVE: f64 = 0.008;

const TARGETS: [(usize, i64); 5] = [(600, 202000), (700, 195600), (800, 190500), (900, 187100), (1000, 186200)];

#[rustfmt::skip]
const GAIN: [i64; 32] = [83450, 53889, 41417, 29114, 24146, 18989, 15405, 13879, 13508, 13297, 12344, 11160, 10457, 10297, 8711, 8585, 8280, 8062, 8062, 5208, 5150, 5150, 4578, 4578, 4218, 4197, 3669, 3362, 1943, 1943, 1434, 0];

pub struct Stopping {
    baseline: f64,
    stop_cost: i64,
    giveup_quality: f64,
}

impl Stopping {
    pub fn new(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Option<Self> {
        let hp = |key: &str| hyperparameters.as_ref().and_then(|m| m.get(key)).and_then(Value::as_f64);
        if hp("early_stop") == Some(0.0) {
            return None;
        }
        let &(_, target_ppm) = TARGETS
            .iter()
            .min_by_key(|t| (t.0 as i64 - challenge.num_nodes as i64).abs())?;
        let target = hp("target_quality").map_or(target_ppm as f64 / 1e6, |t| if t > 1.0 { t / 1e6 } else { t });
        let baseline = baseline_distance(challenge) as f64;
        Some(Self {
            baseline,
            stop_cost: (baseline * (1.0 - target - ABOVE)).floor() as i64,
            giveup_quality: target - BELOW,
        })
    }

    pub fn settled(&self, cost: i64, done: usize, total: usize) -> bool {
        if cost <= self.stop_cost {
            return true;
        }
        if done >= total {
            return false;
        }
        let gain = GAIN[done * GAIN.len() / total] as f64 / 1e6;
        let quality = 1.0 - cost as f64 / self.baseline;
        quality + SCALE * gain < self.giveup_quality
    }
}
