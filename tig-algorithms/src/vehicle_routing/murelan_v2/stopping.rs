use super::progress::Progress;
use super::baseline::baseline_distance;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::Challenge;

const SCALE: f64 = 1.0;
const BELOW: f64 = 0.004;
const BELOW_SENT: f64 = 0.003;
const BELOW_DEFAULT_TARGET_LATE: f64 = 0.003;
const LATE_TRACKS: [usize; 2] = [600, 800];
const ABOVE: f64 = 0.006;
const ABOVE_DEFAULT_TARGET: f64 = 0.0045;
const ABOVE_SENT: f64 = 0.0045;
const ABOVE_SENT_TRACKS: [usize; 3] = [600, 800, 1000];

const TARGETS: [(usize, i64); 5] = [(600, 204500), (700, 198100), (800, 193600), (900, 189600), (1000, 192000)];

#[rustfmt::skip]
const GAIN: [i64; 32] = [83450, 53889, 41417, 29114, 24146, 18989, 15405, 13879, 13508, 13297, 12344, 11160, 10457, 10297, 8711, 8585, 8280, 8062, 8062, 5208, 5150, 5150, 4578, 4578, 4218, 4197, 3669, 3362, 1943, 1943, 1434, 0];

#[rustfmt::skip]
const GAIN_P99: [(usize, [i64; 32]); 5] = [
    (600, [69021, 39098, 29264, 29264, 22321, 16344, 13957, 13957, 12114, 10781, 10781, 10174, 9302, 8004, 8004, 6997, 6119, 6119, 5244, 4055, 3785, 3785, 3414, 3140, 3140, 2193, 1930, 1169, 1169, 962, 579, 579]),
    (700, [69441, 38332, 38332, 23896, 18276, 18276, 14619, 12466, 12466, 10599, 10191, 10191, 9028, 7483, 7483, 6028, 5514, 5514, 4051, 3343, 3343, 3164, 2529, 2529, 2023, 1831, 1831, 1079, 942, 942, 547, 547]),
    (800, [75957, 45537, 30024, 22809, 18615, 14722, 12520, 10519, 9945, 8960, 8960, 7716, 7601, 6414, 5796, 5156, 4697, 4234, 4078, 3742, 2998, 2998, 2646, 2381, 1714, 1538, 1154, 671, 580, 510, 420, 420]),
    (900, [74319, 48188, 32337, 25117, 20620, 16982, 14216, 12784, 11061, 10107, 10107, 9039, 8038, 7381, 6867, 6171, 5886, 5293, 4598, 4297, 3961, 3961, 3420, 3173, 2381, 2088, 1597, 1435, 1119, 803, 528, 528]),
    (1000, [76638, 44447, 30298, 22243, 18352, 14909, 12783, 11583, 10547, 9646, 8643, 7927, 7614, 7042, 6437, 5578, 4752, 4190, 3832, 3745, 3170, 2863, 2434, 1828, 1495, 1493, 1356, 1143, 925, 542, 374, 374]),
];

pub struct Stopping {
    progress: Progress,
    baseline: f64,
    stop_cost: i64,
    giveup_quality: f64,
    giveup_quality_late: f64,
    gain: &'static [i64; 32],
}

impl Stopping {
    pub fn new(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Option<Self> {
        let hp = |key: &str| hyperparameters.as_ref().and_then(|m| m.get(key)).and_then(Value::as_f64);
        if hp("early_stop") == Some(0.0) {
            return None;
        }
        let &(track, target_ppm) = TARGETS
            .iter()
            .min_by_key(|t| (t.0 as i64 - challenge.num_nodes as i64).abs())?;
        let target = hp("target_quality").map_or(target_ppm as f64 / 1e6, |t| if t > 1.0 { t / 1e6 } else { t });
        let default_search = hyperparameters
            .as_ref()
            .map_or(true, |m| m.keys().all(|k| k == "target_quality" || k == "early_stop"));
        let gain = if default_search {
            &GAIN_P99.iter().min_by_key(|t| (t.0 as i64 - challenge.num_nodes as i64).abs())?.1
        } else {
            &GAIN
        };
        let sent = default_search && hp("target_quality").is_some();
        let above = if !default_search {
            ABOVE
        } else if !sent {
            ABOVE_DEFAULT_TARGET
        } else if ABOVE_SENT_TRACKS.contains(&track) {
            ABOVE_SENT
        } else {
            ABOVE
        };
        let baseline = baseline_distance(challenge) as f64;
        Some(Self {
            progress: Progress::new(challenge.num_nodes, baseline, default_search),
            baseline,
            stop_cost: (baseline * (1.0 - target - above)).floor() as i64,
            giveup_quality: target - if sent { BELOW_SENT } else { BELOW },
            giveup_quality_late: target - if sent {
                BELOW_SENT
            } else if default_search && LATE_TRACKS.contains(&track) {
                BELOW_DEFAULT_TARGET_LATE
            } else {
                BELOW
            },
            gain,
        })
    }

    pub fn settled(&mut self, cost: i64, done: usize, total: usize) -> bool {
        if cost <= self.stop_cost {
            return true;
        }
        if done >= total {
            return false;
        }
        let original_gain = self.gain[done * self.gain.len() / total];
        let gain = self.progress.bound(cost, done, total).map_or(original_gain, |g| g.min(original_gain)) as f64 / 1e6;
        let quality = 1.0 - cost as f64 / self.baseline;
        let giveup_quality = if done * 4 >= total { self.giveup_quality_late } else { self.giveup_quality };
        quality + SCALE * gain < giveup_quality
    }
}
