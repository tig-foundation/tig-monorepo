use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};

pub const NUM_SUB_INSTANCES: usize = 8;

/// Score one CUR reconstruction against the optimal rank-`target_rank` error.
///
/// With `q = fnorm / optimal_fnorm`, normalize the multiplicative degradation
/// from SVD quality by `z = ln(max(q, 1)) / ln(target_rank + 1)`, then return
/// `1 / (1 + z)`. This gives score 1 at SVD quality and 1/2 at q = k + 1.
pub fn score_from_errors(fnorm: f64, optimal_fnorm: f64, target_rank: i32) -> Result<f64> {
    if !fnorm.is_finite() || fnorm < 0.0 {
        return Err(anyhow!(
            "CUR reconstruction error must be finite and non-negative, got {}",
            fnorm
        ));
    }
    if !optimal_fnorm.is_finite() || optimal_fnorm <= 0.0 {
        return Err(anyhow!(
            "Optimal rank-k error must be finite and positive, got {}",
            optimal_fnorm
        ));
    }
    if target_rank < 1 {
        return Err(anyhow!("target_rank must be positive, got {}", target_rank));
    }

    let q = fnorm / optimal_fnorm;
    let z = q.max(1.0).ln() / (target_rank as f64 + 1.0).ln();
    Ok(1.0 / (1.0 + z))
}

pub fn aggregate_sub_scores(scores: &[f64]) -> Result<i32> {
    if scores.len() != NUM_SUB_INSTANCES {
        return Err(anyhow!(
            "Expected {} CUR sub-scores, got {}",
            NUM_SUB_INSTANCES,
            scores.len()
        ));
    }
    for (index, &score) in scores.iter().enumerate() {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(anyhow!(
                "CUR sub-score at index {} must be finite and in [0, 1], got {}",
                index,
                score
            ));
        }
    }

    let mean = scores.iter().sum::<f64>() / NUM_SUB_INSTANCES as f64;
    Ok((mean * QUALITY_PRECISION as f64).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::{aggregate_sub_scores, score_from_errors, NUM_SUB_INSTANCES};
    use crate::QUALITY_PRECISION;

    #[test]
    fn score_is_one_at_svd_quality_and_clamps_numerical_improvements() {
        let target_rank = 4;
        let optimal = 2.0;

        assert_eq!(
            score_from_errors(optimal, optimal, target_rank).unwrap(),
            1.0
        );
        assert_eq!(score_from_errors(1.9, optimal, target_rank).unwrap(), 1.0);
    }

    #[test]
    fn score_is_half_at_the_literature_threshold() {
        // q = target_rank + 1 = 5.
        let score = score_from_errors(10.0, 2.0, 4).unwrap();
        assert_eq!(score, 0.5);
    }

    #[test]
    fn score_continuously_distinguishes_results_below_the_threshold() {
        let better = score_from_errors(4.0, 2.0, 4).unwrap();
        let worse = score_from_errors(8.0, 2.0, 4).unwrap();
        assert!(better > worse);
        assert!(worse > 0.5);
    }

    #[test]
    fn score_rejects_invalid_inputs() {
        assert!(score_from_errors(f64::NAN, 1.0, 1).is_err());
        assert!(score_from_errors(1.0, 0.0, 1).is_err());
        assert!(score_from_errors(1.0, 1.0, 0).is_err());
    }

    #[test]
    fn aggregate_is_rounded_arithmetic_mean() {
        let mut scores = vec![1.0; NUM_SUB_INSTANCES];
        scores[0] = 0.5;

        let expected_mean = (NUM_SUB_INSTANCES as f64 - 0.5) / NUM_SUB_INSTANCES as f64;
        let expected_quality = (expected_mean * QUALITY_PRECISION as f64).round() as i32;
        assert_eq!(aggregate_sub_scores(&scores).unwrap(), expected_quality);
    }

    #[test]
    fn aggregate_requires_all_sub_instances() {
        assert!(aggregate_sub_scores(&vec![1.0; NUM_SUB_INSTANCES - 1]).is_err());
    }
}
