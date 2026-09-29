#![cfg(feature = "job_scheduling")]

use std::collections::HashMap;
use tig_challenges::job_scheduling::{Challenge, Solution};

#[test]
fn rejects_schedule_with_wrapped_finish_times() {
    let challenge = Challenge {
        seed: [0; 32],
        num_jobs: 2,
        num_machines: 1,
        num_operations: 2,
        jobs_per_product: vec![2],
        product_processing_times: vec![vec![HashMap::from([(0, 10)]), HashMap::from([(0, 20)])]],
    };
    let valid = Solution {
        job_schedule: vec![vec![(0, 0), (0, 10)], vec![(0, 30), (0, 40)]],
    };
    assert_eq!(challenge.evaluate_makespan(&valid).unwrap(), 60);

    // Previously accepted with makespan 59: the first operation's finish wrapped to zero.
    let forged = Solution {
        job_schedule: vec![
            vec![(0, u32::MAX - 9), (0, 39)],
            vec![(0, u32::MAX - 9), (0, u32::MAX - 19)],
        ],
    };
    let error = challenge.evaluate_makespan(&forged).unwrap_err();
    assert!(error.to_string().contains("overflows u32"), "{error}");
}
