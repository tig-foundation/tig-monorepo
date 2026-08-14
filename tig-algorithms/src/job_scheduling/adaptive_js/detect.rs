use tig_challenges::job_scheduling::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectedTrack {
    Strict,
    Parallel,
    Random,
    Complex,
    Chaotic,
}

pub fn detect_track(challenge: &Challenge) -> DetectedTrack {
    let num_jobs = challenge.num_jobs;
    let num_products = challenge.product_processing_times.len();
    let num_operations = challenge.num_operations;

    let mut total_flex: usize = 0;
    let mut total_ops: usize = 0;
    let mut has_reentrance = false;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        let job_count = challenge.jobs_per_product[p];
        if ops.len() > num_operations {
            has_reentrance = true;
        }
        for op in ops {
            total_flex += op.len() * job_count;
            total_ops += job_count;
        }
    }

    let flex_avg = if total_ops > 0 {
        total_flex as f64 / total_ops as f64
    } else {
        1.0
    };
    let product_ratio = num_products as f64 / (num_jobs as f64).max(1.0);

    if flex_avg >= 6.0 {
        DetectedTrack::Chaotic
    } else if flex_avg >= 2.0 {
        if product_ratio < 0.55 {
            DetectedTrack::Parallel
        } else {
            DetectedTrack::Complex
        }
    } else if has_reentrance {
        DetectedTrack::Strict
    } else {
        DetectedTrack::Random
    }
}
