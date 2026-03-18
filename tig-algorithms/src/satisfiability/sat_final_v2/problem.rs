use tig_challenges::satisfiability::*;

pub fn detect_track(challenge: &Challenge) -> u8 {
    let n = challenge.num_variables;
    let density = challenge.clauses.len() as f64 / n as f64;
    let ratio = (density * 1000.0).round() as u32;

    match (n, ratio) {
        (5000, 4267) => 1,
        (7500, 4267) => 2,
        (10000, 4267) => 3,
        (100000, 4150) => 4,
        (100000, 4200) => 5,
        _ => 0,
    }
}
