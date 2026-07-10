#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C001Track {
    N5000R4267,
    N7500R4267,
    N10000R4267,
    N100000R4150,
    N100000R4200,
    Fallback,
}

pub fn classify_by_shape(num_variables: usize, num_clauses: usize) -> C001Track {
    let Some(ratio_x1000) = rounded_ratio_x1000(num_variables, num_clauses) else {
        return C001Track::Fallback;
    };

    match (num_variables, ratio_x1000) {
        (5_000, 4_267) => C001Track::N5000R4267,
        (7_500, 4_267) => C001Track::N7500R4267,
        (10_000, 4_267) => C001Track::N10000R4267,
        (100_000, 4_150) => C001Track::N100000R4150,
        (100_000, 4_200) => C001Track::N100000R4200,
        _ => C001Track::Fallback,
    }
}

fn rounded_ratio_x1000(num_variables: usize, num_clauses: usize) -> Option<u32> {
    if num_variables == 0 {
        return None;
    }

    let numerator = (num_clauses as u128) * 1_000 + (num_variables as u128 / 2);
    Some((numerator / num_variables as u128) as u32)
}
