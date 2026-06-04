use super::Hyperparameters;

pub(crate) fn default_max_fuel(_hp: &Hyperparameters) -> f64 {
    115_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n5000_default_fuel_matches_r18_probe() {
        assert_eq!(
            default_max_fuel(&Hyperparameters::default()),
            115_000_000_000.0
        );
    }
}
