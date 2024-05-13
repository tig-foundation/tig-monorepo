use tig_utils::*;

#[test]
#[should_panic]
fn test_add_overflow() {
    let _ = PreciseNumber::from_dec_str(
        "115792089237316195423570985008687907853269984665640564039457584007913129639935",
    )
    .unwrap()
        + PreciseNumber::from(1);
}

#[test]
#[should_panic]
fn test_sub_underflow() {
    let _ = PreciseNumber::from(0) - PreciseNumber::from(1);
}

#[test]
#[should_panic]
fn test_mul_overflow() {
    let _ = PreciseNumber::from_dec_str(
        "115792089237316195423570985008687907853269984665640564039457584007913129639935",
    )
    .unwrap()
        * PreciseNumber::from(2);
}

#[test]
#[should_panic]
fn test_div_zero() {
    let _ = PreciseNumber::from(1) / PreciseNumber::from(0);
}

#[test]
fn test_approx_inv_exp() {
    assert_eq!(
        PreciseNumber::approx_inv_exp(PreciseNumber::from(0)),
        PreciseNumber::from(1)
    );
    assert_eq!(
        PreciseNumber::approx_inv_exp(PreciseNumber::from(1) / PreciseNumber::from(2)),
        PreciseNumber::from_dec_str("606530659712633423").unwrap()
    );
    assert_eq!(
        PreciseNumber::approx_inv_exp(PreciseNumber::from(1)),
        PreciseNumber::from_dec_str("367879441171442320").unwrap()
    );
}
