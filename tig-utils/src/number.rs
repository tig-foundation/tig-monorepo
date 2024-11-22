use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};
use uint::construct_uint;

construct_uint! {
    pub struct U256(4);
}

impl U256 {
    pub const fn from_u128(value: u128) -> Self {
        let mut ret = [0; 4];
        ret[0] = value as u64;
        ret[1] = (value >> 64) as u64;
        U256(ret)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PreciseNumber(U256);

impl PreciseNumber {
    const DECIMALS: u32 = 18;
    const PRECISION: U256 = U256::from_u128(10u128.pow(PreciseNumber::DECIMALS));

    pub fn inner(&self) -> &U256 {
        &self.0
    }

    pub fn to_f64(&self) -> f64 {
        let value = self.0.as_u128() as f64;
        value / 10f64.powi(Self::DECIMALS as i32)
    }

    pub fn from<T: Into<U256>>(value: T) -> Self {
        Self(value.into() * PreciseNumber::PRECISION)
    }

    pub fn from_f64(value: f64) -> Self {
        Self(((value * 10f64.powi(PreciseNumber::DECIMALS as i32)) as i128).into())
    }

    pub fn from_dec_str(value: &str) -> Result<Self, uint::FromDecStrErr> {
        Ok(Self(U256::from_dec_str(value)?))
    }

    pub fn from_hex_str(value: &str) -> Result<Self, uint::FromStrRadixErr> {
        Ok(Self(U256::from_str_radix(value, 16)?))
    }

    pub fn approx_inv_exp(x: PreciseNumber) -> PreciseNumber {
        // taylor series approximation of e^-x
        let one = PreciseNumber::from(1);
        let mut positive_terms = one.clone();
        let mut negative_terms = PreciseNumber::from(0);
        let mut numerator = one.clone();
        let mut denominator = one.clone();
        for i in 0..32 {
            numerator = numerator * x;
            denominator = denominator * PreciseNumber::from(i + 1);
            if i % 2 == 0 {
                negative_terms = negative_terms + numerator / denominator;
            } else {
                positive_terms = positive_terms + numerator / denominator;
            }
        }
        positive_terms - negative_terms
    }
}

impl Display for PreciseNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for PreciseNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<'de> Deserialize<'de> for PreciseNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_dec_str(s.as_str()).map_err(|e| serde::de::Error::custom(e))
    }
}

impl Add for PreciseNumber {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        (&self).add(&other)
    }
}

impl<'a> Add<&'a PreciseNumber> for PreciseNumber {
    type Output = Self;

    fn add(self, other: &'a Self) -> Self {
        (&self).add(other)
    }
}

impl<'a> Add<PreciseNumber> for &'a PreciseNumber {
    type Output = PreciseNumber;

    fn add(self, other: PreciseNumber) -> PreciseNumber {
        self.add(&other)
    }
}

impl<'a, 'b> Add<&'a PreciseNumber> for &'b PreciseNumber {
    type Output = PreciseNumber;

    fn add(self, other: &'a PreciseNumber) -> PreciseNumber {
        PreciseNumber(self.0.checked_add(other.0).unwrap())
    }
}

impl Sub for PreciseNumber {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        (&self).sub(&other)
    }
}

impl<'a> Sub<&'a PreciseNumber> for PreciseNumber {
    type Output = Self;

    fn sub(self, other: &'a Self) -> Self {
        (&self).sub(other)
    }
}

impl<'a> Sub<PreciseNumber> for &'a PreciseNumber {
    type Output = PreciseNumber;

    fn sub(self, other: PreciseNumber) -> PreciseNumber {
        self.sub(&other)
    }
}

impl<'a, 'b> Sub<&'a PreciseNumber> for &'b PreciseNumber {
    type Output = PreciseNumber;

    fn sub(self, other: &'a PreciseNumber) -> PreciseNumber {
        PreciseNumber(self.0.checked_sub(other.0).unwrap())
    }
}

impl Mul for PreciseNumber {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<'a> Mul<&'a PreciseNumber> for PreciseNumber {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<'a> Mul<PreciseNumber> for &'a PreciseNumber {
    type Output = PreciseNumber;

    fn mul(self, rhs: PreciseNumber) -> PreciseNumber {
        self.mul(&rhs)
    }
}

impl<'a, 'b> Mul<&'a PreciseNumber> for &'b PreciseNumber {
    type Output = PreciseNumber;

    fn mul(self, rhs: &'a PreciseNumber) -> PreciseNumber {
        PreciseNumber(
            self.0
                .checked_mul(rhs.0)
                .unwrap()
                .checked_div(PreciseNumber::PRECISION)
                .unwrap(),
        )
    }
}

impl Div for PreciseNumber {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl<'a> Div<&'a PreciseNumber> for PreciseNumber {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        (&self).div(rhs)
    }
}

impl<'a> Div<PreciseNumber> for &'a PreciseNumber {
    type Output = PreciseNumber;

    fn div(self, rhs: PreciseNumber) -> PreciseNumber {
        self.div(&rhs)
    }
}

impl<'a, 'b> Div<&'a PreciseNumber> for &'b PreciseNumber {
    type Output = PreciseNumber;

    fn div(self, rhs: &'a PreciseNumber) -> PreciseNumber {
        PreciseNumber(
            self.0
                .checked_mul(PreciseNumber::PRECISION)
                .unwrap()
                .checked_div(rhs.0)
                .unwrap(),
        )
    }
}

impl AddAssign for PreciseNumber {
    fn add_assign(&mut self, other: Self) {
        *self = self.add(&other);
    }
}

impl AddAssign<&PreciseNumber> for PreciseNumber {
    fn add_assign(&mut self, other: &Self) {
        *self = self.add(other);
    }
}

impl SubAssign for PreciseNumber {
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(&other);
    }
}

impl SubAssign<&PreciseNumber> for PreciseNumber {
    fn sub_assign(&mut self, other: &Self) {
        *self = self.sub(other);
    }
}

impl MulAssign for PreciseNumber {
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(&other);
    }
}

impl MulAssign<&PreciseNumber> for PreciseNumber {
    fn mul_assign(&mut self, other: &Self) {
        *self = self.mul(other);
    }
}

impl DivAssign for PreciseNumber {
    fn div_assign(&mut self, other: Self) {
        *self = self.div(&other);
    }
}

impl DivAssign<&PreciseNumber> for PreciseNumber {
    fn div_assign(&mut self, other: &Self) {
        *self = self.div(other);
    }
}

impl PartialEq for PreciseNumber {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<'a> PartialEq<&'a PreciseNumber> for PreciseNumber {
    fn eq(&self, other: &&'a PreciseNumber) -> bool {
        self.0 == other.0
    }
}

impl<'a> PartialEq<PreciseNumber> for &'a PreciseNumber {
    fn eq(&self, other: &PreciseNumber) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for PreciseNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<'a> PartialOrd<&'a PreciseNumber> for PreciseNumber {
    fn partial_cmp(&self, other: &&'a PreciseNumber) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<'a> PartialOrd<PreciseNumber> for &'a PreciseNumber {
    fn partial_cmp(&self, other: &PreciseNumber) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for PreciseNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Eq for PreciseNumber {}

impl<'a> Sum<&'a PreciseNumber> for PreciseNumber {
    fn sum<I: Iterator<Item = &'a PreciseNumber>>(iter: I) -> Self {
        iter.fold(PreciseNumber::from(0), |acc, x| acc + *x)
    }
}

impl Sum for PreciseNumber {
    fn sum<I: Iterator<Item = PreciseNumber>>(iter: I) -> Self {
        iter.fold(PreciseNumber::from(0), |acc, x| acc + x)
    }
}

pub trait PreciseNumberOps {
    fn normalise(&self) -> Vec<PreciseNumber>;
    fn arithmetic_mean(&self) -> PreciseNumber;
    fn variance(&self) -> PreciseNumber;
}

impl<T> PreciseNumberOps for T
where
    T: AsRef<[PreciseNumber]>,
{
    fn normalise(&self) -> Vec<PreciseNumber> {
        let values = self.as_ref();
        let sum: PreciseNumber = values.iter().sum();
        let zero = PreciseNumber::from(0);
        if sum == zero {
            values.iter().map(|_| zero.clone()).collect()
        } else {
            values.iter().map(|&x| x / sum).collect()
        }
    }

    fn arithmetic_mean(&self) -> PreciseNumber {
        let values = self.as_ref();
        let sum: PreciseNumber = values.iter().sum();
        let count = PreciseNumber::from(values.len() as u32);
        sum / count
    }

    fn variance(&self) -> PreciseNumber {
        let values = self.as_ref();
        let mean = self.arithmetic_mean();
        let variance_sum: PreciseNumber = values
            .iter()
            .map(|&x| {
                let diff = if x >= mean { x - mean } else { mean - x };
                diff * diff
            })
            .sum::<PreciseNumber>();
        let count = PreciseNumber::from(values.len() as u32);
        variance_sum / count
    }
}
