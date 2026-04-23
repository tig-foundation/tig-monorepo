use anyhow::Result;
use blake3::Hasher as Blake3;
use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};

/// Cryptographic hash (512-bit) used for anti-grinding commitments.
///
/// `x_eval = H(H(C0) || H(C*))` — changing C* changes x_eval, making
/// input grinding computationally infeasible (Fiat-Shamir).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CryptoHash(#[serde(with = "serde_bytes")] pub [u8; 64]);

impl CryptoHash {
    /// Hashes a serializable value with Blake3 XOF → 512-bit digest.
    pub fn from_serializable<T: serde::Serialize>(value: &T) -> Result<Self> {
        let bytes = bincode::serialize(value)?;
        let mut hasher = Blake3::new();
        hasher.update(&bytes);
        let mut hash = [0u8; 64];
        hasher.finalize_xof().fill(&mut hash);
        Ok(CryptoHash(hash))
    }

    /// Combines two hashes: `H(self || other)`.
    pub fn combine(&self, other: &CryptoHash) -> Self {
        let mut hasher = Blake3::new();
        hasher.update(&self.0);
        hasher.update(&other.0);
        let mut hash = [0u8; 64];
        hasher.finalize_xof().fill(&mut hash);
        CryptoHash(hash)
    }

    /// Hash-to-Field: derives `count` Curve25519 scalars.
    /// `r_i = H(digest || i) mod p`
    pub fn to_scalars(&self, count: usize) -> Vec<Scalar> {
        (0..count)
            .map(|i| {
                let mut hasher = Blake3::new();
                hasher.update(&self.0);
                hasher.update(&(i as u64).to_le_bytes());
                let mut buf = [0u8; 64];
                hasher.finalize_xof().fill(&mut buf);
                Scalar::from_bytes_mod_order_wide(&buf)
            })
            .collect()
    }
}
