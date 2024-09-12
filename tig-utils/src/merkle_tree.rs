use anyhow::{anyhow, Result};
use md5;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleHash(pub [u8; 16]);

impl MerkleHash {
    pub fn null() -> Self {
        Self([0; 16])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleTree {
    pub hashed_leafs: Vec<MerkleHash>,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleBranch(pub Vec<MerkleHash>);

impl fmt::Display for MerkleHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl FromStr for MerkleHash {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 16 {
            return Err(anyhow!("Invalid MerkleHash length"));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        Ok(MerkleHash(arr))
    }
}

impl Serialize for MerkleHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for MerkleHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for MerkleTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let n_hex = format!("{:016x}", self.n);
        let hashes_hex: String = self.hashed_leafs.iter().map(|h| h.to_string()).collect();
        serializer.serialize_str(&format!("{}{}", n_hex, hashes_hex))
    }
}

impl<'de> Deserialize<'de> for MerkleTree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() % 32 != 16 {
            return Err(serde::de::Error::custom("Invalid MerkleTree string length"));
        }
        let (n_hex, hashes_hex) = s.split_at(16);
        let n = u64::from_str_radix(n_hex, 16).map_err(serde::de::Error::custom)? as usize;
        let hashes = hashes_hex
            .chars()
            .collect::<Vec<char>>()
            .chunks(32)
            .map(|chunk| chunk.iter().collect::<String>().parse())
            .collect::<Result<Vec<MerkleHash>, _>>()
            .map_err(serde::de::Error::custom)?;
        Ok(MerkleTree {
            hashed_leafs: hashes,
            n,
        })
    }
}

impl Serialize for MerkleBranch {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hashes_hex: String = self.0.iter().map(|h| h.to_string()).collect();
        serializer.serialize_str(&hashes_hex)
    }
}

impl<'de> Deserialize<'de> for MerkleBranch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() % 32 != 0 {
            return Err(serde::de::Error::custom(
                "Invalid MerkleProof string length",
            ));
        }
        let hashes = s
            .chars()
            .collect::<Vec<char>>()
            .chunks(32)
            .map(|chunk| chunk.iter().collect::<String>().parse())
            .collect::<Result<Vec<MerkleHash>, _>>()
            .map_err(serde::de::Error::custom)?;
        Ok(MerkleBranch(hashes))
    }
}

impl MerkleTree {
    pub fn new(hashed_leafs: Vec<MerkleHash>, n: usize) -> Result<Self> {
        if hashed_leafs.len() > n {
            return Err(anyhow!("Invalid tree size"));
        }
        Ok(Self { hashed_leafs, n })
    }

    pub fn calc_merkle_root(&self) -> MerkleHash {
        let null_hash = MerkleHash::null();
        let mut hashes = self.hashed_leafs.clone();
        while hashes.len() > 1 {
            let mut new_hashes = Vec::new();
            for chunk in hashes.chunks(2) {
                let mut combined = [0u8; 32];
                combined[..16].copy_from_slice(&chunk[0].0);
                combined[16..].copy_from_slice(&chunk.get(1).unwrap_or(&null_hash).0);
                new_hashes.push(MerkleHash(md5::compute(&combined).0));
            }
            hashes = new_hashes;
        }

        hashes[0].clone()
    }

    pub fn calc_merkle_branch(&self, branch_idx: usize) -> Result<MerkleBranch> {
        if branch_idx >= self.n {
            return Err(anyhow!("Invalid branch index"));
        }

        let mut hashes = self.hashed_leafs.clone();
        let null_hash = MerkleHash::null();
        let mut branch = Vec::new();
        let mut idx = branch_idx;

        while hashes.len() > 1 {
            let mut new_hashes = Vec::new();
            for (i, chunk) in hashes.chunks(2).enumerate() {
                let left = &chunk[0];
                let right = chunk.get(1).unwrap_or(&null_hash);

                if idx >> 1 == i {
                    branch.push(if idx % 2 == 0 { right } else { left }.clone());
                }

                let mut combined = [0u8; 32];
                combined[..16].copy_from_slice(&left.0);
                combined[16..].copy_from_slice(&right.0);
                new_hashes.push(MerkleHash(md5::compute(&combined).0));
            }
            hashes = new_hashes;
            idx /= 2;
        }

        Ok(MerkleBranch(branch))
    }
}

impl MerkleBranch {
    pub fn calc_merkle_root(&self, hashed_leaf: &MerkleHash, branch_idx: usize) -> MerkleHash {
        let mut root = hashed_leaf.clone();
        let mut idx = branch_idx;

        for hash in &self.0 {
            let mut combined = [0u8; 32];
            if idx % 2 == 0 {
                combined[..16].copy_from_slice(&root.0);
                combined[16..].copy_from_slice(&hash.0);
            } else {
                combined[..16].copy_from_slice(&hash.0);
                combined[16..].copy_from_slice(&root.0);
            }
            root = MerkleHash(md5::compute(&combined).0);
            idx /= 2;
        }

        root
    }
}
