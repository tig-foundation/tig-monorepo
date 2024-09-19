use anyhow::{anyhow, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleHash(pub [u8; 32]);

impl MerkleHash {
    pub fn null() -> Self {
        Self([0; 32])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleTree {
    pub hashed_leafs: Vec<MerkleHash>,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleBranch(pub Vec<(u8, MerkleHash)>);

impl fmt::Display for MerkleHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl FromStr for MerkleHash {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(anyhow!("Invalid MerkleHash length"));
        }
        let mut arr = [0u8; 32];
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
        if s.len() % 64 != 16 {
            return Err(serde::de::Error::custom("Invalid MerkleTree string length"));
        }
        let (n_hex, hashes_hex) = s.split_at(16);
        let n = u64::from_str_radix(n_hex, 16).map_err(serde::de::Error::custom)? as usize;
        let hashes = hashes_hex
            .chars()
            .collect::<Vec<char>>()
            .chunks(64)
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
        let hashes_hex: String = self
            .0
            .iter()
            .map(|(d, h)| format!("{:02x}{}", d, h.to_string()))
            .collect();
        serializer.serialize_str(&hashes_hex)
    }
}

impl<'de> Deserialize<'de> for MerkleBranch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() % 66 != 0 {
            return Err(serde::de::Error::custom(
                "Invalid MerkleBranch string length",
            ));
        }
        let mut stems = Vec::new();
        for chunk in s.chars().collect::<Vec<char>>().chunks(66) {
            let depth: u8 = u8::from_str_radix(&chunk[..2].iter().collect::<String>(), 16)
                .map_err(serde::de::Error::custom)?;
            let hash = chunk[2..]
                .iter()
                .collect::<String>()
                .parse()
                .map_err(serde::de::Error::custom)?;
            stems.push((depth, hash));
        }
        Ok(MerkleBranch(stems))
    }
}

impl MerkleTree {
    pub fn new(hashed_leafs: Vec<MerkleHash>, n: usize) -> Result<Self> {
        // check n is power of 2
        if n.count_ones() != 1 {
            return Err(anyhow!("Invalid tree size"));
        }
        if hashed_leafs.len() > n {
            return Err(anyhow!("Invalid tree size"));
        }
        Ok(Self { hashed_leafs, n })
    }

    pub fn calc_merkle_root(&self) -> MerkleHash {
        let mut hashes = self.hashed_leafs.clone();
        while hashes.len() > 1 {
            let mut new_hashes = Vec::with_capacity((hashes.len() + 1) / 2);
            for chunk in hashes.chunks(2) {
                let left = &chunk[0];
                let mut next = left.clone();
                if let Some(right) = chunk.get(1) {
                    let mut combined = [0u8; 64];
                    combined[..32].copy_from_slice(&left.0);
                    combined[32..].copy_from_slice(&right.0);
                    next = MerkleHash(blake3::hash(&combined).into());
                }
                new_hashes.push(next);
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
        let mut branch = Vec::new();
        let mut idx = branch_idx;

        let mut depth = 0;
        while hashes.len() > 1 {
            let mut new_hashes = Vec::new();
            for (i, chunk) in hashes.chunks(2).enumerate() {
                let left = &chunk[0];
                let mut next = left.clone();
                if let Some(right) = chunk.get(1) {
                    if idx >> 1 == i {
                        branch.push((depth, if idx % 2 == 0 { right } else { left }.clone()));
                    }

                    let mut combined = [0u8; 64];
                    combined[..32].copy_from_slice(&left.0);
                    combined[32..].copy_from_slice(&right.0);
                    next = MerkleHash(blake3::hash(&combined).into());
                }
                new_hashes.push(next);
            }
            hashes = new_hashes;
            idx /= 2;
            depth += 1;
        }

        Ok(MerkleBranch(branch))
    }
}

impl MerkleBranch {
    pub fn calc_merkle_root(
        &self,
        hashed_leaf: &MerkleHash,
        branch_idx: usize,
    ) -> Result<MerkleHash> {
        let mut root = hashed_leaf.clone();
        let mut idx = branch_idx;

        let mut curr_depth = 0;
        for (depth, hash) in &self.0 {
            if curr_depth > *depth {
                return Err(anyhow!("Invalid branch"));
            }
            while curr_depth != *depth {
                idx /= 2;
                curr_depth += 1;
            }
            let mut combined = [0u8; 64];
            if idx % 2 == 0 {
                combined[..32].copy_from_slice(&root.0);
                combined[32..].copy_from_slice(&hash.0);
            } else {
                combined[..32].copy_from_slice(&hash.0);
                combined[32..].copy_from_slice(&root.0);
            }
            root = MerkleHash(blake3::hash(&combined).into());
            idx /= 2;
            curr_depth += 1;
        }

        Ok(root)
    }
}
