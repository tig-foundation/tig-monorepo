use md5;
use sha3::{Digest, Keccak256};

pub fn md5_from_str(input: &str) -> String {
    md5_from_bytes(input.as_bytes())
}

pub fn md5_from_bytes(input: &[u8]) -> String {
    format!("{:x}", md5::compute(input))
}

pub fn u32_from_str(input: &str) -> u32 {
    let result = md5::compute(input.as_bytes());
    let bytes = result[12..16].try_into().expect("Should not ever panic..");
    u32::from_le_bytes(bytes)
}

pub fn u64_from_str(input: &str) -> u64 {
    let mut hasher = Keccak256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    u64::from_le_bytes(
        result.as_slice()[0..8]
            .try_into()
            .expect("Should not ever panic.."),
    )
}
