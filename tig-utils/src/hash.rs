use md5;
use sha3::{Digest, Keccak512};

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

pub fn u64s_from_str(input: &str) -> [u64; 8] {
    let mut hasher = Keccak512::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();

    let mut output = [0u64; 8];
    for i in 0..8 {
        let bytes = result[i * 8..(i + 1) * 8]
            .try_into()
            .expect("Should not ever panic..");
        output[i] = u64::from_le_bytes(bytes);
    }
    output
}
