use md5;

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
