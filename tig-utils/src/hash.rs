pub fn u8s_from_str(input: &str) -> [u8; 32] {
    blake3::hash(input.as_bytes()).into()
}

pub fn u64s_from_str(input: &str) -> [u64; 4] {
    let u8s = u8s_from_str(input);

    let mut output = [0u64; 4];
    for i in 0..4 {
        let bytes = u8s[i * 8..(i + 1) * 8]
            .try_into()
            .expect("Should not ever panic..");
        output[i] = u64::from_le_bytes(bytes);
    }
    output
}
