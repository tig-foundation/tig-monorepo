#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{*, {CHALLENGE}::*};
use tig_utils::compress_obj;

#[no_mangle]
pub fn entry_point(seed: u32, difficulty: Difficulty, ptr: *mut u8, max_length: usize) {
    let challenge = Challenge::generate_instance(seed, &difficulty).expect("Failed to generate challenge");    
    if let Ok(Some(solution)) = {ALGORITHM}::solve_challenge(&challenge) {
        if challenge.verify_solution(&solution).is_ok() {
            let mut buffer = Vec::<u8>::new();
            let compressed = compress_obj(solution);
            buffer.extend((compressed.len() as u32).to_be_bytes());
            buffer.extend(compressed);                
            if buffer.len() > max_length {
                panic!("Encoded solution exceeds maximum length");
            }

            for (i, &byte) in buffer.iter().enumerate() {
                unsafe { *ptr.add(i) = byte };
            }
        }
    }
}
