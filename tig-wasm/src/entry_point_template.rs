use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{*, {CHALLENGE}::*};
use tig_utils::compress_obj;

#[no_mangle]
pub fn entry_point(seed: u32, difficulty: Difficulty, ptr: *mut u8, max_length: usize) {
    let challenge =
        Challenge::generate_instance(seed, &difficulty).expect("Failed to generate challenge");
    let result: anyhow::Result<Option<Solution>> = {ALGORITHM}::solve_challenge(&challenge);
    let (is_solution, compressed) =
        if let Ok(Some(solution)) = result {
            (
                challenge.verify_solution(&solution).is_ok(),
                compress_obj(solution),
            )
        } else {
            (false, Vec::<u8>::new())
        };
    let mut buffer = Vec::<u8>::new();
    buffer.push(is_solution as u8);
    buffer.extend((compressed.len() as u32).to_be_bytes());
    buffer.extend(compressed);
    for (i, &byte) in buffer.iter().enumerate() {
        if i >= max_length {
            break;
        }
        unsafe { *ptr.add(i) = byte };
    }
}
