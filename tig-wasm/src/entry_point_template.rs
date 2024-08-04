use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{CHALLENGE}::*;
use tig_utils::compress_obj;

#[no_mangle]
pub fn init(len: u32) -> *mut u8 {
    Box::leak(vec![0x00_u8; len as usize].into_boxed_slice()).as_mut_ptr()
}

#[no_mangle]
pub fn entry_point(ptr: *mut u8, len: u32) -> *mut u8 {
    let challenge: Challenge = {
        let challenge_data = unsafe { Vec::from_raw_parts(ptr, len as usize, len as usize) };
        bincode::deserialize(&challenge_data).expect("Failed to deserialize challenge")
    };
    let result: anyhow::Result<Option<Solution>> = {ALGORITHM}::solve_challenge(&challenge);
    if let Ok(Some(solution)) = result {
        let serialized_solution = compress_obj(&solution);
        let solution_length = serialized_solution.len() as u32;
        let solution_ptr = init(solution_length + 4);
        unsafe {
            let solution_length = solution_length.to_le_bytes();
            std::ptr::copy_nonoverlapping(solution_length.as_ptr(), solution_ptr, 4);
            std::ptr::copy_nonoverlapping(
                serialized_solution.as_ptr(),
                solution_ptr.add(4),
                serialized_solution.len(),
            );
        }
        solution_ptr
    } else {
        let solution_ptr = init(4);
        unsafe {
            let solution_length = 0u32.to_le_bytes();
            std::ptr::copy_nonoverlapping(solution_length.as_ptr(), solution_ptr, 4);
        }
        solution_ptr
    }
}
