use tig_structs::core::{BenchmarkSettings, CPUArchitecture, OutputData};
use tig_utils::MerkleHash;

#[test]
fn test_calc_solution_signature() {
    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: "test".to_string(),
        cpu_arch: CPUArchitecture::AMD64,
    };

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(output_data.calc_solution_signature(), 11204800550749450632);
}

#[test]
fn test_calc_seed() {
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        difficulty: vec![1, 2, 3],
    };

    let rand_hash = "random_hash".to_string();
    let nonce = 1337;

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(
        settings.calc_seed(&rand_hash, nonce),
        [
            135, 168, 152, 35, 57, 28, 184, 91, 10, 189, 139, 111, 171, 82, 156, 14, 165, 68, 80,
            41, 169, 236, 42, 41, 198, 73, 124, 78, 130, 216, 168, 67
        ]
    );
}

#[test]
fn test_outputdata_to_merklehash() {
    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: "test".to_string(),
        cpu_arch: CPUArchitecture::AMD64,
    };

    let merkle_hash: MerkleHash = output_data.into();

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(
        merkle_hash,
        MerkleHash([
            79, 126, 186, 90, 12, 111, 100, 8, 120, 150, 225, 176, 200, 201, 125, 150, 58, 122,
            214, 216, 68, 6, 125, 247, 248, 4, 165, 185, 157, 44, 13, 151
        ])
    );
}
