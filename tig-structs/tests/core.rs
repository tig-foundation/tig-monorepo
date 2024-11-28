use serde_json::json;
use tig_structs::core::{BenchmarkSettings, OutputData};
use tig_utils::MerkleHash;

#[test]
fn test_calc_solution_signature() {
    let solution = json!({
        "data_x": 42,
        "data_y": "test"
    })
    .as_object()
    .unwrap()
    .clone();

    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: solution.clone(),
    };

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(output_data.calc_solution_signature(), 11549591319018095145);
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
    let solution = json!({
        "data_x": 42,
        "data_y": "test"
    })
    .as_object()
    .unwrap()
    .clone();

    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: solution.clone(),
    };

    let merkle_hash: MerkleHash = output_data.into();

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(
        merkle_hash,
        MerkleHash([
            207, 29, 184, 163, 158, 22, 137, 73, 72, 58, 24, 246, 67, 9, 44, 20, 32, 22, 86, 206,
            191, 5, 52, 241, 41, 113, 198, 85, 11, 53, 190, 57
        ])
    );
}
