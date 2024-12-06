import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import u64s_from_str, u8s_from_str, jsonify
from common.merkle_tree import MerkleHash
from common.structs import BenchmarkSettings, OutputData

class TestData(unittest.TestCase):
    def test_calc_solution_signature(self):
        solution = {
            "data_x": 42,
            "data_y": "test"
        }

        output_data = OutputData(
            nonce=123,
            runtime_signature=456,
            fuel_consumed=789,
            solution=solution
        )

        # Assert same as Rust version: tig-structs/tests/core.rs
        self.assertEqual(output_data.calc_solution_signature(), 11549591319018095145)

    def test_calc_seed(self):
        settings = BenchmarkSettings(
            player_id="some_player",
            block_id="some_block",
            challenge_id="some_challenge",
            algorithm_id="some_algorithm",
            difficulty=[1, 2, 3]
        )

        rand_hash = "random_hash"
        nonce = 1337

        # Assert same as Rust version: tig-structs/tests/core.rs
        expected = bytes([
            135, 168, 152, 35, 57, 28, 184, 91, 10, 189, 139, 111, 171, 82, 156, 14, 
            165, 68, 80, 41, 169, 236, 42, 41, 198, 73, 124, 78, 130, 216, 168, 67
        ])
        self.assertEqual(settings.calc_seed(rand_hash, nonce), expected)

    def test_outputdata_to_merklehash(self):
        solution = {
            "data_x": 42,
            "data_y": "test"
        }

        output_data = OutputData(
            nonce=123,
            runtime_signature=456,
            fuel_consumed=789,
            solution=solution
        )

        merkle_hash = output_data.to_merkle_hash()

        # Assert same as Rust version: tig-structs/tests/core.rs
        expected = MerkleHash(bytes([
            207, 29, 184, 163, 158, 22, 137, 73, 72, 58, 24, 246, 67, 9, 44, 20,
            32, 22, 86, 206, 191, 5, 52, 241, 41, 113, 198, 85, 11, 53, 190, 57
        ]))
        self.assertEqual(merkle_hash, expected)

if __name__ == '__main__':
    unittest.main()