import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import u64s_from_str, u8s_from_str, jsonify
from common.merkle_tree import MerkleHash
from common.structs import BenchmarkSettings, OutputData

class TestData(unittest.TestCase):
    def test_calc_solution_signature(self):
        output_data = OutputData(
            nonce=123,
            runtime_signature=456,
            fuel_consumed=789,
            solution="test",
            cpu_arch="AMD64"
        )

        # Assert same as Rust version: tig-structs/tests/core.rs
        self.assertEqual(output_data.calc_solution_signature(), 11204800550749450632)

    def test_calc_seed(self):
        settings = BenchmarkSettings(
            player_id="some_player",
            block_id="some_block",
            challenge_id="some_challenge",
            algorithm_id="some_algorithm",
            track_id="a=1,b=2"
        )

        rand_hash = "random_hash"
        nonce = 1337

        # Assert same as Rust version: tig-structs/tests/core.rs
        expected = bytes([
            84, 136, 44, 57, 142, 50, 248, 37, 94, 195, 254, 190, 222, 27, 136, 115, 229, 136, 19,
            207, 7, 208, 15, 193, 111, 99, 209, 131, 27, 189, 226, 175
        ])
        self.assertEqual(settings.calc_seed(rand_hash, nonce), expected)

    def test_outputdata_to_merklehash(self):
        output_data = OutputData(
            nonce=123,
            runtime_signature=456,
            fuel_consumed=789,
            solution="test",
            cpu_arch="AMD64"
        )

        merkle_hash = output_data.to_merkle_hash()

        # Assert same as Rust version: tig-structs/tests/core.rs
        expected = MerkleHash(bytes([
            79, 126, 186, 90, 12, 111, 100, 8, 120, 150, 225, 176, 200, 201, 125, 150, 58, 122,
            214, 216, 68, 6, 125, 247, 248, 4, 165, 185, 157, 44, 13, 151
        ]))
        self.assertEqual(merkle_hash, expected)

if __name__ == '__main__':
    unittest.main()