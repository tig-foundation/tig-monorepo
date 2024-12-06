import unittest
import sys
import os
from blake3 import blake3
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.merkle_tree import MerkleHash, MerkleTree, MerkleBranch

def create_test_hashes() -> List[MerkleHash]:
    return [MerkleHash(blake3(i.to_bytes(4, 'big')).digest()) for i in range(9)]

class TestMerkleTree(unittest.TestCase):
    def test_merkle_tree(self):
        hashes = create_test_hashes()

        tree = MerkleTree(hashes, 16)
        root = tree.calc_merkle_root()
        # Assert same as Rust version: tig-utils/tests/merkle_tree.rs
        self.assertEqual(root, MerkleHash(bytes.fromhex("fa6d5e8cb2667f5e340b8d1a145891859ad34391cd232f4fbc8d28d8d6284e15")))

        branch = tree.calc_merkle_branch(7)
        self.assertEqual(len(branch.stems), 4)
        leaf_hash = hashes[7]
        calculated_root = branch.calc_merkle_root(leaf_hash, 7)
        self.assertEqual(root, calculated_root)

        branch = tree.calc_merkle_branch(8)
        self.assertEqual(len(branch.stems), 1)
        leaf_hash = hashes[8]
        calculated_root = branch.calc_merkle_root(leaf_hash, 8)
        self.assertEqual(root, calculated_root)

    def test_batched_tree(self):
        hashes = create_test_hashes()
        tree = MerkleTree(hashes, 16)

        batches = [MerkleTree(hashes[i:i+4], 4) for i in range(0, len(hashes), 4)]
        batch_roots = [batch.calc_merkle_root() for batch in batches]
        batch_tree = MerkleTree(batch_roots, 4)
        root = tree.calc_merkle_root()
        self.assertEqual(root, batch_tree.calc_merkle_root())
        # Assert same as Rust version: tig-utils/tests/merkle_tree.rs
        self.assertEqual(root, MerkleHash(bytes.fromhex("fa6d5e8cb2667f5e340b8d1a145891859ad34391cd232f4fbc8d28d8d6284e15")))

        branch = tree.calc_merkle_branch(7)
        batch_branch = batches[1].calc_merkle_branch(3)
        batch_branch.stems.extend(
            [(d + 2, h) for d, h in batch_tree.calc_merkle_branch(1).stems]
        )
        self.assertEqual(branch.stems, batch_branch.stems)

        branch = tree.calc_merkle_branch(8)
        batch_branch = batches[2].calc_merkle_branch(0)
        batch_branch.stems.extend(
            [(d + 2, h) for d, h in batch_tree.calc_merkle_branch(2).stems]
        )
        self.assertEqual(branch.stems, batch_branch.stems)

    def test_invalid_tree_size(self):
        hashes = create_test_hashes()
        with self.assertRaises(ValueError):
            MerkleTree(hashes, 8)

    def test_invalid_branch_index(self):
        hashes = create_test_hashes()
        tree = MerkleTree(hashes, 16)
        with self.assertRaises(ValueError):
            tree.calc_merkle_branch(16)

    def test_invalid_branch(self):
        hashes = create_test_hashes()
        tree = MerkleTree(hashes, 16)
        branch = tree.calc_merkle_branch(7)
        branch.stems[0] = (10, branch.stems[0][1])  # Modify depth to an invalid value
        with self.assertRaises(ValueError):
            branch.calc_merkle_root(hashes[7], 7)

    def test_serialization(self):
        hashes = create_test_hashes()
        tree = MerkleTree(hashes, 16)
        branch = tree.calc_merkle_branch(7)

        tree_str = tree.to_str()
        # Assert same as Rust version: tig-utils/tests/merkle_tree.rs
        self.assertEqual(tree_str, "0000000000000010ec2bd03bf86b935fa34d71ad7ebb049f1f10f87d343e521511d8f9e6625620cda4b6064b23dbaa408b171b0fed5628afa267ef40a4f5a806ae2405e85fa6f1c460604abfd7695c05c911fd1ba39654b8381bcee3797692bb863134aa16b68a2c5882f75066fd0398619cdfe6fcfa463ad254ebdecc381c10dd328cb07b498486988d142bfec4b57545a44b809984ab6bee66df2f6d3fb349532199a9daf6a7a2d2f2ce2738e64d2dd1c507c90673c5a3b7d0bb3077a3947a4aa17aa24dc2c48db8c9e67f5bdeaf090a49c34b6fb567d1fa6ffaee939a2c875c510a1d1e6d4a6cb9d8db6bb71b4287b682b768b62a83a92da369d8d66a10980e5e32e4e429aea50cfe342e104404324f40468de99d6f9ad7b8ae4ab228cf1ccd84b4963b12aea5")
        deserialized_tree = MerkleTree.from_str(tree_str)
        self.assertEqual(tree.calc_merkle_root(), deserialized_tree.calc_merkle_root())

        branch_str = branch.to_str()
        # Assert same as Rust version: tig-utils/tests/merkle_tree.rs
        self.assertEqual(branch_str, "00b8c9e67f5bdeaf090a49c34b6fb567d1fa6ffaee939a2c875c510a1d1e6d4a6c01897c33b84ad3657652be252aae642f7c5e1bdf4e22231d013907254e817753d602f94c4d317f59fd4df80655d879260ce43279ae1962953d79c90d6fb26970b27a030cfe342e104404324f40468de99d6f9ad7b8ae4ab228cf1ccd84b4963b12aea5")
        deserialized_branch = MerkleBranch.from_str(branch_str)
        self.assertEqual(
            branch.calc_merkle_root(hashes[7], 7),
            deserialized_branch.calc_merkle_root(hashes[7], 7)
        )

    def test_merkle_hash_serialization(self):
        hash = MerkleHash(bytes([1] * 32))
        serialized = hash.to_str()
        # Assert same as Rust version: tig-utils/tests/merkle_tree.rs
        self.assertEqual(serialized, "0101010101010101010101010101010101010101010101010101010101010101")
        deserialized = MerkleHash.from_str(serialized)
        self.assertEqual(hash, deserialized)

if __name__ == '__main__':
    unittest.main()