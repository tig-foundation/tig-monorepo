#[cfg(test)]
mod tests {
    use blake3::hash;
    use serde_json;
    use tig_utils::{MerkleBranch, MerkleHash, MerkleTree};

    fn create_test_hashes() -> Vec<MerkleHash> {
        (0..14u32)
            .map(|i| MerkleHash(hash(i.to_be_bytes().as_slice()).into()))
            .collect()
    }

    #[test]
    fn test_merkle_tree() {
        let hashes = create_test_hashes();

        let tree = MerkleTree::new(hashes.clone(), hashes.len() + 1).unwrap();
        let root = tree.calc_merkle_root();

        let proof = tree.calc_merkle_branch(7).unwrap();
        let leaf_hash = &hashes[7];
        let calculated_root = proof.calc_merkle_root(leaf_hash, 7);
        assert_eq!(root, calculated_root);
    }

    #[test]
    fn test_invalid_tree_size() {
        let hashes = create_test_hashes();

        let result = MerkleTree::new(hashes, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_branch_index() {
        let hashes = create_test_hashes();

        let tree = MerkleTree::new(hashes, 16).unwrap();
        let result = tree.calc_merkle_branch(16);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization() {
        let hashes = create_test_hashes();

        let tree = MerkleTree::new(hashes.clone(), 16).unwrap();
        let proof = tree.calc_merkle_branch(7).unwrap();

        let tree_json = serde_json::to_string(&tree).unwrap();
        let deserialized_tree: MerkleTree = serde_json::from_str(&tree_json).unwrap();
        assert_eq!(
            tree.calc_merkle_root(),
            deserialized_tree.calc_merkle_root()
        );

        let proof_json = serde_json::to_string(&proof).unwrap();
        let deserialized_proof: MerkleBranch = serde_json::from_str(&proof_json).unwrap();
        assert_eq!(
            proof.calc_merkle_root(&hashes[7], 7),
            deserialized_proof.calc_merkle_root(&hashes[7], 7)
        );
    }

    #[test]
    fn test_merkle_hash_serialization() {
        let hash = MerkleHash([1; 32]);
        let serialized = serde_json::to_string(&hash).unwrap();
        let deserialized: MerkleHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(hash, deserialized);
    }
}
