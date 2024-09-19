use blake3::hash;
use serde_json;
use tig_utils::{MerkleBranch, MerkleHash, MerkleTree};

fn create_test_hashes() -> Vec<MerkleHash> {
    (0..9u32)
        .map(|i| MerkleHash(hash(i.to_be_bytes().as_slice()).into()))
        .collect()
}

#[test]
fn test_merkle_tree() {
    let hashes = create_test_hashes();

    let tree = MerkleTree::new(hashes.clone(), 16).unwrap();
    let root = tree.calc_merkle_root();

    let branch = tree.calc_merkle_branch(7).unwrap();
    assert_eq!(branch.0.len(), 4);
    let leaf_hash = &hashes[7];
    let calculated_root = branch.calc_merkle_root(leaf_hash, 7).unwrap();
    assert_eq!(root, calculated_root);

    let branch = tree.calc_merkle_branch(8).unwrap();
    assert_eq!(branch.0.len(), 1);
    let leaf_hash = &hashes[8];
    let calculated_root = branch.calc_merkle_root(leaf_hash, 8).unwrap();
    assert_eq!(root, calculated_root);
}

#[test]
fn test_batched_tree() {
    let hashes = create_test_hashes();
    let tree = MerkleTree::new(hashes.clone(), 16).unwrap();

    let batches = hashes
        .chunks(4)
        .map(|chunk| MerkleTree::new(chunk.to_vec(), 4).unwrap())
        .collect::<Vec<MerkleTree>>();
    let batch_roots = batches
        .iter()
        .map(|tree| tree.calc_merkle_root())
        .collect::<Vec<MerkleHash>>();
    let batch_tree = MerkleTree::new(batch_roots.clone(), 4).unwrap();
    assert_eq!(tree.calc_merkle_root(), batch_tree.calc_merkle_root());

    let branch = tree.calc_merkle_branch(7).unwrap();
    let mut batch_branch = batches[1].calc_merkle_branch(3).unwrap();
    batch_branch.0.extend(
        batch_tree
            .calc_merkle_branch(1)
            .unwrap()
            .0
            .into_iter()
            .map(|(d, h)| (d + 2, h)),
    );
    assert_eq!(branch, batch_branch);

    let branch = tree.calc_merkle_branch(8).unwrap();
    let mut batch_branch = batches[2].calc_merkle_branch(0).unwrap();
    batch_branch.0.extend(
        batch_tree
            .calc_merkle_branch(2)
            .unwrap()
            .0
            .into_iter()
            .map(|(d, h)| (d + 2, h)),
    );
    assert_eq!(branch, batch_branch);
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
fn test_invalid_branch() {
    let hashes = create_test_hashes();

    let tree = MerkleTree::new(hashes.clone(), 16).unwrap();
    let mut branch = tree.calc_merkle_branch(7).unwrap();
    branch.0.get_mut(0).unwrap().0 = 10;
    let result = branch.calc_merkle_root(&hashes[7], 7);
    assert!(result.is_err());
}

#[test]
fn test_serialization() {
    let hashes = create_test_hashes();

    let tree = MerkleTree::new(hashes.clone(), 16).unwrap();
    let branch = tree.calc_merkle_branch(7).unwrap();

    let tree_json = serde_json::to_string(&tree).unwrap();
    let deserialized_tree: MerkleTree = serde_json::from_str(&tree_json).unwrap();
    assert_eq!(
        tree.calc_merkle_root(),
        deserialized_tree.calc_merkle_root()
    );

    let branch_json = serde_json::to_string(&branch).unwrap();
    let deserialized_branch: MerkleBranch = serde_json::from_str(&branch_json).unwrap();
    assert_eq!(
        branch.calc_merkle_root(&hashes[7], 7).unwrap(),
        deserialized_branch.calc_merkle_root(&hashes[7], 7).unwrap()
    );
}

#[test]
fn test_merkle_hash_serialization() {
    let hash = MerkleHash([1; 32]);
    let serialized = serde_json::to_string(&hash).unwrap();
    let deserialized: MerkleHash = serde_json::from_str(&serialized).unwrap();
    assert_eq!(hash, deserialized);
}
