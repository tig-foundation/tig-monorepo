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
    // Assert same as Python version: tig-benchmarker/tests/merkle_tree.rs
    assert_eq!(
        root,
        MerkleHash(
            hex::decode("fa6d5e8cb2667f5e340b8d1a145891859ad34391cd232f4fbc8d28d8d6284e15")
                .unwrap()
                .try_into()
                .unwrap()
        )
    );

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
    let root = tree.calc_merkle_root();
    assert_eq!(root, batch_tree.calc_merkle_root());
    // Assert same as Python version: tig-benchmarker/tests/merkle_tree.rs
    assert_eq!(
        root,
        MerkleHash(
            hex::decode("fa6d5e8cb2667f5e340b8d1a145891859ad34391cd232f4fbc8d28d8d6284e15")
                .unwrap()
                .try_into()
                .unwrap()
        )
    );

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
    // Assert same as Python version: tig-benchmarker/tests/merkle_tree.rs
    assert_eq!(&tree_json, "\"0000000000000010ec2bd03bf86b935fa34d71ad7ebb049f1f10f87d343e521511d8f9e6625620cda4b6064b23dbaa408b171b0fed5628afa267ef40a4f5a806ae2405e85fa6f1c460604abfd7695c05c911fd1ba39654b8381bcee3797692bb863134aa16b68a2c5882f75066fd0398619cdfe6fcfa463ad254ebdecc381c10dd328cb07b498486988d142bfec4b57545a44b809984ab6bee66df2f6d3fb349532199a9daf6a7a2d2f2ce2738e64d2dd1c507c90673c5a3b7d0bb3077a3947a4aa17aa24dc2c48db8c9e67f5bdeaf090a49c34b6fb567d1fa6ffaee939a2c875c510a1d1e6d4a6cb9d8db6bb71b4287b682b768b62a83a92da369d8d66a10980e5e32e4e429aea50cfe342e104404324f40468de99d6f9ad7b8ae4ab228cf1ccd84b4963b12aea5\"");
    let deserialized_tree: MerkleTree = serde_json::from_str(&tree_json).unwrap();
    assert_eq!(
        tree.calc_merkle_root(),
        deserialized_tree.calc_merkle_root()
    );

    let branch_json = serde_json::to_string(&branch).unwrap();
    // Assert same as Python version: tig-benchmarker/tests/merkle_tree.rs
    assert_eq!(&branch_json, "\"00b8c9e67f5bdeaf090a49c34b6fb567d1fa6ffaee939a2c875c510a1d1e6d4a6c01897c33b84ad3657652be252aae642f7c5e1bdf4e22231d013907254e817753d602f94c4d317f59fd4df80655d879260ce43279ae1962953d79c90d6fb26970b27a030cfe342e104404324f40468de99d6f9ad7b8ae4ab228cf1ccd84b4963b12aea5\"");
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
    // Assert same as Python version: tig-benchmarker/tests/merkle_tree.rs
    assert_eq!(
        &serialized,
        "\"0101010101010101010101010101010101010101010101010101010101010101\""
    );
    let deserialized: MerkleHash = serde_json::from_str(&serialized).unwrap();
    assert_eq!(hash, deserialized);
}
