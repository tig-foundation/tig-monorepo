mod merkle_tree;
use blake3::hash;
use merkle_tree::*;
use hex;
fn main() {
    let leaf_strings: Vec<String> = (0..14u32).map(|i| format!("leaf {i}")).collect();
    let hashes: Vec<MerkleHash> = leaf_strings.into_iter()
    .map(|s| MerkleHash(hash(s.as_bytes()).into()))
    .collect();
    println!("Hashes:");
    for hash in hashes.iter() {
        println!("{}", hash.to_string());
    }
    let merkle_tree = MerkleTree::new(hashes, 14).expect("Can make merkle tree");
    let merkle_root = merkle_tree.calc_merkle_root();
    // Example Merkle proof (two hashes in proof)
    let merkle_proof = merkle_tree.calc_merkle_proof(2).expect("Can generate merkle proof");
   
    println!("\nMerkle Root:\n {}", merkle_root);
    println!("\nMerkle Proof: ");
    for node in merkle_proof.0.iter() {
        println!("{}", node);
    }
   
    // Serialize root and proof
    let serialized_root = serde_json::to_string(&merkle_root).expect("Can serialize Root");
    let serialized_proof = serde_json::to_string(&merkle_proof).expect("Can serialize proof");

    println!("\n USE THE FOLLOWING STRINGS INSIDE THE PYTHON MAIN:");
    println!("\nSerialized Merkle Root: {}", serialized_root);
    println!("Serialized Merkle Proof: {}", serialized_proof);

}