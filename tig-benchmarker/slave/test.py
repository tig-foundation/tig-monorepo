import json
from tig_benchmarker.merkle_tree import MerkleTree
from tig_benchmarker.structs import MerkleProof, OutputData

start_nonce = int(0)
batch_size = int(1024)

leafs = {}
for nonce in range(start_nonce, start_nonce + num_nonces  ):
    file_path = f"output/73a93101f5d72d03d133c443c0bcd7e7_0_1024_1000/{nonce}.json"
    try:
        with open(file_path) as f:
            leafs[nonce] = OutputData.from_dict(json.load(f))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")

merkle_tree = MerkleTree(
                [x.to_merkle_hash() for x in leafs.values()],
                batch_size
            )

merkle_proofs = [
    MerkleProof(
        leaf=leafs[n],
        branch=merkle_tree.calc_merkle_branch(branch_idx=n - start_nonce)
    ).to_dict()
    for n in range(start_nonce, start_nonce + batch_size )
]

# Write to file
with open("merkle_proofs.json", "w") as f:
    json.dump(merkle_proofs, f)
