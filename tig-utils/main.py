import blake3
import binascii
from typing import List


class MerkleHash:
    def __init__(self, value: bytes):
        if len(value) != 32:
            raise ValueError("MerkleHash must be exactly 32 bytes")
        self.value = value

    @classmethod
    def from_hex(cls, hex_str: str):
        return cls(binascii.unhexlify(hex_str))


    @classmethod
    def null(cls):
        return cls(bytes([0] * 32))

    def to_hex(self):
        return binascii.hexlify(self.value).decode()

    def __str__(self):
        return self.to_hex()

    def __eq__(self, other):
        return isinstance(other, MerkleHash) and self.value == other.value

    def __repr__(self):
        return f"MerkleHash({self.to_hex()})"

class MerkleTree:
    def __init__(self, hashed_leafs: List[MerkleHash], n: int):
        if len(hashed_leafs) > n:
            raise ValueError("Invalid tree size")
        self.hashed_leafs = hashed_leafs
        self.n = n

    def serialize(self):
        """Serializes the MerkleTree to a string"""
        # Convert 'n' to a 16-character hexadecimal string (padded)
        n_hex = f"{self.n:016x}"
        # Convert all MerkleHash objects to hex and concatenate
        hashes_hex = ''.join([h.to_hex() for h in self.hashed_leafs])
        # Return the serialized string
        return n_hex + hashes_hex

    @classmethod
    def deserialize(cls, serialized_str: str):
        """Deserializes a MerkleTree from a string"""
        if len(serialized_str) < 16:
            raise ValueError("Invalid MerkleTree string length")

        # Extract the first 16 characters as the hex-encoded size 'n'
        n_hex = serialized_str[:16]
        n = int(n_hex, 16)

        # Extract the remaining part as hex-encoded MerkleHash values
        hashes_hex = serialized_str[16:]

        if len(hashes_hex) % 64 != 0:
            raise ValueError("Invalid MerkleTree hashes length")

        # Split the string into 64-character chunks and convert them to MerkleHash objects
        hashed_leafs = [
            MerkleHash.from_hex(hashes_hex[i:i + 64])
            for i in range(0, len(hashes_hex), 64)
        ]

        return cls(hashed_leafs, n)

    def calc_merkle_root(self) -> MerkleHash:
        null_hash = MerkleHash.null()
        hashes = self.hashed_leafs[:]

        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i+1] if i+1 < len(hashes) else null_hash
                combined = left.value + right.value
                new_hashes.append(MerkleHash(blake3.blake3(combined).digest()))
            hashes = new_hashes

        return hashes[0]

    def calc_merkle_proof(self, branch_idx: int):
        if branch_idx >= self.n:
            raise ValueError("Invalid branch index")

        hashes = self.hashed_leafs[:]
        null_hash = MerkleHash.null()
        proof = []
        idx = branch_idx

        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i+1] if i+1 < len(hashes) else null_hash

                if idx // 2 == i // 2:
                    proof.append(right if idx % 2 == 0 else left)

                combined = left.value + right.value
                new_hashes.append(MerkleHash(blake3.blake3(combined).digest()))
            hashes = new_hashes
            idx //= 2

        return MerkleBranch(proof)

class MerkleBranch:
    def __init__(self, proof_hashes: List[MerkleHash]):
        self.proof_hashes = proof_hashes

    def calc_merkle_root(self, hashed_leaf: MerkleHash, branch_idx: int) -> MerkleHash:
        root = hashed_leaf
        idx = branch_idx

        for hash in self.proof_hashes:
            if idx % 2 == 0:
                combined = root.value + hash.value
            else:
                combined = hash.value + root.value
            root = MerkleHash(blake3.blake3(combined).digest())
            idx //= 2

        return root

    @classmethod
    def deserialize(cls, serialized_str: str):
        """Deserializes a MerkleBranch from a hex string of concatenated MerkleHash values"""
        if len(serialized_str) % 64 != 0:
            raise ValueError("Invalid MerkleProof string length")

        # Split the string into 64-character chunks (32 bytes represented as 64 hex characters)
        hashes = [
            MerkleHash.from_hex(serialized_str[i:i + 64])
            for i in range(0, len(serialized_str), 64)
        ]
        
        return cls(hashes)

    def __repr__(self):
        return f"MerkleBranch({[str(h) for h in self.proof_hashes]})"


# Example usage:
import json
# Example list of hashed leaves
print("Hashes:")
hashed_leafs = [MerkleHash(blake3.blake3(f"leaf {i}".encode()).digest()) for i in range(14)]
for hashleaf in hashed_leafs:
    print(hashleaf.to_hex())
n = len(hashed_leafs)

# Build the Merkle tree
merkle_tree = MerkleTree(hashed_leafs, n)

# Calculate Merkle root
root = merkle_tree.calc_merkle_root()

print("\nMerkle Root:\n", root)

# Generate Merkle proof for a specific leaf
proof = merkle_tree.calc_merkle_proof(2)
print("\nMerkle Proof:")
for node in proof.proof_hashes:
    print(node.to_hex())

print("\nUsing serialized strings from rust: ")

serialized_root = '"bb3b20745d03ce3eaa4603a19056be544bba00f036725d9025205b883c0bf54e"'
serialized_proof = '"ceb50f111fece8844fe4432ed3d19cbce3f54c2ba3994dcd37fe2ceca29791a4af311d272dc334e92c7d626141fa11430dc3b8f55a4911ae1b2542124bdbbef20c2467559ed3061deac0779b0e035514576e2910872b85a84a769087588149a9da007281955a8ed1cbcf3a6f28ec3eb41a385193a7a3a507299032effed88c77"'


# Deserialize Merkle root
root_hex = json.loads(serialized_root)
merkle_root = MerkleHash.from_hex(root_hex)
print("\nDeserialized Merkle Root:", merkle_root)

# Deserialize Merkle proof
proof_str = json.loads(serialized_proof)
proof = MerkleBranch.deserialize(proof_str)
print("\nDeserialized Merkle Proof:")
for node in proof.proof_hashes:
    print(node.to_hex())


# # Verify Merkle proof and calculate root from the proof
calculated_root = proof.calc_merkle_root(hashed_leafs[2], 2)
print("\nCalculated Root from Proof:", calculated_root)

# Check if the root matches
assert calculated_root == root

mt_ser = merkle_tree.serialize()

merkle_tree = MerkleTree.deserialize(mt_ser)
assert merkle_tree.calc_merkle_root() == root
