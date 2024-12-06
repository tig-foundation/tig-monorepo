from blake3 import blake3
from typing import List, Tuple
from .utils import FromStr, u8s_from_str

class MerkleHash(FromStr):
    def __init__(self, value: bytes):
        if len(value) != 32:
            raise ValueError("MerkleHash must be exactly 32 bytes")
        self.value = value

    @classmethod
    def from_str(cls, str: str):
        return cls(bytes.fromhex(str))

    @classmethod
    def null(cls):
        return cls(bytes([0] * 32))

    def to_str(self):
        return self.value.hex()

    def __eq__(self, other):
        return isinstance(other, MerkleHash) and self.value == other.value

    def __repr__(self):
        return f"MerkleHash({self.to_str()})"

class MerkleTree(FromStr):
    def __init__(self, hashed_leafs: List[MerkleHash], n: int):
        if len(hashed_leafs) > n:
            raise ValueError("Invalid tree size")
        if n & (n - 1) != 0:
            raise ValueError("n must be a power of 2")
        self.hashed_leafs = hashed_leafs
        self.n = n

    def to_str(self):
        """Serializes the MerkleTree to a string"""
        n_hex = f"{self.n:016x}"
        hashes_hex = ''.join([h.to_str() for h in self.hashed_leafs])
        return n_hex + hashes_hex

    def __repr__(self):
        return f"MerkleTree([{', '.join([str(h) for h in self.hashed_leafs])}], {self.n})"

    @classmethod
    def from_str(cls, s: str):
        """Deserializes a MerkleTree from a string"""
        if len(s) < 16 or (len(s) - 16) % 64 != 0:
            raise ValueError("Invalid MerkleTree string length")

        n_hex = s[:16]
        n = int(n_hex, 16)

        hashes_hex = s[16:]
        hashed_leafs = [
            MerkleHash.from_str(hashes_hex[i:i + 64])
            for i in range(0, len(hashes_hex), 64)
        ]

        return cls(hashed_leafs, n)

    def calc_merkle_root(self) -> MerkleHash:
        hashes = self.hashed_leafs[:]

        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                result = MerkleHash(left.value)
                if i + 1 < len(hashes):
                    right = hashes[i + 1]
                    combined = left.value + right.value
                    result = MerkleHash(blake3(combined).digest())
                new_hashes.append(result)
            hashes = new_hashes

        return hashes[0]

    def calc_merkle_branch(self, branch_idx: int) -> 'MerkleBranch':
        if branch_idx >= self.n:
            raise ValueError("Invalid branch index")

        hashes = self.hashed_leafs[:]
        branch = []
        idx = branch_idx
        depth = 0

        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                result = MerkleHash(left.value)
                if i + 1 < len(hashes):
                    right = hashes[i + 1]
                    if idx // 2 == i // 2:
                        branch.append((depth, right if idx % 2 == 0 else left))
                    combined = left.value + right.value
                    result = MerkleHash(blake3(combined).digest())
                new_hashes.append(result)
            hashes = new_hashes
            idx //= 2
            depth += 1

        return MerkleBranch(branch)

class MerkleBranch:
    def __init__(self, stems: List[Tuple[int, MerkleHash]]):
        self.stems = stems

    def calc_merkle_root(self, hashed_leaf: MerkleHash, branch_idx: int) -> MerkleHash:
        root = hashed_leaf
        idx = branch_idx
        curr_depth = 0

        for depth, hash in self.stems:
            if curr_depth > depth:
                raise ValueError("Invalid branch")
            while curr_depth != depth:
                idx //= 2
                curr_depth += 1

            if idx % 2 == 0:
                combined = root.value + hash.value
            else:
                combined = hash.value + root.value
            root = MerkleHash(blake3(combined).digest())
            idx //= 2
            curr_depth += 1

        return root

    def to_str(self):
        """Serializes the MerkleBranch to a hex string"""
        return ''.join([f"{depth:02x}{hash.to_str()}" for depth, hash in self.stems])

    def __repr__(self):
        return f"MerkleBranch([{', '.join([f'({depth}, {hash})' for depth, hash in self.stems])}])"

    @classmethod
    def from_str(cls, s: str):
        """Deserializes a MerkleBranch from a hex string"""
        if len(s) % 66 != 0:
            raise ValueError("Invalid MerkleBranch string length")

        stems = []
        for i in range(0, len(s), 66):
            depth = int(s[i:i+2], 16)
            hash_hex = s[i+2:i+66]
            stems.append((depth, MerkleHash.from_str(hash_hex)))

        return cls(stems)