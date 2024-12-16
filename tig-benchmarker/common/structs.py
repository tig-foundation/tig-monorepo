from .merkle_tree import MerkleHash, MerkleBranch
from .utils import FromDict, u64s_from_str, u8s_from_str, jsonify, PreciseNumber
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

Point = Tuple[int, ...]
Frontier = Set[Point]

@dataclass
class AlgorithmDetails(FromDict):
    name: str
    player_id: str
    challenge_id: str
    breakthrough_id: Optional[str]
    type: str
    fee_paid: PreciseNumber

@dataclass
class AlgorithmState(FromDict):
    block_confirmed: int
    round_submitted: int
    round_pushed: Optional[int]
    round_active: Optional[int]
    round_merged: Optional[int]
    banned: bool

@dataclass
class AlgorithmBlockData(FromDict):
    num_qualifiers_by_player: Dict[str, int]
    adoption: PreciseNumber
    merge_points: int
    reward: PreciseNumber

@dataclass
class Algorithm(FromDict):
    id: str
    details: AlgorithmDetails
    state: AlgorithmState
    block_data: Optional[AlgorithmBlockData]

@dataclass
class BenchmarkSettings(FromDict):
    player_id: str
    block_id: str
    challenge_id: str
    algorithm_id: str
    difficulty: Point

    def calc_seed(self, rand_hash: str, nonce: int) -> bytes:
        return u8s_from_str(f"{jsonify(self)}_{rand_hash}_{nonce}")

@dataclass
class PrecommitDetails(FromDict):
    block_started: int
    num_nonces: int
    rand_hash: str
    fee_paid: PreciseNumber

@dataclass
class PrecommitState(FromDict):
    block_confirmed: int

@dataclass
class Precommit(FromDict):
    benchmark_id: str
    details: PrecommitDetails
    settings: BenchmarkSettings
    state: PrecommitState

@dataclass
class BenchmarkDetails(FromDict):
    num_solutions: int
    merkle_root: MerkleHash
    sampled_nonces: List[int]

@dataclass
class BenchmarkState(FromDict):
    block_confirmed: int

@dataclass
class Benchmark(FromDict):
    id: str
    details: BenchmarkDetails
    state: BenchmarkState
    solution_nonces: Optional[Set[int]]

@dataclass
class OutputMetaData(FromDict):
    nonce: int
    runtime_signature: int
    fuel_consumed: int
    solution_signature: int

    @classmethod
    def from_output_data(cls, output_data: 'OutputData') -> 'OutputMetaData':
        return OutputData.to_output_metadata()

    def to_merkle_hash(self) -> MerkleHash:
        return MerkleHash(u8s_from_str(jsonify(self)))

@dataclass
class OutputData(FromDict):
    nonce: int
    runtime_signature: int
    fuel_consumed: int
    solution: dict

    def calc_solution_signature(self) -> int:
        return u64s_from_str(jsonify(self.solution))[0]

    def to_output_metadata(self) -> OutputMetaData:
        return OutputMetaData(
            nonce=self.nonce,
            runtime_signature=self.runtime_signature,
            fuel_consumed=self.fuel_consumed,
            solution_signature=self.calc_solution_signature()
        )

    def to_merkle_hash(self) -> MerkleHash:
        return self.to_output_metadata().to_merkle_hash()

@dataclass
class MerkleProof(FromDict):
    leaf: OutputData
    branch: MerkleBranch

@dataclass
class ProofDetails(FromDict):
    submission_delay: int
    block_active: int

@dataclass
class ProofState(FromDict):
    block_confirmed: int

@dataclass
class Proof(FromDict):
    benchmark_id: str
    details: ProofDetails
    state: ProofState
    merkle_proofs: Optional[List[MerkleProof]]

@dataclass
class FraudState(FromDict):
    block_confirmed: int

@dataclass
class Fraud(FromDict):
    benchmark_id: str
    state: FraudState
    allegation: Optional[str]

@dataclass
class BlockDetails(FromDict):
    prev_block_id: str
    height: int
    round: int
    timestamp: int
    num_confirmed: Dict[str, int]
    num_active: Dict[str, int]

@dataclass
class BlockData(FromDict):
    confirmed_ids: Dict[str, Set[int]]
    active_ids: Dict[str, Set[int]]

@dataclass
class Block(FromDict):
    id: str
    details: BlockDetails
    config: dict
    data: Optional[BlockData]

@dataclass
class ChallengeDetails(FromDict):
    name: str

@dataclass
class ChallengeState(FromDict):
    round_active: int

@dataclass
class ChallengeBlockData(FromDict):
    num_qualifiers: int
    qualifier_difficulties: Set[Point]
    base_frontier: Frontier
    scaled_frontier: Frontier
    scaling_factor: float
    base_fee: PreciseNumber
    per_nonce_fee: PreciseNumber

@dataclass
class Challenge(FromDict):
    id: str
    details: ChallengeDetails
    state: ChallengeState
    block_data: Optional[ChallengeBlockData]

@dataclass
class OPoWBlockData(FromDict):
    num_qualifiers_by_challenge: Dict[str, int]
    cutoff: int
    delegated_weighted_deposit: PreciseNumber
    delegators: Set[str]
    reward_share: float
    imbalance: PreciseNumber
    influence: PreciseNumber
    reward: PreciseNumber

@dataclass
class OPoW(FromDict):
    player_id: str
    block_data: Optional[OPoWBlockData]

@dataclass
class PlayerDetails(FromDict):
    name: Optional[str]
    is_multisig: bool

@dataclass
class PlayerState(FromDict):
    total_fees_paid: PreciseNumber
    available_fee_balance: PreciseNumber
    delegatee: Optional[dict]
    votes: dict
    reward_share: Optional[dict]

@dataclass
class PlayerBlockData(FromDict):
    delegatee: Optional[str]
    reward_by_type: Dict[str, PreciseNumber]
    deposit_by_locked_period: List[PreciseNumber]
    weighted_deposit: PreciseNumber

@dataclass
class Player(FromDict):
    id: str
    details: PlayerDetails
    state: PlayerState
    block_data: Optional[PlayerBlockData]

@dataclass
class BinaryDetails(FromDict):
    compile_success: bool
    download_url: Optional[str]

@dataclass
class BinaryState(FromDict):
    block_confirmed: int

@dataclass
class Binary(FromDict):
    algorithm_id: str
    details: BinaryDetails
    state: BinaryState

@dataclass
class TopUpDetails(FromDict):
    player_id: str
    amount: PreciseNumber
    log_idx: int
    tx_hash: str

@dataclass
class TopUpState(FromDict):
    block_confirmed: int

@dataclass
class TopUp(FromDict):
    id: str
    details: TopUpDetails
    state: TopUpState

@dataclass
class DifficultyData(FromDict):
    num_solutions: int
    num_nonces: int
    difficulty: Point

@dataclass
class DepositDetails(FromDict):
    player_id: str
    amount: PreciseNumber
    log_idx: int
    tx_hash: str
    start_timestamp: int
    end_timestamp: int

@dataclass
class DepositState(FromDict):
    block_confirmed: int

@dataclass
class Deposit(FromDict):
    id: str
    details: DepositDetails
    state: DepositState