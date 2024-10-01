from tig_benchmarker.merkle_tree import MerkleHash, MerkleBranch
from tig_benchmarker.utils import FromDict, u64s_from_str, u8s_from_str, jsonify, PreciseNumber
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

Point = Tuple[int, ...]
Frontier = Set[Point]

@dataclass
class AlgorithmDetails(FromDict):
    name: str
    player_id: str
    challenge_id: str
    tx_hash: str

@dataclass
class AlgorithmState(FromDict):
    block_confirmed: int
    round_submitted: int
    round_pushed: Optional[int]
    round_merged: Optional[int]
    banned: bool

@dataclass
class AlgorithmBlockData(FromDict):
    num_qualifiers_by_player: Dict[str, int]
    adoption: PreciseNumber
    merge_points: int
    reward: PreciseNumber
    round_earnings: PreciseNumber

@dataclass
class Algorithm(FromDict):
    id: str
    details: AlgorithmDetails
    state: AlgorithmState
    block_data: Optional[AlgorithmBlockData]
    code: Optional[str]

@dataclass
class BenchmarkSettings(FromDict):
    player_id: str
    block_id: str
    challenge_id: str
    algorithm_id: str
    difficulty: List[int]

    def calc_seed(self, rand_hash: str, nonce: int) -> bytes:
        return u8s_from_str(f"{jsonify(self)}_{rand_hash}_{nonce}")

@dataclass
class PrecommitDetails(FromDict):
    block_started: int
    num_nonces: Optional[int] # Optional for backwards compatibility
    fee_paid: Optional[int] # Optional for backwards compatibility

@dataclass
class PrecommitState(FromDict):
    block_confirmed: int
    rand_hash: Optional[str] # Optional for backwards compatibility

@dataclass
class Precommit(FromDict):
    benchmark_id: str
    details: PrecommitDetails
    settings: BenchmarkSettings
    state: PrecommitState

@dataclass
class BenchmarkDetails(FromDict):
    num_solutions: int
    merkle_root: Optional[MerkleHash] # Optional for backwards compatibility

@dataclass
class BenchmarkState(FromDict):
    block_confirmed: int
    sampled_nonces: List[int]

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
    branch: Optional[MerkleBranch] # Optional for backwards compatibility

@dataclass
class ProofState(FromDict):
    block_confirmed: int
    submission_delay: int

@dataclass
class Proof(FromDict):
    benchmark_id: str
    state: Optional[ProofState]
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
    eth_block_num: Optional[str] # Optional for backwards compatability
    fees_paid: Optional[PreciseNumber] # Optional for backwards compatability
    num_confirmed_challenges: Optional[int] # Optional for backwards compatability
    num_confirmed_algorithms: Optional[int] # Optional for backwards compatability
    num_confirmed_benchmarks: Optional[int] # Optional for backwards compatability
    num_confirmed_precommits: Optional[int] # Optional for backwards compatability
    num_confirmed_proofs: Optional[int] # Optional for backwards compatability
    num_confirmed_frauds: Optional[int] # Optional for backwards compatability
    num_confirmed_topups: Optional[int] # Optional for backwards compatability
    num_confirmed_wasms: Optional[int] # Optional for backwards compatability
    num_active_challenges: Optional[int] # Optional for backwards compatability
    num_active_algorithms: Optional[int] # Optional for backwards compatability
    num_active_benchmarks: Optional[int] # Optional for backwards compatability
    num_active_players: Optional[int] # Optional for backwards compatability

@dataclass
class BlockData(FromDict):
    confirmed_challenge_ids: Set[int]
    confirmed_algorithm_ids: Set[int]
    confirmed_benchmark_ids: Set[int]
    confirmed_precommit_ids: Set[int]
    confirmed_proof_ids: Set[int]
    confirmed_fraud_ids: Set[int]
    confirmed_topup_ids: Set[int]
    confirmed_wasm_ids: Set[int]
    active_challenge_ids: Set[int]
    active_algorithm_ids: Set[int]
    active_benchmark_ids: Set[int]
    active_player_ids: Set[int]

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
    block_confirmed: int
    round_active: Optional[int]

@dataclass
class ChallengeBlockData(FromDict):
    solution_signature_threshold: int
    num_qualifiers: int
    qualifier_difficulties: Set[Point]
    base_frontier: Frontier
    scaled_frontier: Frontier
    scaling_factor: float
    base_fee: Optional[PreciseNumber]
    per_nonce_fee: Optional[PreciseNumber]

@dataclass
class Challenge(FromDict):
    id: str
    details: ChallengeDetails
    state: ChallengeState
    block_data: Optional[ChallengeBlockData]

@dataclass
class PlayerDetails(FromDict):
    name: str
    is_multisig: bool

@dataclass
class PlayerBlockData(FromDict):
    num_qualifiers_by_challenge: Optional[Dict[str, int]]
    cutoff: Optional[int]
    deposit: Optional[PreciseNumber]
    rolling_deposit: Optional[PreciseNumber]
    qualifying_percent_rolling_deposit: Optional[PreciseNumber]
    imbalance: Optional[PreciseNumber]
    imbalance_penalty: Optional[PreciseNumber]
    influence: Optional[PreciseNumber]
    reward: Optional[PreciseNumber]
    round_earnings: PreciseNumber

@dataclass
class PlayerState(FromDict):
    total_fees_paid: PreciseNumber
    available_fee_balance: PreciseNumber

@dataclass
class Player(FromDict):
    id: str
    details: PlayerDetails
    state: Optional[PlayerState]
    block_data: Optional[PlayerBlockData]

@dataclass
class WasmDetails(FromDict):
    compile_success: bool
    download_url: Optional[str]
    checksum: Optional[str]

@dataclass
class WasmState(FromDict):
    block_confirmed: int

@dataclass
class Wasm(FromDict):
    algorithm_id: str
    details: WasmDetails
    state: WasmState

@dataclass
class TopUpDetails(FromDict):
    player_id: str
    amount: PreciseNumber

@dataclass
class TopUpState(FromDict):
    block_confirmed: int

@dataclass
class TopUp(FromDict):
    id: str
    details: TopUpDetails
    state: TopUpState

@dataclass
class QueryData(FromDict):
    block: Block
    algorithms: Dict[str, Algorithm]
    wasms: Dict[str, Wasm]
    player: Optional[Player]
    precommits: Dict[str, Precommit]
    benchmarks: Dict[str, Benchmark]
    proofs: Dict[str, Proof]
    frauds: Dict[str, Fraud]
    challenges: Dict[str, Challenge]