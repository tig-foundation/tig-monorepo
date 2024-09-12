from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

Point = Tuple[int, ...]
Frontier = Set[Point]

@dataclass
class AlgorithmDetails:
    name: str
    player_id: str
    challenge_id: str
    tx_hash: str

@dataclass
class AlgorithmState:
    block_confirmed: Optional[int] = None
    round_submitted: Optional[int] = None
    round_pushed: Optional[int] = None
    round_merged: Optional[int] = None
    banned: bool = False

@dataclass
class AlgorithmBlockData:
    num_qualifiers_by_player: Optional[Dict[str, int]] = None
    adoption: Optional[int] = None
    merge_points: Optional[int] = None
    reward: Optional[int] = None
    round_earnings: Optional[int] = None

@dataclass
class Algorithm:
    id: str
    details: AlgorithmDetails
    state: Optional[AlgorithmState] = None
    block_data: Optional[AlgorithmBlockData] = None
    code: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Algorithm":
        data = d.pop("block_data")
        return cls(
            id=d.pop("id"),
            details=AlgorithmDetails(**d.pop("details")),
            state=AlgorithmState(**d.pop("state")),
            block_data=AlgorithmBlockData(**data) if data else None,
            code=d.pop("code", None)
        )

@dataclass
class BenchmarkSettings:
    player_id: str
    block_id: str
    challenge_id: str
    algorithm_id: str
    difficulty: List[int]

@dataclass
class BenchmarkDetails:
    block_started: int
    num_solutions: int

@dataclass
class BenchmarkState:
    block_confirmed: Optional[int] = None
    sampled_nonces: Optional[List[int]] = None

@dataclass
class SolutionMetaData:
    nonce: int
    solution_signature: int

@dataclass
class SolutionData:
    nonce: int
    runtime_signature: int
    fuel_consumed: int
    solution: Dict[str, Any]

@dataclass
class Benchmark:
    id: str
    settings: BenchmarkSettings
    details: BenchmarkDetails
    state: Optional[BenchmarkState] = None
    solutions_meta_data: Optional[List[SolutionMetaData]] = None
    solution_data: Optional[SolutionData] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Benchmark":
        solution_data = d.pop("solution_data", None)
        solutions_meta_data = d.pop("solutions_meta_data")
        return cls(
            id=d.pop("id"),
            settings=BenchmarkSettings(**d.pop("settings")),
            details=BenchmarkDetails(**d.pop("details")),
            state=BenchmarkState(**d.pop("state")),
            solutions_meta_data=[SolutionMetaData(**s) for s in solutions_meta_data] if solutions_meta_data else None,
            solution_data=SolutionData(**solution_data) if solution_data else None
        )

@dataclass
class BlockDetails:
    prev_block_id: str
    height: int
    round: int
    eth_block_num: Optional[str] = None

@dataclass
class BlockData:
    mempool_challenge_ids: Set[str] = field(default_factory=set)
    mempool_algorithm_ids: Set[str] = field(default_factory=set)
    mempool_benchmark_ids: Set[str] = field(default_factory=set)
    mempool_proof_ids: Set[str] = field(default_factory=set)
    mempool_fraud_ids: Set[str] = field(default_factory=set)
    mempool_wasm_ids: Set[str] = field(default_factory=set)
    active_challenge_ids: Set[str] = field(default_factory=set)
    active_algorithm_ids: Set[str] = field(default_factory=set)
    active_benchmark_ids: Set[str] = field(default_factory=set)
    active_player_ids: Set[str] = field(default_factory=set)

@dataclass
class Block:
    id: str
    details: BlockDetails
    config: dict
    data: Optional[BlockData] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        data = d.pop("data", None)
        return cls(
            id=d.pop("id"),
            details=BlockDetails(**d.pop("details")),
            config=d.pop("config"),
            data=BlockData(**data) if data else None
        )

@dataclass
class ChallengeDetails:
    name: str

@dataclass
class ChallengeState:
    block_confirmed: Optional[int] = None
    round_active: Optional[int] = None

@dataclass
class ChallengeBlockData:
    solution_signature_threshold: Optional[int] = None
    num_qualifiers: Optional[int] = None
    qualifier_difficulties: Optional[Set[Point]] = None
    base_frontier: Optional[Frontier] = None
    cutoff_frontier: Optional[Frontier] = None
    scaled_frontier: Optional[Frontier] = None
    scaling_factor: Optional[float] = None

@dataclass
class Challenge:
    id: str
    details: ChallengeDetails
    state: Optional[ChallengeState] = None
    block_data: Optional[ChallengeBlockData] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Challenge":
        block_data = d.pop("block_data", None)
        return cls(
            id=d.pop("id"),
            details=ChallengeDetails(**d.pop("details")),
            state=ChallengeState(**d.pop("state")),
            block_data=ChallengeBlockData(**block_data) if block_data else None
        )

@dataclass
class PlayerDetails:
    name: str
    is_multisig: bool

@dataclass
class PlayerBlockData:
    num_qualifiers_by_challenge: Optional[Dict[str, int]] = None
    cutoff: Optional[int] = None
    deposit: Optional[int] = None
    rolling_deposit: Optional[int] = None
    imbalance: Optional[int] = None
    imbalance_penalty: Optional[int] = None
    influence: Optional[int] = None
    reward: Optional[int] = None
    round_earnings: Optional[int] = None
    qualifying_percent_rolling_deposit: Optional[int] = None

@dataclass
class Player:
    id: str
    details: PlayerDetails
    block_data: Optional[PlayerBlockData] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Player":
        data = d.pop("block_data")
        return cls(
            id=d.pop("id"),
            details=PlayerDetails(**d.pop("details")),
            block_data=PlayerBlockData(**data) if data else None
        )

@dataclass
class ProofState:
    block_confirmed: Optional[int] = None
    submission_delay: Optional[int] = None

@dataclass
class Proof:
    benchmark_id: str
    state: Optional[ProofState] = None
    solutions_data: Optional[List[SolutionData]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Proof":
        solutions_data = d.pop("solutions_data")
        return cls(
            benchmark_id=d.pop("benchmark_id"),
            state=ProofState(**d.pop("state")),
            solutions_data=[SolutionData(**s) for s in solutions_data] if solutions_data else None
        )

@dataclass
class FraudState:
    block_confirmed: Optional[int] = None

@dataclass
class Fraud:
    benchmark_id: str
    state: Optional[FraudState] = None
    allegation: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Fraud":
        return cls(
            benchmark_id=d.pop("benchmark_id"),
            state=FraudState(**d.pop("state")),
            allegation=d.pop("allegation", None)
        )

@dataclass
class WasmDetails:
    compile_success: bool
    download_url: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class WasmState:
    block_confirmed: Optional[int] = None

@dataclass
class Wasm:
    algorithm_id: str
    details: WasmDetails
    state: Optional[WasmState] = None
    wasm_blob: Optional[bytes] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Wasm":
        return cls(
            algorithm_id=d.pop("algorithm_id"),
            details=WasmDetails(**d.pop("details")),
            state=WasmState(**d.pop("state")),
            wasm_blob=d.pop("wasm_blob", None)
        )

@dataclass
class QueryData:
    block: Block
    algorithms: Dict[str, Algorithm]
    wasms: Dict[str, Wasm]
    player: Optional[Player]
    benchmarks: Dict[str, Benchmark]
    proofs: Dict[str, Proof]
    frauds: Dict[str, Fraud]
    challenges: Dict[str, Challenge]

@dataclass
class Timestamps:
    start: int
    end: int
    submit: int

@dataclass
class Job:
    download_url: str
    benchmark_id: str
    settings: BenchmarkSettings
    solution_signature_threshold: int
    sampled_nonces: Optional[List[int]]
    wasm_vm_config: dict
    weight: float
    timestamps: Timestamps
    solutions_data: Dict[int, SolutionData]

@dataclass
class State:
    query_data: QueryData
    available_jobs: Dict[str, Job]
    pending_benchmark_jobs: Dict[str, Job]
    pending_proof_jobs: Dict[str, Job]
    submitted_proof_ids: Set[str]
    difficulty_samplers: dict