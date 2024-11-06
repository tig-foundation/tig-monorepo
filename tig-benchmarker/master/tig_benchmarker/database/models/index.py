from decimal import Decimal
from sqlalchemy import (
    Column, DateTime, String, Integer, Boolean, ForeignKey, JSON, Text, Numeric, TIMESTAMP, UniqueConstraint, BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tig_benchmarker.database.init import Base
from tig_benchmarker.structs import * 
import datetime

# Helper functions for PreciseNumber conversions
def precise_to_float(value):
    if isinstance(value, PreciseNumber):
        return float(value.to_float())/10**18
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return None

def float_to_precise(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return PreciseNumber(str(value))
    return PreciseNumber(value)

# Master Config Model
class ConfigModel(Base):
    __tablename__ = 'config'
    id = Column(Integer, primary_key=True, default=1)
    config_data = Column(JSON, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

#BlockModel
class BlockModel(Base):
    __tablename__ = 'blocks'
    id = Column(String, primary_key=True)
    prev_block_id = Column(String, nullable=True)
    height = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    eth_block_num = Column(String, nullable=True)
    fees_paid = Column(Numeric(38, 18), nullable=True)
    num_confirmed_challenges = Column(Integer, nullable=True)
    num_confirmed_algorithms = Column(Integer, nullable=True)
    num_confirmed_benchmarks = Column(Integer, nullable=True)
    num_confirmed_precommits = Column(Integer, nullable=True)
    num_confirmed_proofs = Column(Integer, nullable=True)
    num_confirmed_frauds = Column(Integer, nullable=True)
    num_confirmed_topups = Column(Integer, nullable=True)
    num_confirmed_wasms = Column(Integer, nullable=True)
    num_active_challenges = Column(Integer, nullable=True)
    num_active_algorithms = Column(Integer, nullable=True)
    num_active_benchmarks = Column(Integer, nullable=True)
    num_active_players = Column(Integer, nullable=True)
    config = Column(JSON, nullable=False)
    data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    # prev_block = relationship('BlockModel', remote_side=[id], backref='next_blocks')
    algorithms = relationship('AlgorithmModel', back_populates='block', cascade="all, delete-orphan")
    challenges = relationship('ChallengeModel', back_populates='block', cascade="all, delete-orphan")

    @classmethod
    def from_dataclass(cls, block: Block):
        return cls(
            id=block.id,
            prev_block_id=block.details.prev_block_id,
            height=block.details.height,
            round=block.details.round,
            eth_block_num=block.details.eth_block_num,
            fees_paid=precise_to_float(block.details.fees_paid),
            num_confirmed_challenges=block.details.num_confirmed_challenges,
            num_confirmed_algorithms=block.details.num_confirmed_algorithms,
            num_confirmed_benchmarks=block.details.num_confirmed_benchmarks,
            num_confirmed_precommits=block.details.num_confirmed_precommits,
            num_confirmed_proofs=block.details.num_confirmed_proofs,
            num_confirmed_frauds=block.details.num_confirmed_frauds,
            num_confirmed_topups=block.details.num_confirmed_topups,
            num_confirmed_wasms=block.details.num_confirmed_wasms,
            num_active_challenges=block.details.num_active_challenges,
            num_active_algorithms=block.details.num_active_algorithms,
            num_active_benchmarks=block.details.num_active_benchmarks,
            num_active_players=block.details.num_active_players,
            config=block.config,
            data=block.data
        )

    def to_dataclass(self) -> Block:
        details = BlockDetails(
            prev_block_id=self.prev_block_id,
            height=self.height,
            round=self.round,
            eth_block_num=self.eth_block_num,
            fees_paid=float_to_precise(self.fees_paid),
            num_confirmed_challenges=self.num_confirmed_challenges,
            num_confirmed_algorithms=self.num_confirmed_algorithms,
            num_confirmed_benchmarks=self.num_confirmed_benchmarks,
            num_confirmed_precommits=self.num_confirmed_precommits,
            num_confirmed_proofs=self.num_confirmed_proofs,
            num_confirmed_frauds=self.num_confirmed_frauds,
            num_confirmed_topups=self.num_confirmed_topups,
            num_confirmed_wasms=self.num_confirmed_wasms,
            num_active_challenges=self.num_active_challenges,
            num_active_algorithms=self.num_active_algorithms,
            num_active_benchmarks=self.num_active_benchmarks,
            num_active_players=self.num_active_players
        )
        return Block(
            id=self.id,
            details=details,
            config=self.config,
            data=self.data
        )

# # PlayerModel
class PlayerModel(Base):
    __tablename__ = 'players'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    is_multisig = Column(Boolean, nullable=False)
    total_fees_paid = Column(Numeric(38, 18), nullable=True)
    available_fee_balance = Column(Numeric(38, 18), nullable=True)
    block_data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    # algorithms = relationship('AlgorithmModel', back_populates='player', cascade="all, delete-orphan")
    # precommits = relationship('PrecommitModel', back_populates='player', cascade="all, delete-orphan")
    # benchmarks = relationship('BenchmarkModel', back_populates='player', cascade="all, delete-orphan")
    # proofs = relationship('ProofModel', back_populates='player', cascade="all, delete-orphan")
    # frauds = relationship('FraudModel', back_populates='player', cascade="all, delete-orphan")

    @classmethod
    def from_dataclass(cls, player: Player):
        return cls(
            id=player.id,
            name=player.details.name,
            is_multisig=player.details.is_multisig,
            total_fees_paid=precise_to_float(player.state.total_fees_paid) if player.state else None,
            available_fee_balance=precise_to_float(player.state.available_fee_balance) if player.state else None,
            block_data=player.block_data.__dict__ if player.block_data else {}
        )

    def to_dataclass(self) -> Player:
        details = PlayerDetails(
            name=self.name,
            is_multisig=self.is_multisig
        )
        state = PlayerState(
            total_fees_paid=float_to_precise(self.total_fees_paid),
            available_fee_balance=float_to_precise(self.available_fee_balance)
        ) if self.total_fees_paid is not None else None
        block_data = PlayerBlockData(
            num_qualifiers_by_challenge=self.block_data.get("num_qualifiers_by_challenge"),
            cutoff=self.block_data.get("cutoff"),
            deposit=self.block_data.get("deposit"),
            rolling_deposit=self.block_data.get("rolling_deposit"),
            qualifying_percent_rolling_deposit=self.block_data.get("qualifying_percent_rolling_deposit"),
            imbalance=self.block_data.get("imbalance"),
            imbalance_penalty=self.block_data.get("imbalance_penalty"),
            influence=self.block_data.get("influence"),
            reward=self.block_data.get("reward"),
            round_earnings=self.block_data.get("round_earnings")
        ) if self.block_data else None
        return Player(
            id=self.id,
            details=details,
            state=state,
            block_data=block_data
        )

# ChallengeModel
class ChallengeModel(Base):
    __tablename__ = 'challenges'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    block_confirmed = Column(Integer, nullable=False)
    round_active = Column(Integer, nullable=True)

    # Relationships
    algorithms = relationship('AlgorithmModel', back_populates='challenge', cascade="all, delete-orphan")
    precommits = relationship('PrecommitModel', back_populates='challenge', cascade="all, delete-orphan")
    submit_precommits = relationship('PrecommitRequestModel', back_populates='challenge', cascade="all, delete-orphan")
    block_id = Column(String, ForeignKey('blocks.id'), nullable=True)
    block = relationship('BlockModel', back_populates='challenges')
    difficulty_data = relationship('DifficultyDataModel', back_populates='challenge', cascade="all, delete-orphan")

    @classmethod
    def from_dataclass(cls, challenge: Challenge):
        return cls(
            id=challenge.id,
            name=challenge.details.name,
            block_confirmed=challenge.state.block_confirmed,
            round_active=challenge.state.round_active
        )

    def to_dataclass(self) -> Challenge:
        details = ChallengeDetails(
            name=self.name
        )
        state = ChallengeState(
            block_confirmed=self.block_confirmed,
            round_active=self.round_active
        )
        return Challenge(
            id=self.id,
            details=details,
            state=state,
            block_data=None  # Implement if needed
        )

# AlgorithmModel
class AlgorithmModel(Base):
    __tablename__ = 'algorithms'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    player_id = Column(String, nullable=False)
    challenge_id = Column(String, ForeignKey('challenges.id'), nullable=False)
    tx_hash = Column(String, nullable=True)
    block_confirmed = Column(Integer, nullable=True)
    round_submitted = Column(Integer, nullable=True)
    round_pushed = Column(Integer, nullable=True)
    round_merged = Column(Integer, nullable=True)
    banned = Column(Boolean, nullable=True)
    code = Column(Text, nullable=True)
    block_id = Column(String, ForeignKey('blocks.id'), nullable=True)

    # Relationships
    # player = relationship('PlayerModel', back_populates='algorithms')
    challenge = relationship('ChallengeModel', back_populates='algorithms')
    wasms = relationship('WasmModel', back_populates='algorithm', cascade="all, delete-orphan")
    block = relationship('BlockModel', back_populates='algorithms')

    @classmethod
    def from_dataclass(cls, algorithm: Algorithm):
        return cls(
            id=algorithm.id,
            name=algorithm.details.name,
            player_id=algorithm.details.player_id,
            challenge_id=algorithm.details.challenge_id,
            tx_hash=algorithm.details.tx_hash,
            block_confirmed=algorithm.state.block_confirmed,
            round_submitted=algorithm.state.round_submitted,
            round_pushed=algorithm.state.round_pushed,
            round_merged=algorithm.state.round_merged,
            banned=algorithm.state.banned,
            code=algorithm.code,
            block_id=algorithm.block_data.block_id if algorithm.block_data.block_id else None
        )

    def to_dataclass(self) -> Algorithm:
        details = AlgorithmDetails(
            name=self.name,
            player_id=self.player_id,
            challenge_id=self.challenge_id,
            tx_hash=self.tx_hash
        )
        state = AlgorithmState(
            block_confirmed=self.block_confirmed,
            round_submitted=self.round_submitted,
            round_pushed=self.round_pushed,
            round_merged=self.round_merged,
            banned=self.banned
        )
        return Algorithm(
            id=self.id,
            details=details,
            state=state,
            block_data=None,  # Implement if needed
            code=self.code
        )

# WasmModel
class WasmModel(Base):
    __tablename__ = 'wasms'
    id = Column(Integer, primary_key=True)
    algorithm_id = Column(String, ForeignKey('algorithms.id'), nullable=False)
    compile_success = Column(Boolean, nullable=False)
    download_url = Column(String, nullable=True)
    checksum = Column(String, nullable=True)
    block_confirmed = Column(Integer, nullable=False)

    # Relationships
    algorithm = relationship('AlgorithmModel', back_populates='wasms')

    @classmethod
    def from_dataclass(cls, wasm: Wasm):
        return cls(
            algorithm_id=wasm.algorithm_id,
            compile_success=wasm.details.compile_success,
            download_url=wasm.details.download_url,
            checksum=wasm.details.checksum,
            block_confirmed=wasm.state.block_confirmed
        )

    def to_dataclass(self) -> Wasm:
        details = WasmDetails(
            compile_success=self.compile_success,
            download_url=self.download_url,
            checksum=self.checksum
        )
        state = WasmState(
            block_confirmed=self.block_confirmed
        )
        return Wasm(
            algorithm_id=self.algorithm_id,
            details=details,
            state=state
        )

# PrecommitModel
class PrecommitModel(Base):
    __tablename__ = 'precommits'
    benchmark_id = Column(String, primary_key=True)
    player_id = Column(String, nullable=False)
    block_id = Column(String, ForeignKey('blocks.id'), nullable=False)
    challenge_id = Column(String, ForeignKey('challenges.id'), nullable=False)
    algorithm_id = Column(String, ForeignKey('algorithms.id'), nullable=False)
    difficulty = Column(JSON, nullable=False)
    block_started = Column(Integer, nullable=False)
    num_nonces = Column(Integer, nullable=True)
    fee_paid = Column(Numeric(38, 18), nullable=True)
    rand_hash = Column(String, nullable=True)
    block_confirmed = Column(Integer, nullable=True)

    # Relationships
    # player = relationship('PlayerModel', back_populates='precommits')
    challenge = relationship('ChallengeModel', back_populates='precommits')
    algorithm = relationship('AlgorithmModel')
    block = relationship('BlockModel')

    @classmethod
    def from_dataclass(cls, precommit: Precommit):
        return cls(
            benchmark_id=precommit.benchmark_id,
            player_id=precommit.settings.player_id,
            block_id=precommit.settings.block_id,
            challenge_id=precommit.settings.challenge_id,
            algorithm_id=precommit.settings.algorithm_id,
            difficulty=precommit.settings.difficulty,
            block_started=precommit.details.block_started,
            num_nonces=precommit.details.num_nonces,
            fee_paid=precise_to_float(precommit.details.fee_paid),
            rand_hash=precommit.state.rand_hash,
            block_confirmed=precommit.state.block_confirmed
        )

    def to_dataclass(self) -> Precommit:
        settings = BenchmarkSettings(
            player_id=self.player_id,
            block_id=self.block_id,
            challenge_id=self.challenge_id,
            algorithm_id=self.algorithm_id,
            difficulty=self.difficulty
        )
        details = PrecommitDetails(
            block_started=self.block_started,
            num_nonces=self.num_nonces,
            fee_paid=float_to_precise(self.fee_paid)
        )
        state = PrecommitState(
            block_confirmed=self.block_confirmed,
            rand_hash=self.rand_hash
        )
        return Precommit(
            benchmark_id=self.benchmark_id,
            settings=settings,
            details=details,
            state=state
        )

# BenchmarkModel
class BenchmarkModel(Base):
    __tablename__ = 'benchmarks'
    id = Column(String, primary_key=True)
    num_solutions = Column(Integer, nullable=False)
    merkle_root = Column(String, nullable=True)
    block_confirmed = Column(Integer, nullable=False)
    sampled_nonces = Column(JSON, nullable=True)
    solution_nonces = Column(JSON, nullable=True)

    player_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)

    # Relationships
    # player = relationship('PlayerModel', back_populates='benchmarks')
    proofs = relationship('ProofModel', back_populates='benchmark', cascade="all, delete-orphan")
    frauds = relationship('FraudModel', back_populates='benchmark', cascade="all, delete-orphan")

    @classmethod
    def from_dataclass(cls, benchmark: Benchmark):
        return cls(
            id=benchmark.id,
            num_solutions=benchmark.details.num_solutions,
            merkle_root=benchmark.details.merkle_root,
            block_confirmed=benchmark.state.block_confirmed,
            sampled_nonces=benchmark.state.sampled_nonces,
            solution_nonces=list(benchmark.solution_nonces) if benchmark.solution_nonces else None,
            player_id=benchmark.player_id if hasattr(benchmark, 'player_id') else None
        )

    def to_dataclass(self) -> Benchmark:
        details = BenchmarkDetails(
            num_solutions=self.num_solutions,
            merkle_root=self.merkle_root
        )
        state = BenchmarkState(
            block_confirmed=self.block_confirmed,
            sampled_nonces=self.sampled_nonces
        )
        return Benchmark(
            id=self.id,
            details=details,
            state=state,
            solution_nonces=set(self.solution_nonces) if self.solution_nonces else None
        )

# ProofModel
class ProofModel(Base):
    __tablename__ = 'proofs'
    benchmark_id = Column(String, ForeignKey('benchmarks.id'), primary_key=True)
    block_confirmed = Column(Integer, nullable=True)
    submission_delay = Column(Integer, nullable=True)
    merkle_proofs = Column(JSON, nullable=True)
    player_id = Column(String,  nullable=True)

    # Relationships
    benchmark = relationship('BenchmarkModel', back_populates='proofs')
    # player = relationship('PlayerModel', back_populates='proofs')

    @classmethod
    def from_dataclass(cls, proof: Proof):
        return cls(
            benchmark_id=proof.benchmark_id,
            block_confirmed=proof.state.block_confirmed if proof.state else None,
            submission_delay=proof.state.submission_delay if proof.state else None,
            merkle_proofs=[mp.to_dict() for mp in proof.merkle_proofs] if proof.merkle_proofs else None,
            player_id=proof.player_id if hasattr(proof, 'player_id') else None
        )

    def to_dataclass(self) -> Proof:
        state = ProofState(
            block_confirmed=self.block_confirmed,
            submission_delay=self.submission_delay
        ) if self.block_confirmed is not None else None
        merkle_proofs = [MerkleProof.from_dict(mp) for mp in self.merkle_proofs] if self.merkle_proofs else None
        return Proof(
            benchmark_id=self.benchmark_id,
            state=state,
            merkle_proofs=merkle_proofs
        )

# FraudModel
class FraudModel(Base):
    __tablename__ = 'frauds'
    benchmark_id = Column(String, ForeignKey('benchmarks.id'), primary_key=True)
    block_confirmed = Column(Integer, nullable=False)
    allegation = Column(Text, nullable=True)
    player_id = Column(String, nullable=True)

    # Relationships
    benchmark = relationship('BenchmarkModel', back_populates='frauds')
    # player = relationship('PlayerModel', back_populates='frauds')

    @classmethod
    def from_dataclass(cls, fraud: Fraud):
        return cls(
            benchmark_id=fraud.benchmark_id,
            block_confirmed=fraud.state.block_confirmed,
            allegation=fraud.allegation,
            player_id=fraud.player_id if hasattr(fraud, 'player_id') else None
        )

    def to_dataclass(self) -> Fraud:
        state = FraudState(
            block_confirmed=self.block_confirmed
        )
        return Fraud(
            benchmark_id=self.benchmark_id,
            state=state,
            allegation=self.allegation
        )

# DifficultyDataModel
class DifficultyDataModel(Base):
    __tablename__ = 'difficulty_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    challenge_id = Column(String, ForeignKey('challenges.id'), nullable=False)
    num_solutions = Column(Integer, nullable=False)
    num_nonces = Column(Integer, nullable=False)
    difficulty = Column(JSON, nullable=False)

    # Relationships
    challenge = relationship('ChallengeModel', back_populates='difficulty_data')

    @classmethod
    def from_dataclass(cls, challenge_id: str, difficulty_data: DifficultyData):
        return cls(
            challenge_id=challenge_id,
            num_solutions=difficulty_data.num_solutions,
            num_nonces=difficulty_data.num_nonces,
            difficulty=difficulty_data.difficulty
        )

    def to_dataclass(self) -> DifficultyData:
        return DifficultyData(
            num_solutions=self.num_solutions,
            num_nonces=self.num_nonces,
            difficulty=tuple(self.difficulty)
        )

# TopUpModel
class TopUpModel(Base):
    __tablename__ = 'topups'
    id = Column(String, primary_key=True)
    player_id = Column(String, ForeignKey('players.id'), nullable=False)
    amount = Column(Numeric(38, 18), nullable=False)
    block_confirmed = Column(Integer, nullable=False)

    # Relationships
    player = relationship('PlayerModel')

    @classmethod
    def from_dataclass(cls, topup: TopUp):
        return cls(
            id=topup.id,
            player_id=topup.details.player_id,
            amount=precise_to_float(topup.details.amount),
            block_confirmed=topup.state.block_confirmed
        )

    def to_dataclass(self) -> TopUp:
        details = TopUpDetails(
            player_id=self.player_id,
            amount=float_to_precise(self.amount)
        )
        state = TopUpState(
            block_confirmed=self.block_confirmed
        )
        return TopUp(
            id=self.id,
            details=details,
            state=state
        )
        
#JobModel
class JobModel(Base):
    __tablename__ = 'jobs'
    
    benchmark_id = Column(String, primary_key=True)
    settings = Column(JSON, nullable=False)
    num_nonces = Column(Integer, nullable=False)
    rand_hash = Column(String, nullable=False)
    wasm_vm_config = Column(JSON, nullable=False)
    download_url = Column(String, nullable=False)
    batch_size = Column(Integer, nullable=False)
    challenge = Column(String, nullable=False)
    sampled_nonces = Column(JSON, nullable=True)
    merkle_root = Column(String, nullable=True)
    solution_nonces = Column(JSON, nullable=True)
    merkle_proofs = Column(JSON, nullable=True)
    batch_merkle_proofs = Column(JSON, nullable=True)
    batch_merkle_roots = Column(JSON, nullable=True)
    last_benchmark_submit_time = Column(Integer, nullable=False)
    last_proof_submit_time = Column(Integer, nullable=False)
    last_batch_retry_time = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    
    # Relationships
    batch_results = relationship('BatchResultModel', back_populates='job')
    assigned_batches = relationship('AssignedBatchModel', back_populates='job')
    
    def to_dataclass(self) -> Job:
        return Job(
            benchmark_id=self.benchmark_id,
            settings=BenchmarkSettings.from_dict(self.settings),
            num_nonces=self.num_nonces,
            rand_hash=self.rand_hash,
            wasm_vm_config=self.wasm_vm_config,
            download_url=self.download_url,
            batch_size=self.batch_size,
            challenge=self.challenge,
            sampled_nonces=self.sampled_nonces,
            merkle_root=self.merkle_root,
            solution_nonces=self.solution_nonces,
            merkle_proofs={int(k): MerkleProof.from_dict(v) for k, v in self.merkle_proofs.items()},
            batch_merkle_proofs={int(k): MerkleProof.from_dict(v) for k, v in self.batch_merkle_proofs.items()},
            batch_merkle_roots=[MerkleHash.from_string(root) if root else None for root in self.batch_merkle_roots],
            last_benchmark_submit_time=self.last_benchmark_submit_time,
            last_proof_submit_time=self.last_proof_submit_time,
            last_batch_retry_time=self.last_batch_retry_time
        )
    
    @classmethod
    def from_dataclass(cls, job: Job):
        return cls(
            benchmark_id=job.benchmark_id,
            settings=job.settings.to_dict(),
            num_nonces=job.num_nonces,
            rand_hash=job.rand_hash,
            wasm_vm_config=job.wasm_vm_config,
            download_url=job.download_url,
            batch_size=job.batch_size,
            challenge=job.challenge,
            sampled_nonces=job.sampled_nonces,
            merkle_root=str(job.merkle_root) if job.merkle_root else None,
            solution_nonces=list(job.solution_nonces),
            merkle_proofs={k: v.to_dict() for k, v in job.merkle_proofs.items()},
            batch_merkle_proofs={k: v.to_dict() for k, v in job.batch_merkle_proofs.items()},
            batch_merkle_roots=[str(root) if root else None for root in job.batch_merkle_roots],
            last_benchmark_submit_time=job.last_benchmark_submit_time,
            last_proof_submit_time=job.last_proof_submit_time,
            last_batch_retry_time=job.last_batch_retry_time
        )
    
    @property
    def num_batches(self) -> int:
        return (self.num_nonces + self.batch_size - 1) // self.batch_size
    
    @property
    def sampled_nonces_by_batch_idx(self) -> Dict[int, List[int]]:
        ret = {}
        for nonce in self.sampled_nonces:
            batch_idx = nonce // self.batch_size
            ret.setdefault(batch_idx, []).append(nonce)
        return ret

# AssignedBatch
class AssignedBatchModel(Base):
    __tablename__ = 'assigned_batches'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String, ForeignKey('jobs.benchmark_id'), nullable=False)
    batch_idx = Column(Integer, nullable=False)
    assigned_slave = Column(Integer, ForeignKey('slave_registry.id'), nullable=False)
    submitted_timestamp = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    completed_timestamp = Column(DateTime, nullable=True)
    # batch_result_id = Column(Integer, ForeignKey('batch_results.id'), nullable=True)
    
    # Unique constraint to prevent duplicate assignments
    __table_args__ = (
        UniqueConstraint('benchmark_id', 'batch_idx', name='uq_assigned_batch'),
    )
    
    # Relationships
    batch_result = relationship('BatchResultModel', back_populates='assigned_batch')
    job = relationship('JobModel', back_populates='assigned_batches')
    slave = relationship('SlaveRegistryModel', back_populates='assigned_batches')

# BatchResult
class BatchResultModel(Base):
    __tablename__ = 'batch_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String, ForeignKey('jobs.benchmark_id'), nullable=False)
    start_nonce = Column(Integer, nullable=False)
    merkle_root = Column(String, nullable=False)
    solution_nonces = Column(JSON, nullable=False)
    merkle_proofs = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    assigned_batch_id = Column(Integer, ForeignKey('assigned_batches.id'), nullable=True)
    
    # Relationships
    assigned_batch = relationship('AssignedBatchModel', back_populates='batch_result')
    job = relationship('JobModel', back_populates='batch_results')

# Precommit Request
class PrecommitRequestModel(Base):
    __tablename__ = 'precommit_requests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    challenge_id = Column(String, ForeignKey('challenges.id'), nullable=False)
    settings = Column(JSON, nullable=False)
    num_nonces = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    
    # Relationships
    challenge = relationship('ChallengeModel', back_populates='submit_precommits')

# Benchmark Request
class BenchmarkRequestModel(Base):
    __tablename__ = 'benchmark_requests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey('jobs.benchmark_id'), nullable=False)
    benchmark_id = Column(String, nullable=False)
    merkle_root = Column(String, nullable=False)
    solution_nonces = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    
    # Relationships
    job = relationship('JobModel')

# Proof Request
class ProofRequestModel(Base):
    __tablename__ = 'proof_requests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey('jobs.benchmark_id'), nullable=False)
    benchmark_id = Column(String, nullable=False)
    merkle_proofs = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    
    # Relationships
    job = relationship('JobModel')
    
# # Slave Registry model
class SlaveRegistryModel(Base):
    __tablename__ = 'slave_registry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slave_name = Column(String(255), unique=True, nullable=False)
    num_of_cpus = Column(Integer, nullable=False)
    num_of_threads = Column(Integer, nullable=False)
    memory = Column(BigInteger, nullable=False)
    registered_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slave_name": self.slave_name,
            "num_of_cpus": self.num_of_cpus,
            "num_of_threads": self.num_of_threads,
            "memory": self.memory,
            "registered_at": str(self.registered_at)
        }
    
    # Relationships
    assigned_batches = relationship('AssignedBatchModel', back_populates='slave')

