from sqlalchemy import (
    Column, String, Integer, Boolean, ForeignKey, JSON, Text, Numeric, Float, UniqueConstraint
)
from sqlalchemy.orm import relationship
from database import Base
from structs import * 

# Helper functions for PreciseNumber conversions
def precise_to_float(value):
    return float(value.to_float()) if value else None

def float_to_precise(value):
    return PreciseNumber(value) if value is not None else None

# BlockModel
class BlockModel(Base):
    __tablename__ = 'blocks'
    id = Column(String, primary_key=True)
    prev_block_id = Column(String, ForeignKey('blocks.id'), nullable=True)
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

    # Relationships
    prev_block = relationship('BlockModel', remote_side=[id], backref='next_blocks')
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

# PlayerModel
class PlayerModel(Base):
    __tablename__ = 'players'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    is_multisig = Column(Boolean, nullable=False)
    total_fees_paid = Column(Numeric(38, 18), nullable=True)
    available_fee_balance = Column(Numeric(38, 18), nullable=True)

    # Relationships
    algorithms = relationship('AlgorithmModel', back_populates='player', cascade="all, delete-orphan")
    precommits = relationship('PrecommitModel', back_populates='player', cascade="all, delete-orphan")
    benchmarks = relationship('BenchmarkModel', back_populates='player', cascade="all, delete-orphan")
    proofs = relationship('ProofModel', back_populates='player', cascade="all, delete-orphan")
    frauds = relationship('FraudModel', back_populates='player', cascade="all, delete-orphan")

    @classmethod
    def from_dataclass(cls, player: Player):
        return cls(
            id=player.id,
            name=player.details.name,
            is_multisig=player.details.is_multisig,
            total_fees_paid=precise_to_float(player.state.total_fees_paid) if player.state else None,
            available_fee_balance=precise_to_float(player.state.available_fee_balance) if player.state else None
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
        return Player(
            id=self.id,
            details=details,
            state=state,
            block_data=None  # Implement if needed
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
    player_id = Column(String, ForeignKey('players.id'), nullable=False)
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
    player = relationship('PlayerModel', back_populates='algorithms')
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
            block_id=algorithm.block_data.block_id if algorithm.block_data else None
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
    player_id = Column(String, ForeignKey('players.id'), nullable=False)
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
    player = relationship('PlayerModel', back_populates='precommits')
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

    player_id = Column(String, ForeignKey('players.id'), nullable=False)

    # Relationships
    player = relationship('PlayerModel', back_populates='benchmarks')
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
    player_id = Column(String, ForeignKey('players.id'), nullable=True)

    # Relationships
    benchmark = relationship('BenchmarkModel', back_populates='proofs')
    player = relationship('PlayerModel', back_populates='proofs')

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
    player_id = Column(String, ForeignKey('players.id'), nullable=True)

    # Relationships
    benchmark = relationship('BenchmarkModel', back_populates='frauds')
    player = relationship('PlayerModel', back_populates='frauds')

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
