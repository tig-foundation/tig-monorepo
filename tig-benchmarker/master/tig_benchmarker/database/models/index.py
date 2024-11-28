from decimal import Decimal
from sqlalchemy import (
    Column, DateTime, String, Integer, Boolean, ForeignKey, JSON, Text, Numeric, TIMESTAMP, UniqueConstraint, BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tig_benchmarker.database.init import Base
from tig_benchmarker.structs import * 
import datetime

# Master Config Model
class ConfigModel(Base):
    __tablename__ = 'config'

    id = Column(Integer, primary_key=True, default=1)
    config_data = Column(JSON, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

# Slave Registry model
class SlaveModel(Base):
    __tablename__ = 'slaves'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    num_of_cpus = Column(Integer, nullable=False)
    num_of_threads = Column(Integer, nullable=False)
    memory = Column(BigInteger, nullable=False)
    registered_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)

    # Relationships
    batches = relationship("BatchModel", back_populates="slave", lazy="dynamic")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slave_name": self.name,
            "num_of_cpus": self.num_of_cpus,
            "num_of_threads": self.num_of_threads,
            "memory": self.memory,
            "registered_at": str(self.registered_at)
        }

#JobModel
class JobModel(Base):
    __tablename__ = 'jobs'
    
    benchmark_id = Column(String, primary_key=True)
    settings = Column(JSON, nullable=False)
    num_nonces = Column(Integer, nullable=False)
    rand_hash = Column(String, nullable=False)
    runtime_config = Column(JSON, nullable=False)
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
    batches = relationship("BatchModel", back_populates="job", lazy="joined")

    def to_dataclass(self) -> Job:
        return Job(
            benchmark_id=self.benchmark_id,
            settings=BenchmarkSettings.from_dict(self.settings),
            num_nonces=self.num_nonces,
            rand_hash=self.rand_hash,
            runtime_config=self.runtime_config,
            download_url=self.download_url,
            batch_size=self.batch_size,
            challenge=self.challenge,
            sampled_nonces=self.sampled_nonces,
            merkle_root=self.merkle_root,
            solution_nonces=self.solution_nonces,
            merkle_proofs={int(k): MerkleProof.from_dict(v) for k, v in self.merkle_proofs.items()},
            batch_merkle_proofs={int(k): MerkleProof.from_dict(v) for k, v in self.batch_merkle_proofs.items()},
            batch_merkle_roots=[MerkleHash.from_str(root) if root else None for root in self.batch_merkle_roots],
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
            runtime_config=job.runtime_config,
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
    
  
    
class BatchModel(Base):
    __tablename__ = 'batches'

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String, ForeignKey('jobs.benchmark_id'), nullable=False)
    slave_name = Column(String, ForeignKey('slaves.name'), nullable=False)
    start_nonce = Column(Integer, nullable=False)
    num_nonces = Column(Integer, nullable=False)
    settings = Column(JSON, nullable=False)
    runtime_config = Column(JSON, nullable=False)
    download_url = Column(String, nullable=False)
    rand_hash = Column(String, nullable=False)
    batch_size = Column(Integer, nullable=False)
    sampled_nonces = Column(JSON, nullable=True)
    solution_nonces = Column(JSON, nullable=True)
    merkle_root = Column(String, nullable=True)
    merkle_proofs = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow(), nullable=False)

    # Relationships
    job = relationship("JobModel", back_populates="batches")
    slave = relationship("SlaveModel", back_populates="batches")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "slave_name": self.slave_name,
            "start_nonce": self.start_nonce,
            "num_nonces": self.num_nonces,
            "settings": self.settings,
            "sampled_nonces": self.sampled_nonces,
            "runtime_config": self.runtime_config,
            "download_url": self.download_url,
            "rand_hash": self.rand_hash,
            "batch_size": self.batch_size,
            "sampled_nonces": self.sampled_nonces,
            "merkle_root": self.merkle_root,
            "merkle_proofs": self.merkle_proofs,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at)
        }

