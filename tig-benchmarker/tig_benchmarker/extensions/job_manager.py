import asyncio
import os
import json
import logging
from dataclasses import dataclass
from tig_benchmarker.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Job(FromDict):
    benchmark_id: str
    settings: BenchmarkSettings
    num_nonces: int
    rand_hash: str
    wasm_vm_config: Dict[str, int]
    download_url: str
    batch_size: int
    challenge: str
    sampled_nonces: Optional[List[int]] = field(default_factory=list)
    merkle_root: Optional[MerkleHash] = None
    solution_nonces: List[int] = field(default_factory=list)
    merkle_proofs: Dict[int, MerkleProof] = field(default_factory=dict)
    solution_nonces: List[int] = field(default_factory=list)
    batch_merkle_proofs: Dict[int, MerkleProof] = field(default_factory=dict)
    batch_merkle_roots: List[Optional[MerkleHash]] = None
    last_benchmark_submit_time: int = 0
    last_proof_submit_time: int = 0
    last_batch_retry_time: List[int] = None

    def __post_init__(self):
        self.batch_merkle_roots = [None] * self.num_batches
        self.last_batch_retry_time = [0] * self.num_batches

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

@dataclass
class JobManagerConfig(FromDict):
    backup_folder: str
    batch_sizes: Dict[str, int]

class JobManager:
    def __init__(self, config: JobManagerConfig, jobs: List[Job]):
        self.config = config
        self.jobs = jobs
        os.makedirs(self.config.backup_folder, exist_ok=True)
        for file in os.listdir(self.config.backup_folder):
            if not file.endswith(".json"):
                continue
            file_path = f"{self.config.backup_folder}/{file}"
            logger.info(f"restoring job from {file_path}")
            with open(file_path) as f:
                job = Job.from_dict(json.load(f))
                self.jobs.append(job)

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        challenges: Dict[str, Challenge],
        wasms: Dict[str, Wasm],
        **kwargs
    ):
        job_idxs = {
            j.benchmark_id: idx 
            for idx, j in enumerate(self.jobs)
        }
        # create jobs from confirmed precommits
        challenge_id_2_name = {
            c.id: c.details.name
            for c in challenges.values()
        }
        for benchmark_id, x in precommits.items():
            if (
                benchmark_id in job_idxs or 
                benchmark_id in proofs
            ):
                continue
            logger.info(f"creating job from confirmed precommit {benchmark_id}")
            c_name = challenge_id_2_name[x.settings.challenge_id]
            job = Job(
                benchmark_id=benchmark_id,
                settings=x.settings,
                num_nonces=x.details.num_nonces,
                rand_hash=x.state.rand_hash,
                wasm_vm_config=block.config["wasm_vm"],
                batch_size=self.config.batch_sizes[c_name],
                challenge=c_name,
                download_url=next((w.details.download_url for w in wasms.values() if w.algorithm_id == x.settings.algorithm_id), None)
            )
            job_idxs[benchmark_id] = len(self.jobs)
            self.jobs.append(job)

        # update jobs from confirmed benchmarks
        for benchmark_id, x in benchmarks.items():
            if benchmark_id in proofs:
                continue
            logger.info(f"updating job from confirmed benchmark {benchmark_id}")
            job = self.jobs[job_idxs[benchmark_id]]
            job.sampled_nonces = x.state.sampled_nonces
            for batch_idx in job.sampled_nonces_by_batch_idx:
                job.last_batch_retry_time[batch_idx] = 0

        # prune jobs from confirmed proofs
        prune_idxs = [
            job_idxs[benchmark_id]
            for benchmark_id in proofs
            if benchmark_id in job_idxs
        ] + [
            job_idxs[benchmark_id]
            for benchmark_id in job_idxs
            if benchmark_id not in precommits
        ]
        for idx in sorted(set(prune_idxs), reverse=True):
            job = self.jobs[idx]
            logger.info(f"pruning job {job.benchmark_id}")
            if os.path.exists(f"{self.config.backup_folder}/{job.benchmark_id}.json"):
                os.remove(f"{self.config.backup_folder}/{job.benchmark_id}.json")
            self.jobs.pop(idx)

    def run(self):
        now = int(time.time() * 1000)
        for job in self.jobs:
            if job.merkle_root is not None:
                continue
            num_batches_ready = sum(x is not None for x in job.batch_merkle_roots)
            logger.info(f"benchmark {job.benchmark_id}: (batches: {num_batches_ready} of {job.num_batches} ready, #solutions: {len(job.solution_nonces)})")
            if num_batches_ready != job.num_batches:
                continue
            start_time = min(job.last_batch_retry_time)
            logger.info(f"benchmark {job.benchmark_id}: ready, took {(now - start_time) / 1000} seconds")
            tree = MerkleTree(
                job.batch_merkle_roots,
                1 << (job.num_batches - 1).bit_length()
            )
            job.merkle_root = tree.calc_merkle_root()
        
        for job in self.jobs:
            if (
                job.merkle_root is None or
                len(job.sampled_nonces) == 0 or # benchmark not confirmed
                len(job.merkle_proofs) == len(job.sampled_nonces) # already processed
            ):
                continue
            logger.info(f"proof {job.benchmark_id}: (merkle_proof: {len(job.batch_merkle_proofs)} of {len(job.sampled_nonces)} ready)")
            if len(job.batch_merkle_proofs) != len(job.sampled_nonces):  # not finished
                continue
            logger.info(f"proof {job.benchmark_id}: ready")
            depth_offset = (job.batch_size - 1).bit_length()
            tree = MerkleTree(
                job.batch_merkle_roots, 
                1 << (job.num_batches - 1).bit_length()
            )
            proofs = {}
            sampled_nonces_by_batch_idx = job.sampled_nonces_by_batch_idx
            for batch_idx in sampled_nonces_by_batch_idx:
                upper_stems = [
                    (d + depth_offset, h) 
                    for d, h in tree.calc_merkle_branch(batch_idx).stems
                ]
                for nonce in set(sampled_nonces_by_batch_idx[batch_idx]):
                    proof = job.batch_merkle_proofs[nonce]
                    job.merkle_proofs[nonce] = MerkleProof(
                        leaf=proof.leaf,
                        branch=MerkleBranch(proof.branch.stems + upper_stems)
                    )

        for job in self.jobs:
            file_path = f"{self.config.backup_folder}/{job.benchmark_id}.json"
            logger.debug(f"backing up job to {file_path}")
            with open(file_path, "w") as f:
                json.dump(job.to_dict(), f)