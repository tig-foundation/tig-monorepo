import asyncio
import os
import json
import logging
from dataclasses import dataclass
from tig_benchmarker.event_bus import *
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
    batch_size: int
    num_batches: int
    rand_hash: str
    sampled_nonces: List[int]
    sampled_nonces_by_batch_idx: Dict[int, List[int]]
    wasm_vm_config: Dict[str, int]
    download_url: str
    solution_nonces: List[int]
    batch_merkle_roots: List[Optional[MerkleHash]]
    merkle_proofs: Dict[int, MerkleProof]
    last_retry_time: List[int]
    start_time: int

@dataclass
class JobManagerConfig(FromDict):
    batch_size: int
    ms_delay_between_batch_retries: Optional[int] = None

class Extension:
    def __init__(self, backup_folder: str, job_manager: dict, **kwargs):
        self.backup_folder = backup_folder
        self.config = {k: JobManagerConfig.from_dict(v) for k, v in job_manager.items()}
        for challenge_name, config in self.config.items():
            batch_size = config.batch_size
            assert (batch_size & (batch_size - 1) == 0) and batch_size != 0, f"batch_size {batch_size} for challenge {challenge_name} is not a power of 2"
        self.jobs = {}
        self.wasm_vm_config = {}
        self.challenge_id_2_name = {}
        self.download_urls = {}
        self.benchmark_ready = set()
        self.proof_ready = set()
        self.lock = True
        self._restore_jobs()

    async def on_new_block(
        self, 
        block: Block, 
        precommits: Dict[str, Precommit],
        wasms: Dict[str, Wasm],
        challenges: Dict[str, Challenge],
        **kwargs
    ):
        self.wasm_vm_config = block.config["wasm_vm"]
        self.download_urls = {
            w.algorithm_id: w.details.download_url
            for w in wasms.values()
        }
        prune_jobs = [
            job
            for benchmark_id, job in self.jobs.items()
            if benchmark_id not in precommits
        ]
        for job in prune_jobs:
            self._prune_job(job)
        for c in challenges.values():
            self.challenge_id_2_name[c.id] = c.details.name
        self.lock = False

    async def on_precommit_confirmed(self, precommit: Precommit, **kwargs):
        while self.lock:
            await asyncio.sleep(0.1)
        benchmark_id = precommit.benchmark_id
        if benchmark_id not in self.jobs:
            c_name = self.challenge_id_2_name[precommit.settings.challenge_id]
            batch_size = self.config[c_name].batch_size
            num_batches = (precommit.details.num_nonces + batch_size - 1) // batch_size
            job = Job(
                benchmark_id=benchmark_id,
                settings=precommit.settings,
                num_nonces=precommit.details.num_nonces,
                num_batches=num_batches,
                rand_hash=precommit.state.rand_hash,
                sampled_nonces_by_batch_idx={},
                sampled_nonces=[],
                wasm_vm_config=self.wasm_vm_config,
                download_url=self.download_urls[precommit.settings.algorithm_id],
                batch_size=batch_size,
                solution_nonces=[],
                batch_merkle_roots=[None] * num_batches,
                merkle_proofs={},
                last_retry_time=[0] * num_batches,
                start_time=now()
            )
            self.jobs[benchmark_id] = job
            self._save_job(job)

    async def on_benchmark_confirmed(self, precommit: Precommit, benchmark: Optional[Benchmark], **kwargs):
        await self.on_precommit_confirmed(precommit, **kwargs)
        if benchmark is not None:
            job = self.jobs[precommit.benchmark_id]
            for nonce in benchmark.state.sampled_nonces:
                batch_idx = nonce // job.batch_size
                job.sampled_nonces_by_batch_idx.setdefault(batch_idx, []).append(nonce)
                job.last_retry_time[batch_idx] = 0
            job.sampled_nonces = benchmark.state.sampled_nonces

    async def on_proof_confirmed(self, proof: Proof, **kwargs):
        if (job := self.jobs.get(proof.benchmark_id, None)) is not None:
            self._prune_job(job)

    async def on_update(self):
        if self.lock:
            return
        for benchmark_id, job in self.jobs.items():
            if (
                benchmark_id not in self.proof_ready and 
                len(job.sampled_nonces) > 0 and 
                set(job.sampled_nonces) == set(job.merkle_proofs) and 
                all(x is not None for x in job.batch_merkle_roots)
            ):
                self.proof_ready.add(benchmark_id)
                await self._emit_proof(job)
            elif (
                benchmark_id not in self.benchmark_ready and 
                all(x is not None for x in job.batch_merkle_roots)
            ):
                self.benchmark_ready.add(benchmark_id)
                await self._emit_benchmark(job)
            else:
                await self._emit_batches(job)

    async def on_batch_result(
        self, 
        benchmark_id: str, 
        start_nonce: int, 
        solution_nonces: List[int], 
        merkle_root: MerkleHash, 
        merkle_proofs: List[MerkleProof],
        **kwargs
    ):
        merkle_root = MerkleHash.from_str(merkle_root)
        merkle_proofs = [MerkleProof.from_dict(x) for x in merkle_proofs]

        if benchmark_id not in self.jobs:
            logger.warning(f"job not found for benchmark {benchmark_id}")
        else:
            job = self.jobs[benchmark_id]
            batch_idx = start_nonce // job.batch_size
            # validate batch result (does not check the output data)
            logger.debug(f"validating batch results for {benchmark_id} @ index {batch_idx}")
            assert start_nonce % job.batch_size == 0, "start_nonce not aligned with batch size"
            assert all(start_nonce <= n < start_nonce + job.num_nonces for n in solution_nonces), "solution nonces not in batch"
            left = set(job.sampled_nonces_by_batch_idx.get(batch_idx, []))
            right = set(x.leaf.nonce for x in merkle_proofs)
            if len(left) > 0 and len(right) == 0:
                logger.warning(f"no merkle proofs for batch {batch_idx} of {benchmark_id}")
                return
            assert left == right, f"sampled nonces {left} do not match proofs {right}"
            assert all(x.branch.calc_merkle_root(
                hashed_leaf=x.leaf.to_merkle_hash(),
                branch_idx=x.leaf.nonce - start_nonce, # branch idx of batch tree
            ) == merkle_root for x in merkle_proofs), "merkle proofs do not match merkle root"

            num_nonces = min(job.batch_size, job.num_nonces - start_nonce)
            job.solution_nonces.extend(solution_nonces)
            job.batch_merkle_roots[batch_idx] = merkle_root
            job.merkle_proofs.update({x.leaf.nonce: x for x in merkle_proofs})

            self._save_job(job)

    def _restore_jobs(self):
        path = os.path.join(self.backup_folder, "jobs")
        if not os.path.exists(path):
            logger.info(f"creating backup folder {path}")
            os.makedirs(path, exist_ok=True)
        for file in os.listdir(path):
            if not file.endswith('.json'):
                continue
            file_path = os.path.join(path, file)
            logger.info(f"restoring job from {file_path}")
            with open(file_path) as f:
                job = Job.from_dict(json.load(f))
                self.jobs[job.benchmark_id] = job

    def _save_job(self, job: Job):
        path = os.path.join(self.backup_folder, "jobs", f"{job.benchmark_id}.json")
        with open(path, 'w') as f:
            json.dump(job.to_dict(), f)

    def _prune_job(self, job: Job):
        path = os.path.join(self.backup_folder, "jobs", f"{job.benchmark_id}.json")
        logger.debug(f"pruning job {path}")
        self.jobs.pop(job.benchmark_id, None)
        if os.path.exists(path):
            os.remove(path)
        if job.benchmark_id in self.benchmark_ready:
            self.benchmark_ready.remove(job.benchmark_id)
        if job.benchmark_id in self.proof_ready:
            self.proof_ready.remove(job.benchmark_id)

    async def _emit_proof(self, job: Job):
        # join merkle_proof for the job tree (all batches) with merkle_proof of the batch tree
        depth_offset = (job.batch_size - 1).bit_length()
        tree = MerkleTree(
            job.batch_merkle_roots, 
            1 << (job.num_batches - 1).bit_length()
        )
        proofs = {}
        for batch_idx in job.sampled_nonces_by_batch_idx:
            upper_stems = [
                (d + depth_offset, h) 
                for d, h in tree.calc_merkle_branch(batch_idx).stems
            ]
            for nonce in set(job.sampled_nonces_by_batch_idx[batch_idx]):
                proof = job.merkle_proofs[nonce]
                proofs[nonce] = MerkleProof(
                    leaf=proof.leaf,
                    branch=MerkleBranch(proof.branch.stems + upper_stems)
                )
        c_name = self.challenge_id_2_name[job.settings.challenge_id]
        logger.info(f"proof {job.benchmark_id} ready: (challenge: {c_name}, elapsed: {now() - job.start_time}ms)")
        await emit(
            "proof_ready",
            benchmark_id=job.benchmark_id,
            merkle_proofs=list(proofs.values())
        )

    async def _emit_benchmark(self, job: Job):
        tree = MerkleTree(
            job.batch_merkle_roots,
            1 << (job.num_batches - 1).bit_length()
        )
        root = tree.calc_merkle_root()
        c_name = self.challenge_id_2_name[job.settings.challenge_id]
        logger.info(f"benchmark {job.benchmark_id} ready: (challenge: {c_name}, num_solutions: {len(job.solution_nonces)}, elapsed: {now() - job.start_time}ms)")
        await emit(
            "benchmark_ready",
            benchmark_id=job.benchmark_id,
            merkle_root=root,
            solution_nonces=list(set(job.solution_nonces))
        )

    async def _emit_batches(self, job: Job):
        c_name = self.challenge_id_2_name[job.settings.challenge_id]
        ms_delay_between_batch_retries = self.config[c_name].ms_delay_between_batch_retries or 30000
        now_ = now()
        retry_batch_idxs = [
            batch_idx
            for batch_idx in range(job.num_batches)
            if (
                now_ - job.last_retry_time[batch_idx] >= ms_delay_between_batch_retries and
                (
                    job.batch_merkle_roots[batch_idx] is None or (
                        len(job.sampled_nonces) > 0 and
                        not set(job.sampled_nonces_by_batch_idx.get(batch_idx, [])).issubset(set(job.merkle_proofs))
                    )
                )
            )
        ]
        num_finished = sum(x is not None for x in job.batch_merkle_roots)
        if num_finished != len(job.batch_merkle_roots):
            c_name = self.challenge_id_2_name[job.settings.challenge_id]
            logger.info(f"precommit {job.benchmark_id}: (challenge: {c_name}, progress: {num_finished} of {len(job.batch_merkle_roots)} batches, elapsed: {now_ - job.start_time}ms)")
        if len(retry_batch_idxs) == 0:
            return
        for batch_idx in retry_batch_idxs:
            job.last_retry_time[batch_idx] = now_
            await emit(
                "new_batch",
                benchmark_id=job.benchmark_id,
                settings=job.settings.to_dict(),
                num_nonces=min(
                    job.batch_size,
                    job.num_nonces - batch_idx * job.batch_size
                ),
                start_nonce=batch_idx * job.batch_size,
                batch_size=job.batch_size,
                rand_hash=job.rand_hash,
                sampled_nonces=job.sampled_nonces_by_batch_idx.get(batch_idx, []),
                wasm_vm_config=job.wasm_vm_config,
                download_url=job.download_url,
            )