import os
import json
import time
from dataclasses import dataclass, field
from tig_benchmarker.config import Config
from tig_benchmarker.data_fetcher import QueryData
from tig_benchmarker.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from tig_benchmarker.structs import BenchmarkSettings, MerkleProof, Precommit
from tig_benchmarker.utils import FromDict, u64s_from_str, u8s_from_str, jsonify, PreciseNumber
from typing import Dict, List, Optional, Set, Any, Tuple

@dataclass
class JobDetails(FromDict):
    benchmark_id: str
    settings: BenchmarkSettings
    rand_hash: str
    challenge: str
    algorithm: str
    download_url: str
    batch_size: int
    wasm_vm_config: dict

@dataclass
class Batch(FromDict):
    details: JobDetails
    start_nonce: int
    num_nonces: int
    sampled_nonces: Optional[List[int]] = None

    @property
    def id(self) -> str:
        return f"{self.details.benchmark_id}_{self.start_nonce}"

@dataclass
class BatchResult(FromDict):
    solution_nonces: Set[int]
    merkle_root: MerkleHash
    merkle_proofs: Optional[List[MerkleProof]] = None

@dataclass
class Job(FromDict):
    block_started: int
    details: JobDetails
    num_nonces: int
    results: List[Optional[BatchResult]]
    last_retry: List[int]
    sampled_nonces: Optional[List[int]] = None
    benchmark_ready: bool = False
    proof_ready: bool = False

    def solution_nonces(self) -> Set[int]:
        assert self.benchmark_ready, "Some batches have no results"
        return set().union(
            *(r.solution_nonces for r in self.results)
        )

    def merkle_tree(self) -> MerkleTree:
        assert self.benchmark_ready, "Some batches have no results"
        num_batches = (self.num_nonces + self.details.batch_size - 1) // self.details.batch_size # ceil division
        return MerkleTree(
            [r.merkle_root for r in self.results], 
            1 << (num_batches - 1).bit_length()
        )

    def merkle_root(self) -> MerkleHash:
        tree = self.merkle_tree()
        return tree.calc_merkle_root()

    def merkle_proofs(self) -> List[MerkleProof]:
        assert self.proof_ready, "Some sampled nonces have no proofs"
        depth_offset = (self.details.batch_size - 1).bit_length()
        tree = self.merkle_tree()
        proofs = []
        for batch_idx in set(n // self.details.batch_size for n in self.sampled_nonces):
            upper_stems = [
                (d + depth_offset, h) 
                for d, h in tree.calc_merkle_branch(batch_idx).stems
            ]
            for proof in self.results[batch_idx].merkle_proofs:
                proofs.append(MerkleProof(
                    leaf=proof.leaf,
                    branch=MerkleBranch(proof.branch.stems + upper_stems)
                ))
        return proofs

    def generate_batches(self, duration_before_batch_retry: int) -> Tuple[Dict[str, Batch], bool]:
        sampled_idxs = {}
        for n in (self.sampled_nonces or []):
            batch_idx = n // self.details.batch_size
            sampled_idxs.setdefault(batch_idx, []).append(n)
        now = int(time.time() * 1000)
        batches = {}
        for i, (result, last_retry) in enumerate(zip(self.results, self.last_retry)):
            if now - last_retry <= duration_before_batch_retry:
                continue
            if (
                result is None or 
                (i in sampled_idxs and not result.merkle_proofs)
            ):
                self.last_retry[i] = now
                b = Batch(
                    details=self.details,
                    start_nonce=i * self.details.batch_size,
                    num_nonces=min(
                        self.details.batch_size,
                        self.num_nonces - i * self.details.batch_size
                    ),
                    sampled_nonces=sampled_idxs.get(i, None)
                )
                batches[b.id] = b
        return (batches, len(sampled_idxs) > 0)

    def update_result(self, start_nonce: int, result: BatchResult):
        batch_idx = start_nonce // self.details.batch_size
        self.results[batch_idx] = result
        self.benchmark_ready = all(
            r is not None
            for r in self.results
        )
        self.proof_ready = (
            self.benchmark_ready and
            self.sampled_nonces is not None and 
            all(
                # FIXME: check if proof is valid; check if every nonce has a proof
                len(self.results[n // self.details.batch_size].merkle_proofs) > 0
                for n in self.sampled_nonces
            )
        )

    def update_sampled_nonces(self, sampled_nonces: List[int]):
        if self.sampled_nonces is not None:
            return False
        self.sampled_nonces = sampled_nonces[:]
        for n in sampled_nonces:
            self.last_retry[n // self.details.batch_size] = 0
        return True

    def save(self, folder: str):
        with open(f"{folder}/{self.details.benchmark_id}.json", 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, folder: str, benchmark_id: str) -> 'Job':
        with open(f"{folder}/{benchmark_id}.json", 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def create_from_precommit(
        cls, 
        precommit: Precommit,
        challenge: str,
        algorithm: str,
        download_url: str,
        wasm_vm_config: dict,
        batch_size: int
    ) -> 'BatchStore':
        num_nonces = precommit.details.num_nonces
        num_batches = (num_nonces + batch_size - 1) // batch_size # ceil division
        return cls(
            block_started=precommit.details.block_started,
            details=JobDetails(
                benchmark_id=precommit.benchmark_id,
                settings=precommit.settings,
                challenge=challenge,
                algorithm=algorithm,
                download_url=download_url,
                rand_hash=precommit.state.rand_hash,
                batch_size=batch_size,
                wasm_vm_config=wasm_vm_config
            ),
            num_nonces=num_nonces,
            results=[None] * num_batches,
            last_retry=[0] * num_batches
        )

class JobManager:
    def __init__(self, jobs_folder: str):
        self.jobs_folder = jobs_folder
        self.jobs = {}
        for path in os.listdir(jobs_folder):
            if not path.endswith('.json'):
                continue
            with open(os.path.join(jobs_folder, path)) as f:
                job = Job.from_dict(json.load(f))
                self.jobs[job.details.benchmark_id] = job

    def update_with_query_data(self, query_data: QueryData, config: Config):
        # prune old jobs
        prune_height = query_data.block.details.height - query_data.block.config["benchmark_submissions"]["lifespan_period"]
        for benchmark_id in list(self.jobs):
            job = self.jobs[benchmark_id]
            if (
                job.block_started < prune_height or # prune if too old
                job.details.benchmark_id in query_data.proofs # prune if proof confirmed
            ):
                del self.jobs[benchmark_id]
                os.remove(f"{self.jobs_folder}/{benchmark_id}.json")

        # create jobs for precommits without proofs
        for benchmark_id, precommit in query_data.precommits.items():
            if (
                precommit.details.num_nonces is None or # pre-update benchmark
                benchmark_id in self.jobs or # already created
                benchmark_id in query_data.proofs # proof already confirmed
            ):
                continue
            challenge = query_data.challenges[precommit.settings.challenge_id].details.name
            algorithm = query_data.algorithms[precommit.settings.algorithm_id].details.name
            download_url = query_data.wasms[precommit.settings.algorithm_id].details.download_url
            job = Job.create_from_precommit(
                precommit=precommit, 
                challenge=challenge,
                algorithm=algorithm,
                download_url=download_url,
                wasm_vm_config=query_data.block.config["wasm_vm"],
                batch_size=getattr(config, challenge).batch_size
            )
            print(f"[job_manager] created job {job}")
            self.jobs[benchmark_id] = job

        # update jobs with sampled_nonces
        for benchmark_id, job in self.jobs.items():
            if benchmark_id not in query_data.benchmarks:
                continue
            if job.update_sampled_nonces(query_data.benchmarks[benchmark_id].state.sampled_nonces):
                print(f"[job_manager] updated benchmark '{job.details.benchmark_id}' with sampled nonces {job.sampled_nonces}")

        # save jobs
        for job in self.jobs.values():
            job.save(self.jobs_folder)

    def generate_batches(self, config: Config) -> Tuple[Dict[str, Job], Dict[str, Job]]:
        priority_batches = {}
        batches = {}
        for job in self.jobs.values():
            challenge = job.details.challenge
            d, priority = job.generate_batches(
                duration_before_batch_retry=getattr(config, challenge).duration_before_batch_retry
            )
            if not d:
                continue
            if priority:
                print(f"[job_manager] generated {len(d)} priority batches for benchmark '{job.details.benchmark_id}'")
                priority_batches.update(d)
            else:
                print(f"[job_manager] generated {len(d)} batches for benchmark '{job.details.benchmark_id}'")
                batches.update(d)
        return (batches, priority_batches)