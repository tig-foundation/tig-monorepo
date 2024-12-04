import os
import json
import logging
from dataclasses import dataclass
from tig_benchmarker.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set
from tig_benchmarker.sql import db_conn
import math

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Job(FromDict):
    benchmark_id: str
    settings: BenchmarkSettings
    num_nonces: int
    rand_hash: str
    runtime_config: Dict[str, int]
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
        #os.makedirs(self.config.backup_folder, exist_ok=True)
        #for file in os.listdir(self.config.backup_folder):
        #    if not file.endswith(".json"):
        #         continue
        #    file_path = f"{self.config.backup_folder}/{file}"
        #    logger.info(f"restoring job from {file_path}")
        #    with open(file_path) as f:
        #        job = Job.from_dict(json.load(f))
        #        self.jobs.append(job)

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        challenges: Dict[str, Challenge],
        wasms: Dict[str, Binary],
        **kwargs
    ):
        # create jobs from confirmed precommits
        challenge_id_2_name = {
            c.id: c.details.name
            for c in challenges.values()
        }
        for benchmark_id, x in precommits.items():
            exists = db_conn.fetch_one(
                """
                SELECT benchmark_id FROM jobs 
                WHERE benchmark_id = %s
                """,
                (benchmark_id,)
            )
            if exists:
                continue

            if (
                #benchmark_id in job_idxs or
                benchmark_id in proofs
            ):
                continue
                
            logger.info(f"creating job from confirmed precommit {benchmark_id}")
            c_name = challenge_id_2_name[x.settings.challenge_id]
            #job = Job(
            #    benchmark_id=benchmark_id,
            #    settings=x.settings,
            #    num_nonces=x.details.num_nonces,
            #    rand_hash=x.details.rand_hash,
            #    runtime_config=block.config["benchmarks"]["runtime_configs"]["wasm"],
            #    batch_size=self.config.batch_sizes[c_name],
            #    challenge=c_name,
            #    download_url=next((w.details.download_url for w in wasms.values() if w.algorithm_id == x.settings.algorithm_id), None)
            #)
            # job_idxs[benchmark_id] = len(self.jobs)
            # self.jobs.append(job)

            num_batches = math.ceil(x.details.num_nonces / self.config.batch_sizes[c_name])
            db_conn.execute(
                """
                INSERT INTO jobs (benchmark_id, settings, num_nonces, num_batches, rand_hash, runtime_config, batch_size, challenge, download_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (benchmark_id) DO NOTHING;
                """,
                (
                    benchmark_id,
                    json.dumps(asdict(x.settings)),
                    x.details.num_nonces,
                    num_batches,
                    x.details.rand_hash,
                    json.dumps(block.config["benchmarks"]["runtime_configs"]["wasm"]),
                    self.config.batch_sizes[c_name],
                    c_name,
                    next((w.details.download_url for w in wasms.values() if w.algorithm_id == x.settings.algorithm_id), None)
                )
            )

            for batch_idx in range(num_batches):
                db_conn.execute(
                    """
                    INSERT INTO roots (benchmark_id, batch_idx) VALUES (%s, %s);
                    INSERT INTO proofs (benchmark_id, batch_idx) VALUES (%s, %s);
                    """,
                    (benchmark_id, batch_idx, benchmark_id, batch_idx)
                )


        # update jobs from confirmed benchmarks
        for benchmark_id, x in benchmarks.items():
            if benchmark_id in proofs:
                continue

            # Check if benchmark_id already exists in jobs table
            exists = db_conn.fetch_one(
                """
                SELECT benchmark_id FROM jobs 
                WHERE benchmark_id = %s
                """,
                (benchmark_id,)
            )

            if not exists:
                continue
            
            # Check if sampled_nonces in jobs table is > 0
            result = db_conn.fetch_one(
                """
                SELECT sampled_nonces, num_batches, batch_size
                FROM jobs 
                WHERE benchmark_id = %s
                """,
                (benchmark_id,)
            )

            if result and result['sampled_nonces'] and len(result['sampled_nonces']) > 0:
                continue

            logger.info(f"updating job from confirmed benchmark {benchmark_id}")
            db_conn.execute(
                """
                UPDATE jobs 
                SET sampled_nonces = %s
                WHERE benchmark_id = %s
                """,
                (json.dumps(x.details.sampled_nonces), benchmark_id)
            )

            batch_sampled_nonces = {}
            for nonce in sampled_nonces:
                batch_idx = nonce // batch_size
                batch_sampled_nonces.setdefault(batch_idx, []).append(nonce)

            for batch_idx, sampled_nonces in batch_sampled_nonces.items():
                db_conn.execute(
                    """
                    UPDATE proofs
                        SET sampled_nonces = %s
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s
                    """,
                    (json.dumps(sampled_nonces), benchmark_id, batch_idx)
                )
                
    def run(self):
        now = int(time.time() * 1000)
        #for job in self.jobs:
        #    if job.merkle_root is not None:
        #        continue
        #    num_batches_ready = sum(x is not None for x in job.batch_merkle_roots)
        #    logger.info(f"benchmark {job.benchmark_id}: (batches: {num_batches_ready} of {job.num_batches} ready, #solutions: {len(job.solution_nonces)})")
        #    if num_batches_ready != job.num_batches:
        #        continue
        #    start_time = min(job.last_batch_retry_time)
        #    logger.info(f"benchmark {job.benchmark_id}: ready, took {(now - start_time) / 1000} seconds")
        #    tree = MerkleTree(
        #        job.batch_merkle_roots,
        #        1 << (job.num_batches - 1).bit_length()
        #    )
        #    job.merkle_root = tree.calc_merkle_root()

        rows = db_conn.fetch_all("""
            SELECT A.benchmark_id, JSONB_AGG(R.root) AS batch_merkle_roots
            FROM jobs A
            INNER JOIN roots R ON A.benchmark_id = R.benchmark_id
            INNER JOIN proofs P ON A.benchmark_id = P.benchmark_id AND R.batch_idx = P.batch_idx
            GROUP BY A.benchmark_id
            HAVING COUNT(*) = COUNT(P.proofs)
        """)

        # Calculate merkle roots for completed jobs
        for row in rows:
            benchmark_id = row['benchmark_id']
            batch_merkle_roots = [MerkleHash.from_str(root) for root in row['batch_merkle_roots']]
            num_batches = len(batch_merkle_roots)
            
            tree = MerkleTree(
                batch_merkle_roots,
                1 << (num_batches - 1).bit_length()
            )
            merkle_root = tree.calc_merkle_root()

            # Update the database with calculated merkle root
            db_conn.execute("""
                UPDATE jobs 
                SET merkle_root = %s
                WHERE benchmark_id = %s
            """, (merkle_root.to_str(), benchmark_id))
            
        rows = db_conn.fetch_all("""
            SELECT A.benchmark_id, JSONB_AGG(P.proofs) AS batch_merkle_proofs,
                   A.batch_size, A.num_batches, A.sampled_nonces,
                   JSONB_AGG(R.root) AS batch_merkle_roots
            FROM jobs A
            INNER JOIN roots R ON A.benchmark_id = R.benchmark_id
            INNER JOIN proofs P ON A.benchmark_id = P.benchmark_id AND R.batch_idx = P.batch_idx
            GROUP BY A.benchmark_id
            HAVING COUNT(*) = COUNT(P.proofs)
        """)

        for row in rows:
            benchmark_id = row['benchmark_id']
            batch_merkle_proofs = row['batch_merkle_proofs']
            sampled_nonces = row['sampled_nonces']
            batch_merkle_roots = row['batch_merkle_roots']
            
            logger.info(f"proof {benchmark_id}: (merkle_proof: {len(batch_merkle_proofs)} of {len(sampled_nonces)} ready)")
            
            if (
                len(batch_merkle_proofs) != len(sampled_nonces) or
                any(x is None for x in batch_merkle_roots)
            ):
                continue
                
            logger.info(f"proof {benchmark_id}: ready")
            
            depth_offset = (row['batch_size'] - 1).bit_length()
            tree = MerkleTree(
                batch_merkle_roots,
                1 << (row['num_batches'] - 1).bit_length()
            )
            
            merkle_proofs = {}        
            for batch_idx in range(row['num_batches']):
                sampled_nonces_by_batch_idx = db_conn.fetch_one("""
                    SELECT sampled_nonces
                    FROM proofs
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s
                """, (benchmark_id, batch_idx))
                if not sampled_nonces_by_batch_idx:
                    continue

                sampled_nonces = sampled_nonces_by_batch_idx['sampled_nonces']
                upper_stems = [
                    (d + depth_offset, h)
                    for d, h in tree.calc_merkle_branch(batch_idx).stems
                ]
                for nonce in set(sampled_nonces_by_batch_idx[batch_idx]):
                    proof = batch_merkle_proofs[nonce]
                    merkle_proofs[nonce] = MerkleProof(
                        leaf=proof.leaf,
                        branch=MerkleBranch(proof.branch.stems + upper_stems)
                    )
                    
            # Update database with cxalculated merkle proofs
            db_conn.execute("""
                UPDATE job
                SET merkle_proofs = %s
                WHERE benchmark_id = %s
            """, (json.dumps(merkle_proofs), benchmark_id))