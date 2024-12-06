import os
import json
import logging
from tig_benchmarker.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set
from extensions.sql import db_conn
from extensions.client_manager import get_config
import math

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class JobManager:
    def __init__(self):
        return

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

            config = get_config()["job_manager_config"]

            num_batches = math.ceil(x.details.num_nonces / config["batch_sizes"][c_name])
            db_conn.execute(
                """
                INSERT INTO jobs (benchmark_id, settings, num_nonces, num_batches, rand_hash, runtime_config, batch_size, challenge, download_url, creation_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (benchmark_id) DO NOTHING;
                """,
                (
                    benchmark_id,
                    json.dumps(asdict(x.settings)),
                    x.details.num_nonces,
                    num_batches,
                    x.details.rand_hash,
                    json.dumps(block.config["benchmarks"]["runtime_configs"]["wasm"]),
                    config["batch_sizes"][c_name],
                    c_name,
                    next((w.details.download_url for w in wasms.values() if w.algorithm_id == x.settings.algorithm_id), None),
                    int(time.time())
                )
            )

            for batch_idx in range(num_batches):
                db_conn.execute(
                    """
                    INSERT INTO roots (benchmark_id, batch_idx) VALUES (%s, %s)
                    """,
                    (benchmark_id, batch_idx)
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
                WHERE benchmark_id  = %s
                """,
                (benchmark_id,)
            )

            if result is None or result["sampled_nonces"] is not None:
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
            for nonce in x.details.sampled_nonces:
                batch_idx = nonce // result["batch_size"]
                batch_sampled_nonces.setdefault(batch_idx, []).append(nonce)

            for batch_idx, sampled_nonces in batch_sampled_nonces.items():
                db_conn.execute(
                    """
                    INSERT INTO proofs (sampled_nonces, benchmark_id, batch_idx)
                    VALUES (%s, %s, %s)
                    """,
                    (json.dumps(sampled_nonces), benchmark_id, batch_idx)
                )
                
    def run(self):
        now = int(time.time() * 1000)

        rows = db_conn.fetch_all("""
            SELECT A.benchmark_id, JSONB_AGG(R.root ORDER BY R.batch_idx) AS batch_merkle_roots,
                JSONB_AGG(r.solution_nonces) AS solution_nonces
            FROM jobs A
            INNER JOIN roots R ON A.benchmark_id = R.benchmark_id
            GROUP BY A.benchmark_id
            HAVING COUNT(*) = COUNT(R.root)
        """)

        # Calculate merkle roots for completed jobs
        for row in rows:
            solution_nonces = [x for y in row['solution_nonces'] for x in y]

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
                SET merkle_root = %s,
                    solution_nonces = %s
                WHERE benchmark_id = %s
            """, (merkle_root.to_str(), json.dumps(solution_nonces), benchmark_id))
            
        rows = db_conn.fetch_all("""
            SELECT A.benchmark_id, JSONB_AGG(P.proofs ORDER BY P.batch_idx) AS batch_merkle_proofs,
                A.batch_size, A.num_batches
            FROM jobs A
            INNER JOIN proofs P ON A.benchmark_id = P.benchmark_id
            GROUP BY A.benchmark_id, A.batch_size, A.num_batches
            HAVING COUNT(*) = COUNT(P.proofs)
        """)

        for row in rows:
            benchmark_id = row["benchmark_id"]
            batch_merkle_proofs = [MerkleProof.from_dict(x) for y in row["batch_merkle_proofs"] for x in y]

            batch_merkle_roots = db_conn.fetch_one("""
                SELECT JSONB_AGG(root ORDER BY batch_idx) as batch_merkle_roots
                FROM roots
                WHERE benchmark_id = %s
            """, (benchmark_id,))

            batch_merkle_roots = batch_merkle_roots["batch_merkle_roots"]
            
            #logger.info(f"proof {benchmark_id}: (merkle_proof: {len(batch_merkle_proofs)} ready)")
            
            depth_offset = (row["batch_size"] - 1).bit_length()
            tree = MerkleTree(
                [MerkleHash.from_str(root) for root in batch_merkle_roots],
                1 << (row["num_batches"] - 1).bit_length()
            )
            
            merkle_proofs = []       
            for proof in batch_merkle_proofs:
                batch_idx = proof.leaf.nonce // row["batch_size"]
                upper_stems = [
                    (d + depth_offset, h)
                    for d, h in tree.calc_merkle_branch(batch_idx).stems
                ]
                
                merkle_proofs.append(
                    MerkleProof(
                        leaf=proof.leaf,
                        branch=MerkleBranch(proof.branch.stems + upper_stems)
                    )
                )
                    
            # Update database with cxalculated merkle proofs
            db_conn.execute("""
                UPDATE jobs
                SET merkle_proofs = %s
                WHERE benchmark_id = %s
            """, (json.dumps([x.to_dict() for x in merkle_proofs]), benchmark_id))
