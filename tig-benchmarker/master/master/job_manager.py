import os
import json
import logging
from common.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from common.structs import *
from common.utils import *
from typing import Dict, List, Optional, Set
from master.sql import db_conn
from master.client_manager import CONFIG
import math

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class JobManager:
    def __init__(self):
        pass

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        challenges: Dict[str, Challenge],
        algorithms: Dict[str, Algorithm],
        wasms: Dict[str, Binary],
        **kwargs
    ):
        config = CONFIG["job_manager_config"]
        # create jobs from confirmed precommits
        challenge_id_2_name = {
            c.id: c.details.name
            for c in challenges.values()
        }
        algorithm_id_2_name = {
            a.id: a.details.name
            for a in algorithms.values()
        }
        for benchmark_id, x in precommits.items():
            if (
                benchmark_id in proofs or
                db_conn.fetch_one( # check if job is already created
                    """
                    SELECT 1 
                    FROM job
                    WHERE benchmark_id = %s
                    """,
                    (benchmark_id,)
                )
            ):
                continue
                
            logger.info(f"creating job from confirmed precommit {benchmark_id}")
            c_name = challenge_id_2_name[x.settings.challenge_id]
            a_name = algorithm_id_2_name[x.settings.algorithm_id]

            wasm = wasms.get(x.settings.algorithm_id, None)
            if wasm is None:
                logger.error(f"no wasm found for algorithm_id {x.settings.algorithm_id}")
                continue
            if wasm.details.download_url is None:
                logger.error(f"no download_url found for wasm {wasm.algorithm_id}")
                continue
            num_batches = math.ceil(x.details.num_nonces / config["batch_sizes"][c_name])
            atomic_inserts = [
                (
                    """
                    INSERT INTO job 
                    (
                        benchmark_id, 
                        settings, 
                        num_nonces, 
                        num_batches, 
                        rand_hash, 
                        runtime_config, 
                        batch_size, 
                        challenge,
                        algorithm,
                        download_url,
                        block_started,
                        start_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT)
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
                        a_name,
                        wasm.details.download_url,
                        x.details.block_started
                    )
                ),
                (
                    """
                    INSERT INTO job_data (benchmark_id) VALUES (%s)
                    ON CONFLICT (benchmark_id) DO NOTHING;
                    """,
                    (benchmark_id,)
                )
            ]

            for batch_idx in range(num_batches):
                atomic_inserts += [
                    (
                        """
                        INSERT INTO root_batch (benchmark_id, batch_idx) VALUES (%s, %s)
                        """,
                        (benchmark_id, batch_idx)
                    ),
                    (
                        """
                        INSERT INTO batch_data (benchmark_id, batch_idx) VALUES (%s, %s)
                        """,
                        (benchmark_id, batch_idx)
                    )
                ]
            
            db_conn.execute_many(*atomic_inserts)


        # update jobs from confirmed benchmarks
        for benchmark_id, x in benchmarks.items():
            if (
                benchmark_id in proofs or
                (result := db_conn.fetch_one(
                    """
                    SELECT num_batches, batch_size
                    FROM job
                    WHERE benchmark_id = %s
                        AND sampled_nonces IS NULL
                    """,
                    (benchmark_id,)
                )) is None
            ):
                continue

            logger.info(f"updating job from confirmed benchmark {benchmark_id}")
            atomic_update = [
                (
                    """
                    UPDATE job
                    SET sampled_nonces = %s
                    WHERE benchmark_id = %s
                    """,
                    (json.dumps(x.details.sampled_nonces), benchmark_id)
                )
            ]

            batch_sampled_nonces = {}
            for nonce in x.details.sampled_nonces:
                batch_idx = nonce // result["batch_size"]
                batch_sampled_nonces.setdefault(batch_idx, []).append(nonce)

            for batch_idx, sampled_nonces in batch_sampled_nonces.items():
                atomic_update += [
                    (
                        """
                        INSERT INTO proofs_batch (sampled_nonces, benchmark_id, batch_idx, slave, start_time)
                        SELECT %s, A.benchmark_id, A.batch_idx, A.slave, (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                        FROM root_batch A
                        WHERE A.benchmark_id = %s AND A.batch_idx = %s
                        """,
                        (json.dumps(sampled_nonces), benchmark_id, batch_idx)
                    )
                ]
            
            db_conn.execute_many(*atomic_update)

        # update jobs from confirmed proofs
        if len(proofs) > 0:
            db_conn.execute(
                """
                UPDATE job
                SET end_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                WHERE end_time IS NULL
                    AND benchmark_id IN %s
                """,
                (tuple(proofs),)
            )

        # stop any expired jobs
        db_conn.execute(
            """
            UPDATE job
            SET stopped = true
            WHERE end_time IS NULL
                AND stopped IS NULL
                AND %s >= block_started + 120
            """,
            (block.details.height,)
        )
        
                
    def run(self):
        now = int(time.time() * 1000)

        # Find jobs where all root_batchs are ready
        rows = db_conn.fetch_all(
            """
            WITH ready AS (
                SELECT A.benchmark_id
                FROM root_batch A
                INNER JOIN job B
                    ON B.merkle_root_ready IS NULL
                    AND A.benchmark_id = B.benchmark_id
                GROUP BY A.benchmark_id
                HAVING COUNT(*) = COUNT(A.ready)
            )
            SELECT 
                B.benchmark_id, 
                JSONB_AGG(B.merkle_root ORDER BY B.batch_idx) AS batch_merkle_roots,
                JSONB_AGG(B.solution_nonces) AS solution_nonces
            FROM ready A
            INNER JOIN batch_data B
                ON A.benchmark_id = B.benchmark_id
            GROUP BY B.benchmark_id
            """
        )

        # Calculate merkle roots for completed jobs
        for row in rows:
            solution_nonces = [x for y in row['solution_nonces'] for x in y]

            benchmark_id = row['benchmark_id']
            batch_merkle_roots = [MerkleHash.from_str(root) for root in row['batch_merkle_roots']]
            num_batches = len(batch_merkle_roots)
            
            logger.info(f"job {benchmark_id}: (benchmark ready)")

            tree = MerkleTree(
                batch_merkle_roots,
                1 << (num_batches - 1).bit_length()
            )
            merkle_root = tree.calc_merkle_root()

            # Update the database with calculated merkle root
            db_conn.execute_many(*[
                (
                    """
                    UPDATE job_data
                    SET merkle_root = %s, 
                        solution_nonces = %s
                    WHERE benchmark_id = %s
                    """, 
                    (
                        merkle_root.to_str(), 
                        json.dumps(solution_nonces),
                        benchmark_id
                    )
                ),
                (
                    """
                    UPDATE job
                    SET merkle_root_ready = true
                    WHERE benchmark_id = %s
                    """,
                    (benchmark_id,)
                )
            ])
            
        # Find jobs where all proofs_batchs are ready
        rows = db_conn.fetch_all(
            """
            WITH ready AS (
                SELECT A.benchmark_id
                FROM proofs_batch A
                INNER JOIN job B
                    ON B.merkle_proofs_ready IS NULL
                    AND A.benchmark_id = B.benchmark_id
                GROUP BY A.benchmark_id
                HAVING COUNT(*) = COUNT(A.ready)
            )
            SELECT 
                A.benchmark_id, 
                JSONB_AGG(D.merkle_proofs ORDER BY D.batch_idx) AS batch_merkle_proofs,
                B.batch_size, 
                B.num_batches
            FROM ready A
            INNER JOIN job B 
                ON A.benchmark_id = B.benchmark_id
            INNER JOIN proofs_batch C
                ON A.benchmark_id = C.benchmark_id
            INNER JOIN batch_data D
                ON C.benchmark_id = D.benchmark_id 
                AND C.batch_idx = D.batch_idx
            GROUP BY 
                A.benchmark_id, 
                B.batch_size, 
                B.num_batches
            """
        )

        for row in rows:
            benchmark_id = row["benchmark_id"]
            batch_merkle_proofs = [
                MerkleProof.from_dict(x) 
                for y in row["batch_merkle_proofs"] 
                for x in y
            ]

            batch_merkle_roots = db_conn.fetch_one(
                """
                SELECT JSONB_AGG(merkle_root ORDER BY batch_idx) as batch_merkle_roots
                FROM batch_data
                WHERE benchmark_id = %s
                """, 
                (benchmark_id,)
            )

            batch_merkle_roots = batch_merkle_roots["batch_merkle_roots"]
            
            logger.info(f"job {benchmark_id}: (proof ready)")
            
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
                    
            # Update database with calculated merkle proofs
            db_conn.execute_many(*[
                (
                    """
                    UPDATE job_data
                    SET merkle_proofs = %s
                    WHERE benchmark_id = %s
                    """, 
                    (
                        json.dumps([x.to_dict() for x in merkle_proofs]), 
                        benchmark_id
                    )
                ),
                (
                    """
                    UPDATE job
                    SET merkle_proofs_ready = true
                    WHERE benchmark_id = %s
                    """,
                    (benchmark_id,)
                )
            ])
