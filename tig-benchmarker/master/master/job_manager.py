import os
import json
import logging
import requests
from common.merkle_tree import MerkleHash, MerkleBranch, MerkleTree
from common.structs import *
from common.utils import *
from typing import Dict, List, Optional, Set
from master.sql import get_db_conn
from master.client_manager import CONFIG
import math

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class JobManager:
    def __init__(self):
        self.hash_thresholds = {}
        self.average_solution_ratio = {}

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        challenges: Dict[str, Challenge],
        algorithms: Dict[str, Algorithm],
        binarys: Dict[str, Binary],
        **kwargs
    ):
        api_url = CONFIG["api_url"]
        algo_selection = CONFIG["algo_selection"]
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
                get_db_conn().fetch_one( # check if job is already created
                    """
                    SELECT 1 
                    FROM job
                    WHERE benchmark_id = %s
                    """,
                    (benchmark_id,)
                )
            ):
                continue
                
            if block.details.height - x.details.block_started >= 60:
                logger.info(f"skipping precommit {benchmark_id} as it is over 60 blocks old")
                continue

            logger.info(f"creating job from confirmed precommit {benchmark_id}")
            c_name = challenge_id_2_name[x.settings.challenge_id]
            a_name = algorithm_id_2_name[x.settings.algorithm_id]

            if x.details.block_started not in self.hash_thresholds:
                logger.info(f"fetching hash threshold for block {x.details.block_started}")
                d = requests.get(f"{api_url}/get-challenges?block_id={x.settings.block_id}").json()
                self.hash_thresholds[x.details.block_started] = {
                    c['id']: c['block_data']['hash_threshold']
                    for c in d["challenges"]
                }
                self.average_solution_ratio[x.details.block_started] = {
                    c['id']: c['block_data']['average_solution_ratio']
                    for c in d["challenges"]
                }
            hash_threshold = self.hash_thresholds[x.details.block_started][x.settings.challenge_id]
            average_solution_ratio = self.average_solution_ratio[x.details.block_started][x.settings.challenge_id]

            bin = binarys.get(x.settings.algorithm_id, None)
            if bin is None:
                logger.error(f"batch {x.id}: no binary-blob found for {x.settings.algorithm_id}. skipping job")
                continue
            if bin.details.download_url is None:
                logger.error(f"batch {x.id}: no download_url found for {bin.algorithm_id}. skipping job")
                continue
            batch_size = next(
                (s["batch_size"] for s in algo_selection if s["algorithm_id"] == x.settings.algorithm_id),
                None
            )
            if batch_size is None:
                logger.error(f"batch {x.id}: no batch size found for {x.settings.algorithm_id}. skipping job")
                continue
            num_batches = math.ceil(x.details.num_nonces / batch_size)
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
                        hash_threshold,
                        average_solution_ratio,
                        start_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT)
                    ON CONFLICT (benchmark_id) DO NOTHING;
                    """,
                    (
                        benchmark_id,
                        json.dumps(asdict(x.settings)),
                        x.details.num_nonces,
                        num_batches,
                        x.details.rand_hash,
                        json.dumps(block.config["benchmarks"]["runtime_config"]),
                        batch_size,
                        c_name,
                        a_name,
                        bin.details.download_url,
                        x.details.block_started,
                        hash_threshold.lower(),
                        average_solution_ratio,
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
            
            get_db_conn().execute_many(*atomic_inserts)


        # update jobs from confirmed benchmarks
        for benchmark_id, x in benchmarks.items():
            if (
                benchmark_id in proofs or
                (result := get_db_conn().fetch_one(
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
            
            get_db_conn().execute_many(*atomic_update)

        # update jobs from confirmed proofs
        if len(proofs) > 0:
            get_db_conn().execute(
                """
                UPDATE job
                SET end_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                WHERE end_time IS NULL
                    AND benchmark_id IN %s
                """,
                (tuple(proofs),)
            )

        # stop any expired jobs
        get_db_conn().execute(
            """
            UPDATE job
            SET stopped = true
            WHERE end_time IS NULL
                AND stopped IS NULL
                AND %s >= block_started + 120
            """,
            (block.details.height,)
        )

        # prune old hash thresholds
        for height in list(self.hash_thresholds.keys()):
            if block.details.height - height >= 120:
                del self.hash_thresholds[height]
        
                
    def run(self):
        now = int(time.time() * 1000)

        # Find jobs where all root_batchs are ready
        rows = get_db_conn().fetch_all(
            """
            WITH ready AS (
                SELECT A.benchmark_id
                FROM root_batch A
                INNER JOIN job B
                    ON B.merkle_root_ready IS NULL
                    AND A.benchmark_id = B.benchmark_id
                GROUP BY A.benchmark_id
                HAVING COUNT(*) = COUNT(A.ready)
            ),
            agg_batches AS (
                SELECT 
                    A.benchmark_id, 
                    JSONB_AGG(B.merkle_root ORDER BY B.batch_idx) AS batch_merkle_roots,
                    JSONB_AGG(B.solution_nonces) AS solution_nonces,
                    JSONB_AGG(B.discarded_solution_nonces) AS discarded_solution_nonces,
                    JSONB_AGG(B.hashes) AS hashes
                FROM ready A
                INNER JOIN batch_data B
                    ON A.benchmark_id = B.benchmark_id
                GROUP BY A.benchmark_id
            )
            SELECT
                A.*,
                B.hash_threshold,
                B.average_solution_ratio,
                B.num_nonces
            FROM agg_batches A
            INNER JOIN job B
                ON A.benchmark_id = B.benchmark_id
            """
        )

        # Calculate merkle roots for completed jobs
        for row in rows:
            benchmark_id = row['benchmark_id']
            solution_nonces = [x for y in row['solution_nonces'] for x in y]
            discarded_solution_nonces = [x for y in row['discarded_solution_nonces'] for x in y]
            hashes = [x for y in row['hashes'] for x in y]

            # calc hash threshold based on reliability
            num_solutions = len(solution_nonces) + len(discarded_solution_nonces)
            num_nonces = row["num_nonces"]
            solution_ratio = num_solutions / num_nonces if num_nonces > 0 else 0
            average_solution_ratio = row["average_solution_ratio"]
            if average_solution_ratio == 0:
                reliability = 1.0
            elif solution_ratio == 0:
                reliability = 0.0
            else:
                reliability = min(1.0, solution_ratio / average_solution_ratio)
            # FIXME floating point representation may result in very very minor differences..
            assert 0 <= reliability <= 1.0
            denominator = 1000
            numerator = int(reliability * denominator)
            hash_threshold = (
                int(row["hash_threshold"], 16) // denominator * numerator
            ).to_bytes(32, 'big').hex()
            assert hash_threshold <= row["hash_threshold"]

            # discard solutions
            discarded_solution_nonces = set(discarded_solution_nonces)
            for n, h in zip(solution_nonces, hashes):
                if h > hash_threshold:
                    discarded_solution_nonces.add(n)
            solution_nonces = list(set(solution_nonces) - discarded_solution_nonces)
            discarded_solution_nonces = list(discarded_solution_nonces)

            batch_merkle_roots = [MerkleHash.from_str(root) for root in row['batch_merkle_roots']]
            num_batches = len(batch_merkle_roots)
            
            logger.info(f"job {benchmark_id}: (benchmark ready, #solution_nonces: {len(solution_nonces)}, #discarded_solution_nonces: {len(discarded_solution_nonces)})")

            tree = MerkleTree(
                batch_merkle_roots,
                1 << (num_batches - 1).bit_length()
            )
            merkle_root = tree.calc_merkle_root()

            # Update the database with calculated merkle root
            get_db_conn().execute_many(*[
                (
                    """
                    UPDATE job_data
                    SET merkle_root = %s, 
                        solution_nonces = %s,
                        discarded_solution_nonces = %s
                    WHERE benchmark_id = %s
                    """, 
                    (
                        merkle_root.to_str(), 
                        json.dumps(solution_nonces),
                        json.dumps(discarded_solution_nonces),
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
        rows = get_db_conn().fetch_all(
            """
            WITH ready AS (
                SELECT A.benchmark_id
                FROM proofs_batch A
                INNER JOIN job B
                    ON B.merkle_root_ready
                    AND B.merkle_proofs_ready IS NULL
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

            batch_merkle_roots = get_db_conn().fetch_one(
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
            get_db_conn().execute_many(*[
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
