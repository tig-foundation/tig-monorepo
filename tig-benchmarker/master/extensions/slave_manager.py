import os
import json
import logging
import re
import signal
import time
import random
import threading
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set
from extensions.sql import db_conn
from extensions.client_manager import get_config

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Batch(FromDict):
    benchmark_id: str
    start_nonce: int
    num_nonces: int
    settings: BenchmarkSettings
    sampled_nonces: List[int]
    runtime_config: dict
    download_url: str
    rand_hash: str
    batch_size: int
    batch_idx: int

@dataclass
class BatchRoots(FromDict):
    merkle_root: MerkleHash
    solution_nonces: List[int]
    #merkle_proofs: List[MerkleProof]

@dataclass
class BatchProof(FromDict):
    merkle_proofs: List[MerkleProof]

@dataclass
class SlaveConfig(FromDict):
    name_regex: str
    max_concurrent_batches: int

@dataclass
class SlaveManagerConfig(FromDict):
    port: int
    time_before_batch_retry: int
    slaves: List[SlaveConfig]

class SlaveManager:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs
        self.assigned = {}
        self.concurrent = {}

    def start(self):
        app = FastAPI()

        @app.route('/get-batch', methods=['GET'])
        def get_batch(request: Request):
            config = get_config()["slave_manager_config"]

            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in config["slaves"]):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                return "Unregistered slave", 403

            slave = next((slave for slave in config["slaves"] if re.match(slave["name_regex"], slave_name)), None)

            result = db_conn.fetch_one("""
            SELECT COUNT(*)
            FROM (
                SELECT 1
                FROM roots r
                WHERE r.slave = %s 
                    AND r.end_epoch IS NULL
                UNION ALL
                SELECT 1 
                FROM proofs p
                WHERE p.slave = %s
                    AND p.end_epoch IS NULL
                    AND p.start_epoch IS NOT NULL
            ) AS ongoing_batches
            """, (slave_name, slave_name))
            
            concurrent = result["count"] if result else 0
            if concurrent >= slave.max_concurrent_batches:
                logger.debug(f"{slave_name} get-batch: Max concurrent batches reached")
                raise HTTPException(status_code=503, detail="Max concurrent batches reached")

            # pending proofs for this slave
            result = db_conn.fetch_one("""
            WITH selected AS (
                SELECT p.sampled_nonces AS batch_sampled_nonces, r.*, p.*,
                    j.benchmark_id as b_id, r.batch_idx as b_idx,
                    j.settings, j.num_nonces, j.runtime_config, j.download_url, 
                    j.rand_hash, j.batch_size, j.challenge
                FROM roots r
                INNER JOIN proofs p
                    ON p.benchmark_id = r.benchmark_id AND p.batch_idx = r.batch_idx
                INNER JOIN jobs j
                    ON j.benchmark_id = r.benchmark_id
                WHERE r.end_epoch IS NOT NULL
                    AND p.start_epoch IS NULL
                    AND p.end_epoch IS NULL
                    AND r.slave = %s
                ORDER BY j.block_started DESC
                LIMIT 1
            )
            UPDATE proofs p SET
                start_epoch = EXTRACT(EPOCH FROM NOW()),
                slave = %s
            FROM selected s
            WHERE p.benchmark_id = s.b_id
                AND p.batch_idx = s.b_idx
            RETURNING s.*
            """, (slave_name, slave_name))

            if result is None:
                # if we dont have a pending proof, we can try to get a new one
                result = db_conn.fetch_one("""
                WITH selected AS (
                    SELECT p.sampled_nonces AS batch_sampled_nonces, r.*, p.*, 
                        j.benchmark_id as b_id, r.batch_idx as b_idx,
                        j.settings, j.num_nonces, j.runtime_config, j.download_url, 
                        j.rand_hash, j.batch_size, j.challenge
                    FROM roots r
                    INNER JOIN proofs p
                        ON p.benchmark_id = r.benchmark_id AND p.batch_idx = r.batch_idx
                    INNER JOIN jobs j
                        ON j.benchmark_id = r.benchmark_id
                    WHERE r.end_epoch IS NOT NULL
                        AND p.end_epoch IS NULL
                        AND (
                            r.slave IS NULL
                            OR (EXTRACT(EPOCH FROM NOW()) - p.start_epoch) > %s
                        )
                    ORDER BY j.block_started ASC
                    LIMIT 1
                )
                UPDATE proofs p SET
                    start_epoch = EXTRACT(EPOCH FROM NOW()),
                    slave = %s
                FROM selected s
                WHERE p.benchmark_id = s.b_id
                    AND p.batch_idx = s.b_idx
                RETURNING s.*
                """, (config["time_before_batch_retry"], slave_name))

                if result is None:
                    result = db_conn.fetch_one("""
                    WITH selected AS (
                        SELECT NULL AS batch_sampled_nonces, r.*, 
                            j.benchmark_id as b_id, r.batch_idx as b_idx,
                            j.settings, j.num_nonces, j.runtime_config, j.download_url, 
                            j.rand_hash, j.batch_size, j.challenge
                        FROM roots r
                        INNER JOIN jobs j
                            ON j.benchmark_id = r.benchmark_id
                        WHERE r.end_epoch IS NULL
                            AND (
                                r.start_epoch IS NULL
                                OR (EXTRACT(EPOCH FROM NOW()) - r.start_epoch) > %s
                            )
                        ORDER BY j.block_started ASC
                        LIMIT 1
                    )
                    UPDATE roots r SET
                        start_epoch = EXTRACT(EPOCH FROM NOW()),
                        slave = %s
                    FROM selected s
                    WHERE r.benchmark_id = s.b_id
                        AND r.batch_idx = s.b_idx
                    RETURNING s.*
                    """, (config["time_before_batch_retry"], slave_name))

            if result is None:
                logger.debug(f"{slave_name} get-batch: None available")
                raise HTTPException(status_code=503, detail="No batches available")
 
            selected_challenge = result["challenge"]
            start_nonce = result["b_idx"] * result["batch_size"]

            batch = Batch(
                benchmark_id=result["b_id"],
                start_nonce=start_nonce,
                num_nonces=min(result["batch_size"], result["num_nonces"] - start_nonce),
                settings=result["settings"],
                sampled_nonces=result["batch_sampled_nonces"] if result["batch_sampled_nonces"] is not None else [],
                runtime_config=result["runtime_config"],
                download_url=result["download_url"],
                rand_hash=result["rand_hash"],
                batch_size=result["batch_size"],
                batch_idx=result["b_idx"]
            )

            logger.debug(f"{slave_name} get-batch: (challenge: {selected_challenge}, #batches: 1, batch_ids: [{batch.benchmark_id}])")
            return JSONResponse(content=jsonable_encoder(batch))

        @app.post('/submit-batch-root/{batch_id}')
        async def submit_batch_roots(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            result = db_conn.fetch_one("""
                SELECT slave 
                FROM roots 
                WHERE root IS NULL 
                    AND benchmark_id = %s 
                    AND batch_idx = %s
            """, (benchmark_id, batch_idx))

            if result is None or result['slave'] != slave_name:
                raise HTTPException(status_code=400, detail=f"Slave submitted roots for {batch_id}, but either took too long, or was not assigned this batch.")

            roots = BatchRoots.from_dict(await request.json())
            
            # Update roots table with merkle root and solution nonces
            db_conn.execute("""
                UPDATE roots 
                SET root = %s,
                    solution_nonces = %s,
                    end_epoch = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id = %s 
                AND batch_idx = %s
            """, (roots.merkle_root.to_str(), json.dumps(roots.solution_nonces), benchmark_id, batch_idx))

            return {"status": "OK"}

        @app.post('/submit-batch-proofs/{batch_id}')
        async def submit_batch_proofs(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            result = db_conn.fetch_one("""
                SELECT slave 
                FROM proofs 
                WHERE proofs IS NULL 
                    AND benchmark_id = %s 
                    AND batch_idx = %s
            """, (benchmark_id, batch_idx))

            if result is None or result['slave'] != slave_name:
                raise HTTPException(status_code=400, detail=f"Slave submitted proofs for {batch_id}, but either took too long, or was not assigned this batch.")

            proofs = await request.json()
            
            # Update proofs table with merkle proofs
            db_conn.execute("""
                UPDATE proofs 
                SET proofs = %s,
                    end_epoch = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id = %s 
                AND batch_idx = %s
            """, (json.dumps(proofs["merkle_proofs"]), benchmark_id, batch_idx))

            return {"status": "OK"}
            
        thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=get_config()["slave_manager_config"].port))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:{get_config()["slave_manager_config"].port}")
