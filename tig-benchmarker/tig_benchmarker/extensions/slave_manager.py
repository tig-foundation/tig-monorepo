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
from tig_benchmarker.extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set
from tig_benchmarker.sql import db_conn

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
    max_concurrent_batches: Dict[str, int]

@dataclass
class SlaveManagerConfig(FromDict):
    port: int
    time_before_batch_retry: int
    slaves: List[SlaveConfig]

class SlaveManager:
    def __init__(self, config: SlaveManagerConfig, jobs: List[Job]):
        self.config = config
        self.jobs = jobs
        self.assigned = {}
        self.concurrent = {}

    def start(self):
        app = FastAPI()

        @app.route('/get-batch', methods=['GET'])
        def get_batch(request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                return "Unregistered slave", 403

            slave = next((slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)), None)

            result = db_conn.fetch_one("""
            SELECT COUNT(*) as count, j.challenge
            FROM batch b
            INNER JOIN jobs j 
                ON j.benchmark_id = b.benchmark_id 
            WHERE b.datetime_finish IS NULL
                AND b.slave = %s
            GROUP BY j.challenge
            """, (slave_name,))

            count = result["count"] if result is not None else 0
            selected_challenge = result["challenge"] if result is not None else None

            if selected_challenge is not None and count >= slave.max_concurrent_batches[selected_challenge]:
                logger.debug(f"{slave_name} get-batch: max concurrent batches reached")
                raise HTTPException(status_code=425, detail="Max concurrent batches reached")

            if selected_challenge is None:
                result = db_conn.fetch_one("""
                    WITH selected AS (
                        SELECT b.sampled_nonces AS batch_sampled_nonces, b.*, 
                            j.challenge, j.num_nonces, j.batch_size, j.rand_hash, j.settings, j.runtime_config, j.download_url
                        FROM batch b
                        INNER JOIN jobs j ON j.benchmark_id = b.benchmark_id
                        WHERE b.datetime_finish IS NULL 
                            AND (b.datetime_start IS NULL OR (EXTRACT(EPOCH FROM NOW()) - b.datetime_start) > %s)
                        ORDER BY b.datetime_start ASC
                        LIMIT 1
                    )
                    UPDATE batch b SET
                        datetime_start = EXTRACT(EPOCH FROM NOW()),
                        slave = %s
                    FROM selected s
                    WHERE b.benchmark_id = s.benchmark_id
                        AND b.batch_idx = s.batch_idx
                    RETURNING s.*
                """, (self.config.time_before_batch_retry, slave_name))
            else:
                result = db_conn.fetch_one("""
                    WITH selected AS (
                        SELECT b.sampled_nonces AS batch_sampled_nonces, b.*,
                            j.challenge, j.num_nonces, j.batch_size, j.rand_hash, j.settings, j.runtime_config, j.download_url
                        FROM batch b
                        INNER JOIN jobs j ON j.benchmark_id = b.benchmark_id
                        WHERE b.datetime_finish IS NULL 
                            AND (b.datetime_start IS NULL OR (EXTRACT(EPOCH FROM NOW()) - b.datetime_start) > %s)
                        AND j.challenge = %s
                        ORDER BY b.datetime_start ASC
                        LIMIT 1
                    )
                    UPDATE batch b SET
                        datetime_start = EXTRACT(EPOCH FROM NOW()),
                        slave = %s
                    FROM selected s
                    WHERE b.benchmark_id = s.benchmark_id
                        AND b.batch_idx = s.batch_idx
                    RETURNING s.*
                """, (self.config.time_before_batch_retry, selected_challenge, slave_name))

            if result is None:
                logger.debug(f"{slave_name} get-batch: None available")
                raise HTTPException(status_code=503, detail="No batches available")

            start_nonce = result["batch_idx"] * result["batch_size"]
            batch = Batch(
                benchmark_id=result["benchmark_id"],
                start_nonce=start_nonce,
                num_nonces=min(result["batch_size"], result["num_nonces"] - start_nonce),
                settings=result["settings"],
                sampled_nonces=result["batch_sampled_nonces"] if result["batch_sampled_nonces"] is not None else [],
                runtime_config=result["runtime_config"],
                download_url=result["download_url"],
                rand_hash=result["rand_hash"],
                batch_size=result["batch_size"]
            )

            logger.debug(f"{slave_name} get-batch: (challenge: {selected_challenge}, #batches: 1, batch_ids: [{batch.benchmark_id}])")
            return JSONResponse(content=jsonable_encoder(batch))

        @app.post('/submit-batch-roots/{batch_id}')
        async def submit_batch_roots(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            result = db_conn.fetch_one("""
                SELECT slave 
                FROM batch 
                WHERE merkle_root IS NULL 
                    AND benchmark_id = %s 
                    AND batch_idx = %s
            """, (benchmark_id, batch_idx))

            if result is None or result['slave'] != slave_name:
                raise HTTPException(status_code=400, detail=f"Slave submitted roots for {batch_id}, but either took too long, or was not assigned this batch.")

            roots = BatchRoots.from_dict(await request.json())
            
            # Update batch with merkle root and solution nonces
            db_conn.execute("""
                UPDATE batch 
                SET merkle_root = %s,
                    solution_nonces = %s,
                    datetime_finish = EXTRACT(EPOCH FROM NOW())
                    proof_request_time = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id = %s 
                AND batch_idx = %s
            """, (roots.merkle_root.to_str(), json.dumps(roots.solution_nonces), benchmark_id, batch_idx))

            return {"status": "OK"}

        @app.post('/submit-batch-proofs/{batch_id}')
        async def submit_batch_proofs(batch_id: str, request: Request):
            # might want to introduce a lock here
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            result = db_conn.fetch_one("""
                SELECT slave 
                    FROM batch 
                WHERE merkle_proofs IS NULL
                    AND merkle_root IS NOT NULL
                    AND benchmark_id = %s 
                    AND batch_idx = %s
            """, (benchmark_id, batch_idx))

            if result is None or result["slave"] != slave_name:
                raise HTTPException(status_code=400, detail=f"Slave submitted roots for {batch_id}, but either took too long, or was not assigned this batch.")

            proofs = BatchProof.from_dict(await request.json())
            
            # Get current merkle proofs and merge with new ones
            current_proofs = db_conn.fetch_one("""
                SELECT merkle_proofs
                    FROM jobs
                WHERE benchmark_id = %s
            """, (benchmark_id,))

            merged_proofs = {}
            if current_proofs and current_proofs["merkle_proofs"]:
                merged_proofs.update(current_proofs["merkle_proofs"])
            merged_proofs.update(proofs.merkle_proofs)

            # Update job with merged merkle proofs
            db_conn.execute("""
                UPDATE jobs 
                    SET merkle_proofs = %s
                WHERE benchmark_id = %s;

                UPDATE batch
                SET merkle_proofs = %s,
                    proofs_submitted = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id = %s
                    AND batch_idx = %s
            """, (json.dumps(merged_proofs), benchmark_id, json.dumps(proofs.merkle_proofs), benchmark_id, batch_idx))

            return {"status": "OK"}
            
        thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=self.config.port))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:{self.config.port}")
