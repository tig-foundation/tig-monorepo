import os
import json
import logging
import re
import time
import random
from threading import Thread, Lock
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from common.structs import *
from common.utils import *
from typing import Dict, List, Optional, Set
from master.sql import get_db_conn
from master.client_manager import CONFIG


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class SlaveManager:
    def __init__(self):
        self.batches = []
        self.lock = Lock()

    def run(self):
        with self.lock:
            self.batches = get_db_conn().fetch_all(
                """
                SELECT * FROM (
                    SELECT
                        A.slave,
                        A.start_time,
                        A.end_time,
                        JSONB_BUILD_OBJECT(
                            'id', A.benchmark_id || '_' || A.batch_idx,
                            'benchmark_id', A.benchmark_id,
                            'start_nonce', A.batch_idx * B.batch_size,
                            'num_nonces', LEAST(B.batch_size, B.num_nonces - A.batch_idx * B.batch_size),
                            'settings', B.settings,
                            'sampled_nonces', A.sampled_nonces,
                            'runtime_config', B.runtime_config,
                            'download_url', B.download_url,
                            'rand_hash', B.rand_hash,
                            'batch_size', B.batch_size,
                            'batch_idx', A.batch_idx,
                            'challenge', B.challenge,
                            'algorithm', B.algorithm,
                            'hash_threshold', B.hash_threshold
                        ) AS batch
                    FROM proofs_batch A
                    INNER JOIN job B
                        ON A.ready IS NULL
                        AND B.merkle_root_ready
                        AND B.stopped IS NULL
                        AND A.benchmark_id = B.benchmark_id
                    ORDER BY B.block_started, A.benchmark_id, A.batch_idx
                )
                
                UNION ALL
                
                SELECT * FROM (
                    SELECT
                        A.slave,
                        A.start_time,
                        A.end_time,
                        JSONB_BUILD_OBJECT(
                            'id', A.benchmark_id || '_' || A.batch_idx,
                            'benchmark_id', A.benchmark_id,
                            'start_nonce', A.batch_idx * B.batch_size,
                            'num_nonces', LEAST(B.batch_size, B.num_nonces - A.batch_idx * B.batch_size),
                            'settings', B.settings,
                            'sampled_nonces', NULL,
                            'runtime_config', B.runtime_config,
                            'download_url', B.download_url,
                            'rand_hash', B.rand_hash,
                            'batch_size', B.batch_size,
                            'batch_idx', A.batch_idx,
                            'challenge', B.challenge,
                            'algorithm', B.algorithm,
                            'hash_threshold', B.hash_threshold
                        ) AS batch
                    FROM root_batch A
                    INNER JOIN job B
                        ON A.ready IS NULL
                        AND B.stopped IS NULL
                        AND A.benchmark_id = B.benchmark_id
                    ORDER BY B.block_started, A.benchmark_id, A.batch_idx
                )
                """
            )
            logger.debug(f"Refreshed pending batches. Got {len(self.batches)}")

    def start(self):
        app = FastAPI()

        @app.route('/get-batches', methods=['GET'])
        def get_batch(request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave["name_regex"], slave_name) for slave in CONFIG["slaves"]):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                raise HTTPException(status_code=403, detail="Unregistered slave")

            slave = next((slave for slave in CONFIG["slaves"] if re.match(slave["name_regex"], slave_name)), None)

            concurrent = []
            updates = []
            now = time.time() * 1000
            with self.lock:
                concurrent = [
                    b["batch"] for b in self.batches
                    if b["slave"] == slave_name
                ]
                for b in self.batches:
                    batch = b["batch"]
                    if len(concurrent) >= slave["max_concurrent_batches"]:
                        break
                    if (
                        b["slave"] == slave_name or
                        not re.match(slave["algorithm_id_regex"], b["batch"]["settings"]["algorithm_id"]) or
                        b["end_time"] is not None
                    ):
                        continue
                    if (
                        b["slave"] is None or
                        b["start_time"] is None or
                        (now - b["start_time"]) > CONFIG["time_before_batch_retry"]
                    ):
                        b["slave"] = slave_name
                        b["start_time"] = now
                        table = "root_batch" if batch["sampled_nonces"] is None else "proofs_batch"
                        updates.append((
                            f"""
                            UPDATE {table}
                            SET slave = %s,
                                start_time = %s
                            WHERE benchmark_id = %s
                                AND batch_idx = %s
                            """,
                            (slave_name, now, batch["benchmark_id"], batch["batch_idx"])
                        ))
                        concurrent.append(batch)
            if len(concurrent) == 0:
                logger.debug(f"no batches available for {slave_name}")
            if len(updates) > 0:
                get_db_conn().execute_many(*updates)
            return JSONResponse(content=jsonable_encoder(concurrent))

        @app.post('/submit-batch-root/{batch_id}')
        async def submit_batch_root(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            with self.lock:
                b = next((
                    b for b in self.batches
                    if (
                        b["batch"]["id"] == batch_id and 
                        b["batch"]["sampled_nonces"] is None and
                        b["slave"] == slave_name
                    )
                ), None)
                if b is None:
                    raise HTTPException(
                        status_code=408, 
                        detail=f"Slave submitted roots for {batch_id}, but either took too long, or was not assigned this batch."
                    )
                b["end_time"] = time.time() * 1000

            try:
                result = await request.json()
                merkle_root = MerkleHash.from_str(result["merkle_root"])
                solution_nonces = result["solution_nonces"]
                discarded_solution_nonces = result["discarded_solution_nonces"]
                assert isinstance(solution_nonces, list) and all(isinstance(x, int) for x in solution_nonces)
                assert isinstance(discarded_solution_nonces, list) and all(isinstance(x, int) for x in discarded_solution_nonces)
                hashes = result["hashes"]
                assert (
                    isinstance(hashes, list) and 
                    len(hashes) == len(solution_nonces) and
                    all(
                        isinstance(x, str) and 
                        len(x) == 64 and
                        x <= b["batch"]["hash_threshold"]
                        for x in hashes
                    )
                )
                logger.debug(f"slave {slave_name} submitted root for {batch_id}")
            except Exception as e:
                logger.error(f"slave {slave_name} submitted INVALID root for {batch_id}: {e}")
                raise HTTPException(status_code=400, detail="INVALID root")
            
            # Update roots table with merkle root and solution nonces
            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)
            get_db_conn().execute_many(*[
                (
                    """
                    UPDATE root_batch
                    SET ready = true,
                        end_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s
                    """, 
                    (
                        benchmark_id,
                        batch_idx
                    )
                ),
                (
                    """
                    UPDATE batch_data
                    SET merkle_root = %s,
                        solution_nonces = %s,
                        discarded_solution_nonces = %s,
                        hashes = %s
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s                    
                    """,
                    (
                        merkle_root.to_str(),
                        json.dumps(solution_nonces),
                        json.dumps(discarded_solution_nonces),
                        json.dumps(hashes),
                        benchmark_id,
                        batch_idx
                    )
                )
            ])

            return {"status": "OK"}

        @app.post('/submit-batch-proofs/{batch_id}')
        async def submit_batch_proofs(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            with self.lock:
                b = next((
                    b for b in self.batches
                    if (
                        b["batch"]["id"] == batch_id and 
                        b["batch"]["sampled_nonces"] is not None and
                        b["slave"] == slave_name
                    )
                ), None)
                if b is None:
                    raise HTTPException(
                        status_code=408, 
                        detail=f"Slave submitted proofs for {batch_id}, but either took too long, or was not assigned this batch."
                    )
                b["end_time"] = time.time() * 1000

            try:
                result = await request.json()
                merkle_proofs = [MerkleProof.from_dict(x) for x in result["merkle_proofs"]]
                logger.debug(f"slave {slave_name} submitted proofs for {batch_id}")
            except Exception as e:
                logger.error(f"slave {slave_name} submitted INVALID proofs for {batch_id}: {e}")
                raise HTTPException(status_code=400, detail="INVALID proofs")

            # Update proofs table with merkle proofs
            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)
            get_db_conn().execute_many(*[
                (
                    """
                    UPDATE proofs_batch
                    SET ready = true,
                        end_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s
                    """, 
                    (benchmark_id, batch_idx)
                ),
                (
                    """
                    UPDATE batch_data
                    SET merkle_proofs = %s
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s
                    """, 
                    (
                        json.dumps([x.to_dict() for x in merkle_proofs]), 
                        benchmark_id, 
                        batch_idx
                    )
                )
            ])

            return {"status": "OK"}
            
        thread = Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=5115))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:5115")
