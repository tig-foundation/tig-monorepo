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
from master.sql import db_conn
from master.client_manager import CONFIG


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class SlaveManager:
    def __init__(self):
        self.batches = []
        self.lock = Lock()

    def run(self):
        with self.lock:
            self.batches = db_conn.fetch_all(
                """
                SELECT * FROM (
                    SELECT
                        A.slave,
                        A.start_time,
                        A.end_time,
                        B.challenge,
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
                            'batch_idx', A.batch_idx
                        ) AS batch
                    FROM proofs_batch A
                    INNER JOIN job B
                        ON A.ready IS NULL
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
                        B.challenge,
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
                            'batch_idx', A.batch_idx
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
            config = CONFIG["slave_manager_config"]

            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave["name_regex"], slave_name) for slave in config["slaves"]):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                raise HTTPException(status_code=403, detail="Unregistered slave")

            slave = next((slave for slave in config["slaves"] if re.match(slave["name_regex"], slave_name)), None)

            concurrent = []
            updates = []
            now = time.time() * 1000
            selected_challenges = set(slave["selected_challenges"])
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
                        b["challenge"] not in selected_challenges or
                        b["end_time"] is not None
                    ):
                        continue
                    if (
                        b["slave"] is None or
                        b["start_time"] is None or
                        (now - b["start_time"]) > config["time_before_batch_retry"]
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
                db_conn.execute_many(*updates)
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

            logger.debug((await request.body()).decode())
            result = await request.json()
            logger.debug(f"slave {slave_name} submitted root for {batch_id}")
            
            # Update roots table with merkle root and solution nonces
            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)
            db_conn.execute_many(*[
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
                        solution_nonces = %s
                    WHERE benchmark_id = %s 
                        AND batch_idx = %s                    
                    """,
                    (
                        result["merkle_root"],
                        json.dumps(result["solution_nonces"]),
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

            result = await request.json()
            logger.debug(f"slave {slave_name} submitted proofs for {batch_id}")

            # Update proofs table with merkle proofs
            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)
            db_conn.execute_many(*[
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
                        json.dumps(result["merkle_proofs"]), 
                        benchmark_id, 
                        batch_idx
                    )
                )
            ])

            return {"status": "OK"}
            
        config = CONFIG["slave_manager_config"]
        thread = Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=config["port"]))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:{config['port']}")
