import os
import json
import logging
import re
import signal
import time
import random
import threading
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

@dataclass
class Batch(FromDict):
    id: str
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

class SlaveManager:
    def __init__(self):
        pass

    def start(self):
        app = FastAPI()

        @app.route('/get-batch', methods=['GET'])
        def get_batch(request: Request):
            config = CONFIG["slave_manager_config"]

            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave["name_regex"], slave_name) for slave in config["slaves"]):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                raise HTTPException(status_code=403, detail="Unregistered slave")

            slave = next((slave for slave in config["slaves"] if re.match(slave["name_regex"], slave_name)), None)

            # check how many batches are currently assigned for this slave
            result = db_conn.fetch_all(
                """
                SELECT benchmark_id || '_' || batch_idx AS id
                FROM root_batch
                WHERE slave = %s 
                    AND end_time IS NULL
                
                UNION ALL
                
                SELECT benchmark_id || '_' || batch_idx AS id
                FROM proofs_batch
                WHERE slave = %s
                    AND end_time IS NULL
                    AND start_time IS NOT NULL
                """, 
                (
                    slave_name, 
                    slave_name
                )
            )
            concurrent = len(result) if result else 0
            if concurrent >= slave["max_concurrent_batches"]:
                logger.debug(f"{slave_name} get-batch: Max concurrent batches reached")
                batch_ids = [r["id"] for r in result]
                return JSONResponse(content=jsonable_encoder(batch_ids), status_code=425)

            # find pendings proof where the slave computed the root recently
            result = db_conn.fetch_one(
                """
                WITH selected AS (
                    SELECT
                        A.benchmark_id || '_' || A.batch_idx AS id,
                        A.sampled_nonces, 
                        A.benchmark_id, 
                        A.batch_idx,
                        C.settings, 
                        C.num_nonces, 
                        C.runtime_config, 
                        C.download_url, 
                        C.rand_hash, 
                        C.batch_size,
                        C.challenge
                    FROM proofs_batch A
                    INNER JOIN root_batch B
                        ON A.slave = %s
                        AND A.ready IS NULL
                        AND A.start_time IS NULL
                        AND B.ready
                        AND A.benchmark_id = B.benchmark_id
                        AND A.batch_idx = B.batch_idx
                    INNER JOIN job C
                        ON C.stopped IS NULL
                        AND C.benchmark_id = B.benchmark_id
                        AND C.challenge IN %s
                    ORDER BY C.block_started DESC
                    LIMIT 1
                )
                UPDATE proofs_batch D 
                SET
                    start_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                FROM selected E
                WHERE D.benchmark_id = E.benchmark_id
                    AND D.batch_idx = E.batch_idx
                RETURNING E.*
                """, 
                (
                    slave_name,
                    tuple(slave["selected_challenges"])
                )
            )

            # find any pending proofs
            if result is None:
                result = db_conn.fetch_one(
                    """
                    WITH selected AS (
                        SELECT 
                            A.benchmark_id || '_' || A.batch_idx AS id,
                            A.sampled_nonces, 
                            A.benchmark_id, 
                            A.batch_idx,
                            C.settings, 
                            C.num_nonces, 
                            C.runtime_config,
                            C.download_url, 
                            C.rand_hash,
                            C.batch_size,
                            C.challenge
                        FROM proofs_batch A
                        INNER JOIN root_batch B
                            ON A.ready IS NULL
                            AND B.ready
                            AND (
                                A.slave IS NULL
                                OR ((EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT - A.start_time) > %s
                            )
                            AND A.benchmark_id = B.benchmark_id
                            AND A.batch_idx = B.batch_idx
                        INNER JOIN job C
                            ON C.stopped IS NULL
                            AND C.benchmark_id = B.benchmark_id
                            AND C.challenge IN %s
                        ORDER BY C.block_started ASC
                        LIMIT 1
                    )
                    UPDATE proofs_batch D
                    SET
                        start_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT,
                        slave = %s
                    FROM selected E
                    WHERE D.benchmark_id = E.benchmark_id
                        AND D.batch_idx = E.batch_idx
                    RETURNING E.*
                    """, 
                    (
                        config["time_before_batch_retry"], 
                        tuple(slave["selected_challenges"]),
                        slave_name
                    )
                )

            # find any pending root
            if result is None:
                result = db_conn.fetch_one(
                    """
                    WITH selected AS (
                        SELECT 
                            A.benchmark_id || '_' || A.batch_idx AS id,
                            A.benchmark_id,
                            A.batch_idx,
                            B.settings,
                            B.num_nonces,
                            B.runtime_config,
                            B.download_url, 
                            B.rand_hash,
                            B.batch_size,
                            B.challenge
                        FROM root_batch A
                        INNER JOIN job B
                            ON A.ready IS NULL
                            AND B.stopped IS NULL
                            AND B.challenge IN %s
                            AND (
                                A.start_time IS NULL
                                OR ((EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT - A.start_time) > %s
                            )
                            AND A.benchmark_id = B.benchmark_id
                        ORDER BY B.block_started ASC
                        LIMIT 1
                    )
                    UPDATE root_batch C
                    SET
                        start_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT,
                        slave = %s
                    FROM selected D
                    WHERE C.benchmark_id = D.benchmark_id
                        AND C.batch_idx = D.batch_idx
                    RETURNING D.*
                    """, 
                    (
                        tuple(slave["selected_challenges"]),
                        config["time_before_batch_retry"], 
                        slave_name
                    )
                )

            if result is None:
                logger.debug(f"{slave_name} get-batch: None available")
                raise HTTPException(status_code=503, detail="No batches available")
 
            selected_challenge = result["challenge"]
            start_nonce = result["batch_idx"] * result["batch_size"]

            batch = Batch(
                id=result["id"],
                benchmark_id=result["benchmark_id"],
                start_nonce=start_nonce,
                num_nonces=min(result["batch_size"], result["num_nonces"] - start_nonce),
                settings=result["settings"],
                sampled_nonces=result.get("sampled_nonces"),
                runtime_config=result["runtime_config"],
                download_url=result["download_url"],
                rand_hash=result["rand_hash"],
                batch_size=result["batch_size"],
                batch_idx=result["batch_idx"]
            )

            logger.debug(f"{slave_name} get-batch: (challenge: {selected_challenge}, benchmark_id: {batch.benchmark_id})")
            return JSONResponse(content=jsonable_encoder(batch))

        @app.post('/submit-batch-root/{batch_id}')
        async def submit_batch_root(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            if db_conn.fetch_one(
                """
                SELECT 1
                FROM root_batch
                WHERE ready IS NULL 
                    AND benchmark_id = %s 
                    AND batch_idx = %s
                    AND slave = %s
                """, 
                (benchmark_id, batch_idx, slave_name)
            ) is None:
                raise HTTPException(
                    status_code=408, 
                    detail=f"Slave submitted roots for {batch_id}, but either took too long, or was not assigned this batch."
                )

            logger.debug((await request.body()).decode())
            result = await request.json()
            logger.debug(f"slave {slave_name} submitted root for {benchmark_id} batch {batch_idx}")
            
            # Update roots table with merkle root and solution nonces
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

            benchmark_id, batch_idx = batch_id.split("_")
            batch_idx = int(batch_idx)

            if db_conn.fetch_one(
                """
                SELECT 1 
                FROM proofs_batch
                WHERE ready IS NULL 
                    AND benchmark_id = %s 
                    AND batch_idx = %s
                    AND slave = %s
                """, 
                (benchmark_id, batch_idx, slave_name)
            ) is None:
                raise HTTPException(
                    status_code=408, 
                    detail=f"Slave submitted proofs for {batch_id}, but either took too long, or was not assigned this batch."
                )

            result = await request.json()
            logger.debug(f"slave {slave_name} submitted proofs for {benchmark_id} batch {batch_idx}")

            # Update proofs table with merkle proofs
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
        thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=config["port"]))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:{config['port']}")
