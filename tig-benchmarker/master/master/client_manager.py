import os
import logging
import signal
import threading
import uvicorn
import json
from datetime import datetime
from master.sql import db_conn
from fastapi import FastAPI, Query, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

CONFIG = {}

class ClientManager:
    def __init__(self):
        logger.info("ClientManager initialized and connected to the database.")

        # Fetch initial config from database
        result = db_conn.fetch_one(
            """
            SELECT config FROM config
            LIMIT 1
            """
        )
        CONFIG.update(result["config"])
        self.latest_data = {}

    def on_new_block(self, **kwargs):
        def convert(d):
            if isinstance(d, dict):
                return {k: convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert(v) for v in d]
            elif hasattr(d, "to_dict"):
                return d.to_dict()
            else:
                return d
        self.latest_data = convert(kwargs)

    def setup_routes(self):
        @self.app.get('/get-latest-data')
        async def get_latest_data():
            return JSONResponse(content=self.latest_data)

        @self.app.get('/get-config')
        async def get_config_endpoint():
            return JSONResponse(content=CONFIG)
        
        @self.app.get('/stop/{benchmark_id}')
        async def stop_benchmark(benchmark_id: str):
            try:
                db_conn.execute(
                    """
                    UPDATE job
                    SET stopped = true
                    WHERE benchmark_id = %s
                    """,
                    (benchmark_id,)
                )
                return JSONResponse(content={"message": "Benchmark stopped successfully."})
            except Exception as e:
                logger.error(f"Unexpected error on /stop/{benchmark_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.post("/update-config")
        async def update_config(request: Request):
            logger.debug("Received config update")
            try:
                new_config = await request.json()
                new_config["player_id"] = new_config["player_id"].lower()
                new_config["api_url"] = new_config["api_url"].rstrip('/')
                
                # Update config in database
                db_conn.execute(
                    """
                    DELETE FROM config;
                    INSERT INTO config (config)
                    VALUES (%s)
                    """,
                    (json.dumps(new_config),)
                )
                logger.info(f"Config updated in database: {new_config}")

                CONFIG.update(new_config)

                return JSONResponse(content={"message": "Config updated successfully."})
            except Exception as e:
                logger.error(f"Unexpected error on /update-config: {e}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")
        
        @self.app.get("/get-jobs")
        async def get_jobs():
            result = db_conn.fetch_all(
                """
                WITH recent_jobs AS (
                    SELECT benchmark_id 
                    FROM job
                    ORDER BY block_started DESC
                    LIMIT 100
                ),
                recent_batches AS (
                    SELECT
                        C.benchmark_id,
                        C.batch_idx,
                        JSONB_BUILD_OBJECT(
                            'batch_idx', C.batch_idx,
                            'slave', CASE 
                                WHEN E.batch_idx IS NOT NULL THEN E.slave 
                                ELSE C.slave
                            END,
                            'num_nonces', LEAST(
                                B.batch_size,
                                B.num_nonces - B.batch_size * C.batch_idx
                            ),
                            'num_solutions', JSONB_ARRAY_LENGTH(D.solution_nonces),
                            'status', CASE
                                WHEN B.stopped IS NOT NULL THEN 'STOPPED'
                                WHEN E.batch_idx IS NOT NULL THEN (
                                    CASE
                                        WHEN E.ready = true THEN 'PROOF READY'
                                        WHEN E.start_time IS NOT NULL THEN 'COMPUTING PROOF'
                                        ELSE 'PROOF NOT ASSIGNED'
                                    END
                                )
                                ELSE (
                                    CASE
                                        WHEN C.ready = true THEN 'ROOT READY'
                                        WHEN C.start_time IS NOT NULL THEN 'COMPUTING ROOT'
                                        ELSE 'ROOT NOT ASSIGNED'
                                    END
                                )
                            END,
                            'end_time', CASE
                                WHEN E.batch_idx IS NOT NULL THEN E.end_time
                                ELSE C.end_time
                            END,
                            'start_time', CASE
                                WHEN E.batch_idx IS NOT NULL THEN E.start_time
                                ELSE C.start_time
                            END
                        ) AS batch_data
                    FROM recent_jobs A
                    INNER JOIN job B
                        ON A.benchmark_id = B.benchmark_id
                    INNER JOIN root_batch C
                        ON B.benchmark_id = C.benchmark_id
                    INNER JOIN batch_data D
                        ON C.benchmark_id = D.benchmark_id
                        AND C.batch_idx = D.batch_idx
                    LEFT JOIN proofs_batch E
                        ON C.benchmark_id = E.benchmark_id
                        AND C.batch_idx = E.batch_idx
                ),
                grouped_batches AS (
                    SELECT 
                        benchmark_id,
                        JSONB_AGG(batch_data ORDER BY batch_idx) AS batches,
                        SUM((batch_data->>'num_solutions')::INTEGER) AS num_solutions
                    FROM recent_batches
                    GROUP BY benchmark_id
                )
                SELECT 
                    B.benchmark_id,
                    B.challenge,
                    B.algorithm,
                    B.settings->'difficulty' AS difficulty,
                    B.batch_size,
                    B.num_nonces,
                    A.num_solutions,
                    CASE
                        WHEN B.end_time IS NOT NULL THEN 'COMPLETED'
                        WHEN B.stopped IS NOT NULL THEN 'STOPPED'
                        WHEN B.merkle_proofs_ready = true THEN 'SUBMITTING PROOF'
                        WHEN B.sampled_nonces IS NOT NULL THEN 'COMPUTING PROOF'
                        WHEN B.sampled_nonces IS NULL AND B.merkle_root_ready THEN 'SUBMITTING ROOT'
                        ELSE 'COMPUTING ROOT'
                    END AS status,
                    B.end_time,
                    B.start_time,
                    A.batches,
                    B.end_time IS NULL AND B.stopped IS NULL AS can_stop
                FROM grouped_batches A
                INNER JOIN job B
                    ON A.benchmark_id = B.benchmark_id
                ORDER BY 
                    B.block_started DESC,
                    B.benchmark_id DESC
                """
            )

            return JSONResponse(
                content=[dict(row) for row in result], 
                status_code=200,
                headers = {"Accept-Encoding": "gzip"}
            )

    def start(self):
        def run():
            self.app = FastAPI()
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self.app.add_middleware(
                GZipMiddleware,
                minimum_size=1000,
                compresslevel=5,
            )

            self.setup_routes()

            uvicorn.run(self.app, host="0.0.0.0", port=3336)
        
        server_thread = threading.Thread(target=run, daemon=True)
        server_thread.start()
        logger.info(f"ClientManager started on 0.0.0.0:3336")