import os
import logging
import signal
import threading
import uvicorn
import json
from datetime import datetime
from fastapi import FastAPI, Query, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from extensions.sql import db_conn
from fastapi import Request

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def get_config():
    global master_config
    return master_config if master_config is not None else {}

def set_config(new_config):
    global master_config
    master_config = new_config

def init_config():
    global master_config
    master_config = json.loads("""
    {"player_id":"0x0000000000000000000000000000000000000000","api_key":"0","api_url":"https://testnet-api.tig.foundation","difficulty_sampler_config":{"difficulty_ranges":{"satisfiability":[0,0.5],"vehicle_routing":[0,0.5],"knapsack":[0,0.5],"vector_search":[0,0.5]},"selected_difficulties":{"vehicle_routing":[[1,2],[4,5]]}},"job_manager_config":{"backup_folder":"jobs","batch_sizes":{"satisfiability":8,"vehicle_routing":8,"knapsack":8,"vector_search":8}},"submissions_manager_config":{"time_between_retries":60000},"precommit_manager_config":{"max_pending_benchmarks":4,"algo_selection":{"satisfiability":{"algorithm":"sat_global_opt","num_nonces":40,"weight":1,"base_fee_limit":"10000000000000000"},"vehicle_routing":{"algorithm":"advanced_routing","num_nonces":40,"weight":1,"base_fee_limit":"10000000000000000"},"knapsack":{"algorithm":"classic_quadkp","num_nonces":40,"weight":1,"base_fee_limit":"10000000000000000"},"vector_search":{"algorithm":"invector_hybrid","num_nonces":40,"weight":0,"base_fee_limit":"10000000000000000"}}},"slave_manager_config":{"port":5115,"time_before_batch_retry":60000,"num_nonces_to_sample":0.5,"slaves":[{"name_regex":".*","max_concurrent_batches":4}]}}
    """)

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
        
        if result and result.get("config"):
            set_config(result["config"])
        else:
            init_config()
        
    def verify_api_key(self, x_api_key: str = Header(...)):
        if x_api_key != get_config()["api_key"]:
            raise HTTPException(status_code=403, detail="Could not validate credentials")

    def setup_routes(self):
        @self.app.post("/register-player")
        async def register_player(request: Request):
            try:
                new_player_data = await request.json()
                # Validate the incoming config if needed
                modified_config = get_config()
                if modified_config:
                    modified_config["player_id"] = new_player_data.get("player_id")
                    modified_config["api_key"] = new_player_data.get("api_key")
                
                db_conn.execute(
                    """
                    DELETE FROM config;
                    INSERT INTO config (config)
                    VALUES (%s)
                    """,
                    (json.dumps(modified_config),)
                )
                
                logger.info(f"Player updated: {modified_config}")
                return JSONResponse(content={"message": "Player updated successfully."}, status_code=200)
            except Exception as e:
                logger.error(f"Unexpected error on /update-config: {e}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")

        @self.app.get('/get-config')
        def get_config_endpoint():
            try:
                return JSONResponse(content=get_config(), status_code=200)
            except Exception as e:
                logger.error(f"Database error on /get-config: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.post("/update-config")
        async def update_config(request: Request):
            logger.debug("Received config update")
            try:
                config_update = await request.json()
                new_config = config_update
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

                set_config(new_config)

                return JSONResponse(content={"message": "Config updated successfully."}, status_code=200)
            except Exception as e:
                logger.error(f"Unexpected error on /update-config: {e}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")
        
        @self.app.get("/get-jobs")
        async def get_jobs():
            result = db_conn.fetch_all("""
                SELECT j.num_batches as batches, 
                    j.creation_timestamp as created_at, 
                    j.*
                FROM jobs j
                ORDER BY j.block_started DESC
            """)

            jobs_data = []
            if result:
                for row in result:
                    job_dict = dict(row)
                    job_dict["created_at"] = datetime.fromtimestamp(job_dict["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
                    job_dict["updated_at"] = "0"

                    batches = db_conn.fetch_all("""
                        SELECT 
                            r.batch_idx as batch_number,
                            r.slave as slave_id,
                            r.solution_nonces as solutions,
                            CASE 
                                WHEN p.proofs IS NOT NULL THEN 'COMPLETED'
                                WHEN r.root IS NOT NULL THEN 'PENDING PROOF'
                                WHEN r.slave IS NULL THEN 'PENDING SLAVE'
                                ELSE 'PENDING ROOT'
                            END as status,
                            CASE
                                WHEN p.end_epoch IS NOT NULL THEN (p.end_epoch - r.start_epoch)
                                WHEN r.end_epoch IS NOT NULL THEN (r.end_epoch - r.start_epoch)
                                WHEN r.start_epoch IS NOT NULL THEN (EXTRACT(EPOCH FROM NOW()) - r.start_epoch)
                                ELSE 0
                            END as elapsed_time
                        FROM jobs j
                        LEFT JOIN roots r ON j.benchmark_id = r.benchmark_id 
                        LEFT JOIN proofs p ON r.benchmark_id = p.benchmark_id AND r.batch_idx = p.batch_idx
                        WHERE j.benchmark_id = %s
                        ORDER BY r.batch_idx ASC
                    """, (job_dict["benchmark_id"],))
                    job_dict["batches"] = [dict(batch) for batch in batches] if batches else []
                    for batch in job_dict["batches"]:
                        batch["num_solutions"] = len(batch["solutions"]) if batch["solutions"] else 0
                        batch["slave_id"] = batch["slave_id"] if batch["slave_id"] else "Not yet assigned"
                        batch["elapsed_time"] = int(batch["elapsed_time"]) * 1000 if batch["elapsed_time"] else 0

                        batch["start_nonce"] = batch["batch_number"] * job_dict["batch_size"]
                        batch["num_nonces"] = min(
                            job_dict["batch_size"],
                            job_dict["num_nonces"] - batch["start_nonce"]
                        )

                    jobs_data.append(job_dict)

            return JSONResponse(content=jobs_data, status_code=200)

    def start(self, host="0.0.0.0", port=3336):
        def run():
            self.app = FastAPI()
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            self.setup_routes()

            uvicorn.run(self.app, host=host, port=port)
        
        server_thread = threading.Thread(target=run, daemon=True)
        server_thread.start()
        logger.info(f"ClientManager started on {host}:{port}")

        # Keep the main thread alive
        # try:
        #     while True:
        #         signal.pause()
        # except KeyboardInterrupt:
        #     logger.info("Shutting down ClientManager.")
        #     self.db_session.close()
        #     logger.info("ClientManager shut down successfully.")