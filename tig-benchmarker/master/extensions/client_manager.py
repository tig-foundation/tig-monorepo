import os
import logging
import signal
import threading
import uvicorn
import json
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

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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

        self.setup_routes()
        
    def verify_api_key(self, x_api_key: str = Header(...)):
        if x_api_key != get_config()["api_key"]:
            raise HTTPException(status_code=403, detail="Could not validate credentials")

    async def setup_routes(self):
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

        @self.app.get("/get-config")
        async def get_config():
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
            #jobs = self.db_session.query(JobModel).all()
            #jobs_data = [
            #    {
            #        **job.to_dataclass().to_dict(),
            #        "batches": [ batch.to_dict() for batch in job.batches ],
            #        "created_at": str(job.created_at)
            #    } for job in jobs
            #]

            jobs_data = []
            return JSONResponse(content=jobs_data, status_code=200)

    def start(self, host="0.0.0.0", port=3336):
        def run():
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