import os
import logging
import signal
import threading
from sqlalchemy import desc
import uvicorn
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified
from fastapi import FastAPI, Query, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, RootModel
from tig_benchmarker.database.init import SessionLocal
from tig_benchmarker.database.models.index import ConfigModel, JobModel, SlaveModel
from fastapi import Request


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

# Pydantic Models
class ConfigUpdate(RootModel[dict]):
    pass

class PlayerData(RootModel[dict]):
    pass


# ClientManager Class
class ClientManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.db_session = SessionLocal()
        logger.info("ClientManager initialized and connected to the database.")
        # FastAPI Application
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        self.start_server()
        
    def verify_api_key(self, x_api_key: str = Header(...)):
        if x_api_key != self.api_key:
            raise HTTPException(status_code=403, detail="Could not validate credentials")

    def setup_routes(self):
        @self.app.post("/register-player")
        async def register_player(player_data: PlayerData):
            try:
                new_player_data = player_data.root
                # Validate the incoming config if needed
                config = self.db_session.query(ConfigModel).first()
                if config:
                    config.config_data["player_id"] = new_player_data.get("player_id")
                    config.config_data["api_key"] = new_player_data.get("api_key")
                
                flag_modified(config, "config_data")

                logger.info(f"Player updated: {config.config_data}")

                self.db_session.commit()
                return JSONResponse(content={"message": "Player updated successfully."}, status_code=200)
            except ValidationError as ve:
                logger.error(f"Validation error on /update-config: {ve}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")
            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error on /update-config: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /update-config: {e}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")

        @self.app.get("/get-config")
        async def get_config():
            try:
                config = self.db_session.query(ConfigModel).first()
                if config and config.config_data:
                    return JSONResponse(content=config.config_data, status_code=200)
                else:
                    raise HTTPException(status_code=404, detail="Config not found")
            except SQLAlchemyError as e:
                logger.error(f"Database error on /get-config: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.post("/update-config")
        async def update_config(request: Request):
            print("Received config update")
            logger.debug("Received config update")
            try:
                config_update = await request.json()
                new_config = config_update
                new_config["player_id"] = new_config["player_id"].lower()
                new_config["api_url"] = new_config["api_url"].rstrip('/')
                # Validate the incoming config if needed
                config = self.db_session.query(ConfigModel).first()
                if config:
                    config.config_data = new_config
                else:
                    config = ConfigModel(config_data=new_config)
                    self.db_session.add(config)
                self.db_session.commit()
                return JSONResponse(content={"message": "Config updated successfully."}, status_code=200)
            except ValidationError as ve:
                logger.error(f"Validation error on /update-config: {ve}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")
            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error on /update-config: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /update-config: {e}")
                raise HTTPException(status_code=400, detail="Invalid configuration data")
        
        @self.app.get("/get-jobs")
        async def get_jobs():
            jobs = self.db_session.query(JobModel).all()
            jobs_data = [
                {
                    **job.to_dataclass().to_dict(),
                    "batches": [ batch.to_dict() for batch in job.batches ],
                    "created_at": str(job.created_at)
                } for job in jobs
            ]

            return JSONResponse(content=jobs_data, status_code=200)

    def start_server(self, host="0.0.0.0", port=3336):
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