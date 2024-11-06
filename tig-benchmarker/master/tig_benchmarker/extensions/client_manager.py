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
from tig_benchmarker.database.models.index import AssignedBatchModel, BlockModel, ConfigModel, JobModel, PlayerModel, SlaveRegistryModel


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

        @self.app.get("/get-config", dependencies=[Depends(self.verify_api_key)])
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

        @self.app.post("/update-config", dependencies=[Depends(self.verify_api_key)])
        async def update_config(config_update: ConfigUpdate):
            try:
                new_config = config_update.root
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
            
        @self.app.get("/get-current-block", dependencies=[Depends(self.verify_api_key)])
        async def get_current_block():
            try:
                block = self.db_session.query(BlockModel).order_by(desc(BlockModel.created_at)).first()
                if block :
                    player = self.db_session.query(PlayerModel).order_by(desc(PlayerModel.created_at)).first()
                    if player:
                        return JSONResponse(
                            content={
                                "block": block.to_dataclass().to_dict(),
                                "player": player.to_dataclass().to_dict()
                            },
                            status_code=200
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Player details not found"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        details="Block details not found"
                    )
            except SQLAlchemyError as e:
                logger.error(f"Database error on /get-current-block: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /get-current-block: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            
        @self.app.get("/get-benchmark-jobs", dependencies=[Depends(self.verify_api_key)])
        async def get_benchmark_jobs(limit: int = Query(10, gt=0), page: int = Query(1, gt=0)):

            # TODO: Get Block Data
            # TODO: Get Created Time, Start Time, End Time
            # TODO: Batch Progress
            
            try:
                total_jobs = self.db_session.query(JobModel).count()
                total_pages = (total_jobs + limit - 1) // limit
                if page > total_pages and total_pages != 0:
                    raise HTTPException(status_code=400, detail="Page not found")
                
                jobs = self.db_session.query(JobModel).order_by(desc(JobModel.created_at)).offset((page - 1)*limit).limit(limit).all()

                logger.info(f"Jobs: {jobs[0]}")

                jobs_data = [{**job.to_dataclass().to_dict(), "block_height": job.block.to_dataclass().to_dict()['details']['height'], "assigned_batches": [batch.to_dict() for batch in job.assigned_batches]} for job in jobs]
                response = {
                    "total_jobs": total_jobs,
                    "total_pages": total_pages,
                    "current_page": page,
                    "benchmark_jobs": jobs_data
                }
                return JSONResponse(content=response, status_code=200)
            except SQLAlchemyError as e:
                logger.error(f"Database error on /get-benchmark-jobs: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /get-benchmark-jobs: {e}")
                raise HTTPException(status_code=400, detail="Invalid request parameters")
            
        @self.app.get("/get-batches", dependencies=[Depends(self.verify_api_key)])
        async def get_batches(limit: int = Query(10, gt=0), page: int = Query(1, gt=0)):
            try:
                total_batches = self.db_session.query(AssignedBatchModel).count()
                total_pages = (total_batches + limit - 1) // limit
                if page > total_pages and total_batches != 0:
                    raise HTTPException(status_code=400, detail="Page not found")
                
                batches = self.db_session.query(AssignedBatchModel).order_by(desc(AssignedBatchModel.submitted_timestamp)).offset((page - 1)*limit).limit(limit).all()
                batch_data = [batch.to_dict() for batch in batches]
                response = {
                    "total_batches": total_batches,
                    "total_pages": total_pages,
                    "current_page": page,
                    "batches": batch_data
                }
                return JSONResponse(content=response, status_code=200)
            except SQLAlchemyError as e:
                logger.error(f"Database error on /get-batches: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /get-batches: {e}")
                raise HTTPException(status_code=400, detail="Invalid request parameters")
            
        @self.app.get("/get-slaves", dependencies=[Depends(self.verify_api_key)])
        async def get_slaves(limit: int = Query(10, gt=0), page: int = Query(1, gt=0)):
            try:
                total_slaves = self.db_session.query(SlaveRegistryModel).count()
                total_pages = (total_slaves + limit - 1) // limit
                if page > total_pages and total_slaves != 0:
                    raise HTTPException(status_code=400, detail="Page not found")
                
                slaves = self.db_session.query(SlaveRegistryModel).order_by(desc(SlaveRegistryModel.registered_at)).offset((page - 1)*limit).limit(limit).all()
                slaves_data = [slave.to_dict() for slave in slaves]
                response = {
                    "total_slaves": total_slaves,
                    "total_pages": total_pages,
                    "current_page": page,
                    "slaves": slaves_data
                }
                return JSONResponse(content=response, status_code=200)
            except SQLAlchemyError as e:
                logger.error(f"Database error on /get-slaves: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"Unexpected error on /get-slaves: {e}")
                raise HTTPException(status_code=400, detail="Invalid request parameters")

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