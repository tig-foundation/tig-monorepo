import os
import logging
import random
import re
import signal
import threading
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from tig_benchmarker.database.init import SessionLocal, engine
from sqlalchemy.orm.attributes import flag_modified
from typing import Dict, List, Optional, Set
import datetime
from tig_benchmarker.database.models.index import JobModel, ConfigModel, SlaveModel, BatchModel
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, ValidationError, RootModel
from typing import Annotated
import time

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Batch(FromDict):
    benchmark_id: str
    start_nonce: int
    num_nonces: int
    settings: BenchmarkSettings
    sampled_nonces: List[int]
    wasm_vm_config: dict
    download_url: str
    rand_hash: str
    batch_size: int

@dataclass
class BatchResult(FromDict):
    merkle_root: MerkleHash
    solution_nonces: List[int]

@dataclass
class BatchMerkleProof(FromDict):
    merkle_proofs: List[MerkleProof]

@dataclass
class Slave(FromDict):
    name: str
    num_of_cpus: int
    num_of_threads: int
    memory: int

@dataclass
class SlaveConfig(FromDict):
    name_regex: str
    max_concurrent_batches: Dict[str, int]

@dataclass
class SlaveManagerConfig(FromDict):
    port: int
    time_before_batch_retry: int
    num_nonces_to_sample: float
    slaves: List[SlaveConfig]

@dataclass
class AssignedBatch(FromDict):
    benchmark_id: str
    batch_idx: int
    start_nonce: int
    num_nonces: int
    settings: BenchmarkSettings
    sampled_nonces: List[int]
    wasm_vm_config: dict
    download_url: str
    rand_hash: str
    batch_size: int
    assigned_slave: str
    submitted_timestamp: str  # ISO formatted string
    completed_timestamp: Optional[str] = None
    batch_result_id: Optional[int] = None
    

class Slave(RootModel[dict]):
    pass

class BatchResultData(RootModel[dict]):
    pass
class SlaveManager:
    def __init__(self, config: SlaveManagerConfig):
        self.config = config
        # Initialize database session
        self.db_session = SessionLocal()
        logger.info("SlaveManager initialized and connected to the database.")
        # Initialize FastAPI app
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/register-slave")
        def register_slave( slave_data: Slave):
            try:
                slave = slave_data.root

                logger.info(f"Registering: { slave.get("name")}: {slave_data}")

                name = slave.get("name")
                num_of_cpus = int(slave.get("num_of_cpus"))
                num_of_threads = int(slave.get("num_of_threads"))
                memory = int(slave.get("memory"))

                slaveModel = SlaveModel(
                    name=name,
                    num_of_cpus=num_of_cpus,
                    num_of_threads=num_of_threads, 
                    memory = memory
                )

                self.db_session.add(slaveModel)
                self.db_session.flush()

                default_config = {
                    "name_regex": str(slaveModel.id),
                    "max_concurrent_batches": {
                        "satisfiability": 1,
                        "vehicle_routing": 1,
                        "knapsack": 1,
                        "vector_search": 1
                    }
                }

                config = self.db_session.query(ConfigModel).first()
                if config:
                    config.config_data["slave_manager_config"]["slaves"].append(default_config)

                flag_modified(config, "config_data")

                self.db_session.commit()
                
                return JSONResponse(content={"detail": "OK", "id": slaveModel.id}, status_code=200)
            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error during register_slave: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
        
        @self.app.get("/get-batches")
        def get_batches(user_agent: Annotated[str | None, Header()], request: Request):
            # Extract User-Agent header to identify the slave
            slave_id = user_agent
            if not slave_id:
                logger.warning("Missing User-Agent header in get-batches request.")
                raise HTTPException(status_code=403, detail="User-Agent header is required.")

            # Find the matching slave configuration
            matching_slaves = [slave for slave in self.config.slaves if re.match(slave.name_regex, slave_id)]
            if not matching_slaves:
                logger.warning(f"Slave '{slave_id}' does not match any registered regex patterns.")
                raise HTTPException(status_code=403, detail="Unregistered slave.")

            slave = matching_slaves[0]
            
            now = int(time.time() * 1000)
            batches = []
            selected_challenge = None
            max_concurrent_batches = None
            
            try:
                with self.db_session.begin():
                    # Fetch jobs from the database that are pending (merkle_root is None)
                    pending_jobs = self.db_session.query(JobModel).filter_by(merkle_root=None).all()
                    slaveModel = self.db_session.query(SlaveModel).filter_by(id=slave_id).first()

                    if not slaveModel:
                        logger.warning(f"Slave '{slave_id}' requested batches but no matching slave found.")
                        return HTTPException(status_code=404, detail="Slave not Registered.")
                    
                    for job in pending_jobs:
                        
                        # Check if the job's challenge is handled by this slave
                        if job.challenge not in slave.max_concurrent_batches:
                            continue

                        if selected_challenge and job.challenge != selected_challenge:
                            continue  # Only handle one challenge at a time per request

                        # Get max concurrent batches for this challenge
                        max_concurrent_batches = slave.max_concurrent_batches[job.challenge]

                        # Count currently assigned batches for this challenge and slave
                        current_assigned = slaveModel.batches.count()

                        logger.info(f"Current assigned batches: {current_assigned}")

                        available_slots = max_concurrent_batches - current_assigned
                        if available_slots <= 0:
                            continue  # No available slots for this challenge

                        # Fetch available batches for this job
                        for batch_idx in range(job.num_batches):
                            if available_slots <= 0:
                                break  # Reached the max concurrent batches

                            start_nonce = batch_idx * job.batch_size
                            num_nonces = min(job.batch_size, job.num_nonces - start_nonce)

                            # Check if this batch is already assigned
                            existing_assignment = self.db_session.query(BatchModel).filter_by(
                                benchmark_id=job.benchmark_id,
                                start_nonce=start_nonce,
                            ).first()

                            logger.info(f"Current assigned batches: {current_assigned}")

                            if existing_assignment:
                                continue  # Skip if already completed
                                # if existing_assignment.completed_timestamp:

                            # Check if batch is ready
                            if not (
                                now - job.last_batch_retry_time[batch_idx] > self.config.time_before_batch_retry and
                                (
                                    job.batch_merkle_roots[batch_idx] is None or
                                    not job.sampled_nonces or
                                    not set(job.sampled_nonces[batch_idx]).issubset(job.merkle_proofs)
                                )
                            ):
                                continue  # Batch not ready
                            
                            # Assign the batch to this slave
                            assigned_timestamp = datetime.datetime.utcnow()

                            # assigned_batch = AssignedBatchModel(
                            #     benchmark_id=job.benchmark_id,
                            #     batch_idx=batch_idx,
                            #     assigned_slave=slave_id,
                            #     submitted_timestamp=assigned_timestamp,
                            #     completed_timestamp=None
                            #     # batch_result_id=None
                            # )
                            
                            # Prepare the Batch data to return
                         

                            batchSettings = BenchmarkSettings.from_dict(job.settings)

                            batchModel = BatchModel(
                                benchmark_id=job.benchmark_id,
                                slave_id=slave_id,
                                start_nonce=start_nonce,
                                num_nonces=num_nonces,
                                settings=batchSettings.to_dict(),
                                wasm_vm_config=job.wasm_vm_config,
                                download_url=job.download_url,
                                rand_hash=job.rand_hash,
                                batch_size=job.batch_size,
                                sampled_nonces=job.sampled_nonces[batch_idx] if batch_idx in job.sampled_nonces else [],
                            )
                            
                            self.db_session.add(batchModel)

                            batch = Batch(
                                benchmark_id=job.benchmark_id,
                                start_nonce=start_nonce,
                                num_nonces=num_nonces,
                                settings=batchSettings,
                                sampled_nonces=job.sampled_nonces[batch_idx] if batch_idx in job.sampled_nonces else [],
                                wasm_vm_config=job.wasm_vm_config,
                                download_url=job.download_url,
                                rand_hash=job.rand_hash,
                                batch_size=job.batch_size
                            )
                            
                            batches.append(batch.to_dict())
                            selected_challenge = job.challenge
                            available_slots -= 1
                    
                    logger.info(f"Batches: {batches}")

                    if not batches:
                        logger.debug(f"Slave '{slave_id}' requested batches but none are available.")
                        return JSONResponse(content={"detail": "No batches available"}, status_code=503)
                    else:
                        # Commit the assignments
                        self.db_session.commit()
                        logger.debug(f"Slave '{slave_id}' received {len(batches)} batch(es) for challenge '{selected_challenge}'.")
                        return JSONResponse(content=batches, status_code=200)
            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error during get_batches: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
        
        @self.app.post("/submit-batch-result/{batch_id}")
        def submit_batch_result(batch_id: str, batch_result_data: BatchResultData, request: Request, user_agent: Annotated[str | None, Header()]):
            Session = sessionmaker(bind=engine)
            session = Session()
            # Extract User-Agent header to identify the slave
            slave_name = user_agent
            if not slave_name:
                logger.warning("Missing User-Agent header in submit-batch-result request.")
                raise HTTPException(status_code=403, detail="User-Agent header is required.")

            # Find the matching slave configuration
            matching_slaves = [slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)]
            if not matching_slaves:
                logger.warning(f"Slave '{slave_name}' does not match any registered regex patterns.")
                raise HTTPException(status_code=403, detail="Unregistered slave.")

            slave = matching_slaves[0]  # Assuming one match is sufficient
            try:
                benchmark_id, start_nonce_str = batch_id.split("_")
                start_nonce = int(start_nonce_str)
            except ValueError:
                logger.warning(f"Invalid batch_id format: {batch_id}")
                raise HTTPException(status_code=400, detail="Invalid batch_id format.")
            
            try:
                with session.begin():
                    # Fetch the job from the database
                    job = session.query(JobModel).filter_by(benchmark_id=benchmark_id).first()
                    if not job:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result for non-existent job '{benchmark_id}'.")
                        raise HTTPException(status_code=400, detail="Invalid benchmark_id.")

                    # Determine the batch index
                    batch_idx = start_nonce // job.batch_size
                    if batch_idx >= job.num_batches:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result with invalid start_nonce '{start_nonce}'.")
                        raise HTTPException(status_code=400, detail="Invalid start_nonce.")
                    
                    # Fetch the AssignedBatchModel
                    assigned_batch = session.query(AssignedBatchModel).filter_by(
                        benchmark_id=benchmark_id,
                        batch_idx=batch_idx,
                        assigned_slave=slave_name,
                        completed_timestamp=None
                    ).first()

                    logger.info(f"Assigned batch: {assigned_batch}")
                    
                    if not assigned_batch:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result for an unassigned or already completed batch '{batch_id}'.")
                        raise HTTPException(status_code=400, detail="Batch not assigned or already completed.")
                    
                    # Get body from request
                    result = BatchResult.from_dict(batch_result_data.root)
                    
                    # Update JobModel's batch_merkle_roots
                    job.batch_merkle_roots[batch_idx] = str(result.merkle_root.to_str())  # Assuming MerkleHash can be converted to string

                    # Update solution_nonces
                    job.solution_nonces = list(set(job.solution_nonces + result.solution_nonces))
                        
                    # Create BatchResultModel
                    batch_result = BatchResultModel(
                        benchmark_id=benchmark_id,
                        start_nonce=start_nonce,
                        merkle_root=str(result.merkle_root.to_str()),
                        solution_nonces=result.solution_nonces,
                        merkle_proofs=[],
                        assigned_batch=assigned_batch  # Link to AssignedBatchModel
                    )
                    session.add(batch_result)
                    session.flush() # To get batch_result.id
                    
                    # Update AssignedBatchModel with completed_timestamp and batch_result_id
                    assigned_batch.completed_timestamp = datetime.datetime.utcnow()
                    # assigned_batch.batch_result_id = batch_result.id

                    # TODO: Filter sampled nonces based on start nonce and batch size
                    if job.sampled_nonces is None:
                        non_solution_nonces = list(set(range(start_nonce, start_nonce + job.batch_size)) - set(result.solution_nonces))
                        random.shuffle(result.solution_nonces)
                        random.shuffle(non_solution_nonces)
                        # num_nonces_to_sample = int(len(result.solution_nonces) * self.config.num_nonces_to_sample)
                        sampled = (result.solution_nonces + non_solution_nonces)[:10]
                    else:
                        sampled = job.sampled_nonces

                    logger.info(f"Sampled nonces: {sampled}")
                    
                    # Commit the transaction
                    session.commit()

                    logger.debug(f"Slave '{slave_name}' submitted batch result for benchmark '{benchmark_id}', start_nonce '{start_nonce}'.")
                    return JSONResponse(content={"detail": "OK", "sample_nonces": sampled}, status_code=200)
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error during submit_batch_result: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
            except HTTPException as he:
                raise he
            except Exception as e:
                session.rollback()
                logger.error(f"Unexpected error during submit_batch_result: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
            
        @self.app.post('/submit-merkle-proofs/{batch_id}')
        def get_merkle_proofs(batch_id, request: Request):
            # Extract User-Agent header to identify the slave
            slave_name = request.headers.get('User-Agent')
            if not slave_name:
                logger.warning("Missing User-Agent header in submit-batch-result request.")
                raise HTTPException(status_code=403, detail="User-Agent header is required.")

            # Find the matching slave configuration
            matching_slaves = [slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)]
            if not matching_slaves:
                logger.warning(f"Slave '{slave_name}' does not match any registered regex patterns.")
                raise HTTPException(status_code=403, detail="Unregistered slave.")

            slave = matching_slaves[0]  # Assuming one match is sufficient
            try:
                benchmark_id, start_nonce_str = batch_id.split("_")
                start_nonce = int(start_nonce_str)
            except ValueError:
                logger.warning(f"Invalid batch_id format: {batch_id}")
                raise HTTPException(status_code=400, detail="Invalid batch_id format.")
            
            try:
                with self.db_session.begin():
                    # Fetch the job from the database
                    job = self.db_session.query(JobModel).filter_by(benchmark_id=benchmark_id).first()
                    if not job:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result for non-existent job '{benchmark_id}'.")
                        raise HTTPException(status_code=400, detail="Invalid benchmark_id.")
                    
                    # Get Request body
                    result = BatchMerkleProof.from_dict(request.json())
                    
                    # Set the batch result's merkle proofs
                    job.merkle_proofs = result.merkle_proofs

                    # Commit the transaction
                    self.db_session.commit()
                    logger.debug(f"Slave '{slave_name}' submitted batch result for benchmark '{benchmark_id}', start_nonce '{start_nonce}'.")
                    return JSONResponse(content={"detail": "OK"}, status_code=200)

            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error during submit_batch_result: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
            except HTTPException as he:
                raise he
            except Exception as e:
                self.db_session.rollback()
                logger.error(f"Unexpected error during submit_batch_result: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
    
    def start(self, host="0.0.0.0"):
        def run():
            uvicorn.run(self.app, host=host, port=self.config.port)
            
        # Start the FastAPI server in a separate thread
        server_thread = threading.Thread(target=run, daemon=True)
        server_thread.start()
        logger.info(f"SlaveManager started on 0.0.0.0:{self.config.port}")

        # Keep the main thread alive
        # try:
        #     while True:
        #         signal.pause()
        # except KeyboardInterrupt:
        #     logger.info("Shutting down SlaveManager.")
        #     self.db_session.close()
        #     logger.info("SlaveManager shut down successfully.")