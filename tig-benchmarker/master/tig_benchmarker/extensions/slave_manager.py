import os
import logging
import random
import re
import signal
import threading
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from database.init import SessionLocal
from typing import Dict, List, Optional, Set
import datetime
from database.models.index import JobModel, BatchResultModel, AssignedBatchModel, SlaveRegistryModel

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
    
class SlaveManager:
    def __init__(self, config: SlaveManagerConfig):
        self.config = config
        # Initialize database session
        self.db_session = SessionLocal()
        logger.info("SlaveManager initialized and connected to the database.")
        # Initialize FastAPI app
        self.app = FastAPI()
        self.setup_routes()
    
    def start(self):
        @self.app.post("/register-slave")
        def register_slave(slave: Slave, request: Request):
            # Extract User-Agent header to identify the slave
            slave_name = request.headers.get('User-Agent')
            if not slave_name:
                logger.warning("Missing User-Agent header in register-slave request.")
                raise HTTPException(status_code=403, detail="User-Agent header is required.")
            
            # Save slave details in the database
            slave = SlaveRegistryModel(
                name=slave.name,
                num_of_cpus=slave.num_of_cpus,
                num_of_threads=slave.num_of_threads,
            )
            self.db_session.add(slave)
            self.db_session.commit()
            
            return JSONResponse(content={"detail": "OK", "id": slave.id}, status_code=200)


        @self.app.get("/get-batches")
        def get_batches(request: Request):
            # Extract User-Agent header to identify the slave
            slave_name = request.headers.get('User-Agent')
            if not slave_name:
                logger.warning("Missing User-Agent header in get-batches request.")
                raise HTTPException(status_code=403, detail="User-Agent header is required.")

            # Find the matching slave configuration
            matching_slaves = [slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)]
            if not matching_slaves:
                logger.warning(f"Slave '{slave_name}' does not match any registered regex patterns.")
                raise HTTPException(status_code=403, detail="Unregistered slave.")

            slave = matching_slaves[0]
            
            now = int(datetime.time.time() * 1000)
            batches = []
            selected_challenge = None
            max_concurrent_batches = None
            
            try:
                with self.db_session.begin():
                    # Fetch jobs from the database that are pending (merkle_root is None)
                    pending_jobs = self.db_session.query(JobModel).filter_by(merkle_root=None).all()
                    
                    for job in pending_jobs:
                        # Check if the job's challenge is handled by this slave
                        if job.challenge not in slave.max_concurrent_batches:
                            continue

                        if selected_challenge and job.challenge != selected_challenge:
                            continue  # Only handle one challenge at a time per request

                        # Get max concurrent batches for this challenge
                        max_concurrent_batches = slave.max_concurrent_batches[job.challenge]
                        
                        # Count currently assigned batches for this challenge and slave
                        current_assigned = self.db_session.query(AssignedBatchModel).filter_by(
                            benchmark_id=job.benchmark_id,
                            assigned_slave=slave_name,
                            completed_timestamp=None
                        ).count()

                        available_slots = max_concurrent_batches - current_assigned
                        if available_slots <= 0:
                            continue  # No available slots for this challenge

                        # Fetch available batches for this job
                        for batch_idx in range(job.num_batches):
                            if available_slots <= 0:
                                break  # Reached the max concurrent batches
                            
                            # Check if this batch is already assigned
                            existing_assignment = self.db_session.query(AssignedBatchModel).filter_by(
                                benchmark_id=job.benchmark_id,
                                batch_idx=batch_idx
                            ).first()
                            if existing_assignment:
                                continue  # Batch already assigned
                            
                            # Check if batch is ready
                            if not (
                                now - job.last_batch_retry_time[batch_idx] > self.config.time_before_batch_retry and
                                (
                                    job.batch_merkle_roots[batch_idx] is None or
                                    not set(job.sampled_nonces_by_batch_idx.get(batch_idx, [])).issubset(job.merkle_proofs)
                                )
                            ):
                                continue  # Batch not ready
                            
                            # Assign the batch to this slave
                            assigned_timestamp = datetime.datetime.utcnow()
                            assigned_batch = AssignedBatchModel(
                                benchmark_id=job.benchmark_id,
                                batch_idx=batch_idx,
                                assigned_slave=slave_name,
                                submitted_timestamp=assigned_timestamp,
                                completed_timestamp=None,
                                batch_result_id=None
                            )
                            self.db_session.add(assigned_batch)
                            
                            # Prepare the Batch data to return
                            start_nonce = batch_idx * job.batch_size
                            num_nonces = min(job.batch_size, job.num_nonces - start_nonce)
                            
                            batch = Batch(
                                benchmark_id=job.benchmark_id,
                                start_nonce=start_nonce,
                                num_nonces=num_nonces,
                                settings=BenchmarkSettings.from_dict(job.settings),
                                sampled_nonces=job.sampled_nonces_by_batch_idx.get(batch_idx, []),
                                wasm_vm_config=job.wasm_vm_config,
                                download_url=job.download_url,
                                rand_hash=job.rand_hash,
                                batch_size=job.batch_size
                            )
                            
                            batches.append(batch.to_dict())
                            selected_challenge = job.challenge
                            available_slots -= 1
                    
                    if not batches:
                        logger.debug(f"Slave '{slave_name}' requested batches but none are available.")
                        return JSONResponse(content={"detail": "No batches available"}, status_code=503)
                    else:
                        # Commit the assignments
                        self.db_session.commit()
                        logger.debug(f"Slave '{slave_name}' received {len(batches)} batch(es) for challenge '{selected_challenge}'.")
                        return JSONResponse(content=batches, status_code=200)
            except SQLAlchemyError as e:
                self.db_session.rollback()
                logger.error(f"Database error during get_batches: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")
        
        @self.app.post("/submit-batch-result/{batch_id}")
        def submit_batch_result(batch_id: str, result: BatchResult, request: Request):
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

                    # Determine the batch index
                    batch_idx = start_nonce // job.batch_size
                    if batch_idx >= job.num_batches:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result with invalid start_nonce '{start_nonce}'.")
                        raise HTTPException(status_code=400, detail="Invalid start_nonce.")
                    
                    # Fetch the AssignedBatchModel
                    assigned_batch = self.db_session.query(AssignedBatchModel).filter_by(
                        benchmark_id=benchmark_id,
                        batch_idx=batch_idx,
                        assigned_slave=slave_name,
                        completed_timestamp=None
                    ).first()
                    
                    if not assigned_batch:
                        logger.warning(f"Slave '{slave_name}' submitted a batch result for an unassigned or already completed batch '{batch_id}'.")
                        raise HTTPException(status_code=400, detail="Batch not assigned or already completed.")
                    
                    # Update JobModel's batch_merkle_roots
                    job.batch_merkle_roots[batch_idx] = str(result.merkle_root)  # Assuming MerkleHash can be converted to string

                    # Update solution_nonces
                    job.solution_nonces = list(set(job.solution_nonces + result.solution_nonces))

                    # Update batch_merkle_proofs
                    for proof in result.merkle_proofs:
                        job.batch_merkle_proofs[str(proof.leaf.nonce)] = proof.to_dict()
                        
                    # Create BatchResultModel
                    batch_result = BatchResultModel(
                        benchmark_id=benchmark_id,
                        start_nonce=start_nonce,
                        merkle_root=str(result.merkle_root),
                        solution_nonces=result.solution_nonces,
                        merkle_proofs=[proof.to_dict() for proof in result.merkle_proofs],
                        assigned_batch=assigned_batch  # Link to AssignedBatchModel
                    )
                    self.db_session.add(batch_result)
                    self.db_session.flush() # To get batch_result.id
                    
                    # Update AssignedBatchModel with completed_timestamp and batch_result_id
                    assigned_batch.completed_timestamp = datetime.datetime.utcnow()
                    assigned_batch.batch_result_id = batch_result.id

                    
                    if job.sample_nonces is None:
                        non_solution_nonces = list(set(range(start_nonce, start_nonce + job.batch_size)) - set(result.solution_nonces))
                        random.shuffle(result.solution_nonces)
                        random.shuffle(non_solution_nonces)
                        num_nonces_to_sample = int(len(result.solution_nonces) * self.config.num_nonces_to_sample)
                        sampled = (result.solution_nonces + non_solution_nonces)[:num_nonces_to_sample]
                    else:
                        sampled = job.sample_nonces
                    
                    # Commit the transaction
                    self.db_session.commit()

                    logger.debug(f"Slave '{slave_name}' submitted batch result for benchmark '{benchmark_id}', start_nonce '{start_nonce}'.")
                    return JSONResponse(content={"detail": "OK", "sample_nonces": sampled}, status_code=200)
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
            
        @self.app.post('/merkle-proofs/{batch_id}')
        def get_merkle_proofs(batch_id, result: BatchMerkleProof, request: Request):
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
    
        def start_server(app: FastAPI):
            uvicorn.run(app, host="0.0.0.0", port=self.config.port)
            
        # Start the FastAPI server in a separate thread
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        logger.info(f"Webserver started on 0.0.0.0:{self.config.port}")

        # Keep the main thread alive
        try:
            while True:
                signal.pause()
        except KeyboardInterrupt:
            logger.info("Shutting down SlaveManager.")
            self.db_session.close()
            logger.info("SlaveManager shut down successfully.")