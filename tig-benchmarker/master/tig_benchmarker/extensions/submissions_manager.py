from datetime import datetime
import logging
import json
import os
import requests
from tig_benchmarker.extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Union

import threading

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from tig_benchmarker.database.init import SessionLocal, engine
from tig_benchmarker.database.models.index import (
    JobModel,
    PrecommitRequestModel,
    BenchmarkRequestModel,
    ProofRequestModel
)

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class SubmitPrecommitRequest(FromDict):
    settings: BenchmarkSettings
    num_nonces: int

@dataclass
class SubmitBenchmarkRequest(FromDict):
    benchmark_id: str
    merkle_root: MerkleHash
    solution_nonces: Set[int]

@dataclass
class SubmitProofRequest(FromDict):
    benchmark_id: str
    merkle_proofs: List[MerkleProof]

@dataclass
class SubmissionsManagerConfig(FromDict):
    time_between_retries: int

class SubmissionsManager:
    def __init__(self, config: SubmissionsManagerConfig, api_url: str, api_key: str):
        self.config = config
        self.api_url = api_url
        self.api_key = api_key
        self.db_session = SessionLocal()
        logger.info("SubmissionsManager initialized and connected to the database.")
        
    def _post(self, submission_type: str, req: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]):
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "tig-benchmarker-py/v0.2"
        }
        if submission_type == "precommit":
            logger.info(f"submitting {submission_type}")
        else:
            logger.info(f"submitting {submission_type} '{req.benchmark_id}'")
        logger.debug(f"{req}")

        try:
            response = requests.post(f"{self.api_url}/submit-{submission_type}", json=req.to_dict(), headers=headers)
            text = response.text
            if response.status_code == 200:
                logger.info(f"submitted {submission_type} successfully")
            elif response.headers.get("Content-Type") == "text/plain":
                logger.error(f"status {response.status_code} when submitting {submission_type}: {text}")
            else:
                logger.error(f"status {response.status_code} when submitting {submission_type}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting {submission_type}: {e}")
   
    def handle_precommit(self, submit_precommit_req: SubmitPrecommitRequest):
        try:
            # Create a PrecommitRequestModel entry
            precommit_entry = PrecommitRequestModel(
                job_id=submit_precommit_req.settings.challenge_id,  # Assuming challenge_id corresponds to job_id
                settings=submit_precommit_req.settings.to_dict(),
                num_nonces=submit_precommit_req.num_nonces,
                timestamp=datetime.utcnow()
            )
            logger.info(f"Precommit Entry: {precommit_entry}")
            # self.db_session.add(precommit_entry)

            # Submit the precommit request
            success = self._post("precommit", submit_precommit_req)

            if success:
                self.db_session.commit()
                logger.debug(f"PrecommitRequestModel entry created with id {precommit_entry.id}")
                logger.info(f"Precommit request for job_id '{submit_precommit_req.settings.challenge_id}' submitted successfully.")
            else:
                logger.error(f"Failed to submit precommit request for job_id '{submit_precommit_req.settings.challenge_id}'.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during precommit submission: {e}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error during precommit submission: {e}")

    def handle_benchmark_submissions(self, now: int):
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            # Fetch jobs eligible for benchmark submission
            eligible_jobs = session.query(JobModel).filter(
                JobModel.merkle_root.isnot(None),
                func.coalesce(func.jsonb_array_length(JobModel.sampled_nonces), 0) == 0,
                JobModel.last_benchmark_submit_time < now - self.config.time_between_retries
            ).all()

            if not eligible_jobs:
                logger.debug("No Benchmark submissions to process")
                return

            for job in eligible_jobs:
                logger.debug(f"Processing Benchmark submission for job_id '{job.benchmark_id}'")

                # Create a BenchmarkRequestModel entry
                benchmark_entry = BenchmarkRequestModel(
                    job_id=job.benchmark_id,
                    benchmark_id=job.benchmark_id,  # Assuming benchmark_id is same as job_id
                    merkle_root=str(job.merkle_root),
                    solution_nonces=list(job.solution_nonces),
                    timestamp=datetime.utcnow()
                )
                logger.info(f"Benchmark Entry: {benchmark_entry}")
                # self.db_session.add(benchmark_entry)
                
                # Update job's last_benchmark_submit_time
                job.last_benchmark_submit_time = now

                # Prepare the submission request
                submit_benchmark_req = SubmitBenchmarkRequest(
                    benchmark_id=job.benchmark_id,
                    merkle_root=job.merkle_root,
                    solution_nonces=job.solution_nonces
                )

                # Submit the benchmark request
                success = self._post("benchmark", submit_benchmark_req)

                if success:
                    self.db_session.commit()
                    logger.info(f"Benchmark request for job_id '{job.benchmark_id}' submitted successfully.")
                else:
                    logger.error(f"Failed to submit benchmark request for job_id '{job.benchmark_id}'.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during benchmark submissions: {e}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error during benchmark submissions: {e}")

    def handle_proof_submissions(self, now: int):
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            # Fetch jobs eligible for proof submission
            eligible_jobs = session.query(JobModel).filter(
                func.coalesce(func.jsonb_array_length(JobModel.sampled_nonces), 0) != 0,
                func.coalesce(func.jsonb_array_length(JobModel.merkle_proofs), 0) == func.coalesce(func.jsonb_array_length(JobModel.sampled_nonces), 0),
                JobModel.last_proof_submit_time < now - self.config.time_between_retries
            ).all()

             

            if not eligible_jobs:
                logger.debug("No Proof submissions to process")
                return

            for job in eligible_jobs:
                logger.debug(f"Processing Proof submission for job_id '{job.benchmark_id}'")

                # # Create a ProofRequestModel entry
                proof_entry = ProofRequestModel(
                    job_id=job.benchmark_id,
                    benchmark_id=job.benchmark_id,  # Assuming benchmark_id is same as job_id
                    merkle_proofs=list(job.merkle_proofs.values()),
                    timestamp=datetime.utcnow()
                )
                logger.info(f"Proof Entry: {proof_entry}")
                # self.db_session.add(proof_entry)

                # Update job's last_proof_submit_time
                job.last_proof_submit_time = now

                # Prepare the submission request
                submit_proof_req = SubmitProofRequest(
                    benchmark_id=job.benchmark_id,
                    merkle_proofs=list(job.merkle_proofs.values())
                )

                # Submit the proof request
                success = self._post("proof", submit_proof_req)

                if success:
                    self.db_session.commit()
                    logger.info(f"Proof request for job_id '{job.benchmark_id}' submitted successfully.")
                else:
                    logger.error(f"Failed to submit proof request for job_id '{job.benchmark_id}'.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during proof submissions: {e}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error during proof submissions: {e}")
    
    def run(self, submit_precommit_req: Optional[SubmitPrecommitRequest]):
        now = int(time.time() * 1000)
        # Submit Precommit Request if provided
        if submit_precommit_req:
            logger.debug("Processing Precommit Request")
            precommit_thread = threading.Thread(target=self.handle_precommit, args=(submit_precommit_req,))
            precommit_thread.start()
        else:
            logger.debug("No Precommit Request to process")

        # Submit Benchmark Requests
        benchmark_thread = threading.Thread(target=self.handle_benchmark_submissions, args=(now,))
        benchmark_thread.start()
        
        # Submit Proof Requests
        proof_thread = threading.Thread(target=self.handle_proof_submissions, args=(now,))
        proof_thread.start()