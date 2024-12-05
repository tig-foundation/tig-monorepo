import requests
import threading
import logging
import json
import os
from extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Union
from extensions.sql import db_conn
from extensions.client_manager import get_config

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
    def __init__(self, api_url: str, api_key: str, jobs: List[Job]):
        self.jobs = jobs
        self.api_url = api_url
        self.api_key = api_key
        
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
        
        resp = requests.post(f"{self.api_url}/submit-{submission_type}", json=req.to_dict(), headers=headers)
        if resp.status_code == 200:
            logger.info(f"submitted {submission_type} successfully")
        elif resp.headers.get("Content-Type") == "text/plain":
            logger.error(f"status {resp.status_code} when submitting {submission_type}: {resp.text}")
        else:
            logger.error(f"status {resp.status_code} when submitting {submission_type}")

    def _post_thread(self, submission_type: str, req: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]):
        thread = threading.Thread(target=self._post, args=(submission_type, req))
        thread.start()

    def run(self, submit_precommit_req: Optional[SubmitPrecommitRequest]):
        now = int(time.time() * 1000)
        if submit_precommit_req is None:
            logger.debug("no precommit to submit")
        else:
            self._post_thread("precommit", submit_precommit_req)

        config = get_config()["submissions_manager_config"]
        benchmark_to_submit = db_conn.fetch_one(
            """
            WITH updated AS (
                UPDATE jobs
                SET last_submit_time = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id IN (
                    SELECT benchmark_id 
                    FROM jobs
                    WHERE (
                        last_submit_time IS NULL 
                        OR (EXTRACT(EPOCH FROM NOW()) - last_submit_time) > %s
                    ) AND merkle_root IS NOT NULL
                    ORDER BY block_started
                    LIMIT 1
                )
                RETURNING benchmark_id, merkle_root, solution_nonces
            )
            SELECT benchmark_id, merkle_root, solution_nonces FROM updated
            """,
            (config["time_between_retries"],)
        )

        if benchmark_to_submit:
            benchmark_id = benchmark_to_submit["benchmark_id"]
            merkle_root = benchmark_to_submit["merkle_root"] 
            solution_nonces = benchmark_to_submit["solution_nonces"]

            self._post_thread("benchmark", SubmitBenchmarkRequest(
                benchmark_id=benchmark_id,
                merkle_root=merkle_root,
                solution_nonces=solution_nonces
            ))
        else:
            logger.debug("no benchmark to submit")

        proof_to_submit = db_conn.fetch_one(
            """
            WITH updated AS (
                UPDATE jobs
                SET last_proof_submit_time = EXTRACT(EPOCH FROM NOW())
                WHERE benchmark_id IN (
                    SELECT benchmark_id 
                    FROM jobs
                    WHERE (
                        last_proof_submit_time IS NULL 
                        OR (EXTRACT(EPOCH FROM NOW()) - last_proof_submit_time) > %s
                    ) AND merkle_proofs IS NOT NULL
                    ORDER BY block_started
                    LIMIT 1
                )
                RETURNING benchmark_id, merkle_proofs
            )
            SELECT benchmark_id, merkle_proofs FROM updated
            """,
            (config["time_between_retries"],)
        )

        if proof_to_submit:
            benchmark_id = proof_to_submit["benchmark_id"]
            merkle_proofs = proof_to_submit["merkle_proofs"]

            self._post_thread("proof", SubmitProofRequest(
                benchmark_id=benchmark_id,
                merkle_proofs=merkle_proofs
            ))
        else:
            logger.debug("no proof to submit")
