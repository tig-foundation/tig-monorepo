import requests
import threading
import logging
import json
import os
from tig_benchmarker.extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Union

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
    def __init__(self, config: SubmissionsManagerConfig, api_url: str, api_key: str, jobs: List[Job]):
        self.config = config
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

        for job in self.jobs:
            if (
                job.merkle_root is not None and
                len(job.sampled_nonces) == 0 and
                now - job.last_benchmark_submit_time > self.config.time_between_retries
            ):
                job.last_benchmark_submit_time = now
                self._post_thread("benchmark", SubmitBenchmarkRequest(
                    benchmark_id=job.benchmark_id,
                    merkle_root=job.merkle_root.to_str(),
                    solution_nonces=job.solution_nonces
                ))
                break
        else:
            logger.debug("no benchmark to submit")

        for job in self.jobs:
            if (
                len(job.sampled_nonces) > 0 and
                len(job.merkle_proofs) == len(job.sampled_nonces) and
                now - job.last_proof_submit_time > self.config.time_between_retries
            ):
                job.last_proof_submit_time = now
                self._post_thread("proof", SubmitProofRequest(
                    benchmark_id=job.benchmark_id,
                    merkle_proofs=list(job.merkle_proofs.values())
                ))
                break
        else:
            logger.debug("no proof to submit")