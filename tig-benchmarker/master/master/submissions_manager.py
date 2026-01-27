import brotli
import requests
import threading
import logging
import os
from common.structs import *
from common.utils import *
from typing import Union, Set, List, Dict, Optional
from master.sql import get_db_conn
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class TrackSettings(FromDict):
    num_bundles: int
    hyperparameters: Optional[dict]
    fuel_budget: int

@dataclass
class SubmitPrecommitRequest(FromDict):
    settings: BenchmarkSettings
    track_settings: Dict[str, TrackSettings]

@dataclass
class SubmitBenchmarkRequest(FromDict):
    benchmark_id: str
    stopped: bool
    merkle_root: Optional[MerkleHash]
    solution_quality: Optional[List[int]]

@dataclass
class SubmitProofRequest(FromDict):
    benchmark_id: str
    merkle_proofs: List[MerkleProof]

class SubmissionsManager:
    def __init__(self):
        pass
        
    def _post(self, submission_type: str, req: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]):
        api_key = CONFIG["api_key"]
        api_url = CONFIG["api_url"]

        headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "tig-benchmarker-py/v0.2"
        }
        if submission_type == "precommit":
            logger.info(f"submitting {submission_type}")
        else:
            logger.info(f"submitting {submission_type} '{req.benchmark_id}'")
        logger.debug(f"{req}")
        
        data = jsonify(req)
        if len(data) > 10 * 1024:
            headers.update({
                'Content-Encoding': 'br',
                'Accept-Encoding': 'br',
            })
            data = brotli.compress(data.encode())
        resp = requests.post(f"{api_url}/submit-{submission_type}", data=data, headers=headers)
        if resp.status_code == 200:
            logger.info(f"submitted {submission_type} successfully")
        elif resp.headers.get("Content-Type") == "text/plain":
            logger.error(f"status {resp.status_code} when submitting {submission_type}: {resp.text}")
        else:
            logger.error(f"status {resp.status_code} when submitting {submission_type}")

    def _post_thread(self, submission_type: str, req: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]):
        thread = threading.Thread(target=self._post, args=(submission_type, req))
        thread.start()

    def on_new_block(self, 
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        **kwargs
    ):
        if len(benchmarks) > 0:
            get_db_conn().execute(
                """
                UPDATE job
                SET benchmark_submitted = true
                WHERE benchmark_id IN %s
                """,
                (tuple(benchmarks),)
            )
        
        if len(proofs) > 0:
            get_db_conn().execute(
                """
                UPDATE job
                SET proof_submitted = true
                WHERE benchmark_id IN %s
                """,
                (tuple(proofs),)
            )

    def run(self, submit_precommit_req: Optional[SubmitPrecommitRequest]):
        now = int(time.time() * 1000)
        if submit_precommit_req is None:
            logger.debug("no precommit to submit")
        else:
            self._post_thread("precommit", submit_precommit_req)

        benchmark_to_submit = get_db_conn().fetch_one(
            """
            WITH updated AS (
                UPDATE job
                SET benchmark_submit_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                WHERE benchmark_id IN (
                    SELECT benchmark_id 
                    FROM job
                    WHERE (merkle_root_ready OR stopped)
                        AND end_time IS NULL
                        AND benchmark_submitted IS NULL
                        AND (
                            benchmark_submit_time IS NULL 
                            OR ((EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT - benchmark_submit_time) > %s
                        ) 
                    ORDER BY block_started
                    LIMIT 1
                )
                RETURNING benchmark_id, stopped
            )
            SELECT
                A.benchmark_id, 
                A.stopped,
                B.merkle_root,
                B.solution_quality
            FROM updated A
            INNER JOIN job_data B
                ON A.benchmark_id = B.benchmark_id
            """,
            (CONFIG["time_between_resubmissions"],)
        )

        if benchmark_to_submit:
            benchmark_id = benchmark_to_submit["benchmark_id"]
            merkle_root = benchmark_to_submit["merkle_root"] 
            solution_quality = benchmark_to_submit["solution_quality"]

            if benchmark_to_submit["stopped"]:
                self._post_thread("benchmark", SubmitBenchmarkRequest(
                    benchmark_id=benchmark_id,
                    stopped=True,
                    merkle_root=None,
                    solution_quality=None,
                ))
            else:
                self._post_thread("benchmark", SubmitBenchmarkRequest(
                    benchmark_id=benchmark_id,
                    stopped=False,
                    merkle_root=merkle_root,
                    solution_quality=solution_quality,
                ))
        else:
            logger.debug("no benchmark to submit")

        proof_to_submit = get_db_conn().fetch_one(
            """
            WITH updated AS (
                UPDATE job
                SET proof_submit_time = (EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT
                WHERE benchmark_id IN (
                    SELECT benchmark_id 
                    FROM job
                    WHERE merkle_proofs_ready
                        AND stopped IS NULL
                        AND proof_submitted IS NULL
                        AND (
                            proof_submit_time IS NULL 
                            OR ((EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT - proof_submit_time) > %s
                        )
                    ORDER BY block_started
                    LIMIT 1
                )
                RETURNING benchmark_id
            )
            SELECT 
                B.benchmark_id, 
                B.merkle_proofs 
            FROM updated A
            INNER JOIN job_data B
                ON A.benchmark_id = B.benchmark_id
            """,
            (CONFIG["time_between_resubmissions"],)
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
