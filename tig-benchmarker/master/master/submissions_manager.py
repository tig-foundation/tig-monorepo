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
class SubmitPrecommitRequest(FromDict):
    settings: BenchmarkSettings
    num_nonces: int

@dataclass
class SubmitBenchmarkRequest(FromDict):
    benchmark_id: str
    merkle_root: MerkleHash
    non_solution_nonces: Optional[List[int]]
    discarded_solution_nonces: Optional[List[int]]
    solution_nonces: Optional[List[int]]

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
                    WHERE merkle_root_ready
                        AND stopped IS NULL
                        AND benchmark_submitted IS NULL
                        AND (
                            benchmark_submit_time IS NULL 
                            OR ((EXTRACT(EPOCH FROM NOW()) * 1000)::BIGINT - benchmark_submit_time) > %s
                        ) 
                    ORDER BY block_started
                    LIMIT 1
                )
                RETURNING benchmark_id, num_nonces
            )
            SELECT
                A.benchmark_id, 
                A.num_nonces,
                B.merkle_root,
                B.solution_nonces,
                B.discarded_solution_nonces
            FROM updated A
            INNER JOIN job_data B
                ON A.benchmark_id = B.benchmark_id
            """,
            (CONFIG["time_between_resubmissions"],)
        )

        if benchmark_to_submit:
            benchmark_id = benchmark_to_submit["benchmark_id"]
            num_nonces = benchmark_to_submit["num_nonces"]
            merkle_root = benchmark_to_submit["merkle_root"] 
            non_solution_nonces = list(
                set(range(num_nonces)) - 
                set(benchmark_to_submit["solution_nonces"]) - 
                set(benchmark_to_submit["discarded_solution_nonces"])
            )
            solution_nonces = benchmark_to_submit["solution_nonces"]
            discarded_solution_nonces = benchmark_to_submit["discarded_solution_nonces"]

            max_size = max(len(non_solution_nonces), len(solution_nonces), len(discarded_solution_nonces))
            if len(solution_nonces) == max_size:
                solution_nonces = None
            elif len(discarded_solution_nonces) == max_size:
                discarded_solution_nonces = None
            else:
                non_solution_nonces = None

            self._post_thread("benchmark", SubmitBenchmarkRequest(
                benchmark_id=benchmark_id,
                merkle_root=merkle_root,
                non_solution_nonces=non_solution_nonces,
                solution_nonces=solution_nonces,
                discarded_solution_nonces=discarded_solution_nonces,
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
