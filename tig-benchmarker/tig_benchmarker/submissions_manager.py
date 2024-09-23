import aiohttp
import time
from tig_benchmarker.job_manager import *
from tig_benchmarker.data_fetcher import *
from tig_benchmarker.structs import *
from typing import Union

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

class SubmissionsManager:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.time_benchmark_submitted = {}
        self.time_proof_submitted = {}
    
    async def post(
        self,
        req: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]
    ):
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "tig-benchmarker-py/v0.2"
        }

        now = int(time.time() * 1000)
        if isinstance(req, SubmitPrecommitRequest):
            submission_type = "precommit"
        elif isinstance(req, SubmitBenchmarkRequest):
            submission_type = "benchmark"
            self.time_benchmark_submitted[req.benchmark_id] = now
        elif isinstance(req, SubmitProofRequest):
            submission_type = "proof"
            self.time_proof_submitted[req.benchmark_id] = now
        else:
            raise ValueError(f"Invalid request type: {type(req)}")

        d = req.to_dict()
        if submission_type == "proof":
            print(f"[submissions_manager] submitting {submission_type}")
        else:
            print(f"[submissions_manager] submitting {submission_type}: {d}")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/submit-{submission_type}", json=d, headers=headers) as resp:
                text = await resp.text()
                if resp.status == 200:
                    print(f"[submissions_manager] submitted {submission_type} successfully")
                else:
                    if resp.headers.get("Content-Type") == "text/plain":
                        print(f"[submissions_manager] failed to submit {submission_type}: (status: {resp.status}, body: {text})")
                    else:
                        print(f"[submissions_manager] failed to submit {submission_type}: (status: {resp.status})")
                    submission_type != "precommit" and print(f"[submissions_manager] requeueing request")
                print(f"[submissions_manager] done. took {(int(time.time() * 1000) - now) / 1000} seconds")
                return text

    def find_benchmark_to_submit(self, query_data: QueryData, job_manager: JobManager) -> Optional[Job]:
        now = int(time.time() * 1000)
        ready_to_submit = sorted([
            (
                self.time_benchmark_submitted.get(benchmark_id, 0),
                benchmark_id
            )
            for benchmark_id, job in job_manager.jobs.items()
            if (
                benchmark_id not in query_data.benchmarks and # benchmark not confirmed
                now - self.time_benchmark_submitted.get(benchmark_id, 0) >= 60000 and # not submitted in last 60 seconds
                job.benchmark_ready # benchmark is ready to submit
            )
        ])
        if len(ready_to_submit) > 0:
            benchmark_id = ready_to_submit[0][1]
            ret = job_manager.jobs[benchmark_id]
            print(f"[submissions_manager] found benchmark to submit: {benchmark_id}")
        else:
            print(f"[submissions_manager] no benchmark to submit")
            ret = None
        return ret

    def find_proof_to_submit(self, query_data: QueryData, job_manager: JobManager) -> Optional[Job]:
        now = int(time.time() * 1000)
        ready_to_submit = sorted([
            (
                self.time_proof_submitted.get(benchmark_id, 0),
                benchmark_id
            )
            for benchmark_id, job in job_manager.jobs.items()
            if (
                benchmark_id not in query_data.proofs and # proof not confirmed
                now - self.time_proof_submitted.get(benchmark_id, 0) >= 60000 and # not submitted in last 60 seconds
                job.proof_ready # proof is ready to submit
            )
        ])
        if len(ready_to_submit) > 0:
            benchmark_id = ready_to_submit[0][1]
            ret = job_manager.jobs[benchmark_id]
            print(f"[submissions_manager] found proof to submit: {benchmark_id}")
        else:
            print(f"[submissions_manager] no proof to submit")
            ret = None
        return ret