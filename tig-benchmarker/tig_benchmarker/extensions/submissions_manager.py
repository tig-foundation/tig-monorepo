import aiohttp
import logging
import json
import os
from tig_benchmarker.event_bus import *
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
    clear_precommits_submission_on_new_block: bool = True
    max_retries: Optional[int] = None
    ms_delay_between_retries: int = 60000

@dataclass
class PendingSubmission:
    last_retry_time: int
    retries: int
    request: Union[SubmitPrecommitRequest, SubmitBenchmarkRequest, SubmitProofRequest]

class Extension:
    def __init__(self, api_url: str, api_key: str, backup_folder: str, **kwargs):
        self.api_url = api_url
        self.api_key = api_key
        self.backup_folder = backup_folder
        if (submissions_manager := kwargs.get("submissions_manager", None)):
            self.config = SubmissionsManagerConfig.from_dict(submissions_manager)
        else:
            self.config = SubmissionsManagerConfig()
        self.pending_submissions  = {
            "precommit": [],
            "benchmark": [],
            "proof": []
        }
        self._restore_pending_submissions()

    def _restore_pending_submissions(self):
        pending_benchmarks = []
        pending_proofs = []
        for submission_type in ["benchmark", "proof"]:
            path = os.path.join(self.backup_folder, submission_type)
            if not os.path.exists(path):
                logger.info(f"creating backup folder {path}")
                os.makedirs(path, exist_ok=True)
            for file in os.listdir(path):
                if not file.endswith(".json"):
                    continue
                file_path = os.path.join(path, file)
                logger.info(f"restoring {submission_type} from {file_path}")
                with open(file_path) as f:
                    d = json.load(f)
                submission = PendingSubmission(
                    last_retry_time=0,
                    retries=0,
                    request=SubmitBenchmarkRequest.from_dict(d) if submission_type == "benchmark" else SubmitProofRequest.from_dict(d)
                )
                self.pending_submissions[submission_type].append(submission)

    async def on_precommit_ready(self, settings: BenchmarkSettings, num_nonces: int, **kwargs):
        self.pending_submissions["precommit"].append(
            PendingSubmission(
                last_retry_time=0,
                retries=0,
                request=SubmitPrecommitRequest(settings=settings, num_nonces=num_nonces)
            )
        )

    async def on_benchmark_ready(self, benchmark_id: str, merkle_root: MerkleHash, solution_nonces: Set[int], **kwargs):
        if any(
            x.request.benchmark_id == benchmark_id
            for x in self.pending_submissions["benchmark"]
        ):
            logger.warning(f"benchmark {benchmark_id} already pending submission")
        else:
            request = SubmitBenchmarkRequest(
                benchmark_id=benchmark_id,
                merkle_root=merkle_root,
                solution_nonces=solution_nonces
            )
            self.pending_submissions["benchmark"].append(
                PendingSubmission(
                    last_retry_time=0,
                    retries=0,
                    request=request
                )
            )
            with open(os.path.join(self.backup_folder, "benchmark", f"{benchmark_id}.json"), "w") as f:
                json.dump(request.to_dict(), f)
    
    async def on_proof_ready(self, benchmark_id: str, merkle_proofs: List[MerkleProof], **kwargs):
        if any(
            x.request.benchmark_id == benchmark_id
            for x in self.pending_submissions["proof"]
        ):
            logger.warning(f"proof {benchmark_id} already in pending submissions")
        else:
            request = SubmitProofRequest(
                benchmark_id=benchmark_id,
                merkle_proofs=merkle_proofs
            )
            self.pending_submissions["proof"].append(
                PendingSubmission(
                    last_retry_time=0,
                    retries=0,
                    request=request
                )
            )
            with open(os.path.join(self.backup_folder, "proof", f"{benchmark_id}.json"), "w") as f:
                json.dump(request.to_dict(), f)

    async def on_new_block(
        self, 
        block: Block, 
        precommits: Dict[str, Precommit],
        **kwargs
    ):
        if self.config.clear_precommits_submission_on_new_block:
            logger.debug(f"clearing {len(self.pending_submissions['precommit'])} pending precommits")
            self.pending_submissions["precommit"].clear()

        for submission_type in ["benchmark", "proof"]:
            filtered_submissions = []
            for submission in self.pending_submissions[submission_type]:
                if (
                    submission.request.benchmark_id not in precommits or # is expired
                    submission.request.benchmark_id in kwargs[submission_type + "s"] # is confirmed
                ):
                    self._prune_pending_submission(submission_type, submission.request.benchmark_id)
                else:
                    filtered_submissions.append(submission)
            self.pending_submissions[submission_type] = filtered_submissions

    def _prune_pending_submission(self, submission_type: str, benchmark_id: str):
        logger.debug(f"removing {submission_type} {benchmark_id} from pending submissions")
        path = os.path.join(self.backup_folder, submission_type, f"{benchmark_id}.json")
        if os.path.exists(path):
            os.remove(path)

    async def on_update(self):
        if (
            len(self.pending_submissions['precommit']) == 0 and
            len(self.pending_submissions['benchmark']) == 0 and
            len(self.pending_submissions['proof']) == 0
        ):
            return

        logger.debug(f"pending submissions: (#precommits: {len(self.pending_submissions['precommit'])}, #benchmarks: {len(self.pending_submissions['benchmark'])}, #proofs: {len(self.pending_submissions['proof'])})")

        now_ = now()
        for submission_type in ["precommit", "benchmark", "proof"]:
            if len(self.pending_submissions[submission_type]) > 0:
                self.pending_submissions[submission_type] = sorted(self.pending_submissions[submission_type], key=lambda x: x.last_retry_time)
                if now_ - self.pending_submissions[submission_type][0].last_retry_time < self.config.ms_delay_between_retries:
                    logger.debug(f"no {submission_type} ready for submission")
                else:
                    s = self.pending_submissions[submission_type].pop(0)
                    asyncio.create_task(self.submit(s))
                    if submission_type == "precommit":
                        logger.info(f"submitting precommit {s.request}")
                    else:
                        logger.info(f"submitting {submission_type} '{s.request.benchmark_id}'")
        
    async def submit(
        self,
        submission: PendingSubmission, 
    ):
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "tig-benchmarker-py/v0.2"
        }

        if isinstance(submission.request, SubmitPrecommitRequest):
            submission_type = "precommit"
        elif isinstance(submission.request, SubmitBenchmarkRequest):
            submission_type = "benchmark"
        elif isinstance(submission.request, SubmitProofRequest):
            submission_type = "proof"
        else:
            raise ValueError(f"Invalid request type: {type(submission.request)}")

        d = submission.request.to_dict()
        logger.debug(f"submitting {submission_type}: {d}")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/submit-{submission_type}", json=d, headers=headers) as resp:
                text = await resp.text()
                if resp.status == 200:
                    logger.info(f"submitted {submission_type} successfully")
                    if submission_type != "precommit":
                        self._prune_pending_submission(submission_type, submission.request.benchmark_id)
                    await emit(f"submit_{submission_type}_success", text=text, **d)
                else:
                    if resp.headers.get("Content-Type") == "text/plain":
                        logger.error(f"status {resp.status} when submitting {submission_type}: {text}")
                    else:
                        logger.error(f"status {resp.status} when submitting {submission_type}")
                    if 500 <= resp.status <= 599 and (
                        self.config.max_retries is None or 
                        submission.retries + 1 < self.config.max_retries
                    ):
                        submission.retries += 1
                        submission.last_retry_time = now()
                        self.pending_submissions[submission_type].append(submission)
                    await emit(f"submit_{submission_type}_error", text=text, status=resp.status, request=submission.request)