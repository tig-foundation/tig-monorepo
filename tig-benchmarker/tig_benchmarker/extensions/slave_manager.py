import os
import json
import logging
import re
import signal
import time
import random
import threading
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from tig_benchmarker.extensions.job_manager import Job
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Batch(FromDict):
    benchmark_id: str
    start_nonce: int
    num_nonces: int
    settings: BenchmarkSettings
    sampled_nonces: List[int]
    runtime_config: dict
    download_url: str
    rand_hash: str
    batch_size: int

@dataclass
class BatchResult(FromDict):
    merkle_root: MerkleHash
    solution_nonces: List[int]
    merkle_proofs: List[MerkleProof]

@dataclass
class SlaveConfig(FromDict):
    name_regex: str
    max_concurrent_batches: Dict[str, int]

@dataclass
class SlaveManagerConfig(FromDict):
    port: int
    time_before_batch_retry: int
    slaves: List[SlaveConfig]

class SlaveManager:
    def __init__(self, config: SlaveManagerConfig, jobs: List[Job]):
        self.config = config
        self.jobs = jobs
        self.assigned = {}
        self.concurrent = {}

    def start(self):
        app = FastAPI()

        @app.route('/get-batch', methods=['GET'])
        def get_batch(request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                return "Unregistered slave", 403

            slave = next((slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)), None)

            now = int(time.time() * 1000)
            batch = None
            
            concurrent = self.concurrent.get(slave_name, {"count": 0, "challenge": None})
            selected_challenge = concurrent["challenge"]

            count = concurrent["count"]
            if selected_challenge is not None and count >= slave.max_concurrent_batches[selected_challenge]:
                logger.debug(f"{slave_name} get-batch: max concurrent batches reached")
                raise HTTPException(status_code=425, detail="Max concurrent batches reached")

            for job in self.jobs:
                if selected_challenge is not None and job.challenge != selected_challenge:
                    continue
                sampled_nonces_by_batch_idx = job.sampled_nonces_by_batch_idx
                for batch_idx in range(job.num_batches):
                    if not (
                        now - job.last_batch_retry_time[batch_idx] > self.config.time_before_batch_retry and
                        (
                            job.batch_merkle_roots[batch_idx] is None or
                            not set(sampled_nonces_by_batch_idx.get(batch_idx, [])).issubset(job.merkle_proofs)
                        )
                    ):
                        continue
                    job.last_batch_retry_time[batch_idx] = now
                    selected_challenge = job.challenge
                    start_nonce = batch_idx * job.batch_size
                    batch = Batch(
                        benchmark_id=job.benchmark_id,
                        start_nonce=start_nonce,
                        num_nonces=min(job.batch_size, job.num_nonces - start_nonce),
                        settings=job.settings.to_dict(),
                        sampled_nonces=sampled_nonces_by_batch_idx.get(batch_idx, []),
                        runtime_config=job.runtime_config,
                        download_url=job.download_url,
                        rand_hash=job.rand_hash,
                        batch_size=job.batch_size
                    )
                    break
                if batch is not None:
                    break

            if batch is None:
                logger.debug(f"{slave_name} get-batch: None available")
                print('batch is none')

                raise HTTPException(status_code=503, detail="No batches available")
            else:
                batch_id = f"{batch.benchmark_id}_{batch.start_nonce}"
                if (old_slave_name := self.assigned.pop(batch_id, None)) is not None:
                    logger.warning(f"{old_slave_name} was assigned batch {batch_id} but did not submit result")
                    self.concurrent[old_slave_name]["count"] -= 1
                self.assigned[batch_id] = slave_name
                if slave_name not in self.concurrent:
                    self.concurrent[slave_name] = {"challenge": selected_challenge, "count": 0}

                self.concurrent[slave_name]["count"] += 1
                self.assigned[batch_id] = slave_name
                logger.debug(f"{slave_name} get-batch: (challenge: {selected_challenge}, #batches: 1, batch_ids: [{batch.benchmark_id}])")
                return JSONResponse(content=jsonable_encoder(batch))

        @app.post('/submit-batch-result/{batch_id}')
        async def submit_batch_result(batch_id: str, request: Request):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                raise HTTPException(status_code=403, detail="User-Agent header is required")
            if slave_name != self.assigned.get(batch_id, None):
                raise HTTPException(status_code=400, detail=f"Slave submitted result for {batch_id}, but either took too long, or was not assigned this batch.")
            self.assigned.pop(batch_id)
            self.concurrent[slave_name]["count"] -= 1
            if self.concurrent[slave_name]["count"] == 0:
                self.concurrent.pop(slave_name)

            benchmark_id, start_nonce = batch_id.split("_")
            start_nonce = int(start_nonce)
            result = BatchResult.from_dict(await request.json())
            job = next((job for job in self.jobs if job.benchmark_id == benchmark_id), None)
            logger.debug(f"{slave_name} submit-batch-result: (benchmark_id: {benchmark_id}, start_nonce: {start_nonce}, #solutions: {len(result.solution_nonces)}, #proofs: {len(result.merkle_proofs)})")
            if job is None:
                logger.warning(f"{slave_name} submit-batch-result: no job found with benchmark_id {benchmark_id}")
                raise HTTPException(status_code=400, detail="Invalid benchmark_id")
            batch_idx = start_nonce // job.batch_size
            job.batch_merkle_roots[batch_idx] = result.merkle_root
            job.solution_nonces = list(set(job.solution_nonces + result.solution_nonces))
            job.batch_merkle_proofs.update({
                x.leaf.nonce: x
                for x in result.merkle_proofs
            })
            return {"status": "OK"}

        thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=self.config.port))
        thread.daemon = True
        thread.start()

        logger.info(f"webserver started on 0.0.0.0:{self.config.port}")
