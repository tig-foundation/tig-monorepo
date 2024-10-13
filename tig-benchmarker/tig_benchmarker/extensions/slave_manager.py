import asyncio
import os
import json
import logging
import re
import signal
from quart import Quart, request, jsonify
from hypercorn.config import Config
from hypercorn.asyncio import serve
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
    wasm_vm_config: dict
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

    def start(self):
        app = Quart(__name__)

        @app.route('/get-batches', methods=['GET'])
        async def get_batches():
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batches request")
                return "Unregistered slave", 403
            
            slave = next((slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)), None)

            now = int(time.time() * 1000)
            batches = []
            selected_challenge = None
            max_concurrent_batches = None
            for job in self.jobs:
                if (
                    job.challenge not in slave.max_concurrent_batches or 
                    (selected_challenge is not None and job.challenge != selected_challenge)
                ):
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
                    max_concurrent_batches = slave.max_concurrent_batches[job.challenge]
                    start_nonce = batch_idx * job.batch_size
                    batches.append(Batch(
                        benchmark_id=job.benchmark_id,
                        start_nonce=start_nonce,
                        num_nonces=min(job.batch_size, job.num_nonces - start_nonce),
                        settings=job.settings.to_dict(),
                        sampled_nonces=sampled_nonces_by_batch_idx.get(batch_idx, []),
                        wasm_vm_config=job.wasm_vm_config,
                        download_url=job.download_url,
                        rand_hash=job.rand_hash,
                        batch_size=job.batch_size
                    ))
                    if len(batches) >= max_concurrent_batches:
                        break
                if max_concurrent_batches is not None and len(batches) >= max_concurrent_batches:
                    break

            if len(batches) == 0:
                logger.debug(f"{slave_name} get-batches: None available")
                return "No batches available", 503
            else:
                logger.debug(f"{slave_name} get-batches: Assigning {len(batches)} {selected_challenge} batches")
                return jsonify([b.to_dict() for b in batches])

        @app.route('/submit-batch-result/<batch_id>', methods=['POST'])
        async def submit_batch_result(batch_id):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting submit-batch-result request")
            benchmark_id, start_nonce = batch_id.split("_")
            start_nonce = int(start_nonce)
            result = BatchResult.from_dict(await request.json)
            job = next((job for job in self.jobs if job.benchmark_id == benchmark_id), None)
            logger.debug(f"{slave_name} submit-batch-result: (benchmark_id: {benchmark_id}, start_nonce: {start_nonce}, #solutions: {len(result.solution_nonces)}, #proofs: {len(result.merkle_proofs)})")
            if job is None:
                logger.warning(f"{slave_name} submit-batch-result: no job found with benchmark_id {benchmark_id}")
                return "Invalid benchmark_id", 400
            batch_idx = start_nonce // job.batch_size
            job.batch_merkle_roots[batch_idx] = result.merkle_root
            job.solution_nonces = list(set(job.solution_nonces + result.solution_nonces))
            job.batch_merkle_proofs.update({
                x.leaf.nonce: x
                for x in result.merkle_proofs
            })
            return "OK"

        config = Config()
        config.bind = [f"0.0.0.0:{self.config.port}"]

        exit_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, exit_event.set)
        loop.add_signal_handler(signal.SIGTERM, exit_event.set)
        asyncio.create_task(serve(app, config, shutdown_trigger=exit_event.wait))
        logger.info(f"webserver started on {config.bind[0]}")