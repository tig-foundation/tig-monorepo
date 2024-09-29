import asyncio
import os
import json
import logging
from dataclasses import dataclass
from collections import deque
from enum import Enum
from quart import Quart, request, jsonify
from hypercorn.config import Config
from hypercorn.asyncio import serve
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class Status(Enum):
    QUEUED = 0
    PROCESSING = 1
    FINISHED = 2
    PRIORITY_QUEUED = 3
    PRIORITY_PROCESSING = 4
    PRIORITY_FINISHED = 5

class Extension:
    def __init__(self, port: int, **kwargs):
        self.port = port
        self.batch_status = {}
        self.batches = deque()
        self.priority_batches = deque()
        self._start_server()

    def _start_server(self):
        app = Quart(__name__)

        @app.route('/get-batch', methods=['GET'])
        async def get_batch():
            batch = None
            while batch is None:
                if len(self.priority_batches):
                    batch = self.priority_batches.popleft()
                elif len(self.batches):
                    batch = self.batches.popleft()
                else:
                    return "No batches available", 404
                benchmark_id = batch['benchmark_id']
                start_nonce = batch['start_nonce']
                is_priority = len(batch['sampled_nonces']) > 0
                batch_status = self.batch_status.get(benchmark_id, {}).get(start_nonce, None)
                if batch_status is None:
                    batch = None
                elif is_priority and batch_status in {
                    Status.PRIORITY_FINISHED
                }: 
                    batch = None
                elif not is_priority and batch_status in {
                    Status.FINISHED, 
                    Status.PRIORITY_QUEUED,
                    Status.PRIORITY_PROCESSING,
                    Status.PRIORITY_FINISHED
                }:
                    batch = None

            batch_id = f"{benchmark_id}_{start_nonce}"
            if is_priority:
                self.batch_status[benchmark_id][start_nonce] = Status.PRIORITY_PROCESSING
                logger.info(f"serving priority batch {batch_id} to {request.headers['User-Agent']}")
            else:
                self.batch_status[benchmark_id][start_nonce] = Status.PROCESSING
                logger.info(f"serving batch {batch_id} to {request.headers['User-Agent']}")
            
            return jsonify(batch)

        @app.route('/submit-batch-result/<batch_id>', methods=['POST'])
        async def submit_batch_result(batch_id):
            benchmark_id, start_nonce = batch_id.split("_")
            start_nonce = int(start_nonce)
            batch_status = self.batch_status.get(benchmark_id, {}).get(start_nonce, None)
            data = await request.json
            is_priority = len(data["merkle_proofs"]) > 0
            if batch_status is None:
                return "OK"
            elif is_priority and batch_status == Status.PRIORITY_FINISHED:
                return "OK"
            elif not is_priority and batch_status in {
                Status.FINISHED, 
                Status.PRIORITY_QUEUED, 
                Status.PRIORITY_PROCESSING, 
                Status.PRIORITY_FINISHED
            }:
                return "OK"

            if is_priority:
                self.batch_status[benchmark_id][start_nonce] = Status.PRIORITY_FINISHED
                logger.info(f"received results for priority batch {batch_id} from {request.headers['User-Agent']}")
            else:
                self.batch_status[benchmark_id][start_nonce] = Status.FINISHED
                logger.info(f"received results for batch {batch_id} from {request.headers['User-Agent']}")
            await emit("batch_result", benchmark_id=benchmark_id, start_nonce=start_nonce, **data)
            return "OK"

        config = Config()
        config.bind = [f"0.0.0.0:{self.port}"]

        self._server_task = asyncio.create_task(serve(app, config))
        logger.info(f"webserver started on {config.bind[0]}")

    async def on_new_batch(self, **batch):
        benchmark_id = batch['benchmark_id']
        start_nonce = batch['start_nonce']
        is_priority = len(batch['sampled_nonces']) > 0
        batch_status = self.batch_status.get(benchmark_id, {}).get(start_nonce, None)
        if is_priority and batch_status in {
            Status.PRIORITY_QUEUED, 
            Status.PRIORITY_FINISHED
        }:
            return
        elif not is_priority and batch_status in {
            Status.QUEUED, 
            Status.FINISHED, 
            Status.PRIORITY_QUEUED, 
            Status.PRIORITY_PROCESSING, 
            Status.PRIORITY_FINISHED
        }:
            return

        if is_priority:
            self.batch_status.setdefault(benchmark_id, {})[start_nonce] = Status.PRIORITY_QUEUED
            self.priority_batches.append(batch)
        else:
            self.batch_status.setdefault(benchmark_id, {})[start_nonce] = Status.QUEUED
            self.batches.append(batch)

    async def on_new_block(self, precommits: Dict[str, Precommit], **kwargs):
        logger.info(f"#batches in queue (normal: {len(self.batches)}, priority: {len(self.priority_batches)})")
        for benchmark_id in list(self.batch_status):
            if benchmark_id in precommits:
                continue
            self.batch_status.pop(benchmark_id)