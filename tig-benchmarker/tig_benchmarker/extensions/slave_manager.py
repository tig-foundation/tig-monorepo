import asyncio
import os
import json
import logging
from dataclasses import dataclass
from collections import deque
from quart import Quart, request, jsonify
from hypercorn.config import Config
from hypercorn.asyncio import serve
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class Extension:
    def __init__(self, port: int, **kwargs):
        self.port = port
        self.batch_ids = set()
        self.batches = deque()
        self.priority_batches = deque()
        self._start_server()
        subscribe("new_batch", self.on_new_batch)
        subscribe("new_block", self.on_new_block)

    def _start_server(self):
        app = Quart(__name__)

        @app.route('/get-batch', methods=['GET'])
        async def get_batch():
            if len(self.priority_batches) == 0 and len(self.batches) == 0:
                return "No batches available", 404
            batch = self.priority_batches.popleft() if len(self.priority_batches) else self.batches.popleft()
            batch_id = f"{batch['benchmark_id']}_{batch['start_nonce']}"
            self.batch_ids.remove(batch_id)
            logger.info(f"serving batch {batch_id} to {request.headers['User-Agent']}")
            return jsonify(batch)

        @app.route('/submit-batch-result/<batch_id>', methods=['POST'])
        async def submit_batch_result(batch_id):
            logger.info(f"received results for batch {batch_id} from {request.headers['User-Agent']}")
            benchmark_id, start_nonce = batch_id.split("_")
            data = await request.json
            await emit("batch_result", benchmark_id=benchmark_id, start_nonce=int(start_nonce), **data)
            return "OK"

        config = Config()
        config.bind = [f"0.0.0.0:{self.port}"]

        self._server_task = asyncio.create_task(serve(app, config))
        logger.info(f"webserver started on {config.bind[0]}")

    async def on_new_batch(self, **batch):
        batch_id = f"{batch['benchmark_id']}_{batch['start_nonce']}"
        if batch_id in self.batch_ids:
            return
        self.batch_ids.add(batch_id)
        if len(batch['sampled_nonces']) > 0:
            self.priority_batches.append(batch)
        else:
            self.batches.append(batch)

    async def on_new_block(self, **kwargs):
        logger.info(f"{len(self.batches)} batches in queue")