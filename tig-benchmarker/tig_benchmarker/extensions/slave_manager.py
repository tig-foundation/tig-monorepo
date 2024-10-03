import asyncio
import os
import json
import logging
import re
from dataclasses import dataclass
from collections import deque
from enum import Enum
from quart import Quart, request, jsonify
from hypercorn.config import Config
from hypercorn.asyncio import serve
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class Status(Enum):
    QUEUED = 0
    PROCESSING = 1
    FINISHED = 2
    PRIORITY_QUEUED = 3
    PRIORITY_PROCESSING = 4
    PRIORITY_FINISHED = 5

@dataclass
class SlaveConfig(FromDict):
    name_regex: str
    challenge_selection: Optional[List[str]]

@dataclass
class SlaveManagerConfig(FromDict):
    slaves: List[SlaveConfig]

class Extension:
    def __init__(self, port: int, exit_event: asyncio.Event, slave_manager: dict, **kwargs):
        self.port = port
        self.config = SlaveManagerConfig.from_dict(slave_manager)
        self.exit_event = exit_event
        self.batch_status = {}
        self.batches = {}
        self.priority_batches = {}
        self.challenge_name_2_id = {}
        self.lock = True
        self._start_server()

    def _start_server(self):
        app = Quart(__name__)

        @app.route('/get-batch', methods=['GET'])
        async def get_batch():
            if self.lock:
                return "Slave manager is not ready", 503
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting get-batch request")
                return "Unregistered slave", 403
            
            if (slave := next((slave for slave in self.config.slaves if re.match(slave.name_regex, slave_name)), None)) is not None:
                challenge_selection = slave.challenge_selection
            else:
                challenge_selection = None
                
            batch = None
            while batch is None:
                if challenge_selection is not None:
                    for c_name in challenge_selection:
                        c_id = self.challenge_name_2_id[c_name]
                        if len(self.priority_batches.get(c_id, [])):
                            batch, _ = self.priority_batches[c_id].popleft()
                            break
                        elif len(self.batches.get(c_id, [])):
                            batch, _ = self.batches[c_id].popleft()
                            break
                    else:
                        return "No batches available", 404
                else:
                    if (non_empty_queues := [q for q in self.priority_batches.values() if q]):
                        batch, _ = min(non_empty_queues, key=lambda q: q[0][1]).popleft()
                    elif (non_empty_queues := [q for q in self.batches.values() if q]):
                        batch, _ = min(non_empty_queues, key=lambda q: q[0][1]).popleft()
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
                logger.debug(f"{slave_name} got priority batch {batch_id}")
            else:
                self.batch_status[benchmark_id][start_nonce] = Status.PROCESSING
                logger.debug(f"{slave_name} got batch {batch_id}")
            
            return jsonify(batch)

        @app.route('/submit-batch-result/<batch_id>', methods=['POST'])
        async def submit_batch_result(batch_id):
            if (slave_name := request.headers.get('User-Agent', None)) is None:
                return "User-Agent header is required", 403
            if not any(re.match(slave.name_regex, slave_name) for slave in self.config.slaves):
                logger.warning(f"slave {slave_name} does not match any regex. rejecting submit-batch-result request")
            benchmark_id, start_nonce = batch_id.split("_")
            start_nonce = int(start_nonce)
            batch_status = self.batch_status.get(benchmark_id, {}).get(start_nonce, None)
            data = await request.json
            is_priority = len(data["merkle_proofs"]) > 0
            if batch_status is None:
                return "Redundant result", 404
            elif is_priority and batch_status == Status.PRIORITY_FINISHED:
                return "Redundant result", 404
            elif not is_priority and batch_status in {
                Status.FINISHED, 
                Status.PRIORITY_QUEUED, 
                Status.PRIORITY_PROCESSING, 
                Status.PRIORITY_FINISHED
            }:
                return "Redundant result", 404

            if is_priority:
                self.batch_status[benchmark_id][start_nonce] = Status.PRIORITY_FINISHED
                logger.debug(f"{slave_name} returned priority batch {batch_id}")
            else:
                self.batch_status[benchmark_id][start_nonce] = Status.FINISHED
                logger.debug(f"{slave_name} returned batch {batch_id}")
            await emit("batch_result", benchmark_id=benchmark_id, start_nonce=start_nonce, **data)
            return "OK"

        config = Config()
        config.bind = [f"0.0.0.0:{self.port}"]

        self._server_task = asyncio.create_task(serve(app, config, shutdown_trigger=self.exit_event.wait))
        logger.info(f"webserver started on {config.bind[0]}")

    async def on_new_batch(self, **batch):
        benchmark_id = batch['benchmark_id']
        start_nonce = batch['start_nonce']
        challenge_id = batch['settings']['challenge_id']
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

        now_ = now()
        if is_priority:
            self.batch_status.setdefault(benchmark_id, {})[start_nonce] = Status.PRIORITY_QUEUED
            self.priority_batches.setdefault(challenge_id, deque()).append((batch, now_))
        else:
            self.batch_status.setdefault(benchmark_id, {})[start_nonce] = Status.QUEUED
            self.batches.setdefault(challenge_id, deque()).append((batch, now_))

    async def on_update(self):
        logger.info(f"#batches in queue (normal: {sum(len(x) for x in self.batches.values())}, priority: {sum(len(x) for x in self.priority_batches.values())})")

    async def on_new_block(self, precommits: Dict[str, Precommit], challenges: Dict[str, Challenge], **kwargs):
        for benchmark_id in list(self.batch_status):
            if benchmark_id in precommits:
                continue
            self.batch_status.pop(benchmark_id)

        self.challenge_name_2_id = {c.details.name: c.id for c in challenges.values()}
        challenge_names = set(self.challenge_name_2_id)
        for slave in self.config.slaves:
            if slave.challenge_selection is None:
                continue
            assert set(slave.challenge_selection).issubset(challenge_names), f"challenge_selection for slave regex '{slave.name_regex}' is not a subset of {challenge_names}"
        self.lock = False