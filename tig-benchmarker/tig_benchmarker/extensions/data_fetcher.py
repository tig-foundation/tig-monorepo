import aiohttp
import asyncio
import json
import logging
import os
from datetime import datetime
from tig_benchmarker.structs import *
from tig_benchmarker.event_bus import *
from typing import Dict, Any

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

async def _get(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        logger.debug(f"fetching from {url}")
        async with session.get(url) as resp:
            text = await resp.text()
            if resp.status == 200:
                return json.loads(text)
            else:
                if resp.headers.get("Content-Type") == "text/plain":
                    err_msg = f"status code {resp.status} from {url}: {text}"
                else:
                    err_msg = f"status code {resp.status} from {url}"
                logger.error(err_msg)
                raise Exception(err_msg)

class Extension:
    def __init__(self, api_url: str, player_id: str, **kwargs):
        self.api_url = api_url
        self.player_id = player_id
        self._cache = None
        self.lock = False

    async def on_update(self):
        if self.lock:
            return
        self.lock = True
        logger.debug("fetching latest block")
        block_data = await _get(f"{self.api_url}/get-block")
        block = Block.from_dict(block_data["block"])

        if self._cache is not None and block.id == self._cache["block"].id:
            logger.debug("no new block data")
            self.lock = False
            return

        logger.info(f"new block @ height {block.details.height}, fetching data")
        tasks = [
            _get(f"{self.api_url}/get-algorithms?block_id={block.id}"),
            _get(f"{self.api_url}/get-players?player_type=benchmarker&block_id={block.id}"),
            _get(f"{self.api_url}/get-benchmarks?player_id={self.player_id}&block_id={block.id}"),
            _get(f"{self.api_url}/get-challenges?block_id={block.id}")
        ]
        
        algorithms_data, players_data, benchmarks_data, challenges_data = await asyncio.gather(*tasks)

        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data["algorithms"]}
        wasms = {w["algorithm_id"]: Wasm.from_dict(w) for w in algorithms_data["wasms"]}
        
        player = next((Player.from_dict(p) for p in players_data["players"] if p["id"] == self.player_id), None)
        
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data["precommits"]}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data["benchmarks"]}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data["proofs"]}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data["frauds"]}
        for benchmark_id, precommit in precommits.items():
            benchmark = benchmarks.get(benchmark_id, None)
            proof = proofs.get(benchmark_id, None)
            if proof is not None:
                if  self._cache is None or benchmark_id not in self._cache["proofs"]:
                    await emit(
                        "proof_confirmed", 
                        precommit=precommit,
                        benchmark=benchmark,
                        proof=proof,
                    )
            elif benchmark is not None:
                if  self._cache is None or benchmark_id not in self._cache["benchmarks"]:
                    await emit(
                        "benchmark_confirmed", 
                        precommit=precommit,
                        benchmark=benchmark,
                    )
            elif self._cache is None or benchmark_id not in self._cache["precommits"]:
                await emit(
                    "precommit_confirmed", 
                    precommit=precommit
                )        
        
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}

        self._cache = {
            "block": block,
            "algorithms": algorithms,
            "wasms": wasms,
            "player": player,
            "precommits": precommits,
            "benchmarks": benchmarks,
            "proofs": proofs,
            "frauds": frauds,
            "challenges": challenges
        }
        await emit("new_block", **self._cache)
        self.lock = False
