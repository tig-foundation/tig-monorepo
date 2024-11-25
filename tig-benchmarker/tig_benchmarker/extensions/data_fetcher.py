import aiohttp
import asyncio
import json
import logging
import os
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
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

class DataFetcher:
    def __init__(self, api_url: str, player_id: str):
        self.api_url = api_url
        self.player_id = player_id
        self.last_fetch = 0
        self._cache = None

    async def run(self) -> dict:
        logger.debug("fetching latest block")
        block_data = await _get(f"{self.api_url}/get-block")
        block = Block.from_dict(block_data["block"])

        if self._cache is not None and block.id == self._cache["block"].id:
            logger.debug("no new block data")
            return self._cache

        logger.info(f"new block @ height {block.details.height}, fetching data")
        tasks = [
            _get(f"{self.api_url}/get-algorithms?block_id={block.id}"),
            _get(f"{self.api_url}/get-benchmarks?player_id={self.player_id}&block_id={block.id}"),
            _get(f"{self.api_url}/get-challenges?block_id={block.id}")
        ]
        
        algorithms_data, benchmarks_data, challenges_data = await asyncio.gather(*tasks)

        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data["algorithms"]}
        wasms = {w["algorithm_id"]: Binary.from_dict(w) for w in algorithms_data["binarys"]}
        
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data["precommits"]}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data["benchmarks"]}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data["proofs"]}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data["frauds"]}        
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}
        
        tasks = [
            _get(f"{self.api_url}/get-difficulty-data?block_id={block.id}&challenge_id={c_id}")
            for c_id in challenges
        ]
        difficulty_data = await asyncio.gather(*tasks)
        difficulty_data = {
            c_id: [DifficultyData.from_dict(x) for x in d["data"]]
            for c_id, d in zip(challenges, difficulty_data)
        }

        self._cache = {
            "block": block,
            "algorithms": algorithms,
            "wasms": wasms,
            "precommits": precommits,
            "benchmarks": benchmarks,
            "proofs": proofs,
            "frauds": frauds,
            "challenges": challenges,
            "difficulty_data": difficulty_data
        }
        return self._cache
