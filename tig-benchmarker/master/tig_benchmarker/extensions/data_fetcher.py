import json
import logging
import os
import requests
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

# async def _get(url: str) -> Dict[str, Any]:
#     async with aiohttp.ClientSession() as session:
#         logger.debug(f"fetching from {url}")
#         async with session.get(url) as resp:
#             text = resp.text()
#             if resp.status == 200:
#                 return json.loads(text)
#             else:
#                 if resp.headers.get("Content-Type") == "text/plain":
#                     err_msg = f"status code {resp.status} from {url}: {text}"
#                 else:
#                     err_msg = f"status code {resp.status} from {url}"
#                 logger.error(err_msg)
#                 raise Exception(err_msg)
            
def _get(url: str) -> Dict[str, Any]:
    logger.debug(f"fetching from {url}")
    resp = requests.get(url)
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        if resp.headers.get("Content-Type") == "text/plain":
            err_msg = f"status code {resp.status_code} from {url}: {resp.text}"
        else:
            err_msg = f"status code {resp.status_code} from {url}"
        logger.error(err_msg)
        raise Exception(err_msg)


class DataFetcher:
    def __init__(self, api_url: str, player_id: str):
        self.api_url = api_url
        self.player_id = player_id
        self.last_fetch = 0
        self._cache = None

    def run(self) -> dict:
        logger.debug("fetching latest block")
        block_data = _get(f"{self.api_url}/get-block")
        block = Block.from_dict(block_data["block"])

        if self._cache is not None and block.id == self._cache["block"].id:
            logger.debug("no new block data")
            return self._cache

        logger.info(f"new block @ height {block.details.height}, fetching data")
        urls = [
            f"{self.api_url}/get-algorithms?block_id={block.id}",
            f"{self.api_url}/get-players?player_type=benchmarker&block_id={block.id}",
            f"{self.api_url}/get-benchmarks?player_id={self.player_id}&block_id={block.id}",
            f"{self.api_url}/get-challenges?block_id={block.id}"
        ]
        
        with ThreadPoolExecutor() as executor:
             algorithms_data, players_data, benchmarks_data, challenges_data = list(executor.map(_get, urls))

        print(algorithms_data)

        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data["algorithms"]}
        wasms = {w["algorithm_id"]: Wasm.from_dict(w) for w in algorithms_data["wasms"]}
        
        player = next((Player.from_dict(p) for p in players_data["players"] if p["id"] == self.player_id), None)
        
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data["precommits"]}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data["benchmarks"]}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data["proofs"]}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data["frauds"]}        
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}
        
        tasks = [
            f"{self.api_url}/get-difficulty-data?block_id={block.id}&challenge_id={c_id}"
            for c_id in challenges
        ]

        with ThreadPoolExecutor() as executor:
            difficulty_data = list(executor.map(_get, tasks))

        difficulty_data = {
            c_id: [DifficultyData.from_dict(x) for x in d["data"]]
            for c_id, d in zip(challenges, difficulty_data)
        }

        self._cache = {
            "block": block,
            "algorithms": algorithms,
            "wasms": wasms,
            "player": player,
            "precommits": precommits,
            "benchmarks": benchmarks,
            "proofs": proofs,
            "frauds": frauds,
            "challenges": challenges,
            "difficulty_data": difficulty_data
        }

        return self._cache
