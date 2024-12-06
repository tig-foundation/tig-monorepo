import requests
import json
import logging
import os
from common.structs import *
from common.utils import *
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def _get(url: str) -> Dict[str, Any]:
    logger.debug(f"Fetching from {url}")
    resp = requests.get(url, timeout=10)  # Added timeout for robustness
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
    def __init__(self):
        self.last_fetch = 0
        self._cache = None

    def run(self) -> dict:
        config = CONFIG
        logger.debug("fetching latest block")
        block_data = _get(f"{config['api_url']}/get-block")
        block = Block.from_dict(block_data["block"])

        if self._cache is not None and block.id == self._cache["block"].id:
            logger.debug("no new block data")
            return self._cache

        logger.info(f"new block @ height {block.details.height}, fetching data")
        tasks = [
            f"{config['api_url']}/get-algorithms?block_id={block.id}",
            f"{config['api_url']}/get-benchmarks?player_id={config['player_id']}&block_id={block.id}",
            f"{config['api_url']}/get-challenges?block_id={block.id}"
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor: # Defined max workers as there are 4 process to be executed in parallel.
            algorithms_data, benchmarks_data, challenges_data = list(executor.map(_get, tasks))

        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data["algorithms"]}
        wasms = {w["algorithm_id"]: Binary.from_dict(w) for w in algorithms_data["binarys"]}
        
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data["precommits"]}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data["benchmarks"]}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data["proofs"]}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data["frauds"]}        
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}
        
        # Fetch difficulty data for each challenge
        difficulty_urls = [
            f"{config['api_url']}/get-difficulty-data?block_id={block.id}&challenge_id={c_id}"
            for c_id in challenges
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            difficulty_responses = list(executor.map(_get, difficulty_urls))
        
        difficulty_data = {
            c_id: [DifficultyData.from_dict(d) for d in resp.get("data", [])]
            for c_id, resp in zip(challenges, difficulty_responses)
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
