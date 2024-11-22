import json
import logging
import os
import requests
from sqlalchemy import false, null
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, api_url: str, player_id: str):
        self.api_url = api_url
        self.player_id = player_id

    def run(self) -> dict:
        logger.debug("Fetching latest block")
        block_data = _get(f"{self.api_url}/get-block?include_data")
        block = Block.from_dict({k: v for k, v in block_data["block"].items() if k != 'data'})
        block.data = block_data["block"]["data"]
        
        # logger.info(f"New block detected @ height {block.details.height}, fetching data")
        urls = [
            f"{self.api_url}/get-algorithms?block_id={block.id}",
            f"{self.api_url}/get-opow?block_id={block.id}",
            f"{self.api_url}/get-benchmarks?player_id={self.player_id}&block_id={block.id}",
            f"{self.api_url}/get-challenges?block_id={block.id}",
            f"{self.api_url}/get-player-data?player_id={self.player_id}&block_id={block.id}"
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor: # Defined max workers as there are 4 process to be executed in parallel.
            algorithms_data, opow_data, benchmarks_data, challenges_data, player_data = list(executor.map(_get, urls))

        # Parse algorithms and wasms
        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data.get("algorithms", [])}
        binarys = {w["algorithm_id"]: Binary.from_dict(w) for w in algorithms_data.get("binarys", [])}


        
        # Parse player
        dummy_player = {
            "id": self.player_id,
            "details": {
                "name": self.player_id,
                "is_multisig": False
            },
            "state": {
                "total_fees_paid":  0,
                "available_fee_balance": 0,
                "votes": {}
            }
        }

        # logger.info(f"Player: {dummy_player}")

        player = Player.from_dict(player_data["player"]) if player_data["player"] is not None else Player.from_dict(dummy_player)
        
        # Parse precommits, benchmarks, proofs, frauds
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data.get("precommits", [])}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data.get("benchmarks", [])}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data.get("proofs", [])}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data.get("frauds", [])}

        # Parse challenges
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data.get("challenges", [])}
        
        # Fetch difficulty data for each challenge
        difficulty_urls = [
            f"{self.api_url}/get-difficulty-data?block_id={block.id}&challenge_id={c_id}"
            for c_id in challenges
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            difficulty_responses = list(executor.map(_get, difficulty_urls))
        
        difficulty_data = {
            c_id: [DifficultyData.from_dict(d) for d in resp.get("data", [])]
            for c_id, resp in zip(challenges, difficulty_responses)
        }

        # Return data
        return {
            "block": block,
            "algorithms": algorithms,
            "wasms": binarys,
            "player": player,
            "precommits": precommits,
            "benchmarks": benchmarks,
            "proofs": proofs,
            "frauds": frauds,
            "challenges": challenges,
            "difficulty_data": difficulty_data
        }