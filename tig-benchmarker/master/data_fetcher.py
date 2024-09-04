import aiohttp
import asyncio
import json
from typing import Dict, Any
from master.data import *
from master.config import *

_CACHE = {}

async def run(state: State):
    while True:
        try:
            print(f"[data_fetcher] querying API")
            query_data = await _execute()
            print(f"[data_fetcher] done")
            state.query_data = query_data
        except Exception as e:
            print(f"[data_fetcher] error: {e}")
        finally:
            await asyncio.sleep(10)

async def _get(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            txt = await response.text()
            if response.status != 200:
                raise Exception(f"error {response.status} fetching from {url}:\n\t{txt}")
            return json.loads(txt)

async def _execute() -> QueryData:
    block_data = await _get(f"{API_URL}/get-block")
    block = Block.from_dict(block_data["block"])

    if block.id in _CACHE:
        print(f"[data_fetcher] no new block data")
        return _CACHE[block.id]

    print(f"[data_fetcher] new block @ height {block.details.height}, fetching data")
    tasks = [
        _get(f"{API_URL}/get-algorithms?block_id={block.id}"),
        _get(f"{API_URL}/get-players?player_type=benchmarker&block_id={block.id}"),
        _get(f"{API_URL}/get-benchmarks?player_id={PLAYER_ID}&block_id={block.id}"),
        _get(f"{API_URL}/get-challenges?block_id={block.id}")
    ]
    
    algorithms_data, players_data, benchmarks_data, challenges_data = await asyncio.gather(*tasks)

    algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data["algorithms"]}
    wasms = {w["algorithm_id"]: Wasm.from_dict(w) for w in algorithms_data["wasms"]}
    
    player = next((Player.from_dict(p) for p in players_data["players"] if p["id"] == PLAYER_ID), None)
    
    benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data["benchmarks"]}
    proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data["proofs"]}
    frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data["frauds"]}
    
    challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}

    data = QueryData(
        block=block,
        algorithms=algorithms,
        wasms=wasms,
        player=player,
        benchmarks=benchmarks,
        proofs=proofs,
        frauds=frauds,
        challenges=challenges
    )

    _CACHE[block.id] = data

    return data