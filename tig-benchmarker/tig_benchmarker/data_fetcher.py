import aiohttp
import asyncio
import json
from datetime import datetime
from tig_benchmarker.structs import *
from typing import Dict, Any

async def _get(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            txt = await response.text()
            if response.status != 200:
                raise Exception(f"error {response.status} fetching from {url}:\n\t{txt}")
            return json.loads(txt)

@dataclass
class QueryData(FromDict):
    block: Block
    algorithms: Dict[str, Algorithm]
    wasms: Dict[str, Wasm]
    player: Optional[Player]
    precommits: Dict[str, Precommit]
    benchmarks: Dict[str, Benchmark]
    proofs: Dict[str, Proof]
    frauds: Dict[str, Fraud]
    challenges: Dict[str, Challenge]

class DataFetcher:
    def __init__(self, api_url: str, player_id: str):
        self.api_url = api_url
        self.player_id = player_id
        self._cache = None

    async def fetch(self) -> 'QueryData':
        start = datetime.now()
        print(f"[data_fetcher] querying API")
        block_data = await _get(f"{self.api_url}/get-block")
        block = Block.from_dict(block_data["block"])

        if self._cache is not None and block.id == self._cache.block.id:
            print(f"[data_fetcher] no new block data")
            return self._cache

        print(f"[data_fetcher] new block @ height {block.details.height}, fetching data")
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
        
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data["challenges"]}

        data = QueryData(
            block=block,
            algorithms=algorithms,
            wasms=wasms,
            player=player,
            precommits=precommits,
            benchmarks=benchmarks,
            proofs=proofs,
            frauds=frauds,
            challenges=challenges
        )
        print(f"[data_fetcher] done. took {(datetime.now() - start).total_seconds()} seconds")
        self._cache = data
        return data
