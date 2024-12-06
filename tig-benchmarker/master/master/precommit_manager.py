import os
import logging
import random
from dataclasses import dataclass
from master.submissions_manager import SubmitPrecommitRequest
from common.structs import *
from common.utils import FromDict
from typing import Dict, List, Optional, Set
from master.sql import db_conn
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class AlgorithmSelectionConfig(FromDict):
    algorithm: str
    base_fee_limit: PreciseNumber
    num_nonces: int
    weight: float

@dataclass
class PrecommitManagerConfig(FromDict):
    max_pending_benchmarks: int
    algo_selection: Dict[str, AlgorithmSelectionConfig]

class PrecommitManager:
    def __init__(self):
        self.last_block_id = None
        self.num_precommits_submitted = 0
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.curr_base_fees = {}

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        challenges: Dict[str, Challenge],
        algorithms: Dict[str, Algorithm],
        difficulty_data: Dict[str, List[DifficultyData]],
        **kwargs
    ):
        self.last_block_id = block.id
        self.num_precommits_submitted = 0
        self.challenge_name_2_id = {
            c.details.name: c.id
            for c in challenges.values()
        }
        self.algorithm_name_2_id = {
            f"{challenges[a.details.challenge_id].details.name}_{a.details.name}": a.id
            for a in algorithms.values()
        }
        self.curr_base_fees = {
            c.details.name: c.block_data.base_fee
            for c in challenges.values()
            if c.block_data is not None
        }
        benchmark_stats_by_challenge = {
            c.details.name: {
                "solutions": 0,
                "nonces": 0,
                "qualifiers": 0
            }
            for c in challenges.values()
            if c.block_data is not None
        }
        for benchmark in benchmarks.values():
            precommit = precommits[benchmark.id]
            c_name = challenges[precommit.settings.challenge_id].details.name
            benchmark_stats_by_challenge[c_name]["solutions"] += benchmark.details.num_solutions
            benchmark_stats_by_challenge[c_name]["nonces"] += precommit.details.num_nonces

        for c_name, x in benchmark_stats_by_challenge.items():
            avg_nonces_per_solution = (x["nonces"] // x["solutions"]) if x["solutions"] > 0 else 0
            logger.info(f"benchmark stats for {c_name}: (#nonces: {x['nonces']}, #solutions: {x['solutions']}, #qualifiers: {x['qualifiers']}, avg_nonces_per_solution: {avg_nonces_per_solution})")

        aggregate_difficulty_data = {
            c_id: {
                "nonces": sum(
                    x.num_nonces if x.difficulty in challenges[c_id].block_data.qualifier_difficulties else 0
                    for x in difficulty_data
                ),
                "solutions": sum(
                    x.num_solutions if x.difficulty in challenges[c_id].block_data.qualifier_difficulties else 0
                    for x in difficulty_data
                ),
            }
            for c_id, difficulty_data in difficulty_data.items()
        }
        for c_id, x in aggregate_difficulty_data.items():
            avg_nonces_per_solution = (x["nonces"] // x["solutions"]) if x["solutions"] > 0 else 0
            logger.info(f"global qualifier difficulty stats for {challenges[c_id].details.name}: (#nonces: {x['nonces']}, #solutions: {x['solutions']}, avg_nonces_per_solution: {avg_nonces_per_solution})")

    def run(self, difficulty_samples: Dict[str, List[int]]) -> SubmitPrecommitRequest:
        num_pending_jobs = db_conn.fetch_one(
            """
            SELECT COUNT(*) 
            FROM job
            WHERE merkle_proofs_ready IS NULL
                AND stopped IS NULL
            """
        )["count"]

        config = CONFIG["precommit_manager_config"]

        num_pending_benchmarks = num_pending_jobs + self.num_precommits_submitted
        if  num_pending_benchmarks >= config["max_pending_benchmarks"]:
            logger.debug(f"number of pending benchmarks has reached max of {config['max_pending_benchmarks']}")
            return
        selections = [
            (c_name, x) for c_name, x in config["algo_selection"].items()
            #if self.curr_base_fees[c_name] <= x.base_fee_limit
        ]
        if len(selections) == 0:
            logger.warning("No challenges under base fee limit")
            return None
        logger.debug(f"Selecting challenge from: {[(c_name, x['weight']) for c_name, x in selections]}")
        selection = random.choices(selections, weights=[x["weight"] for _, x in selections])[0]
        c_id = self.challenge_name_2_id[selection[0]]
        a_id = self.algorithm_name_2_id[f"{selection[0]}_{selection[1]['algorithm']}"]
        self.num_precommits_submitted += 1
        req = SubmitPrecommitRequest(
            settings=BenchmarkSettings(
                challenge_id=c_id,
                algorithm_id=a_id,
                player_id=CONFIG["player_id"],
                block_id=self.last_block_id,
                difficulty=difficulty_samples[selection[0]]
            ),
            num_nonces=selection[1]["num_nonces"]
        )
        logger.info(f"Created precommit (challenge: {selection[0]}, algorithm: {selection[1]['algorithm']}, difficulty: {req.settings.difficulty}, num_nonces: {req.num_nonces})")
        return req