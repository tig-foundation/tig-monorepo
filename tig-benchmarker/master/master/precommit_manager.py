import os
import logging
import random
from dataclasses import dataclass
from master.submissions_manager import SubmitPrecommitRequest
from common.structs import *
from common.utils import FromDict
from typing import Dict, List, Optional, Set
from master.sql import get_db_conn
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class PrecommitManager:
    def __init__(self):
        self.last_block_id = None
        self.num_precommits_submitted = 0
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}

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
        num_pending_jobs = get_db_conn().fetch_one(
            """
            SELECT COUNT(*) 
            FROM job
            WHERE merkle_proofs_ready IS NULL
                AND stopped IS NULL
            """
        )["count"]

        algo_selection = CONFIG["algo_selection"]

        num_pending_benchmarks = num_pending_jobs + self.num_precommits_submitted
        if  num_pending_benchmarks >= CONFIG["max_concurrent_benchmarks"]:
            logger.debug(f"number of pending benchmarks has reached max of {CONFIG['max_concurrent_benchmarks']}")
            return
        logger.debug(f"Selecting algorithm from: {[(x['algorithm_id'], x['weight']) for x in algo_selection]}")
        selection = random.choices(algo_selection, weights=[x["weight"] for x in algo_selection])[0]
        a_id = selection["algorithm_id"]
        c_id = a_id[:4]
        self.num_precommits_submitted += 1
        req = SubmitPrecommitRequest(
            settings=BenchmarkSettings(
                challenge_id=c_id,
                algorithm_id=a_id,
                player_id=CONFIG["player_id"],
                block_id=self.last_block_id,
                difficulty=difficulty_samples[a_id]
            ),
            num_nonces=selection["num_nonces"]
        )
        logger.info(f"Created precommit (algorithm_id: {a_id}, difficulty: {req.settings.difficulty}, num_nonces: {req.num_nonces})")
        return req