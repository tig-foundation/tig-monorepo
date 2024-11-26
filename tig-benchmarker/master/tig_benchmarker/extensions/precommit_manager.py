import os
import logging
import random
from dataclasses import dataclass
from tig_benchmarker.extensions.job_manager import Job
from tig_benchmarker.extensions.submissions_manager import SubmitPrecommitRequest
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import Dict, List, Optional, Set

from tig_benchmarker.database.init import SessionLocal
from tig_benchmarker.database.models.index import JobModel
from sqlalchemy.exc import SQLAlchemyError

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
    def __init__(self, config: PrecommitManagerConfig, player_id: str):
        self.config = config
        self.player_id = player_id
        self.last_block_id = None
        self.num_precommits_submitted = 0
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.curr_base_fees = {}
        # Initialize database session
        self.db_session = SessionLocal()
        logger.info("PrecommitManager initialized and connected to the database.")

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        challenges: Dict[str, Challenge],
        algorithms: Dict[str, Algorithm],
        player: Optional[Player],
        difficulty_data: Dict[str, List[DifficultyData]],
        **kwargs
    ):
        self.last_block_id = block.id
        self.num_precommits_submitted = 0
        self.challenge_name_2_id = {
            c.details.name: c.id
            for c in challenges.values()
        }
        self.algorithm_name_2_id = {}
        for a in algorithms.values():
            challenge_id = a.details.challenge_id
            if challenge_id not in challenges:
                logger.warning(f"Challenge ID '{challenge_id}' not found in challenges dictionary.")
                continue
            challenge_name = challenges[challenge_id].details.name
            key = f"{challenge_name}_{a.details.name}"
            self.algorithm_name_2_id[key] = a.id

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
            precommit = precommits.get(benchmark.id)
            if not precommit:
                logger.warning(f"No precommit found for benchmark ID '{benchmark.id}'.")
                continue
            c_id = precommit.settings.challenge_id
            if c_id not in challenges:
                logger.warning(f"Challenge ID '{c_id}' for benchmark '{benchmark.id}' not found.")
                continue
            c_name = challenges[c_id].details.name
            benchmark_stats_by_challenge[c_name]["solutions"] += benchmark.details.num_solutions
            benchmark_stats_by_challenge[c_name]["nonces"] += precommit.details.num_nonces

        #if player is not None and player.block_data is not None and player.block_data.reward:
        #    logger.info(f"player earnings: (latest: {player.block_data.reward.to_float()}, round: {player.block_data.round_earnings.to_float()})")
        #    logger.info(f"player stats: (cutoff: {player.block_data.cutoff}, imbalance: {player.block_data.imbalance.to_float() * 100}%)")
        #    for c_id, num_qualifiers in player.block_data.num_qualifiers_by_challenge.items():
        #        if c_id not in challenges:
        #            logger.warning(f"Challenge ID '{c_id}' in player.block_data.num_qualifiers_by_challenge not found.")
        #            continue
        #        c_name = challenges[c_id].details.name
        #        benchmark_stats_by_challenge[c_name]["qualifiers"] = num_qualifiers

        if player is not None and player.state is not None:
            logger.info(f"player fee balance: (available: {player.state.available_fee_balance.to_float()}, paid: {player.state.total_fees_paid.to_float()})")

        #for c_name, x in benchmark_stats_by_challenge.items():
        #    avg_nonces_per_solution = (x["nonces"] // x["solutions"]) if x["solutions"] > 0 else 0
        #    logger.info(f"benchmark stats for {c_name}: (#nonces: {x['nonces']}, #solutions: {x['solutions']}, #qualifiers: {x['qualifiers']}, avg_nonces_per_solution: {avg_nonces_per_solution})")

        #if player is not None and player.block_data and any(x['qualifiers'] == player.block_data.cutoff for x in benchmark_stats_by_challenge.values()):
        #    c_name = min(benchmark_stats_by_challenge, key=lambda x: benchmark_stats_by_challenge[x]['solutions'])
        #    logger.warning(f"recommend finding more solutions for challenge {c_name} to avoid being cut off")

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
            if c_id in challenges
        }
        for c_id, x in aggregate_difficulty_data.items():
            c_name = challenges[c_id].details.name
            avg_nonces_per_solution = (x["nonces"] // x["solutions"]) if x["solutions"] > 0 else 0
            logger.info(f"global qualifier difficulty stats for {c_name}: (#nonces: {x['nonces']}, #solutions: {x['solutions']}, avg_nonces_per_solution: {avg_nonces_per_solution})")

    def run(self, difficulty_samples: Dict[str, List[int]]) -> Optional[SubmitPrecommitRequest]:
        try:
            # Fetch all pending jobs (i.e. Merkle Root is None)
            pending_jobs = self.db_session.query(JobModel).filter_by(merkle_root=None).all()
            num_pending_benchmarks = len(pending_jobs) + self.num_precommits_submitted
            if num_pending_benchmarks >= self.config.max_pending_benchmarks:
                logger.debug(f"number of pending benchmarks has reached max of {self.config.max_pending_benchmarks}")
                return None
            selections = [
                (c_name, x) for c_name, x in self.config.algo_selection.items()
                if c_name in self.curr_base_fees and self.curr_base_fees[c_name] <= x.base_fee_limit
            ]
            if len(selections) == 0:
                logger.warning("No challenges under base fee limit")
                return None
            logger.debug(f"Selecting challenge from: {[(c_name, x.weight) for c_name, x in selections]}")
            selection = random.choices(selections, weights=[x.weight for _, x in selections])[0]
            c_name = selection[0]
            if c_name not in self.challenge_name_2_id:
                logger.warning(f"Challenge name '{c_name}' not found in challenge_name_2_id mapping.")
                return None
            c_id = self.challenge_name_2_id[c_name]
            algo_key = f"{c_name}_{selection[1].algorithm}"
            if algo_key not in self.algorithm_name_2_id:
                logger.warning(f"Algorithm key '{algo_key}' not found in algorithm_name_2_id mapping.")
                return None
            a_id = self.algorithm_name_2_id[algo_key]
            self.num_precommits_submitted += 1
            if c_name not in difficulty_samples:
                logger.warning(f"No difficulty samples provided for challenge '{c_name}'.")
                return None
            req = SubmitPrecommitRequest(
                settings=BenchmarkSettings(
                    challenge_id=c_id,
                    algorithm_id=a_id,
                    player_id=self.player_id,
                    block_id=self.last_block_id,
                    difficulty=difficulty_samples[c_name]
                ),
                num_nonces=selection[1].num_nonces
            )
            logger.info(f"Created precommit (challenge: {c_name}, algorithm: {selection[1].algorithm}, difficulty: {req.settings.difficulty}, num_nonces: {req.num_nonces})")
            return req
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during run: {e}")
            return None
        except KeyError as e:
            logger.error(f"KeyError during run: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during run: {e}")
            return None
