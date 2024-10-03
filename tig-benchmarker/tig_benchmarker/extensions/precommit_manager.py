import asyncio
import os
import logging
import random
from dataclasses import dataclass
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class AlgorithmSelectionConfig(FromDict):
    algorithm: str
    base_fee_limit: PreciseNumber
    num_nonces: int
    weight: Optional[float]

@dataclass
class PrecommitManagerConfig(FromDict):
    max_unresolved_precommits: int
    algo_selection: Dict[str, AlgorithmSelectionConfig]

class Extension:
    def __init__(self, player_id: str, backup_folder: str, precommit_manager: dict, **kwargs):
        self.player_id = player_id
        self.config = PrecommitManagerConfig.from_dict(precommit_manager)
        self.last_block_id = None
        self.num_unresolved_precommits = None
        self.curr_base_fees = {}
        self.difficulty_samples = {}
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.percent_qualifiers_by_challenge = {}
        self.num_solutions_by_challenge = {}
        self.lock = True

    async def on_new_block(
        self, 
        block: Block, 
        challenges: Dict[str, Challenge], 
        algorithms: Dict[str, Algorithm], 
        precommits: Dict[str, Precommit], 
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        player: Optional[Player],
        **kwargs
    ):
        if self.last_block_id == block.id:
            return
        self.num_unresolved_precommits = sum(1 for benchmark_id in precommits if benchmark_id not in proofs)
        self.last_block_id = block.id
        for c in challenges.values():
            c_name = c.details.name
            assert c_name in self.config.algo_selection, f"missing algorithm selection for challenge {c_name}"
            self.challenge_name_2_id[c_name] = c.id
            a_name = self.config.algo_selection[c_name].algorithm
            a = next(
                (
                    a for a in algorithms.values()
                    if a.details.challenge_id == c.id and a.details.name == a_name
                ),
                None
            )
            assert a is not None, f"selected non-existent algorithm {a_name} for challenge {c_name}"
            self.algorithm_name_2_id[f"{c_name}_{a_name}"] = a.id
            if c.block_data is not None:
                self.curr_base_fees[c_name] = c.block_data.base_fee
        logger.info(f"current base_fees: {self.curr_base_fees}")
        if (
            player is None or 
            player.block_data is None or 
            player.block_data.num_qualifiers_by_challenge is None
        ):
            self.percent_qualifiers_by_challenge = {
                c.details.name: 0
                for c in challenges.values()
                if c.block_data is not None
            }
        else:
            self.percent_qualifiers_by_challenge = {
                c.details.name: player.block_data.num_qualifiers_by_challenge.get(c.id, 0) / c.block_data.num_qualifiers
                for c in challenges.values()
                if c.block_data is not None
            }
        logger.info(f"percent_qualifiers_by_challenge: {self.percent_qualifiers_by_challenge}")
        self.num_solutions_by_challenge = {
            c.details.name: 0
            for c in challenges.values()
            if c.block_data is not None
        }
        for benchmark_id, benchmark in benchmarks.items():
            precommit = precommits[benchmark_id]
            c_name = challenges[precommit.settings.challenge_id].details.name
            num_solutions = benchmark.details.num_solutions
            self.num_solutions_by_challenge[c_name] += num_solutions
        logger.info(f"num_solutions_by_challenge: {self.num_solutions_by_challenge}")
        self.lock = False

    async def on_difficulty_samples(self, challenge_id: str, samples: list, **kwargs):
        self.difficulty_samples[challenge_id] = samples

    async def on_update(self):
        if self.lock:
            return
        if self.num_unresolved_precommits >= self.config.max_unresolved_precommits:
            logger.debug(f"reached max unresolved precommits: {self.config.max_unresolved_precommits}")
            return
        algo_selection = [
            (c_name, selection)
            for c_name, selection in self.config.algo_selection.items()
            if self.curr_base_fees[c_name] <= selection.base_fee_limit
        ]
        if len(algo_selection) == 0:
            logger.warning("no challenges within base fee limit")
            return
        if any(x[1].weight is not None for x in algo_selection):
            logger.debug(f"using config weights to randomly select a challenge + algorithm: {self.config.algo_selection}")
            selection = random.choices(algo_selection, weights=[x[1].weight or 1e-12 for x in algo_selection])[0]
        elif (max_percent := max(v for v in self.percent_qualifiers_by_challenge.values())) > 0:
            logger.debug(f"using percent qualifiers to randomly select a challenge + algorithm: {self.percent_qualifiers_by_challenge}")
            selection = random.choices(
                algo_selection,
                weights=[max_percent - self.percent_qualifiers_by_challenge.get(x[0], 0) + 1e-12 for x in algo_selection]
            )[0]
        else:
            logger.debug(f"using number of solutions to randomly select a challenge + algorithm: {self.num_solutions_by_challenge}")
            max_solution = max(v for v in self.num_solutions_by_challenge.values())
            selection = random.choices(
                algo_selection,
                weights=[max_solution - self.num_solutions_by_challenge.get(x[0], 0) + 1e-12 for x in algo_selection]
            )[0]
        c_name = selection[0]
        c_id = self.challenge_name_2_id[c_name]
        a_name = selection[1].algorithm
        a_id = self.algorithm_name_2_id[f"{c_name}_{a_name}"]
        num_nonces = selection[1].num_nonces

        difficulty_samples = self.difficulty_samples.get(c_id, [])
        if len(difficulty_samples) == 0:
            return
        difficulty = difficulty_samples.pop()
        settings = BenchmarkSettings(
            player_id=self.player_id,
            block_id=self.last_block_id,
            algorithm_id=a_id,
            challenge_id=c_id,
            difficulty=difficulty
        )
        logger.debug(f"created precommit: (settings: {settings}, num_nonces: {num_nonces})")
        await emit(
            "precommit_ready",
            settings=settings,
            num_nonces=num_nonces
        )
        