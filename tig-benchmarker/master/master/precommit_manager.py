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

    def on_new_block(self, block: Block, **kwargs):
        self.last_block_id = block.id
        self.num_precommits_submitted = 0
        self.challenge_configs = block.config["challenges"]

    def run(self) -> SubmitPrecommitRequest:
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
        if c_id not in self.challenge_configs:
            logger.error(f"Invalid selected challenge_id '{c_id}'. Valid challenge_ids: {sorted(self.challenge_configs)}")
            return
        challenge_config = self.challenge_configs[c_id]
        selected_track_ids = sorted(set(selection["selected_track_ids"]) & set(challenge_config["active_tracks"]))
        if len(selected_track_ids) == 0:
            selected_track_ids = sorted(challenge_config["active_tracks"])
        selection["selected_track_ids"] = selected_track_ids
        
        if selection["num_bundles"] < challenge_config["min_num_bundles"]:
            selection["num_bundles"] = challenge_config["min_num_bundles"]

        for k, v in challenge_config["runtime_config_limits"].items():
            if (
                selection["runtime_config"].get(k) is None or 
                selection["runtime_config"][k] < 0 or
                selection["runtime_config"][k] > v
            ):
                selection["runtime_config"][k] = v

        self.num_precommits_submitted += 1
        req = SubmitPrecommitRequest(
            settings=BenchmarkSettings(
                challenge_id=c_id,
                algorithm_id=a_id,
                player_id=CONFIG["player_id"],
                block_id=self.last_block_id,
                track_id=random.choice(selection["selected_track_ids"]),
            ),
            num_bundles=selection["num_bundles"],
            hyperparameters=selection["hyperparameters"],
            runtime_config=selection["runtime_config"],
        )
        logger.info(f"Created precommit (algorithm_id: {a_id}, track: {req.settings.track_id}, num_bundles: {req.num_bundles}, hyperparameters: {req.hyperparameters}, runtime_config: {req.runtime_config})")
        return req