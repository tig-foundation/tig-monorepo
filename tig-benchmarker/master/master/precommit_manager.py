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
        for t_id in set(selection["track_settings"]) - set(challenge_config["active_tracks"]):
            selection["track_settings"].pop(t_id)
        for t_id in set(challenge_config["active_tracks"]) - set(selection["track_settings"]):
            selection["track_settings"][t_id] = {}
        
        for t_id in set(challenge_config["active_tracks"]):
            for k in set(selection["track_settings"][t_id]) - {"num_bundles", "hyperparameters", "fuel_budget"}:
                selection["track_settings"][t_id].pop(k)
            if selection["track_settings"][t_id].get("num_bundles", 0) < challenge_config["min_num_bundles"]:
                selection["track_settings"][t_id]["num_bundles"] = challenge_config["min_num_bundles"]
            if (
                selection["track_settings"][t_id].get("fuel_budget") is None or 
                selection["track_settings"][t_id]["fuel_budget"] < 0 or
                selection["track_settings"][t_id]["fuel_budget"] > challenge_config["max_fuel_budget"]
            ):
                selection["track_settings"][t_id]["fuel_budget"] = challenge_config["max_fuel_budget"]
            if "hyperparameters" not in selection["track_settings"][t_id]:
                selection["track_settings"][t_id]["hyperparameters"] = None

        self.num_precommits_submitted += 1
        req = SubmitPrecommitRequest(
            settings=BenchmarkSettings(
                challenge_id=c_id,
                algorithm_id=a_id,
                player_id=CONFIG["player_id"],
                block_id=self.last_block_id,
                track_id="",
            ),
            track_settings=selection["track_settings"],
        )
        logger.info(f"Created precommit with algorithm: {a_id}")
        return req