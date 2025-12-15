import os
import logging
import json
from master.submissions_manager import SubmitPrecommitRequest
from common.structs import *
from typing import Dict, List, Optional
from master.sql import get_db_conn
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class PrecommitManager:
    def __init__(self):
        self.last_block_id = None
        self.num_precommits_submitted = 0
        self.num_precommits_submitted_by_challenge: Dict[str, int] = {}
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.challenge_configs: Dict[str, dict] = {}
        self._challenge_config_by_id: Dict[str, dict] = {}

        self._challenge_rr_pool: List[str] = []
        self._challenge_rr_index = 0
        self._algo_rr_current_by_challenge: Dict[str, Dict[str, float]] = {}
        self._algo_rr_weight_by_challenge: Dict[str, Dict[str, float]] = {}
        self._algo_allowed_tracks_by_key: Dict[str, List[str]] = {}
        self._algo_item_by_key: Dict[str, dict] = {}
        self._algo_track_rr_index_by_key: Dict[str, int] = {}
        self._scheduler_fingerprint: Optional[str] = None

    def on_new_block(self, block: Block, **kwargs):
        self.last_block_id = block.id
        self.num_precommits_submitted = 0
        self.num_precommits_submitted_by_challenge = {}
        self.challenge_configs = block.config["challenges"]
        self._challenge_config_by_id = {
            self._normalize_id(c_id): cfg for c_id, cfg in (self.challenge_configs or {}).items()
        }

    def _normalize_id(self, value: str) -> str:
        return value.lower() if isinstance(value, str) else value

    def _get_max_concurrent_jobs_for_challenge(self, challenge_id: str) -> Optional[int]:
        challenge_id = self._normalize_id(challenge_id)
        overrides = CONFIG.get("max_concurrent_jobs_per_challenge_overrides", {}) or {}
        for k in (challenge_id, challenge_id.upper()):
            if k in overrides:
                return overrides[k]

        default_limit = CONFIG.get("max_concurrent_jobs_per_challenge", None)
        return default_limit

    def _algo_item_key(self, item: dict) -> str:
        payload = {
            "algorithm_id": item.get("algorithm_id"),
            "num_bundles": item.get("num_bundles"),
            "selected_track_ids": item.get("selected_track_ids") or [],
            "hyperparameters": item.get("hyperparameters"),
            "runtime_config": item.get("runtime_config") or {},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _config_fingerprint(self) -> str:
        active_tracks_by_challenge = {
            self._normalize_id(c_id): sorted((c_cfg.get("active_tracks") or {}).keys())
            for c_id, c_cfg in (self._challenge_config_by_id or {}).items()
        }
        payload = {
            "algo_selection": CONFIG.get("algo_selection") or [],
            "active_tracks_by_challenge": active_tracks_by_challenge,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _ensure_scheduler(self) -> None:
        fingerprint = self._config_fingerprint()
        if fingerprint == self._scheduler_fingerprint:
            return
        self._scheduler_fingerprint = fingerprint

        self._challenge_rr_pool = []
        self._algo_rr_weight_by_challenge = {}
        self._algo_item_by_key = {}
        self._algo_allowed_tracks_by_key = {}

        algo_selection = CONFIG.get("algo_selection") or []
        for c_id, challenge_config in (self._challenge_config_by_id or {}).items():
            active_tracks = set((challenge_config.get("active_tracks") or {}).keys())
            if not active_tracks:
                continue

            weights_by_key: Dict[str, float] = {}
            for item in algo_selection:
                a_id = item.get("algorithm_id")
                if not isinstance(a_id, str) or len(a_id) < 4:
                    continue
                if self._normalize_id(a_id[:4]) != c_id:
                    continue

                weight = item.get("weight", 0)
                if weight is None or float(weight) <= 0:
                    continue

                selected_track_ids = item.get("selected_track_ids") or []
                if selected_track_ids:
                    allowed_tracks = [t for t in selected_track_ids if t in active_tracks]
                else:
                    allowed_tracks = sorted(active_tracks)

                if not allowed_tracks:
                    continue

                key = self._algo_item_key(item)
                weights_by_key[key] = float(weight)
                self._algo_item_by_key[key] = item
                self._algo_allowed_tracks_by_key[key] = allowed_tracks

            if weights_by_key:
                self._algo_rr_weight_by_challenge[c_id] = weights_by_key
                self._algo_rr_current_by_challenge.setdefault(c_id, {})
                self._challenge_rr_pool.append(c_id)

        self._challenge_rr_pool.sort()
        if self._challenge_rr_pool:
            self._challenge_rr_index %= len(self._challenge_rr_pool)
        else:
            self._challenge_rr_index = 0

        # prune stale state
        active_challenges = set(self._challenge_rr_pool)
        for c_id in list(self._algo_rr_current_by_challenge.keys()):
            if c_id not in active_challenges:
                del self._algo_rr_current_by_challenge[c_id]
                continue
            active_keys = set(self._algo_rr_weight_by_challenge.get(c_id, {}).keys())
            current_map = self._algo_rr_current_by_challenge[c_id]
            for key in list(current_map.keys()):
                if key not in active_keys:
                    del current_map[key]

        for key in list(self._algo_track_rr_index_by_key.keys()):
            if key not in self._algo_item_by_key:
                del self._algo_track_rr_index_by_key[key]

    def _next_algo_key_for_challenge(self, challenge_id: str) -> Optional[str]:
        weights = self._algo_rr_weight_by_challenge.get(challenge_id) or {}
        if not weights:
            return None

        current = self._algo_rr_current_by_challenge.setdefault(challenge_id, {})
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return None

        selected_key = None
        selected_score = None
        for key, w in weights.items():
            current[key] = current.get(key, 0.0) + w
            score = current[key]
            if selected_score is None or score > selected_score:
                selected_key = key
                selected_score = score

        if selected_key is None:
            return None

        current[selected_key] = current.get(selected_key, 0.0) - total_weight
        return selected_key

    def _next_track_for_algo_key(self, algo_key: str) -> Optional[str]:
        tracks = self._algo_allowed_tracks_by_key.get(algo_key) or []
        if not tracks:
            return None
        idx = self._algo_track_rr_index_by_key.get(algo_key, 0) % len(tracks)
        self._algo_track_rr_index_by_key[algo_key] = idx + 1
        return tracks[idx]

    def _pending_jobs_by_challenge(self) -> Dict[str, int]:
        rows = get_db_conn().fetch_all(
            """
            SELECT settings->>'challenge_id' AS challenge_id, COUNT(*) AS count
            FROM job
            WHERE merkle_proofs_ready IS NULL
                AND stopped IS NULL
            GROUP BY settings->>'challenge_id'
            """
        )
        result: Dict[str, int] = {}
        for row in rows:
            c_id = self._normalize_id(row["challenge_id"])
            result[c_id] = int(row["count"])
        return result

    def run(self) -> Optional[SubmitPrecommitRequest]:
        self._ensure_scheduler()
        if not self._challenge_rr_pool:
            logger.debug("no eligible challenges/tracks in config to precommit with")
            return

        pending_jobs_by_challenge = self._pending_jobs_by_challenge()

        num_pending_jobs_total = sum(pending_jobs_by_challenge.values())
        num_pending_benchmarks = num_pending_jobs_total + self.num_precommits_submitted
        if num_pending_benchmarks >= CONFIG["max_concurrent_benchmarks"]:
            logger.debug(f"number of pending benchmarks has reached max of {CONFIG['max_concurrent_benchmarks']}")
            return

        selection = None
        c_id = None
        challenge_config = None
        track_id = None
        a_id = None

        for _ in range(len(self._challenge_rr_pool)):
            c_id = self._challenge_rr_pool[self._challenge_rr_index % len(self._challenge_rr_pool)]
            self._challenge_rr_index = (self._challenge_rr_index + 1) % len(self._challenge_rr_pool)

            max_for_challenge = self._get_max_concurrent_jobs_for_challenge(c_id)
            if max_for_challenge is not None:
                pending_for_challenge = pending_jobs_by_challenge.get(c_id, 0) + self.num_precommits_submitted_by_challenge.get(c_id, 0)
                if pending_for_challenge >= int(max_for_challenge):
                    continue

            algo_key = self._next_algo_key_for_challenge(c_id)
            if algo_key is None:
                continue

            track_id = self._next_track_for_algo_key(algo_key)
            if track_id is None:
                continue

            selection = dict(self._algo_item_by_key[algo_key])
            selection["runtime_config"] = dict(selection.get("runtime_config") or {})
            a_id = selection["algorithm_id"]
            challenge_config = self._challenge_config_by_id[c_id]
            break

        if selection is None:
            logger.debug("no eligible algorithms to precommit with (concurrency / challenge activity / track config)")
            return
        
        if selection["num_bundles"] < challenge_config["min_num_bundles"]:
            selection["num_bundles"] = challenge_config["min_num_bundles"]

        if (
            len(selection["runtime_config"]) > 1 or
            selection["runtime_config"].get("max_fuel") is None or 
            selection["runtime_config"]["max_fuel"] < 0 or
            selection["runtime_config"]["max_fuel"] > challenge_config["runtime_config_limits"]["max_fuel"]
        ):
            selection["runtime_config"] = {"max_fuel": challenge_config["runtime_config_limits"]["max_fuel"]}

        self.num_precommits_submitted += 1
        self.num_precommits_submitted_by_challenge[c_id] = self.num_precommits_submitted_by_challenge.get(c_id, 0) + 1
        req = SubmitPrecommitRequest(
            settings=BenchmarkSettings(
                challenge_id=c_id,
                algorithm_id=a_id,
                player_id=CONFIG["player_id"],
                block_id=self.last_block_id,
                track_id=track_id,
            ),
            num_bundles=selection["num_bundles"],
            hyperparameters=selection["hyperparameters"],
            runtime_config={
                **challenge_config["runtime_config_limits"],
                **selection["runtime_config"]
            },
        )
        logger.info(f"Created precommit (algorithm_id: {a_id}, track: {req.settings.track_id}, num_bundles: {req.num_bundles}, hyperparameters: {req.hyperparameters}, runtime_config: {req.runtime_config})")
        return req
