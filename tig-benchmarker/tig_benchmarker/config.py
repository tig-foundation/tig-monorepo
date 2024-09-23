from dataclasses import dataclass
from tig_benchmarker.data_fetcher import QueryData
from tig_benchmarker.utils import FromDict, PreciseNumber
import json

@dataclass
class ChallengeConfig(FromDict):
    algorithm: str
    num_nonces: int
    batch_size: int
    duration_before_batch_retry: int
    weight: float
    base_fee_limit: PreciseNumber

@dataclass
class Config(FromDict):
    max_precommits_per_block: int
    satisfiability: ChallengeConfig
    vehicle_routing: ChallengeConfig
    knapsack: ChallengeConfig
    vector_search: ChallengeConfig
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            return cls.from_dict(json.load(f))

    def validate(self, query_data: QueryData):
        assert self.max_precommits_per_block > 0, "max_precommits_per_block must be greater than 0"
        for c in query_data.challenges.values():
            algorithms = {
                a.details.name: a 
                for a in query_data.algorithms.values() 
                if a.details.challenge_id == c.id
            }
            assert hasattr(self, c.details.name), f"Missing config for challenge '{c.details.name}'"
            challenge_config = getattr(self, c.details.name)
            assert challenge_config.algorithm in algorithms, f"Invalid algorithm '{challenge_config.algorithm}' for challenge '{c.details.name}'"
            assert challenge_config.num_nonces > 0, "num_nonces must be greater than 0"
            assert challenge_config.batch_size > 0, "batch_size must be greater than 0"
            assert challenge_config.batch_size & (challenge_config.batch_size - 1) == 0, "batch_size must be a power of 2"

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def refresh_and_validate(self, query_data: QueryData):
        print(f"[config_manager] reloading config from '{self.config_path}'")
        self.config = Config.load(self.config_path)
        print(f"[config_manager] loaded {self.config}")
        print(f"[config_manager] validating")
        self.config.validate(query_data)
        print(f"[config_manager] OK")