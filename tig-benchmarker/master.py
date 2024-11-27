import argparse
import asyncio
import json
import logging
import os
from tig_benchmarker.extensions.data_fetcher import *
from tig_benchmarker.extensions.difficulty_sampler import *
from tig_benchmarker.extensions.job_manager import *
from tig_benchmarker.extensions.precommit_manager import *
from tig_benchmarker.extensions.slave_manager import *
from tig_benchmarker.extensions.submissions_manager import *
from tig_benchmarker.utils import FromDict

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class Config(FromDict):
    player_id: str
    api_key: str
    api_url: str
    difficulty_sampler_config: DifficultySamplerConfig
    job_manager_config: JobManagerConfig
    precommit_manager_config: PrecommitManagerConfig
    slave_manager_config: SlaveManagerConfig
    submissions_manager_config: SubmissionsManagerConfig

async def main(config: Config):
    last_block_id = None
    jobs = []

    config.player_id = config.player_id.lower()
    config.api_url = config.api_url.rstrip("/")

    data_fetcher = DataFetcher(config.api_url, config.player_id)
    difficulty_sampler = DifficultySampler(config.difficulty_sampler_config)
    job_manager = JobManager(config.job_manager_config, jobs)
    precommit_manager = PrecommitManager(config.precommit_manager_config, config.player_id, jobs)
    submissions_manager = SubmissionsManager(config.submissions_manager_config, config.api_url, config.api_key, jobs)
    slave_manager = SlaveManager(config.slave_manager_config, jobs)
    slave_manager.start()

    while True:
        try:
            data = await data_fetcher.run()
            if data["block"].id != last_block_id:
                last_block_id = data["block"].id
                difficulty_sampler.on_new_block(**data)
                job_manager.on_new_block(**data)
                precommit_manager.on_new_block(**data)
            job_manager.run()
            samples = difficulty_sampler.run()
            submit_precommit_req = precommit_manager.run(samples)
            submissions_manager.run(submit_precommit_req)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"{e}")
        finally:
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Benchmarker")
    parser.add_argument("config_path", help="Path to the configuration JSON file")
    parser.add_argument("--verbose", action='store_true', help="Print debug logs")
    
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    if not os.path.exists(args.config_path):
        logger.error(f"config file not found at path: {args.config_path}")
        sys.exit(1)
    with open(args.config_path, "r") as f:
        config = json.load(f)
        config = Config.from_dict(config)
    asyncio.run(main(config))