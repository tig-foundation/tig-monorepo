import argparse
import json
import logging
import os
import threading
import time
from extensions.data_fetcher import *
from extensions.difficulty_sampler import *
from extensions.job_manager import *
from extensions.precommit_manager import *
from extensions.slave_manager import *
from extensions.submissions_manager import *
from extensions.client_manager import *
from tig_benchmarker.utils import FromDict

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    last_block_id = None
    jobs = []

    client_manager = ClientManager()
    client_manager.start()

    config = get_config()
    print(config)

    data_fetcher = DataFetcher(config["api_url"], config["player_id"])
    difficulty_sampler = DifficultySampler()
    job_manager = JobManager(jobs)
    precommit_manager = PrecommitManager(config["player_id"], jobs)
    submissions_manager = SubmissionsManager(config["api_url"], config["api_key"], jobs)
    
    slave_manager = SlaveManager(jobs)
    slave_manager.start()

    while True:
        try:
            data = data_fetcher.run()
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
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Benchmarker")
    parser.add_argument("config_path", help="Path to the configuration JSON file")
    parser.add_argument("--verbose", action='store_true', help="Print debug logs")
    
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    
    #if not os.path.exists(args.config_path):
    #    logger.error(f"config file not found at path: {args.config_path}")
    #    sys.exit(1)
    #with open(args.config_path, "r") as f:
    #    config = json.load(f)
    #    config = Config.from_dict(config)
    
    main()