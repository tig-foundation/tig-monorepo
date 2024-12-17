import argparse
import json
import logging
import os
import threading
import time
from master.data_fetcher import *
from master.difficulty_sampler import *
from master.job_manager import *
from master.precommit_manager import *
from master.slave_manager import *
from master.submissions_manager import *
from master.client_manager import *
from common.utils import FromDict

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    last_block_id = None

    client_manager = ClientManager()
    client_manager.start()

    data_fetcher = DataFetcher()
    difficulty_sampler = DifficultySampler()
    job_manager = JobManager()
    precommit_manager = PrecommitManager()
    submissions_manager = SubmissionsManager()
    
    slave_manager = SlaveManager()
    slave_manager.start()

    while True:
        try:
            data = data_fetcher.run()
            if data["block"].id != last_block_id:
                last_block_id = data["block"].id
                client_manager.on_new_block(**data)
                difficulty_sampler.on_new_block(**data)
                job_manager.on_new_block(**data)
                submissions_manager.on_new_block(**data)
                precommit_manager.on_new_block(**data)
            job_manager.run()
            samples = difficulty_sampler.run()
            submit_precommit_req = precommit_manager.run(samples)
            submissions_manager.run(submit_precommit_req)
            slave_manager.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"{e}")
        finally:
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Benchmarker")
    
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if os.environ.get("VERBOSE") else logging.INFO
    )
    
    main()