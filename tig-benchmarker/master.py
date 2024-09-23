import asyncio
import argparse
import os
import random
import sys
from quart import Quart, request, jsonify
from hypercorn.config import Config
from hypercorn.asyncio import serve
from tig_benchmarker.config import ConfigManager
from tig_benchmarker.data_fetcher import DataFetcher
from tig_benchmarker.difficulty_manager import DifficultyManager
from tig_benchmarker.job_manager import JobManager, Batch, BatchResult
from tig_benchmarker.structs import BenchmarkSettings
from tig_benchmarker.submissions_manager import SubmissionsManager, SubmitBenchmarkRequest, SubmitPrecommitRequest, SubmitProofRequest

app = Quart(__name__)

# Global variables
config_manager = None
job_manager = None
data_fetcher = None
submissions_manager = None
batches = {}
priority_batches = {}
difficulty_manager = None

@app.route('/get-batch', methods=['GET'])
async def get_batch():
    if not priority_batches and not batches:
        return jsonify({"error": "No batches available"}), 404
    _, batch = priority_batches.popitem() if priority_batches else batches.popitem()
    return jsonify(batch.to_dict())

@app.route('/submit-batch-result/<batch_id>', methods=['POST'])
async def submit_batch_result(batch_id):
    benchmark_id, start_nonce = batch_id.split("_")
    job_manager.jobs[benchmark_id].update_result(
        start_nonce=int(start_nonce),
        result=BatchResult.from_dict(await request.json)
    )
    return "OK"

async def main(
    player_id: str,
    api_key: str,
    config_path: str, 
    jobs_folder: str, 
    api_url: str
):
    global config_manager, job_manager, data_fetcher, submissions_manager, batches, priority_batches, difficulty_manager

    config_manager = ConfigManager(config_path)
    job_manager = JobManager(jobs_folder)
    data_fetcher = DataFetcher(api_url, player_id)
    submissions_manager = SubmissionsManager(api_url, api_key)
    difficulty_manager = DifficultyManager()
    latest_block_id = None
    num_precommits_submitted = 0

    while True:
        try:
            query_data = await data_fetcher.fetch()
            
            if latest_block_id is None or latest_block_id != query_data.block.id:
                num_precommits_submitted = 0
                difficulty_manager.update_with_query_data(query_data)
                config_manager.refresh_and_validate(query_data)
                job_manager.update_with_query_data(query_data, config_manager.config)
                latest_block_id = query_data.block.id
            ret = job_manager.generate_batches(config_manager.config)
            batches.update(ret[0])
            priority_batches.update(ret[1])

            submissions = []
            print(f"[main] {num_precommits_submitted} precommits already submitted for block '{latest_block_id}' (max: {config_manager.config.max_precommits_per_block})")
            if num_precommits_submitted < config_manager.config.max_precommits_per_block:
                challenge_weights = []
                for c in query_data.challenges.values():
                    challenge_config = getattr(config_manager.config, c.details.name)
                    print(f"[main] challenge: {c.details.name}, base_fee: {c.block_data.base_fee}, config.base_fee_limit: {challenge_config.base_fee_limit}")
                    if c.block_data.base_fee < challenge_config.base_fee_limit:
                        challenge_weights.append((
                            c.details.name,
                            challenge_config.weight
                        ))
                if len(challenge_weights) == 0:
                    print("[main] no challenges with base_fee below limit")
                else:
                    print(f"[main] weighted sampling challenges: {challenge_weights}")
                    challenge = random.choices(
                        [c for c, _ in challenge_weights],
                        weights=[w for _, w in challenge_weights]
                    )[0]
                    challenge_config = getattr(config_manager.config, challenge)
                    print(f"[main] selected challenge: {challenge}, algorithm: {challenge_config.algorithm}, num_nonces: {challenge_config.num_nonces}")
                    difficulty = difficulty_manager.sample(challenge)
                    num_precommits_submitted += 1
                    req = SubmitPrecommitRequest(
                        settings=BenchmarkSettings(
                            player_id=player_id,
                            algorithm_id=next(a.id for a in query_data.algorithms.values() if a.details.name == challenge_config.algorithm),
                            challenge_id=next(c.id for c in query_data.challenges.values() if c.details.name == challenge),
                            difficulty=difficulty,
                            block_id=query_data.block.id
                        ),
                        num_nonces=challenge_config.num_nonces
                    )
                    submissions.append(submissions_manager.post(req))

            if (job := submissions_manager.find_benchmark_to_submit(query_data, job_manager)) is not None:
                req = SubmitBenchmarkRequest(
                    benchmark_id=job.details.benchmark_id,
                    merkle_root=job.merkle_root(),
                    solution_nonces=job.solution_nonces()
                )
                submissions.append(submissions_manager.post(req))

            if (job := submissions_manager.find_proof_to_submit(query_data, job_manager)) is not None:
                req = SubmitProofRequest(
                    benchmark_id=job.details.benchmark_id,
                    merkle_proofs=job.merkle_proofs()
                )
                submissions.append(submissions_manager.post(req))

            await asyncio.gather(*submissions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[main] error: {e}")
        finally:
            print(f"[main] sleeping 5s")
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Benchmarker")
    parser.add_argument("player_id", help="Player ID")
    parser.add_argument("api_key", help="API Key")
    parser.add_argument("config_path", help="Path to the configuration JSON file")
    parser.add_argument("jobs_folder", help="Folder to save jobs until their proofs are submitted (create your own folder)")
    parser.add_argument("--port", type=int, default=5115, help="Port to run the server on (default: 5115)")
    parser.add_argument("--api", default="https://mainnet-api.tig.foundation", help="API URL (default: https://mainnet-api.tig.foundation)")
    
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"[main] fatal error: config file not found at path: {args.config_path}")
        sys.exit(1)
    if not os.path.exists(args.jobs_folder):
        print(f"[main] fatal error: jobs folder not found at path: {args.jobs_folder}")
        sys.exit(1)

    config = Config()
    config.bind = [f"localhost:{args.port}"]

    async def start():
        await asyncio.gather(
            serve(app, config), 
            main(args.player_id, args.api_key, args.config_path, args.jobs_folder, args.api)
        )

    asyncio.run(start())