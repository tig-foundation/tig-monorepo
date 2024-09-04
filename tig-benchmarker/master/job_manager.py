import asyncio
import random
from master.config import *
from master.data import *
from master.utils import *
from collections import Counter

async def run(state: State):
    while True:
        try:
            print(f"[job_manager] checking status of running jobs")
            await _execute(state)
            print(f"[job_manager] done")
        except Exception as e:
            print(f"[job_manager] error: {e}")
        finally:
            await asyncio.sleep(5)

async def _execute(state: State):
    block = state.query_data.block
    challenges = state.query_data.challenges
    algorithms = state.query_data.algorithms
    wasms = state.query_data.wasms
    available_jobs = state.available_jobs
    pending_benchmark_jobs = state.pending_benchmark_jobs

    challenge_map = dict(
        **{
            c.id: c.details.name
            for c in challenges.values()
        },
        **{
            c.details.name: c.id
            for c in challenges.values()
        }
    )
    algorithm_map = dict(
        **{
            a.id: a.details.name
            for a in algorithms.values()
        },
        **{
            f"{a.details.challenge_id}_{a.details.name}": a.id
            for a in algorithms.values()
        }
    )

    n = now()
    finished_jobs = [
        benchmark_id
        for benchmark_id, job in available_jobs.items()
        if n >= job.timestamps.end
    ]
    for benchmark_id in list(available_jobs): 
        job = available_jobs[benchmark_id]
        if n >= job.timestamps.end:
            print(f"[job_manager] job {benchmark_id} FINISHED")
            pending_benchmark_jobs[benchmark_id] = available_jobs.pop(benchmark_id)

    job_counter = Counter(
        f"{challenge_map[job.settings.challenge_id]}_{algorithm_map[job.settings.algorithm_id]}"
        for job in available_jobs.values()
    )
    print(f"[job_manager] jobs counter: {job_counter}")
    new_jobs = []
    for challenge_name, selected_algorithms in JOBS.items():
        challenge_id = challenge_map[challenge_name]
        for algorithm_name, job_config in selected_algorithms.items():
            algorithm_id = algorithm_map.get(f"{challenge_id}_{algorithm_name}", None)
            assert algorithm_id is not None, f"Algorithm '{algorithm_name}' for challenge '{challenge_name}' does not exist"
            num_jobs = job_counter[f"{challenge_name}_{algorithm_name}"]
            if num_jobs >= job_config["num_jobs"]:
                continue
            weight = job_config["weight"] / job_config["num_jobs"]
            download_url = wasms[algorithm_id].details.download_url
            assert download_url is not None, f"Download URL for algorithm '{algorithm_id}' is None"
            timestamps = Timestamps(
                start=n,
                end=n + job_config["benchmark_duration"],
                submit=n + job_config["benchmark_duration"] + job_config["wait_slave_duration"]
            )
            for _ in range(job_config["num_jobs"] - num_jobs):
                # FIXME use difficulty sampler
                difficulty = random.choice(challenges[challenge_id].block_data.qualifier_difficulties)
                benchmark_id = f"{challenge_name}_{algorithm_name}_{difficulty[0]}_{difficulty[1]}_{now()}"
                print(f"[job_manager] job: {benchmark_id} CREATED")
                job = Job(
                    download_url=download_url,
                    benchmark_id=benchmark_id,
                    settings=BenchmarkSettings(
                        algorithm_id=algorithm_id,
                        challenge_id=challenge_id,
                        difficulty=difficulty,
                        player_id=PLAYER_ID,
                        block_id=block.id
                    ),
                    solution_signature_threshold=challenges[challenge_id].block_data.solution_signature_threshold,
                    sampled_nonces=None,
                    wasm_vm_config=block.config["wasm_vm"],
                    weight=weight,
                    timestamps=timestamps,
                    solutions_data={}
                )
                new_jobs.append(job)

    available_jobs.update({job.benchmark_id: job for job in new_jobs})