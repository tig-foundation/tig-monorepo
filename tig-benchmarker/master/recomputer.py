import aiohttp
import asyncio
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Any
from datetime import datetime
from master.data import *
from master.config import *
from master.utils import now
from subprocess import Popen, PIPE

async def run(state: State):
    while True:
        job = None
        try:
            print(f"[recomputer] finding proof to recompute")
            job = await _find_proof_to_recompute(state)
            if job is None:
                print(f"[recomputer] no proof to recompute")
            else:
                print(f"[recomputer] re-computing proof {job.benchmark_id} with {len(job.sampled_nonces)} sampled nonces")
                await _execute(state, job)
            print(f"[recomputer] done")
        except Exception as e:
            print(f"[recomputer] error: {e}")
        finally:
            if job is None:
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(0.1)

async def _execute(state: State, job: Job):
    block = state.query_data.block
    challenge_name = next((c.details.name for c in state.query_data.challenges.values() if c.id == job.settings.challenge_id), None)
    assert challenge_name is not None, f"challenge {job.settings.challenge_id} not found"
    algorithm_name = next((a.details.name for a in state.query_data.algorithms.values() if a.id == job.settings.algorithm_id), None)
    assert algorithm_name is not None, f"algorithm {job.settings.algorithm_id} not found"

    wasm_path = f"{TIG_ALGORITHMS_FOLDER}/wasms/{challenge_name}/{algorithm_name}.wasm"
    if os.path.exists(wasm_path):
        print(f"[recomputer] wasm already downloaded for algorithm '{job.settings.algorithm_id}'")
    else:
        print(f"[recomputer] downloading wasm for algorithm '{job.settings.algorithm_id}' from {job.download_url}")
        os.makedirs(os.path.dirname(wasm_path), exist_ok=True)
        async with aiohttp.ClientSession() as session:
            async with session.get(job.download_url) as response:
                if response.status == 200:
                    with open(wasm_path, "wb") as f:
                        f.write(await response.read())
                else:
                    raise Exception(f"status HTTP {response.status}, error downloading {job.download_url}")

    processes = {}
    for nonce in job.sampled_nonces:
        print(f"[recomputer] proof {job.benchmark_id}, nonce {nonce} RECOMPUTING")
        process = Popen(
            [TIG_WORKER_PATH, 'compute_solution', json.dumps(asdict(job.settings)), str(nonce), wasm_path, "--fuel", str(block.config["wasm_vm"]["max_fuel"])],
            stdout=PIPE,
            stderr=PIPE
        )
        processes[nonce] = process

    failed = False
    while len(processes) > 0 and not failed:
        for nonce in list(processes):
            p = processes[nonce]
            return_code = p.poll()
            if return_code is None:
                continue
            elif return_code == 0:
                try:
                    solution_data = p.stdout.read().decode()
                    job.solutions_data[nonce] = SolutionData(**json.loads(solution_data))
                    print(f"[recomputer] proof {job.benchmark_id}, nonce {nonce} DONE")
                except:
                    print(f"[recomputer] proof {job.benchmark_id}, nonce {nonce} ERROR: {p.stderr.read().decode()}")
                    failed = True
            else:
                print(f"[recomputer] proof {job.benchmark_id}, nonce {nonce} ERROR: {p.stderr.read().decode()}")
                failed = True
            processes.pop(nonce)                
        time.sleep(0.1)

    if failed:
        for nonce in processes:
            processes[nonce].kill()  
    else:
        print(f"[recomputer] proof {job.benchmark_id} re-computed successfully")
        state.pending_proof_jobs[job.benchmark_id] = job

async def _find_proof_to_recompute(state: State) -> Job:
    block = state.query_data.block
    wasms = state.query_data.wasms
    benchmarks = state.query_data.benchmarks
    proofs = state.query_data.proofs
    frauds = state.query_data.frauds
    pending_proof_jobs = state.pending_proof_jobs
    submitted_proof_ids = state.submitted_proof_ids

    for benchmark_id, benchmark in benchmarks.items():
        if (
            benchmark_id not in pending_proof_jobs and
            benchmark_id not in submitted_proof_ids and
            benchmark_id not in frauds and
            benchmark_id not in proofs and
            benchmark.state is not None
        ):
            print(f"[recomputer] found proof to recompute: {benchmark_id}")
            settings = benchmark.settings
            sampled_nonces = benchmark.state.sampled_nonces
            assert sampled_nonces is not None
            download_url = wasms[settings.algorithm_id].details.download_url
            assert download_url is not None

            n = now()
            return Job(
                benchmark_id=benchmark_id,
                download_url=download_url,
                settings=settings,
                solution_signature_threshold=None,
                sampled_nonces=sampled_nonces,
                wasm_vm_config=block.config["wasm_vm"],
                weight=0.0,
                timestamps=Timestamps(
                    start=n,
                    end=n,
                    submit=n
                ),
                solutions_data={}
            )