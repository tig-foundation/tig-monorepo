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


async def _execute(state: State) -> Job:
    block = state.query_data.block
    wasms = state.query_data.wasms
    benchmarks = state.query_data.benchmarks
    proofs = state.query_data.benchmarks
    frauds = state.query_data.benchmarks
    challenges = state.query_data.challenges
    algorithms = state.query_data.algorithms
    pending_proof_jobs = state.pending_proof_jobs
    submitted_proof_ids = state.submitted_proof_ids

    challenge_id_to_name = {
        c.id: c.details.name
        for c in challenges.values()
    }
    algorithm_id_to_name = {
        a.id: a.details.name
        for a in algorithms.values()
    }

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

            challenge_name = challenge_id_to_name[settings.challenge_id]
            algorithm_name = algorithm_id_to_name[settings.algorithm_id]

            wasm_path = f"{TIG_ALGORITHMS_FOLDER}/wasms/{challenge_name}/{algorithm_name}.wasm"
            if os.path.exists(wasm_path):
                print(f"[recomputer] wasm already downloaded for algorithm '{settings.algorithm_id}'")
            else:
                print(f"[recomputer] downloading wasm for algorithm '{settings.algorithm_id}' from {download_url}")
                os.makedirs("wasms", exist_ok=True)
                async with aiohttp.ClientSession() as session:
                    async with session.get(download_url) as response:
                        with open(wasm_path, "wb") as f:
                            f.write(await response.read())

            processes = {}
            for nonce in sampled_nonces:
                print(f"[recomputer] proof {benchmark_id}, nonce {nonce} RECOMPUTING")
                process = Popen(
                    [TIG_WORKER_PATH, 'compute_solution', json.dumps(asdict(settings)), str(nonce), wasm_path, "--fuel", str(block.config["wasm_vm"]["max_fuel"])],
                    stdout=PIPE,
                    stderr=PIPE
                )
                processes[nonce] = process

            solutions_data = {}
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
                            solutions_data[nonce] = SolutionData(**json.loads(solution_data))
                            print(f"[recomputer] proof {benchmark_id}, nonce {nonce} DONE")
                        except:
                            print(f"[recomputer] proof {benchmark_id}, nonce {nonce} ERROR: {p.stderr.read().decode()}")
                            failed = True
                    else:
                        print(f"[recomputer] proof {benchmark_id}, nonce {nonce} ERROR: {p.stderr.read().decode()}")
                        failed = True
                    processes.pop(nonce)                
                time.sleep(0.1)

            if failed:
                for nonce in processes:
                    processes[nonce].kill()  
            else:
                print(f"[recomputer] proof {benchmark_id} re-computed successfully")
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
                    solutions_data=solutions_data
                )