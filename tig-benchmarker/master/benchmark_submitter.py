import aiohttp
import asyncio
import json
import random
from dataclasses import asdict
from typing import Dict, Any, Optional
from master.data import *
from master.config import *
from master.utils import *

async def run(state: State):
    while True:
        try:
            print(f"[benchmark_submitter] finding benchmark to submit")
            job = await _find_benchmark_to_submit(state)
            if job is None:
                print(f"[benchmark_submitter] no benchmark to submit")
            else:
                print(f"[benchmark_submitter] submitting benchmark {job.benchmark_id} with {len(job.solutions_data)} solutions")
                await _execute(state, job)
        except Exception as e:
            print(f"[benchmark_submitter] error: {e}")
        finally:
            await asyncio.sleep(5)

async def _execute(state: State, job: Job):
    solutions_meta_data = [
        SolutionMetaData(
            solution_signature=u32_from_str(json.dumps(asdict(s), sort_keys=True, separators=(',', ':'))),
            nonce=s.nonce
        )
        for s in job.solutions_data.values()
    ]
    headers = {
        "X-Api-Key": API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "tig-benchmarker-py/v0.1"
    }
    random_nonce = random.choice(list(job.solutions_data))
    payload = {
        "settings": asdict(job.settings),
        "solutions_meta_data": [asdict(s) for s in solutions_meta_data],
        "solution_data": asdict(job.solutions_data[random_nonce])
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/submit-benchmark", json=payload, headers=headers) as resp:
            if resp.status == 200:
                benchmark_id = await resp.json()["benchmark_id"]
                print(f"[benchmark_submitter] benchmark {job.benchmark_id} submitted successfully. benchmark_id={benchmark_id}")
                job.benchmark_id = benchmark_id
                state.pending_proof_jobs[benchmark_id] = job
            elif 400 <= resp.status < 500:
                resp_text = await resp.text()
                print(f"[benchmark_submitter] benchmark {job.benchmark_id} failed to submit: {resp_text}")
            elif resp.status >= 500:
                resp_text = await resp.text()
                print(f"[benchmark_submitter] benchmark {job.benchmark_id} failed to submit: {resp_text}")
                print(f"[benchmark_submitter] benchmark {job.benchmark_id} will be retried in 10 seconds")
                job.timestamps.submit = now() + 10000
                state.pending_benchmark_jobs[job.benchmark_id] = job

async def _find_benchmark_to_submit(state: State) -> Optional[Job]:
    pending_benchmark_jobs = state.pending_benchmark_jobs
    benchmarks = state.query_data.benchmarks
    proofs = state.query_data.proofs
    frauds = state.query_data.frauds
    
    n = now()
    pending_benchmark_ids = sorted([
        (job.timestamps.submit, benchmark_id)
        for benchmark_id, job in state.pending_benchmark_jobs.items()
        if n >= job.timestamps.submit
    ])
    for _, benchmark_id in pending_benchmark_ids:
        job = pending_benchmark_jobs.pop(benchmark_id)
        if (
            benchmark_id in benchmarks or
            benchmark_id in proofs or
            benchmark_id in frauds or
            len(job.solutions_data) == 0
        ):
            continue
        return job