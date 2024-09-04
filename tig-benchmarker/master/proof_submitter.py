import aiohttp
import asyncio
import json
from dataclasses import asdict
from typing import Dict, Any, Optional
from master.data import *
from master.utils import *
from master.config import *

async def run(state: State):
    while True:
        try:
            print(f"[proof_submitter] finding proof to submit")
            job = await _find_proof_to_submit(state)
            if job is None:
                print(f"[proof_submitter] no proof to submit")
            else:
                print(f"[proof_submitter] submitting proof {job.benchmark_id} with {len(job.sampled_nonces)} sampled nonces")
                await _execute(state, job)
        except Exception as e:
            print(f"[proof_submitter] error: {e}")
        finally:
            await asyncio.sleep(5)

async def _execute(state: State, job: Job):
    solutions_data = [
        job.solutions_data[nonce]
        for nonce in job.sampled_nonces
    ]
    headers = {
        "X-Api-Key": API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "tig-benchmarker-py/v0.1"
    }
    payload = {
        "benchmark_id": job.benchmark_id,
        "solutions_data": [asdict(s) for s in solution_data]
    }

    state.submitted_proof_ids.add(job.benchmark_id)
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/submit-proof", json=payload, headers=headers) as resp:
            if resp.status == 200:
                print(f"[proof_submitter] proof {job.benchmark_id} submitted successfully")
            elif 400 <= resp.status < 500:
                resp_text = await resp.text()
                print(f"[proof_submitter] proof {job.benchmark_id} failed to submit: {resp_text}")
            elif resp.status >= 500:
                resp_text = await resp.text()
                print(f"[proof_submitter] proof {job.benchmark_id} failed to submit: {resp_text}")
                print(f"[proof_submitter] proof {job.benchmark_id} will be retried in 10 seconds")
                job.timestamps.submit = now() + 10000
                state.pending_benchmark_jobs[job.benchmark_id] = job
                state.submitted_proof_ids.remove(job.benchmark_id)

async def _find_proof_to_submit(state: State) -> Optional[Job]:
    pending_proof_jobs = state.pending_proof_jobs
    benchmarks = state.query_data.benchmarks
    proofs = state.query_data.proofs
    frauds = state.query_data.frauds

    n = now()
    pending_proof_ids = sorted([
        (job.timestamps.submit, benchmark_id)
        for benchmark_id, job in state.pending_proof_jobs.items()
        if n >= job.timestamps.submit
    ])
    for _, benchmark_id in pending_proof_ids:
        if (
            benchmark_id in benchmarks and
            benchmarks[benchmark_id].state is not None and
            benchmarks[benchmark_id].state.sampled_nonces is not None
        ):
            job = pending_proof_jobs.pop(benchmark_id)
            if benchmark_id in proofs or benchmark_id in frauds:
                continue
            job.sampled_nonces = benchmarks[benchmark_id].state.sampled_nonces
            return job