import argparse
import json
import os
import randomname
import requests
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

@dataclass
class BenchmarkSettings:
    player_id: str
    block_id: str
    challenge_id: str
    algorithm_id: str
    difficulty: List[int]

def main(
    master_ip: str,
    tig_worker_path: str,
    wasm_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int,
    api_url: str
):
    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    if not os.path.exists(wasm_folder):
        raise FileNotFoundError(f"WASM folder not found at path: {wasm_folder}")

    headers = {
        "User-Agent": slave_name
    }
    get_job_url = f"http://{master_ip}:{master_port}/get-job"
    submit_results_url = f"http://{master_ip}:{master_port}/submit-results"
    
    while True:
        try:
            # Step 1: Query for job
            start = datetime.now()
            print(f"Fetching Job: url={get_job_url}, headers={headers}")
            response = requests.get(get_job_url, headers=headers)
            response.raise_for_status()
            job = response.json()
            job = Job(settings=BenchmarkSettings(**job.pop("settings")), **job)
            print(f"Fetching Job: took {(datetime.now() - start).total_seconds()} seconds")
            print(f"Job: {job}")
            
            # Step 2: Download WASM
            wasm_path = os.path.join(wasm_folder, f"{job.settings.algorithm_id}.wasm")
            if not os.path.exists(wasm_path):
                start = datetime.now()
                download_url = f"{api_url}/get-wasm-blob?algorithm_id={job.settings.algorithm_id}"
                print(f"Downloading WASM: {download_url}")
                response = requests.get(download_url)
                response.raise_for_status()
                with open(wasm_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloading WASM: took {(datetime.now() - start).total_seconds()} seconds")
            print(f"WASM Path: {wasm_path}")
            
            # Step 3: Run tig-worker
            start = datetime.now()
            cmd = [
                tig_worker_path, "compute_batch",
                json.dumps(asdict(job.settings)), 
                job.rand_hash, 
                str(job.start_nonce), 
                str(job.num_nonces),
                str(job.batch_size), 
                wasm_path,
                "--mem", str(job.wasm_vm_config["max_mem"]),
                "--fuel", str(job.wasm_vm_config["max_fuel"]),
                "--workers", str(num_workers),
            ]
            if job.sampled_nonces:
                cmd += ["--sampled", *map(str, job.sampled_nonces)]
            print(f"Running Command: {' '.join(cmd)}")
            cmd_start = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = json.loads(result.stdout)
            print(f"Running Command: took {(datetime.now() - cmd_start).total_seconds()} seconds")
            print(f"Result: {result}")
            
            # Step 4: Submit results
            start = datetime.now()
            print(f"Submitting Results: url={submit_results_url}/{job.id}, headers={headers}")
            submit_url = f"{submit_results_url}/{job.id}"
            submit_response = requests.post(submit_url, json=result, headers=headers)
            submit_response.raise_for_status()
            print(f"Submitting Results: took {(datetime.now() - cmd_start).total_seconds()} seconds")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("master_ip", help="IP address of the master")
    parser.add_argument("tig_worker_path", help="Path to tig-worker executable")
    parser.add_argument("wasm_folder", help="Path to folder to download WASMs")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    parser.add_argument("--api", type=str, default="https://mainnet-api.tig.foundation", help="TIG API URL (default: https://mainnet-api.tig.foundation)")
    
    args = parser.parse_args()
    
    main(args.master_ip, args.tig_worker_path, args.wasm_folder, args.workers, args.name, args.port, args.api)