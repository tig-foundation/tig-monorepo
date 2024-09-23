import argparse
import json
import os
import randomname
import requests
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from tig_benchmarker.job_manager import Batch, BatchResult
from typing import List, Optional

def main(
    master_ip: str,
    tig_worker_path: str,
    tig_algorithms_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int
):
    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    if not os.path.exists(tig_algorithms_folder):
        raise FileNotFoundError(f"tig-algorithms folder not found at path: {tig_algorithms_folder}")

    headers = {
        "User-Agent": slave_name
    }
    
    while True:
        try:
            # Step 1: Query for job
            start = datetime.now()
            get_batch_url = f"http://{master_ip}:{master_port}/get-batch"
            print(f"Fetching Job: url={get_batch_url}, headers={headers}")
            response = requests.get(get_batch_url, headers=headers)
            response.raise_for_status()
            print(f"Fetching Job: took {(datetime.now() - start).total_seconds()} seconds")
            batch = Batch.from_dict(response.json())
            print(f"Batch: {batch}")
            
            # Step 2: Download WASM
            challenge_folder = os.path.join(tig_algorithms_folder, "wasms", batch.details.challenge)
            os.makedirs(challenge_folder, exist_ok=True)
            wasm_path = os.path.join(challenge_folder, f"{batch.details.algorithm}.wasm")
            if not os.path.exists(wasm_path):
                start = datetime.now()
                print(f"Downloading WASM: (from: {batch.details.download_url}, to: {wasm_path})")
                response = requests.get(batch.details.download_url)
                response.raise_for_status()
                with open(wasm_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloading WASM: took {(datetime.now() - start).total_seconds()} seconds")
            print(f"WASM Path: {wasm_path}")
            
            # Step 3: Run tig-worker
            start = datetime.now()
            cmd = [
                tig_worker_path, "compute_batch",
                json.dumps(asdict(batch.details.settings)), 
                batch.details.rand_hash, 
                str(batch.start_nonce), 
                str(batch.num_nonces),
                str(batch.details.batch_size), 
                wasm_path,
                "--mem", str(batch.details.wasm_vm_config["max_memory"]),
                "--fuel", str(batch.details.wasm_vm_config["max_fuel"]),
                "--workers", str(num_workers),
            ]
            if batch.sampled_nonces:
                cmd += ["--sampled", *map(str, batch.sampled_nonces)]
            print(f"Running Command: {' '.join(cmd)}")
            cmd_start = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = BatchResult.from_dict(json.loads(result.stdout))
            print(f"Running Command: took {(datetime.now() - cmd_start).total_seconds()} seconds")
            print(f"Result: {result}")
            
            # Step 4: Submit results
            start = datetime.now()
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-result/{batch.id}"
            print(f"Submitting Results: url={submit_url}, headers={headers}")
            submit_response = requests.post(submit_url, json=result.to_dict(), headers=headers)
            submit_response.raise_for_status()
            print(f"Submitting Results: took {(datetime.now() - cmd_start).total_seconds()} seconds")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("master_ip", help="IP address of the master")
    parser.add_argument("tig_worker_path", help="Path to tig-worker executable")
    parser.add_argument("tig_algorithms_folder", help="Path to tig-algorithms folder. Used to save WASMs")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    
    args = parser.parse_args()
    
    main(args.master_ip, args.tig_worker_path, args.tig_algorithms_folder, args.workers, args.name, args.port)