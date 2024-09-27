import argparse
import json
import os
import logging
import randomname
import requests
import subprocess
import time

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
logging.basicConfig(
    format='%(levelname)s - [%(name)s] - %(message)s',
    level=logging.INFO
)

def main(
    master_ip: str,
    tig_worker_path: str,
    download_wasms_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int
):
    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    os.makedirs(download_wasms_folder, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }
    
    while True:
        try:
            # Step 1: Query for job
            start = time.time()
            get_batch_url = f"http://{master_ip}:{master_port}/get-batch"
            logger.info(f"fetching job from {get_batch_url}")
            resp = requests.get(get_batch_url, headers=headers)
            if resp.status_code != 200:
                if resp.headers.get("Content-Type") == "text/plain":
                    raise Exception(f"status {resp.status_code} when fetching job: {resp.text}")
                else:
                    raise Exception(f"status {resp.status_code} when fetching job")
            logger.debug(f"fetching job: took {time.time() - start} seconds")
            batch = resp.json()
            batch_id = f"{batch['benchmark_id']}_{batch['start_nonce']}"
            logger.info(f"batch {batch_id}: {batch}")
            
            # Step 2: Download WASM
            wasm_path = os.path.join(download_wasms_folder, f"{batch['settings']['algorithm_id']}.wasm")
            if not os.path.exists(wasm_path):
                start = time.time()
                logger.info(f"downloading WASM from {batch['download_url']}")
                resp = requests.get(batch['download_url'])
                if resp.status_code != 200:
                    if resp.headers.get("Content-Type") == "text/plain":
                        raise Exception(f"status {resp.status_code} when downloading WASM: {resp.text}")
                    else:
                        raise Exception(f"status {resp.status_code} when downloading WASM")
                with open(wasm_path, 'wb') as f:
                    f.write(resp.content)
                logger.debug(f"downloading WASM: took {time.time() - start} seconds")
            logger.info(f"WASM Path: {wasm_path}")
            
            # Step 3: Run tig-worker
            start = time.time()
            cmd = [
                tig_worker_path, "compute_batch",
                json.dumps(batch["settings"]), 
                batch["rand_hash"], 
                str(batch["start_nonce"]), 
                str(batch["num_nonces"]),
                str(batch["batch_size"]), 
                wasm_path,
                "--mem", str(batch["wasm_vm_config"]["max_memory"]),
                "--fuel", str(batch["wasm_vm_config"]["max_fuel"]),
                "--workers", str(num_workers),
            ]
            if batch["sampled_nonces"]:
                cmd += ["--sampled", *map(str, batch["sampled_nonces"])]
            logger.info(f"running Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = json.loads(result.stdout)
            logger.debug(f"running command took {time.time() - start} seconds")
            logger.info(f"result: {result}")
            
            # Step 4: Submit results
            start = time.time()
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-result/{batch_id}"
            logger.info(f"posting results to {submit_url}")
            resp = requests.post(submit_url, json=result, headers=headers)
            if resp.status_code != 200:
                if resp.headers.get("Content-Type") == "text/plain":
                    raise Exception(f"status {resp.status_code} when downloading WASM: {resp.text}")
                else:
                    raise Exception(f"status {resp.status_code} when downloading WASM")
            logger.debug(f"posting results took {time.time() - start} seconds")
            
        except Exception as e:
            logger.error(e)
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("master_ip", help="IP address of the master")
    parser.add_argument("tig_worker_path", help="Path to tig-worker executable")
    parser.add_argument("--download", type=str, default="wasms", help="Folder to download WASMs to")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    
    args = parser.parse_args()
    
    main(args.master_ip, args.tig_worker_path, args.download, args.workers, args.name, args.port)