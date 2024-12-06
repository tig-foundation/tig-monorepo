import argparse
import json
import os
import logging
import randomname
import requests
import subprocess
import time
import threading
from tig_benchmarker.structs import OutputData, MerkleProof
from tig_benchmarker.merkle_tree import MerkleTree

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def now():
    return int(time.time() * 1000)

def download_wasm(session, download_url, wasm_path):
    if not os.path.exists(wasm_path):
        start = now()
        logger.info(f"downloading WASM from {download_url}")
        resp = session.get(download_url)
        if resp.status_code != 200:
            raise Exception(f"status {resp.status_code} when downloading WASM: {resp.text}")
        with open(wasm_path, 'wb') as f:
            f.write(resp.content)
        logger.debug(f"downloading WASM: took {now() - start}ms")
    logger.debug(f"WASM Path: {wasm_path}")

def run_tig_worker(tig_worker_path, batch, wasm_path, num_workers, output_path):
    start = now()
    cmd = [
        tig_worker_path, "compute_batch",
        json.dumps(batch["settings"]), 
        batch["rand_hash"], 
        str(batch["start_nonce"]), 
        str(batch["num_nonces"]),
        str(batch["batch_size"]), 
        wasm_path,
        "--mem", str(batch["runtime_config"]["max_memory"]),
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--workers", str(num_workers),
        "--output", f"{output_path}/{batch['benchmark_id']}_{batch['batch_idx']}",
    ]
    #if batch["sampled_nonces"]:
    #    cmd += ["--sampled", *map(str, batch["sampled_nonces"])]
    logger.info(f"computing batch: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"tig-worker failed: {stderr.decode()}")
    result = json.loads(stdout.decode())
    logger.info(f"computing batch took {now() - start}ms")
    logger.debug(f"batch result: {result}")
    return result

def process_batch(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers, output_path):
    batch_id = None
    result = None
    try:
        batch_id = f"{batch['benchmark_id']}_{batch['batch_idx']}"
        logger.info(f"Processing batch {batch_id}: {batch}")

        # Step 3: Run tig-worker only if output folder does not exist
        output_folder = f"{output_path}/{batch['benchmark_id']}_{batch['batch_idx']}"
        if not os.path.exists(output_folder):
            wasm_path = os.path.join(download_wasms_folder, f"{batch['settings']['algorithm_id']}.wasm")
            download_wasm(session, batch['download_url'], wasm_path)

            result = run_tig_worker(tig_worker_path, batch, wasm_path, num_workers, output_path)
        else:
            logger.info(f"Output folder {output_folder} already exists, skipping tig-worker") 

        if len(batch["sampled_nonces"]) > 0:
            leafs = {}
            for nonce in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"]):
                with open(f"{output_folder}/{nonce}.json") as f:
                    leafs[nonce] = OutputData.from_dict(json.load(f))
                
            merkle_tree = MerkleTree(
                [x.to_merkle_hash() for x in leafs.values()],
                batch["batch_size"]
            )

            proofs_to_submit = [
                MerkleProof(
                    leaf=leafs[n],
                    branch=merkle_tree.calc_merkle_branch(branch_idx=n - batch["start_nonce"])
                ).to_dict()
                for n in batch["sampled_nonces"]
            ]
            # send proofs
            start = now()
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-proofs/{batch_id}"
            logger.info(f"posting results to {submit_url}")
            resp = session.post(submit_url, json={"merkle_proofs": proofs_to_submit}, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"status {resp.status_code} when posting proofs to master: {resp.text}")
            logger.debug(f"posting proofs took {now() - start} ms")
        else:
            # send root
            start = now()
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-root/{batch_id}"
            logger.info(f"posting results to {submit_url}")
            resp = session.post(submit_url, json=result, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"status {resp.status_code} when posting root to master: {resp.text}")
            logger.debug(f"posting root took {now() - start} ms")
            
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")

pending_jobs = []
pending_jobs_lock = threading.Lock()
def process_pending_jobs():
    while True:
        if len(pending_jobs) == 0:
            time.sleep(1)
            continue
        to_remove = []
        for i, (batch, thread) in enumerate(pending_jobs):
            if not thread.is_alive():
                to_remove.append(i)

        if len(to_remove) > 0:
            pending_jobs_lock.acquire()
            for i in reversed(to_remove):
                pending_jobs.pop(i)
            pending_jobs_lock.release()

def main(
    master_ip: str,
    tig_worker_path: str,
    download_wasms_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int,
    output_path: str,
):
    print(f"Starting slave {slave_name}")

    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    os.makedirs(download_wasms_folder, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }
    
    has_exceeded_pending_jobs_limit = False
    last_pending_jobs = 0
    session = requests.Session()
    while True:
        try:
            if has_exceeded_pending_jobs_limit:
                if len(pending_jobs) < last_pending_jobs:
                    has_exceeded_pending_jobs_limit = False
                else:
                    time.sleep(0.5)
                    continue

            # Step 1: Query for job
            start = now()
            get_batch_url = f"http://{master_ip}:{master_port}/get-batch"
            logger.info(f"fetching job from {get_batch_url}")
            resp = session.get(get_batch_url, headers=headers)
            if resp.status_code == 425: # too early
                has_exceeded_pending_jobs_limit = True
                last_pending_jobs = len(pending_jobs)
                raise Exception(f"too many pending jobs, cooling down for 5 seconds")
            elif resp.status_code != 200:
                raise Exception(f"status {resp.status_code} when fetching job: {resp.text}")
            
            batch = resp.json()
            print(f"Fetched batch: {batch}")
            logger.debug(f"fetching job: took {now() - start}ms")

            thread = threading.Thread(
                target=process_batch,
                args=(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers, output_path)
            )
            thread.start()
            
            pending_jobs_lock.acquire()
            pending_jobs.append((batch, thread))
            pending_jobs_lock.release()

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("master_ip", help="IP address of the master")
    parser.add_argument("tig_worker_path", help="Path to tig-worker executable")
    parser.add_argument("--download", type=str, default="wasms", help="Folder to download WASMs to (default: wasms)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    parser.add_argument("--verbose", action='store_true', help="Print debug logs")
    parser.add_argument("--output", type=str, default="results", help="Folder to output results to (default: output)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    threading.Thread(target=process_pending_jobs, daemon=True).start()
    main(args.master_ip, args.tig_worker_path, args.download, args.workers, args.name, args.port, args.output)
