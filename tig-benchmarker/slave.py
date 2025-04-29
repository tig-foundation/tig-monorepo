#!/usr/bin/env python3

import argparse
import json
import os
import logging
import randomname
import requests
import shutil
import subprocess
import time
import zlib
from threading import Thread
from common.structs import OutputData, MerkleProof
from common.merkle_tree import MerkleTree, MerkleHash

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
PENDING_BATCH_IDS = set()
PROCESSING_BATCH_IDS = set()
READY_BATCH_IDS = set()
FINISHED_BATCH_IDS = {}

def now():
    return int(time.time() * 1000)

def download_library(download_url, library_path):
    if not os.path.exists(library_path):
        start = now()
        logger.info(f"downloading algorithm library from {download_url}")
        resp = requests.get(download_url)
        if resp.status_code != 200:
            raise Exception(f"status {resp.status_code} when downloading algorithm library: {resp.text}")
        with open(library_path, 'wb') as f:
            f.write(resp.content)
        logger.debug(f"downloading algorithm library: took {now() - start}ms")
    logger.debug(f"Library Path: {library_path}")


def run_tig_worker(tig_worker_path, tig_runtime_path, batch, library_path, num_workers, output_path):
    start = now()
    cmd = [
        tig_worker_path,
        tig_runtime_path,
        json.dumps(batch["settings"]),
        batch["rand_hash"],
        str(batch["start_nonce"]), 
        str(batch["num_nonces"]),
        str(batch["batch_size"]),
        library_path,
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--workers", str(num_workers),
        "--output", f"{output_path}/{batch['id']}",
    ]
    logger.info(f"computing batch: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while True:
        ret = process.poll()
        if ret is not None:
            if ret != 0:
                PROCESSING_BATCH_IDS.remove(batch["id"])
                raise Exception(f"tig-worker failed with return code {ret}")
            
            stdout, stderr = process.communicate()
            result = json.loads(stdout.decode())
            logger.info(f"computing batch {batch['id']} took {now() - start}ms")
            logger.debug(f"batch {batch['id']} result: {result}")
            with open(f"{output_path}/{batch['id']}/result.json", "w") as f:
                json.dump(result, f)
            
            PROCESSING_BATCH_IDS.remove(batch["id"])
            READY_BATCH_IDS.add(batch["id"])
            break

        elif batch["id"] not in PROCESSING_BATCH_IDS:
            process.kill()
            logger.info(f"batch {batch['id']} stopped")
            break
        
        time.sleep(0.1)
    

def purge_folders(output_path, ttl):
    n = now()
    purge_batch_ids = [
        batch_id
        for batch_id, finish_time in FINISHED_BATCH_IDS.items()
        if n >= finish_time + (ttl * 1000)
    ]
    if len(purge_batch_ids) == 0:
        time.sleep(5)
        return

    for batch_id in purge_batch_ids:
        if os.path.exists(f"{output_path}/{batch_id}"):
            logger.info(f"purging batch {batch_id}")
            shutil.rmtree(f"{output_path}/{batch_id}", ignore_errors=True)
        FINISHED_BATCH_IDS.pop(batch_id)


def send_results(headers, master_ip, master_port, tig_worker_path, downloads_folder, num_workers, output_path):
    try:
        batch_id = READY_BATCH_IDS.pop()
    except KeyError:
        logger.debug("No batches to send")
        time.sleep(1)
        return

    if now() - FINISHED_BATCH_IDS.get(batch_id, 0) < 10000:
        logger.debug(f"Batch {batch_id} submitted recently")
        return
    
    output_folder = f"{output_path}/{batch_id}"
    with open(f"{output_folder}/batch.json") as f:
        batch = json.load(f)

    if (
        not os.path.exists(f"{output_folder}/result.json")
        or not os.path.exists(f"{output_folder}/data.zlib")
        or not os.path.exists(f"{output_folder}/hashes.zlib")
    ):
        if os.path.exists(f"{output_folder}/result.json"):
            os.remove(f"{output_folder}/result.json")
        logger.debug(f"Batch {batch_id} flagged as ready, but missing nonce files")
        PENDING_BATCH_IDS.add(batch_id)
        return

    if batch["sampled_nonces"] is None:
        with open(f"{output_folder}/result.json") as f:
            result = json.load(f)
        with open(f"{output_folder}/hashes.zlib", "rb") as f:
            hashes = json.loads(zlib.decompress(f.read()).decode())
        hash_threshold = batch["hash_threshold"].lower()
        within_threshold_solutions = [
            n for n in result["solution_nonces"]
            if hashes[n - batch["start_nonce"]].lower() <= hash_threshold
        ]
        logger.info(f"Batch {batch_id} has {len(within_threshold_solutions)} out of {len(result['solution_nonces'])} solutions within threshold")
        result["solution_nonces"] = within_threshold_solutions

        submit_url = f"http://{master_ip}:{master_port}/submit-batch-root/{batch_id}"
        logger.info(f"posting root to {submit_url}")
        resp = requests.post(submit_url, headers=headers, json=result)
        if resp.status_code == 200:
            FINISHED_BATCH_IDS[batch_id] = now()
            logger.info(f"successfully posted root for batch {batch_id}")
        elif resp.status_code == 408: # took too long 
            FINISHED_BATCH_IDS[batch_id] = now()
            logger.error(f"status {resp.status_code} when posting root for batch {batch_id} to master: {resp.text}")
        else:
            logger.error(f"status {resp.status_code} when posting root for batch {batch_id} to master: {resp.text}")
            READY_BATCH_IDS.add(batch_id) # requeue
            time.sleep(2)

    else:
        with open(f"{output_folder}/hashes.zlib", "rb") as f:
            hashes = json.loads(zlib.decompress(f.read()).decode())
        with open(f"{output_folder}/data.zlib", "rb") as f:
            leafs = json.loads(zlib.decompress(f.read()).decode())
        
        merkle_tree = MerkleTree(
            [MerkleHash.from_str(x) for x in hashes],
            batch["batch_size"]
        )

        proofs_to_submit = [
            dict(
                leaf=leafs[n - batch["start_nonce"]],
                branch=merkle_tree.calc_merkle_branch(branch_idx=n - batch["start_nonce"]).to_str()
            )
            for n in batch["sampled_nonces"]
        ]
        
        submit_url = f"http://{master_ip}:{master_port}/submit-batch-proofs/{batch_id}"
        logger.info(f"posting proofs to {submit_url}")
        resp = requests.post(submit_url, headers=headers, json={"merkle_proofs": proofs_to_submit})
        if resp.status_code == 200:
            FINISHED_BATCH_IDS[batch_id] = now()
            logger.info(f"successfully posted proofs for batch {batch_id}")
        elif resp.status_code == 408: # took too long 
            FINISHED_BATCH_IDS[batch_id] = now()
            logger.error(f"status {resp.status_code} when posting proofs for batch {batch_id} to master: {resp.text}")
        else:
            logger.error(f"status {resp.status_code} when posting proofs for batch {batch_id} to master: {resp.text}")
            READY_BATCH_IDS.add(batch_id) # requeue
            time.sleep(2)


def process_batch(tig_worker_path, tig_runtime_path, downloads_folder, num_workers, output_path):
    try:
        batch_id = PENDING_BATCH_IDS.pop()
    except KeyError:
        logger.debug("No pending batches")
        time.sleep(1)
        return

    if (
        batch_id in PROCESSING_BATCH_IDS or
        batch_id in READY_BATCH_IDS
    ):
        return
    
    if os.path.exists(f"{output_path}/{batch_id}/result.json"):
        logger.info(f"Batch {batch_id} already processed")
        READY_BATCH_IDS.add(batch_id)
        return
    PROCESSING_BATCH_IDS.add(batch_id)

    with open(f"{output_path}/{batch_id}/batch.json") as f:
        batch = json.load(f)

    
    library_path = os.path.join(
        downloads_folder,
        batch['settings']['algorithm_id']
    )
    download_library(batch['download_url'], library_path)
    
    Thread(
        target=run_tig_worker,
        args=(tig_worker_path, tig_runtime_path, batch, library_path, num_workers, output_path)
    ).start()


def poll_batches(headers, master_ip, master_port, output_path):    
    get_batches_url = f"http://{master_ip}:{master_port}/get-batches"
    logger.info(f"fetching batches from {get_batches_url}")
    resp = requests.get(get_batches_url, headers=headers)

    if resp.status_code == 200:
        batches = resp.json()
        root_batch_ids = [batch['id'] for batch in batches if batch['sampled_nonces'] is None]
        proofs_batch_ids = [batch['id'] for batch in batches if batch['sampled_nonces'] is not None]
        logger.info(f"root batches: {root_batch_ids}")
        logger.info(f"proofs batches: {proofs_batch_ids}")
        for batch in batches:
            output_folder = f"{output_path}/{batch['id']}"
            os.makedirs(output_folder, exist_ok=True)
            with open(f"{output_folder}/batch.json", "w") as f:
                json.dump(batch, f)
        PENDING_BATCH_IDS.clear()
        PENDING_BATCH_IDS.update(root_batch_ids + proofs_batch_ids)
        for batch_id in PROCESSING_BATCH_IDS - set(root_batch_ids + proofs_batch_ids):
            logger.info(f"stopping batch {batch_id}")
            PROCESSING_BATCH_IDS.remove(batch_id)
        time.sleep(5)

    else:
        logger.error(f"status {resp.status_code} when fetching batch: {resp.text}")
        time.sleep(5)


def wrap_thread(func, *args):
    logger.info(f"Starting thread for {func.__name__}")
    while True:
        try:
            func(*args)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            time.sleep(5)


def main(
    master_ip: str,
    tig_worker_path: str,
    tig_runtime_path: str,
    downloads_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int,
    output_path: str,
    ttl: int,
):
    print(f"Starting slave with config:")
    print(f"  Slave Name: {slave_name}")
    print(f"  Master IP: {master_ip}")
    print(f"  Master Port: {master_port}")
    print(f"  Worker Path: {tig_worker_path}")
    print(f"  Runtime Path: {tig_runtime_path}")
    print(f"  Downloads Folder: {downloads_folder}")
    print(f"  Number of Workers: {num_workers}")
    print(f"  Output Path: {output_path}")
    print(f"  TTL: {ttl}")
    print(f"  Verbose: {args.verbose}")

    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    if not os.path.exists(tig_runtime_path):
        raise FileNotFoundError(f"tig-runtime not found at path: {tig_runtime_path}")
    os.makedirs(downloads_folder, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }

    Thread(
        target=wrap_thread,
        args=(process_batch, tig_worker_path, tig_runtime_path, downloads_folder, num_workers, output_path)
    ).start()

    Thread(
        target=wrap_thread,
        args=(send_results, headers, master_ip, master_port, tig_worker_path, downloads_folder, num_workers, output_path)
    ).start()

    Thread(
        target=wrap_thread,
        args=(purge_folders, output_path, ttl)
    ).start()

    wrap_thread(poll_batches, headers, master_ip, master_port, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("--tig_worker_path", type=str, default="tig-worker", help="Path to tig-worker executable")
    parser.add_argument("--tig_runtime_path", type=str, default="tig-runtime", help="Path to tig-runtime executable")
    parser.add_argument("--master", type=str, default="0.0.0.0", help="IP address of the master (default: 0.0.0.0)")
    parser.add_argument("--download", type=str, default="libs", help="Folder to download algorithm libraries to (default: libs)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    parser.add_argument("--verbose", action='store_true', help="Print debug logs")
    parser.add_argument("--output", type=str, default="results", help="Folder to output results to (default: results)")
    parser.add_argument("--ttl", type=int, default=300, help="(Time To Live) Seconds to retain results (default: 300)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    main(args.master, args.tig_worker_path, args.tig_runtime_path, args.download, args.workers, args.name, args.port, args.output, args.ttl)
