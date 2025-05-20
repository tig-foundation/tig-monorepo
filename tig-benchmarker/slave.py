#!/usr/bin/env python3

import argparse
import io
import json
import os
import platform
import logging
import randomname
import re
import requests
import shutil
import subprocess
import sys
import tarfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from threading import Thread
from common.structs import OutputData, MerkleProof
from common.merkle_tree import MerkleTree, MerkleHash

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
PENDING_BATCH_IDS = set()
PROCESSING_BATCH_IDS = set()
READY_BATCH_IDS = set()
FINISHED_BATCH_IDS = {}
TOTAL_COSTS = {"cpu": 0, "gpu": 0}
if (CPU_ARCH := platform.machine().lower()) in ["x86_64", "amd64"]:
    CPU_ARCH = "amd64"
elif CPU_ARCH in ["arm64", "aarch64"]:
    CPU_ARCH = "aarch64"
else:
    print(f"Unsupported CPU architecture: {CPU_ARCH}")
    sys.exit(1)
HAS_GPU = subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0

def now():
    return int(time.time() * 1000)

def download_library(downloads_folder, batch):
    challenge_folder = f"{downloads_folder}/{batch['challenge']}"
    so_path = f"{challenge_folder}/{CPU_ARCH}/{batch['algorithm']}.so"
    ptx_path = f"{challenge_folder}/ptx/{batch['algorithm']}.ptx"
    if not os.path.exists(so_path):
        start = now()
        logger.info(f"downloading {batch['algorithm']}.tar.gz from {batch['download_url']}")
        resp = requests.get(batch['download_url'], stream=True)
        if resp.status_code != 200:
            raise Exception(f"status {resp.status_code} when downloading algorithm library: {resp.text}")
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            tar.extractall(path=challenge_folder)
        logger.debug(f"downloading {batch['algorithm']}.tar.gz took {now() - start}ms")

    if not os.path.exists(ptx_path):
        return so_path, None
    elif not HAS_GPU:
        raise Exception(f"Algorithm {batch['algorithm']} requires GPU support, but GPU not found")
    else:
        return so_path, ptx_path


def run_tig_runtime(nonce, tig_runtime_path, batch, so_path, ptx_path, output_path):
    start = now()
    output_file = f"{output_path}/{batch['id']}/{nonce}.json"
    cmd = [
        tig_runtime_path,
        json.dumps(batch["settings"]),
        batch["rand_hash"],
        str(nonce),
        so_path,
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--output", output_file,
    ]
    if ptx_path is not None:
        cmd += ["--ptx", ptx_path]
    logger.debug(f"computing batch: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while True:
        ret = process.poll()
        if ret is not None:
            # exit codes:
            # 0 - success
            # 1 - runtime error
            # 85 - no solution
            # 86 - invalid solution
            # 87 - out of fuel
            if (ret == 1 or ret == 87) and not os.path.exists(output_file):
                d = OutputData(
                    nonce=nonce,
                    runtime_signature=0,
                    fuel_consumed=(ret == 87) and (batch["runtime_config"]["max_fuel"] + 1),
                    solution={},
                    cpu_arch=CPU_ARCH,
                )
                with open(output_file, "wb") as f:
                    json.dump(d.to_dict(), f)

            assert os.path.exists(output_file), f"Exit code {ret}. Output file does not exist"
            if ret == 1:
                raise Exception(process.stderr.read().decode())
            
            break

        elif batch["id"] not in PROCESSING_BATCH_IDS:
            process.kill()
            logger.debug(f"batch {batch['id']}, nonce {nonce} stopped")
            break
        
        time.sleep(0.1)

    logger.debug(f"batch {batch['id']}, nonce {nonce} finished, took {now() - start}ms")


def compute_merkle_root(batch, output_path):
    start = now()
    while True:
        if batch["id"] not in PROCESSING_BATCH_IDS:
            logger.info(f"batch {batch['id']} stopped")
            break

        processing_nonces = set(
            n for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"])
            if not os.path.exists(f"{output_path}/{batch['id']}/{n}.json")
        )
        if len(processing_nonces) > 0:
            logger.debug(f"batch {batch['id']} still processing nonces: {processing_nonces}")
            time.sleep(1.5)
            continue

        hashes = []
        solution_nonces = []
        for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"]):
            with open(f"{output_path}/{batch['id']}/{n}.json", "r") as f:
                d = OutputData.from_dict(json.load(f))
                if len(d.solution) > 0:
                    solution_nonces.append(n)
                hashes.append(d.to_merkle_hash())

        merkle_tree = MerkleTree(hashes, batch["batch_size"])
        with open(f"{output_path}/{batch['id']}/hashes.zlib", "wb") as f:
            hashes = [h.to_str() for h in hashes]
            f.write(zlib.compress(json.dumps(hashes).encode()))
        with open(f"{output_path}/{batch['id']}/result.json", "w") as f:
            result = {
                "solution_nonces": list(solution_nonces),
                "merkle_root": merkle_tree.calc_merkle_root().to_str(),
            }
            logger.debug(f"batch {batch['id']} result: {result}")
            json.dump(result, f)
        logger.info(f"batch {batch['id']} done, took: {now() - start}ms")
        
        PROCESSING_BATCH_IDS.remove(batch["id"])
        READY_BATCH_IDS.add(batch["id"])
        break
    

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


def send_results(headers, master_ip, master_port, output_path):
    try:
        batch_id = READY_BATCH_IDS.pop()
    except KeyError:
        logger.debug("no batches to send")
        time.sleep(1)
        return

    if now() - FINISHED_BATCH_IDS.get(batch_id, 0) < 10000:
        logger.debug(f"batch {batch_id} submitted recently")
        return
    
    output_folder = f"{output_path}/{batch_id}"
    with open(f"{output_folder}/batch.json") as f:
        batch = json.load(f)

    if (
        not os.path.exists(f"{output_folder}/result.json")
        or not all(
            os.path.exists(f"{output_folder}/{n}.json") 
            for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"])
        )
        or not os.path.exists(f"{output_folder}/hashes.zlib")
    ):
        if os.path.exists(f"{output_folder}/result.json"):
            os.remove(f"{output_folder}/result.json")
        logger.debug(f"batch {batch_id} flagged as ready, but missing nonce files")
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
        logger.info(f"batch {batch_id} has {len(within_threshold_solutions)} out of {len(result['solution_nonces'])} solutions within threshold")
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
        leafs = []
        for n in batch["sampled_nonces"]:
            with open(f"{output_folder}/{n}.json", "r") as f:
                leafs.append(json.load(f))
        
        merkle_tree = MerkleTree(
            [MerkleHash.from_str(x) for x in hashes],
            batch["batch_size"]
        )

        proofs_to_submit = [
            dict(
                leaf=leaf,
                branch=merkle_tree.calc_merkle_branch(branch_idx=n - batch["start_nonce"]).to_str()
            )
            for n, leaf in zip(batch["sampled_nonces"], leafs)
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


def process_batch(pool, tig_runtime_path, downloads_folder, config, output_path):
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
        logger.info(f"batch {batch_id} already processed")
        READY_BATCH_IDS.add(batch_id)
        return
    PROCESSING_BATCH_IDS.add(batch_id)

    with open(f"{output_path}/{batch_id}/batch.json") as f:
        batch = json.load(f)

    so_path, ptx_path = download_library(downloads_folder, batch)
    c = next(
        (
            x for x in config["algorithms"]
            if re.match(x["id_regex"], batch["settings"]["algorithm_id"])
        ),
        None
    )
    if c is None:
        logger.error(f"Algorithm {batch['settings']['algorithm_id']} does not match any regex in the config")
        return

    logger.info(f"batch {batch['id']} started")
    pool.submit(compute_merkle_root, batch, output_path)

    def process_nonce(nonce):
        logger.debug(f"batch {batch['id']}, nonce {nonce} started: (cpu_cost {c['cpu_cost']}, gpu_cost {c['gpu_cost']})")
        try:
            run_tig_runtime(nonce, tig_runtime_path, batch, so_path, ptx_path, output_path)
        except Exception as e:
            logger.error(f"batch {batch['id']}, nonce {nonce}, runtime error: {e}")
        finally:
            TOTAL_COSTS["cpu"] -= c["cpu_cost"]
            TOTAL_COSTS["gpu"] -= c["gpu_cost"]

    nonce = batch["start_nonce"]
    while nonce < batch["start_nonce"] + batch["num_nonces"]:
        if batch["id"] not in PROCESSING_BATCH_IDS:
            logger.info(f"batch {batch['id']} stopped")
            break

        if (
            TOTAL_COSTS["cpu"] + c["cpu_cost"] <= config["cpus"] and 
            TOTAL_COSTS["gpu"] + c["gpu_cost"] <= config["gpus"]
        ):
            TOTAL_COSTS["cpu"] += c["cpu_cost"]
            TOTAL_COSTS["gpu"] += c["gpu_cost"]
            pool.submit(process_nonce, nonce)
            nonce += 1
        else:
            time.sleep(0.1)


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
    tig_runtime_path: str,
    downloads_folder: str,
    config_path: str,
    slave_name: str,
    master_port: int,
    output_path: str,
    ttl: int,
):
    if not os.path.exists(tig_runtime_path):
        logger.error(f"tig-runtime not found at path: {tig_runtime_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at path: {config_path}")
        sys.exit(1)
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

    print(f"Starting slave with config:")
    print(f"  Slave Name: {slave_name}")
    print(f"  Master IP: {master_ip}")
    print(f"  Master Port: {master_port}")
    print(f"  CPU Architecture: {CPU_ARCH}")
    print(f"  GPU Available: {HAS_GPU}")
    print(f"  Runtime Path: {tig_runtime_path}")
    print(f"  Downloads Folder: {downloads_folder}")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"  Output Path: {output_path}")
    print(f"  TTL: {ttl}")
    print(f"  Verbose: {args.verbose}")

    os.makedirs(downloads_folder, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }

    pool = ThreadPoolExecutor(max_workers=config["max_workers"])
    Thread(
        target=wrap_thread,
        args=(process_batch, pool, tig_runtime_path, downloads_folder, config, output_path)
    ).start()

    Thread(
        target=wrap_thread,
        args=(send_results, headers, master_ip, master_port, output_path)
    ).start()

    Thread(
        target=wrap_thread,
        args=(purge_folders, output_path, ttl)
    ).start()

    wrap_thread(poll_batches, headers, master_ip, master_port, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--tig_runtime_path", type=str, default=shutil.which("tig-runtime"), help="Path to tig-runtime executable")
    parser.add_argument("--master", type=str, default="0.0.0.0", help="IP address of the master (default: 0.0.0.0)")
    parser.add_argument("--download", type=str, default="libs", help="Folder to download algorithm libraries to (default: libs)")
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

    main(args.master, args.tig_runtime_path, args.download, args.config, args.name, args.port, args.output, args.ttl)
