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
from queue import Queue
from glob import glob
from threading import Thread
from common.structs import OutputData, MerkleProof
from common.merkle_tree import MerkleTree, MerkleHash

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
PENDING_BATCH_IDS = set()
PROCESSING_BATCH_IDS = {}
READY_BATCH_IDS = set()
FINISHED_BATCH_IDS = {}
TOTAL_COST = [0]
if (CPU_ARCH := platform.machine().lower()) in ["x86_64", "amd64"]:
    CPU_ARCH = "amd64"
elif CPU_ARCH in ["arm64", "aarch64"]:
    CPU_ARCH = "arm64"
else:
    print(f"Unsupported CPU architecture: {CPU_ARCH}")
    sys.exit(1)

def now():
    return int(time.time() * 1000)

def download_library(algorithms_dir, batch):
    challenge_folder = f"{algorithms_dir}/{batch['challenge']}"
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
    else:
        return so_path, ptx_path


def run_tig_runtime(nonce, batch, so_path, ptx_path, results_dir):
    output_file = f"{results_dir}/{batch['id']}/{nonce}.json"
    start = now()
    cmd = [
        "docker", "exec", batch["challenge"], "tig-runtime",
        json.dumps(batch["settings"], separators=(',',':')),
        batch["rand_hash"],
        str(nonce),
        so_path,
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--output", output_file,
    ]
    if ptx_path is not None:
        cmd += [
            "--ptx", ptx_path,
        ]
    logger.debug(f"computing batch: {' '.join(cmd[:4] + [f"'{cmd[4]}'"] + cmd[5:])}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while True:
        ret = process.poll()
        if ret is not None:
            # exit codes:
            # 0 - success
            # 84 - runtime error
            # 85 - no solution
            # 86 - invalid solution
            # 87 - out of fuel
            if (ret == 84 or ret == 87) and not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    json.dump(dict(
                        nonce=nonce,
                        runtime_signature=0,
                        fuel_consumed=(ret == 87) and (batch["runtime_config"]["max_fuel"] + 1),
                        solution={},
                        cpu_arch=CPU_ARCH
                    ), f)

            if ret not in {0, 84, 85, 86, 87}:
                logger.error(f"batch {batch['id']}, nonce {nonce} failed with exit code {ret}: {process.stderr.read().decode()}")
            
            break

        elif batch["id"] not in PROCESSING_BATCH_IDS:
            process.kill()
            logger.debug(f"batch {batch['id']}, nonce {nonce} stopped")
            break
        
        time.sleep(0.1)

    logger.debug(f"batch {batch['id']}, nonce {nonce} finished, took {now() - start}ms")


def compute_merkle_roots(results_dir):
    for batch_id in list(PROCESSING_BATCH_IDS):
        job = PROCESSING_BATCH_IDS[batch_id]
        batch = job["batch"]
        start = job["start"]
        q = job["q"]

        num_processing = batch["num_nonces"] - len(job["finished"])
        if num_processing > 0:
            logger.debug(f"batch {batch['id']} still processing {num_processing} nonces")
            time.sleep(1.5)
            continue

        try:
            hashes = []
            solution_nonces = []
            discarded_solution_nonces = []
            for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"]):
                with open(f"{results_dir}/{batch['id']}/{n}.json", "r") as f:
                    d = OutputData.from_dict(json.load(f))
                    h = d.to_merkle_hash()
                    if len(d.solution) > 0:
                        if h.to_str() <= batch["hash_threshold"]:
                            solution_nonces.append(n)
                        else:
                            discarded_solution_nonces.append(n)
                    hashes.append(d.to_merkle_hash())

            merkle_tree = MerkleTree(hashes, batch["batch_size"])
            with open(f"{results_dir}/{batch['id']}/hashes.zlib", "wb") as f:
                hashes = [h.to_str() for h in hashes]
                f.write(zlib.compress(json.dumps(hashes).encode()))
            with open(f"{results_dir}/{batch['id']}/result.json", "w") as f:
                result = {
                    "solution_nonces": solution_nonces,
                    "discarded_solution_nonces": discarded_solution_nonces,
                    "merkle_root": merkle_tree.calc_merkle_root().to_str(),
                }
                logger.debug(f"batch {batch['id']} result: {result}")
                json.dump(result, f)
            logger.info(f"batch {batch['id']} done, took: {now() - start}ms")
            
            READY_BATCH_IDS.add(batch["id"])
        except Exception as e:
            logger.error(f"batch {batch['id']}, error computing merkle root : {e}")
        finally:
            PROCESSING_BATCH_IDS.pop(batch["id"], None)
            return
        
    time.sleep(1)
    

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


def send_results(headers, master_ip, master_port, results_dir):
    try:
        batch_id = READY_BATCH_IDS.pop()
    except KeyError:
        logger.debug("no batches to send")
        time.sleep(1)
        return

    if now() - FINISHED_BATCH_IDS.get(batch_id, 0) < 10000:
        logger.debug(f"batch {batch_id} submitted recently")
        return
    
    output_folder = f"{results_dir}/{batch_id}"
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
        result["hashes"] = [
            hashes[n - batch["start_nonce"]].lower()
            for n in result["solution_nonces"]
        ]

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


def process_batch(algorithms_dir, config, results_dir):
    try:
        batch_id = PENDING_BATCH_IDS.pop()
    except KeyError:
        logger.debug("no pending batches")
        time.sleep(1)
        return

    if (
        batch_id in PROCESSING_BATCH_IDS or
        batch_id in READY_BATCH_IDS
    ):
        return
    
    if os.path.exists(f"{results_dir}/{batch_id}/result.json"):
        logger.info(f"batch {batch_id} already processed")
        READY_BATCH_IDS.add(batch_id)
        return
    
    with open(f"{results_dir}/{batch_id}/batch.json") as f:
        batch = json.load(f)

    containers = set(subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines())
    if batch["challenge"] not in containers:
        logger.error(f"Error processing batch {batch_id}: Challenge container {batch['challenge']} not found. Did you start it with 'docker-compose up {batch['challenge']}'?")
        return
    
    c = next(
        (
            x for x in config["algorithms"]
            if re.match(x["id_regex"], batch["settings"]["algorithm_id"])
        ),
        None
    )
    if c is None:
        logger.error(f"Error processing batch {batch_id}: Algorithm {batch['settings']['algorithm_id']} does not match any regex in the config")
        return
    
    q = Queue()
    for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"]):
        q.put(n)
    so_path, ptx_path = download_library(algorithms_dir, batch)
    logger.info(f"batch {batch['id']} started")
    PROCESSING_BATCH_IDS[batch_id] = {
        "batch": batch,
        "so_path": so_path,
        "ptx_path": ptx_path,
        "q": q,
        "finished": set(),
        "cost": c["cost"],
        "start": now(),
    }

    
def process_nonces(config, results_dir):
    for batch_id in list(PROCESSING_BATCH_IDS):
        job = PROCESSING_BATCH_IDS[batch_id]        
        q = job["q"]
        batch = job["batch"]
        so_path = job["so_path"]
        ptx_path = job["ptx_path"]
        cost = job["cost"]
        if TOTAL_COST[0] + cost <= config["max_cost"]:
            try:
                nonce = q.get_nowait()
                break
            except:
                continue
    else:
        time.sleep(1)
        return
    
    TOTAL_COST[0] += cost
    logger.debug(f"batch {batch_id}, nonce {nonce} started: (cost {cost})")
    try:
        run_tig_runtime(nonce, batch, so_path, ptx_path, results_dir)
    except Exception as e:
        logger.error(f"batch {batch_id}, nonce {nonce}, runtime error: {e}")
    finally:
        TOTAL_COST[0] -= cost
        job["finished"].add(nonce)


def poll_batches(headers, master_ip, master_port, results_dir):    
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
            output_folder = f"{results_dir}/{batch['id']}"
            os.makedirs(output_folder, exist_ok=True)
            with open(f"{output_folder}/batch.json", "w") as f:
                json.dump(batch, f)
        PENDING_BATCH_IDS.clear()
        PENDING_BATCH_IDS.update(root_batch_ids + proofs_batch_ids)
        for batch_id in set(PROCESSING_BATCH_IDS) - set(root_batch_ids + proofs_batch_ids):
            logger.info(f"stopping batch {batch_id}")
            PROCESSING_BATCH_IDS.pop(batch_id, None)
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


def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at path: {config_path}")
        sys.exit(1)
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)
    
    slave_name = os.getenv("SLAVE_NAME") or randomname.get_name()
    master_ip = os.getenv("MASTER_IP") or "0.0.0.0"
    if (master_port := os.getenv("MASTER_PORT")) is None:
        logger.error("MASTER_PORT environment variable not set")
        sys.exit(1)
    master_port = int(master_port)

    algorithms_dir = "algorithms"
    results_dir = "results"
    ttl = int(os.getenv("TTL"))
    num_workers = int(os.getenv("NUM_WORKERS"))

    print(f"Starting slave with config:")
    print(f"  Slave Name: {slave_name}")
    print(f"  Master: {master_ip}:{master_port}")
    print(f"  CPU Architecture: {CPU_ARCH}")
    print(f"  Algorithms Dir: {algorithms_dir}")
    print(f"  Results Dir: {results_dir}")
    print(f"  TTL: {ttl}")
    print(f"  Workers: {num_workers}")
    print(f"  Config: {json.dumps(config, indent=2)}")

    os.makedirs(algorithms_dir, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }

    Thread(
        target=wrap_thread,
        args=(process_batch, algorithms_dir, config, results_dir)
    ).start()

    for _ in range(num_workers):
        Thread(
            target=wrap_thread, 
            args=(process_nonces, config, results_dir)
        ).start()

    Thread(
        target=wrap_thread,
        args=(compute_merkle_roots, results_dir)
    ).start()

    Thread(
        target=wrap_thread,
        args=(send_results, headers, master_ip, master_port, results_dir)
    ).start()

    Thread(
        target=wrap_thread,
        args=(purge_folders, results_dir, ttl)
    ).start()

    wrap_thread(poll_batches, headers, master_ip, master_port, results_dir)


if __name__ == "__main__":    
    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if os.getenv("VERBOSE") else logging.INFO
    )

    main()
