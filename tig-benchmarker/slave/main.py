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
    output_dir = f"{results_dir}/{batch['id']}"
    output_file = f"{output_dir}/{nonce}.json"
    settings = json.dumps(batch["settings"], separators=(',',':'))
    start = now()
    cmd = [
        "docker", "exec", batch["challenge"], "tig-runtime",
        settings,
        batch["rand_hash"],
        str(nonce),
        so_path,
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--output", output_dir,
    ]
    if batch["hyperparameters"] is not None:
        cmd += [
            "--hyperparameters", json.dumps(batch["hyperparameters"], separators=(',',':')),
        ]
    if ptx_path is not None:
        cmd += [
            "--ptx", ptx_path,
        ]
    logger.debug(f"computing nonce: {' '.join(cmd[:4] + [f"'{cmd[4]}'"] + cmd[5:])}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while True:
        ret = process.poll()
        if ret is not None:
            exit_codes = {
                0: "success",
                # 82: "cuda out of memory",
                # 83: "host out of memory",
                84: "runtime error",
                # 85: "no solution",
                # 86: "invalid solution",
                87: "out of fuel",
            }

            if not os.path.exists(output_file):
                if ret == 0:
                    raise Exception(f"no output")
                else:
                    raise Exception(f"failed with exit code {ret}: {process.stderr.read().decode()}")
                
            start = now()
            cmd = [
                "docker", "exec", batch["challenge"], "tig-verifier",
                settings,
                batch["rand_hash"],
                str(nonce),
                output_file,
            ]
            if ptx_path is not None:
                cmd += [
                    "--ptx", ptx_path,
                ]
            logger.debug(f"verifying nonce: {' '.join(cmd[:4] + [f"'{cmd[4]}'"] + cmd[5:])}")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                raise Exception(f"invalid solution (exit code: {ret.returncode}, stderr: {ret.stderr.strip()})")
            
            last_line = ret.stdout.strip().splitlines()[-1]
            if not last_line.startswith("quality: "):
                raise Exception(f"failed to find quality in tig-verifier output")
            try:
                quality = int(last_line[len("quality: "):])
            except:
                raise Exception(f"failed to parse quality from tig-verifier output")
            logger.debug(f"batch {batch['id']}, nonce {nonce} valid solution with quality {quality}")
            with open(output_file, "r") as f:
                d = json.load(f)
                d["quality"] = quality
            with open(output_file, "w") as f:
                json.dump(d, f)
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
            solution_quality = []
            hashes = []
            for n in range(batch["start_nonce"], batch["start_nonce"] + batch["num_nonces"]):
                with open(f"{results_dir}/{batch['id']}/{n}.json", "r") as f:
                    d = json.load(f)
                    solution_quality.append(d.pop("quality"))
                    hashes.append(OutputData.from_dict(d).to_merkle_hash())

            merkle_tree = MerkleTree(hashes, batch["batch_size"])
            with open(f"{results_dir}/{batch['id']}/hashes.zlib", "wb") as f:
                hashes = [h.to_str() for h in hashes]
                f.write(zlib.compress(json.dumps(hashes).encode()))
            with open(f"{results_dir}/{batch['id']}/result.json", "w") as f:
                result = {
                    "solution_quality": solution_quality,
                    "merkle_root": merkle_tree.calc_merkle_root().to_str(),
                }
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
    with open(f"{output_folder}/result.json") as f:
        result = json.load(f)

    if result.get("error") is not None:
        if batch["sampled_nonces"] is None:
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-root/{batch_id}"
        else:
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-proofs/{batch_id}"
        logger.info(f"posting error for failed batch to {submit_url}")
        resp = requests.post(submit_url, headers=headers, json=result)
        if resp.status_code == 200:
            os.remove(f"{output_folder}/result.json")
            logger.info(f"successfully posted error for batch {batch_id}")
        elif resp.status_code == 408: # took too long
            os.remove(f"{output_folder}/result.json")
            logger.error(f"status {resp.status_code} when posting error for batch {batch_id} to master: {resp.text}")
        else:
            logger.error(f"status {resp.status_code} when posting error for batch {batch_id} to master: {resp.text}")
            READY_BATCH_IDS.add(batch_id) # requeue
            time.sleep(2)

    elif batch["sampled_nonces"] is None:
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


def process_batch(algorithms_dir, results_dir):
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
        "start": now(),
    }

    
def process_nonces(results_dir):
    for batch_id in list(PROCESSING_BATCH_IDS):
        job = PROCESSING_BATCH_IDS[batch_id]        
        q = job["q"]
        batch = job["batch"]
        so_path = job["so_path"]
        ptx_path = job["ptx_path"]
        try:
            nonce = q.get_nowait()
            break
        except:
            continue
    else:
        time.sleep(1)
        return
    
    logger.debug(f"batch {batch_id}, nonce {nonce} started")
    try:
        run_tig_runtime(nonce, batch, so_path, ptx_path, results_dir)
        job["finished"].add(nonce)
    except Exception as e:
        msg = f"batch {batch_id}, nonce {nonce}, runtime error: {e}"
        logger.error(msg)
        with open(f"{results_dir}/{batch['id']}/result.json", "w") as f:
            json.dump({"error": msg}, f)
        READY_BATCH_IDS.add(batch["id"])
        PROCESSING_BATCH_IDS.pop(batch["id"], None)

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

    os.makedirs(algorithms_dir, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }

    Thread(
        target=wrap_thread,
        args=(process_batch, algorithms_dir, results_dir)
    ).start()

    for _ in range(num_workers):
        Thread(
            target=wrap_thread, 
            args=(process_nonces, results_dir)
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
