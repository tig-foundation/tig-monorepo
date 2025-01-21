from common.structs import *
import requests
import json
import random
import os
import subprocess

print("THIS IS AN EXAMPLE SCRIPT TO VERIFY A BATCH")
TIG_WORKER_PATH = input("Enter path of tig-worker executable: ")
NUM_WORKERS = int(input("Enter number of workers: "))
if not os.path.exists(TIG_WORKER_PATH):
    raise FileNotFound[ERROR](f"tig-worker not found at path: {TIG_WORKER_PATH}")
MASTER_IP = input("Enter Master IP: ")
MASTER_PORT = input("Enter Master Port: ")

jobs = requests.get(f"http://{MASTER_IP}:{MASTER_PORT}/get-jobs").json()
for i, j in enumerate(jobs):
    print(f"{i + 1}) benchmark_id: {j['benchmark_id']}, challenge: {j['challenge']}, algorithm: {j['algorithm']}, status: {j['status']}")
job_idx = int(input("Enter the index of the batch you want to verify: ")) - 1

for i, b in enumerate(jobs[job_idx]["batches"]):
    print(f"{i + 1}) slave: {b['slave']}, num_solutions: {b['num_solutions']}, status: {b['status']}")
batch_idx = int(input("Enter the index of the batch you want to verify: ")) - 1

benchmark_id = jobs[job_idx]["benchmark_id"]
url = f"http://{MASTER_IP}:{MASTER_PORT}/get-batch-data/{benchmark_id}_{batch_idx}"
print(f"Fetching batch data from {url}")
data = requests.get(url).json()

batch = data['batch']
merkle_root = data['merkle_root']
solution_nonces = data['solution_nonces']
merkle_proofs = data['merkle_proofs']
if merkle_proofs is not None:
    merkle_proofs = {x['leaf']['nonce']: x for x in merkle_proofs}

if (
    merkle_proofs is None or
    len(solution_nonces := set(merkle_proofs) & set(solution_nonces)) == 0
):
    print("No solution data to verify for this batch")
else:
    for nonce in solution_nonces:
        print(f"Verifying solution for nonce: {nonce}")
        cmd = [
            TIG_WORKER_PATH, "verify_solution",
            json.dumps(batch['settings'], separators=(',', ':')), 
            batch["rand_hash"], 
            str(nonce), 
            json.dumps(merkle_proofs[nonce]['leaf']['solution'], separators=(',', ':')),
        ]
        print(f"Running cmd: {' '.join(cmd)}")
        ret = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if ret.returncode == 0:
            print(f"[SUCCESS]: {ret.stdout.decode()}")
        else:
            print(f"[ERROR]: {ret.stderr.decode()}")

if merkle_root is not None:
    download_url = batch["download_url"]
    print(f"Downloading WASM from {download_url}")
    resp = requests.get(download_url)
    if resp.status_code != 200:
        raise Exception(f"status {resp.status_code} when downloading WASM: {resp.text}")
    wasm_path = f'{batch["settings"]["algorithm_id"]}.wasm'
    with open(wasm_path, 'wb') as f:
        f.write(resp.content)
    print(f"WASM Path: {wasm_path}")
    print("")

if merkle_proofs is None:
    print("No merkle proofs to verify for this batch")
else:
    for nonce in merkle_proofs:
        print(f"Verifying output data for nonce: {nonce}")
        cmd = [
            TIG_WORKER_PATH, "compute_solution",
            json.dumps(batch['settings'], separators=(',', ':')), 
            batch["rand_hash"], 
            str(nonce), 
            wasm_path,
            "--mem", str(batch["runtime_config"]["max_memory"]),
            "--fuel", str(batch["runtime_config"]["max_fuel"]),
        ]
        print(f"Running cmd: {' '.join(cmd)}")
        ret = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = json.loads(ret.stdout.decode())
        expected = json.dumps(merkle_proofs[nonce]['leaf'], separators=(',', ':'), sort_keys=True)
        actual = json.dumps(out, separators=(',', ':'), sort_keys=True)
        if expected == actual:
            print(f"[SUCCESS]: output data match")
        else:
            print(f"[ERROR]: output data mismatch")
            print(f"Batch data: {expected}")
            print(f"Recomputed: {actual}")
        print(f"")

if merkle_root is None:
    print("No merkle root to verify for this batch")
else:
    print("Verifying merkle root")
    cmd = [
        TIG_WORKER_PATH, "compute_batch",
        json.dumps(batch["settings"]), 
        batch["rand_hash"], 
        str(batch["start_nonce"]), 
        str(batch["num_nonces"]),
        str(batch["batch_size"]), 
        wasm_path,
        "--mem", str(batch["runtime_config"]["max_memory"]),
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--workers", str(NUM_WORKERS),
    ]
    print(f"Running cmd: {' '.join(cmd)}")
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if ret.returncode == 0:
        out = json.loads(ret.stdout.decode())
        if out["merkle_root"] == merkle_root:
            print(f"[SUCCESS]: merkle root match")
        else:
            print(f"[ERROR]: merkle root mismatch")
            print(f"Batch data: {expected}")
            print(f"Recomputed: {actual}")
    else:
        print(f"[ERROR]: {ret.stderr.decode()}")

print("")
print("FINISHED")