import argparse
import io
import json
import os
import platform
import random
import requests
import subprocess
import sys
import tarfile
import time

API_URL = "https://mainnet-api.tig.foundation"

if not os.path.exists(".env"):
    raise Exception("Expecting .env file in local directory. The same as used by docker-compose -f slave.yml ...")
if (ALGO_DIR := next((l.split("=")[1] for l in open(".env").read().splitlines() if l.startswith("ALGORITHMS_DIR")), None)) is None:
    raise Exception("Expecting ALGORITHMS_DIR in .env file")

CPU_VENDOR = "Unknown"
if os.path.exists("/proc/cpuinfo"):
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()
    if "GenuineIntel" in cpuinfo:
        CPU_VENDOR = "Intel"
    elif "AuthenticAMD" in cpuinfo:
        CPU_VENDOR = "AMD"
    elif "ARM" in cpuinfo or "AArch64" in cpuinfo:
        CPU_VENDOR = "ARM"

if (CPU_ARCH := platform.machine().lower()) in ["x86_64", "amd64"]:
    CPU_ARCH = "amd64"
elif CPU_ARCH in ["arm64", "aarch64"]:
    CPU_ARCH = "arm64"
else:
    print(f"Unsupported CPU architecture: {CPU_ARCH}")
    sys.exit(1)

HAS_GPU = subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0

COMPUTE_TYPES = {
    "Intel": {"aws_t3", "aws_c7i", "aws_m7i", "g4dn"},
    "AMD": {"aws_t3a", "aws_c7a", "aws_m7a", "g4dn"},
    "ARM": {"aws_t4g", "aws_c7g", "aws_m7g"},
    "Unknown": set(),
}

def now():
    return int(round(time.time() * 1000))

def download_library(challenge: str, algorithm: str, download_url: str):
    challenge_folder = f"{ALGO_DIR}/{challenge}"
    so_path = f"{challenge_folder}/{CPU_ARCH}/{algorithm}.so"
    ptx_path = f"{challenge_folder}/ptx/{algorithm}.ptx"
    if not os.path.exists(so_path):
        start = now()
        print(f"downloading {algorithm}.tar.gz from {download_url}")
        resp = requests.get(download_url, stream=True)
        if resp.status_code != 200:
            raise Exception(f"status {resp.status_code} when downloading algorithm library: {resp.text}")
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            tar.extractall(path=challenge_folder)
        print(f"downloading {algorithm}.tar.gz took {now() - start}ms")

    if not os.path.exists(ptx_path):
        return so_path, None
    else:
        return so_path, ptx_path

def run_tig_runtime(nonce, batch, so_path, ptx_path):
    settings = json.dumps(batch["settings"], separators=(',',':'))
    start = now()
    cmd = [
        "docker", "exec", batch["challenge"], "tig-runtime",
        settings,
        batch["rand_hash"],
        str(nonce),
        so_path,
        "--fuel", str(batch["fuel_budget"]),
    ]
    if batch["hyperparameters"] is not None:
        cmd += [
            "--hyperparameters", json.dumps(batch["hyperparameters"], separators=(',',':')),
        ]
    if ptx_path is not None:
        cmd += [
            "--ptx", ptx_path,
        ]
    print(f"computing nonce: {' '.join(cmd[:4] + [f"'{cmd[4]}'"] + cmd[5:])}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    while True:
        try:
            _, stderr = process.communicate(timeout=0.1)
            ret = process.returncode
            if ret != 0:
                return f"tig-runtime failed with exit code {ret}: {stderr}"
                
            start = now()
            cmd = [
                "docker", "exec", batch["challenge"], "tig-verifier",
                settings,
                batch["rand_hash"],
                str(nonce),
                f"{nonce}.json",
            ]
            if ptx_path is not None:
                cmd += [
                    "--ptx", ptx_path,
                ]
            print(f"verifying nonce: {' '.join(cmd[:4] + [f"'{cmd[4]}'"] + cmd[5:])}")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                return f"tig-verifier failed with exit code {ret.returncode}: {ret.stderr.strip()}"
            
            last_line = ret.stdout.strip().splitlines()[-1]
            if not last_line.startswith("quality: "):
                return f"failed to find quality in tig-verifier output"
            try:
                quality = int(last_line[len("quality: "):])
            except:
                return f"failed to parse quality from tig-verifier output"
            print(f"method verification finished, took {now() - start}ms")
            return quality
        except subprocess.TimeoutExpired:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Method Verification Tool")
    parser.add_argument("--benchmark_id", type=str, help="Benchmark ID")
    parser.add_argument("--nonce", type=int, help="Nonce to compute")
    parser.add_argument("--testnet", action="store_true", help="Use testnet API")
    args = parser.parse_args()
    
    if args.testnet:
        API_URL = "https://testnet-api.tig.foundation"

    block = requests.get(f"{API_URL}/get-block").json()["block"]
    gpu_challenges = {k for k, v in block["config"]["challenges"].items() if v["type"] == "gpu"}
    r = block["details"]["round"]
    c_names = {
        x["id"]: x["config"]["name"] for x in requests.get(f"{API_URL}/get-challenges?block_id={block['id']}").json()["challenges"]
        if x["state"]["round_active"] <= r
    }
    d = requests.get(f"{API_URL}/get-algorithms?block_id={block['id']}").json()
    a_names = {x["id"]: x["details"]["name"] for x in d["codes"]}
    download_urls = {x["algorithm_id"]: x["details"]["download_url"] for x in d["binarys"]}

    if args.benchmark_id is None or args.nonce is None:
        print("No benchmark_id or nonce provided. Selecting random:")
        c_id, c_name = random.choice(list(c_names.items()))
        opow = [
            (k, int(v["total"]))
            for k, v in requests.get(f"{API_URL}/get-round-emissions?round={r}").json()["opow"].items()
        ]
        benchmarker = random.choices(opow, weights=[x[1] for x in opow])[0][0]
        benchmark_ids = requests.get(f"{API_URL}/get-reportable-benchmark-ids?round={r}&player_id={benchmarker}&challenge_id={c_id}").json()["benchmark_ids"]
        benchmark_id = random.choice(benchmark_ids)
        d = requests.get(f"{API_URL}/get-benchmark-data?benchmark_id={benchmark_id}").json()
        precommit = d["precommit"]
        nonce = random.randint(0, precommit['details']['num_nonces'] - 1)
        compute_type = d["precommit"]["details"]["compute_type"]
        a_id = d["precommit"]["settings"]["algorithm_id"]
        a_name = a_names[a_id]
        hyperparameters = d["precommit"]["details"]["hyperparameters"]
        expected_quality = d["benchmark"]["solution_quality"][nonce]
        print(f"\tBenchmark ID: {benchmark_id}")
        print(f"\tNonce: {nonce}")
        print(f"\tCompute Type: {compute_type}")
        print(f"\tBenchmarker: {benchmarker}")
        print(f"\tChallenge: {c_name} ({c_id})")
        print(f"\tAlgorithm: {a_name} ({a_id})")
        print(f"\tHyperparameters: {hyperparameters}")
        print(f"\tExpected Quality: {expected_quality}")
        print()
    else:
        print("Using provided benchmark_id and nonce:")
        benchmark_id = args.benchmark_id
        nonce = args.nonce
        print(f"\tBenchmark ID: {benchmark_id}")
        print(f"\tNonce: {nonce}")
        d = requests.get(f"{API_URL}/get-benchmark-data?benchmark_id={benchmark_id}").json()
        precommit = d["precommit"]
        if precommit is None:
            print(f"\nerror: No data found for benchmark_id {benchmark_id}.")
            sys.exit(1)
        if precommit["details"]["num_nonces"] <= nonce:
            print(f"\nerror: Nonce {nonce} is out of range for benchmark_id {benchmark_id}..")
            sys.exit(1)
        compute_type = d["precommit"]["details"]["compute_type"]
        benchmarker = d["precommit"]["settings"]["player_id"]
        c_id = d["precommit"]["settings"]["challenge_id"]
        c_name = c_names[c_id]
        a_id = d["precommit"]["settings"]["algorithm_id"]
        a_name = a_names[a_id]
        hyperparameters = d["precommit"]["details"]["hyperparameters"]
        expected_quality = d["benchmark"]["solution_quality"][nonce]
        print(f"\tCompute Type: {compute_type}")
        print(f"\tBenchmarker: {benchmarker}")
        print(f"\tChallenge: {c_name} ({c_id})")
        print(f"\tAlgorithm: {a_name} ({a_id})")
        print(f"\tHyperparameters: {hyperparameters}")
        print(f"\tExpected Quality: {expected_quality}")
        print()

    if c_id in gpu_challenges and not HAS_GPU:
        print(f"error: Challenge {c_name} ({c_id}) requires GPU, but no GPU detected on this machine.")
        sys.exit(1)

    target_vendor = next((v for v, types in COMPUTE_TYPES.items() if compute_type in types), None)
    if compute_type not in COMPUTE_TYPES[CPU_VENDOR]:
        print(f"warning: Benchmark precommitted to compute type {compute_type} ({target_vendor}), but this machine has {CPU_VENDOR} CPU.")
        print(f"         This may result in a quality mismatch, but the verification will still be attempted.")
        print()
        
    if (
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0 or 
        subprocess.run(["docker", "exec", c_name, "true"], capture_output=True).returncode != 0
    ):
        print(f"error: Cannot find docker container named '{c_name}'. Please start it first using 'docker compose -f slave.yml up {c_name}'")
        sys.exit(1)
    
    so_path, ptx_path = download_library(c_name, a_name, download_urls[a_id])
    batch = {
        "id": benchmark_id,
        "challenge": c_name,
        "settings": precommit["settings"],
        "rand_hash": precommit["details"]["rand_hash"],
        "fuel_budget": precommit["details"]["fuel_budget"],
        "hyperparameters": hyperparameters,
    }
    actual_quality = run_tig_runtime(nonce, batch, so_path, ptx_path)
    print()
    if actual_quality != expected_quality:
        print(f"Qualities mismatch:")
        print(f"\tExpected: {expected_quality}")
        print(f"\tActual: {actual_quality}")
        print()
        print(f"To report this (requires 1 TIG fee), run the following command:")
        print(f"curl -X POST {API_URL}/submit-report -H 'X-Api-Key: <YOUR_API_KEY>' -d '{{\"benchmark_id\": \"{benchmark_id}\", \"nonce\": {nonce}}}'")
    else:
        print(f"Success! Actual quality matches expected quality: {actual_quality}")