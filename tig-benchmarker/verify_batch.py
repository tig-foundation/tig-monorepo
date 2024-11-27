import argparse
import json
import random
import sys
import time
from subprocess import Popen, PIPE
from typing import List, Optional, Set
from tig_benchmarker.structs import BenchmarkSettings
from tig_benchmarker.extensions.slave_manager import BatchResult

class FraudException(Exception):
    pass

def rust_compatible_rng(seed: bytes):
    rng = None
    for i in range(8):
        s = int.from_bytes(seed[i*4:(i+1)*4], 'little', signed=False)
        if rng is not None:
            s = int(s * rng.random())
        rng = random.Random(s)
    return rng

def verify_nonces(
    result: BatchResult,
    start_nonce: int,
    num_nonces: int,
    sampled: Optional[List[int]] = None,
    num_random: Optional[int] = None
):
    for nonce in result.solution_nonces:
        if start_nonce > nonce or nonce >= start_nonce + num_nonces:
            raise FraudException(f"Solution nonce {nonce} out of range [{start_nonce}, {start_nonce + num_nonces}).")
            
    for proof in result.merkle_proofs:
        if start_nonce > proof.leaf.nonce or proof.leaf.nonce >= start_nonce + num_nonces:
            raise FraudException(f"Merkle proof nonce {proof.leaf.nonce} out of range [{start_nonce}, {start_nonce + num_nonces}).")

    if num_random is not None:
        rng = rust_compatible_rng(result.merkle_root.value)
        expected = set()
        nonces = list(range(start_nonce, start_nonce + num_nonces))
        for _ in range(num_random):
            idx = int(rng.random() * len(nonces))
            nonces[idx], nonces[-1] = nonces[-1], nonces[idx]
            expected.add(nonces.pop())
        actual = set(p.leaf.nonce for p in result.merkle_proofs)
        if expected != actual:
            raise FraudException(f"Merkle proof nonces {actual} do not match expected samples {expected}.")

    elif sampled is not None:
        expected, actual = set(sampled), set(p.leaf.nonce for p in result.merkle_proofs)
        if expected != actual:
            raise FraudException(f"Merkle proof nonces {actual} do not match expected samples {expected}.")

def verify_merkle_branches(result: BatchResult, start_nonce: int):
    for proof in result.merkle_proofs:
        try:
            expected = result.merkle_root
            actual = proof.branch.calc_merkle_root(
                proof.leaf.to_merkle_hash(), 
                proof.leaf.nonce - start_nonce
            )
        except Exception as e:
            raise FraudException(f"Failed to calculate merkle root from merkle proof {proof.leaf.nonce}: {str(e)}")
        if expected != actual:
            raise FraudException(f"Merkle root from branch {proof.leaf.nonce} does not match the expected merkle root.")

def verify_solutions(
    tig_worker_path: str,
    result: BatchResult, 
    settings: BenchmarkSettings, 
    rand_hash: str,
):
    solution_nonces = set(result.solution_nonces) & set(p.leaf.nonce for p in result.merkle_proofs)
    merkle_proofs = {p.leaf.nonce: p for p in result.merkle_proofs}
    processes = {}
    
    try:
        for nonce in solution_nonces:
            proof = merkle_proofs[nonce]
            process = Popen(
                [
                    tig_worker_path, 
                    'verify_solution', 
                    json.dumps(settings.to_dict()), 
                    rand_hash, 
                    str(nonce), 
                    json.dumps(proof.solution)
                ],
                stdout=PIPE,
                stderr=PIPE
            )
            processes[nonce] = process
        
        while processes:
            for nonce in list(processes):
                p = processes[nonce]
                return_code = p.poll()
                if return_code is None:
                    continue
                if p.returncode != 0:
                    error_msg = p.stderr.read().decode().rstrip()
                    raise FraudException(f"Solution verification failed for nonce {nonce}: {error_msg}")
                processes.pop(nonce)
            time.sleep(0.1)

    finally:
        for process in processes.values():
            process.kill()

def verify_runtime_signatures(
    tig_worker_path: str,
    wasm_path: str,
    result: BatchResult,
    settings: BenchmarkSettings, 
    rand_hash: str,
    max_fuel: int,
    max_mem: int,
):
    processes = {}
    
    try:
        for proof in result.merkle_proofs:
            nonce = proof.leaf.nonce            
            process = Popen(
                [
                    tig_worker_path, 
                    'compute_solution', 
                    json.dumps(settings.to_dict()), 
                    rand_hash, str(nonce), 
                    wasm_path,
                    "--fuel", str(max_fuel),
                    "--mem", str(max_mem)
                ],
                stdout=PIPE,
                stderr=PIPE
            )
            expected = json.dumps(proof.leaf.to_dict(), sort_keys=True)
            processes[nonce] = (process, expected)

        while processes:
            for nonce in list(processes):
                p, expected = processes[nonce]
                return_code = p.poll()
                if return_code is None:
                    continue
                
                try:
                    actual = p.stdout.read().decode()
                    actual = json.dumps(json.loads(actual), sort_keys=True)
                except json.JSONDecodeError:#
                    raise FraudException(f"Failed to decode solution data for nonce {nonce}")
                
                if expected != actual:
                    raise FraudException(f"OutputData for nonce {nonce} does not match recomputed.")
                processes.pop(nonce)
            time.sleep(0.1)

    finally:
        for nonce, (process, _) in processes.items():
            process.kill()

def main():
    parser = argparse.ArgumentParser(description='Verify batch results')
    parser.add_argument('tig_worker_path', type=str, help='Path to the tig-worker executable')
    parser.add_argument('batch_result', type=str, help='JSON string containing batch result')
    parser.add_argument('settings', type=str, help='JSON string containing benchmark settings')
    parser.add_argument('rand_hash', type=str, help='Random hash value')
    parser.add_argument('start_nonce', type=int, help='Starting nonce value')
    parser.add_argument('num_nonces', type=int, help='Number of nonces')
    parser.add_argument('wasm_path', type=str, help='Path to algorithm wasm blob')
    parser.add_argument('--fuel', type=int, default=2000000000, help='Max fuel')
    parser.add_argument('--mem', type=int, default=1000000000, help='Max memory')
    parser.add_argument('--sampled', type=int, nargs='*', help='List of sampled nonces')
    parser.add_argument('--random', type=int, help='Number of random samples')
    args = parser.parse_args()
    
    result = BatchResult.from_dict(json.loads(args.batch_result))
    settings = BenchmarkSettings.from_dict(json.loads(args.settings))
    
    try:
        verify_nonces(
            result=result,
            start_nonce=args.start_nonce,
            num_nonces=args.num_nonces,
            sampled=args.sampled,
            num_random=args.random
        )
        verify_merkle_branches(result=result, start_nonce=args.start_nonce)
        verify_solutions(
            tig_worker_path=args.tig_worker_path,
            result=result,
            settings=settings,
            rand_hash=args.rand_hash
        )
        verify_runtime_signatures(
            tig_worker_path=args.tig_worker_path,
            wasm_path=args.wasm_path,
            result=result,
            settings=settings,
            rand_hash=args.rand_hash,
            max_fuel=args.fuel,
            max_mem=args.mem
        )
        print(f"Verification successful")
        
    except FraudException as e:
        print(f"Verification failed")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()