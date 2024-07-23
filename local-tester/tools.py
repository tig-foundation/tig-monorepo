import os
import subprocess
import time

def run_test(algorithm_name):
    result = subprocess.run(f"../target/release/tig-worker -- settings.json wasm/{algorithm_name}.wasm {os.environ["TIMER"]} {algorithm_name}", shell=True, capture_output=True, text=True)
    with open("test_results.txt", "a") as file:
        file.write(result.stdout + "\n")
    
def build_worker():
    _ = subprocess.run(f"cargo build -p tig-worker --release", shell=True, capture_output=True, text=True)
    print(f"tig-worker successfully built")

def run_batch(workers):
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]