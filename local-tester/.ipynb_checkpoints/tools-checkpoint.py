import os
import subprocess
import time

def run_test(algorithm_name):
    result = subprocess.run(f"cargo run -p testing-worker --release -- settings.json wasm/{algorithm_name}.wasm {os.environ["TIMER"]} {algorithm_name}", shell=True, capture_output=True, text=True)
    with open("test_results.txt", "a") as file:
        file.write(result.stdout + "\n")
    
def run_batch(workers):
    for worker in workers:
        worker.start()
    [worker.join() for worker in workers]