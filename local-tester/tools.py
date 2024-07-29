import time
import subprocess

def get_results(test_result, raw_result):
    test_result["instances"] = raw_result[0]
    test_result["solutions"] = raw_result[1]
    test_result["invalid_solutions"] = raw_result[0] - (raw_result[1] + raw_result[2])
    test_result["errors"] = raw_result[2]
    return test_result

def loop_entry(algorithm_name, raw_result, timer):
    worker_result = [0,0]
    t_end = time.time() + timer
    while time.time() < t_end:
        raw_result[0] += 1
        run_result = run_test(algorithm_name, raw_result[0])
        if run_result == 0:
            worker_result[0] += 1
        elif run_result == 2:
            worker_result[1] += 1

    raw_result[1] += worker_result[0]
    raw_result[2] += worker_result[1]

def run_test(algorithm_name, i):
    result = subprocess.run(f"../target/release/tig-worker -- settings.json wasm/{algorithm_name}.wasm {i}", shell=True, capture_output=True, text=True)
    return result.returncode

def build_worker():
    _ = subprocess.run(f"cargo build -p tig-worker --release", shell=True, capture_output=True, text=True)
    print(f"tig-worker successfully built")
    