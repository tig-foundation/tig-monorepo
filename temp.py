import subprocess
import multiprocessing

def run_subprocess(i):
    command = f"./target/release/tig-worker -- ./local-tester/settings.json ./local-tester/wasm/clarke_wright.wasm {i}"
    result = subprocess.run(command, shell=True, timeout=)
    return result.returncode

def main():
    with multiprocessing.Pool(4) as pool:
        results = pool.map(run_subprocess, range(11))
    
    for i, returncode in enumerate(results):
        print(f"Subprocess {i} returned {returncode}")

if __name__ == "__main__":
    main()