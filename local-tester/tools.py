import subprocess
import os
import sys
import requests

def run_test(i):
    result = subprocess.run(f"../target/release/tig-worker -- settings.json wasm/{os.environ["algorithm_name"]}.wasm {i}", shell=True, capture_output=True, text=True)
    return result.returncode

def build_worker():
    _ = subprocess.run(f"cargo build -p tig-worker --release", shell=True, capture_output=True, text=True)
    print(f"tig-worker successfully built")
    
def check_algorithm_existence(algorithm_name, algorithm_id_dict):
    if os.path.exists(f"./wasm/{algorithm_name}.wasm") or algorithm_name in algorithm_id_dict.values():
        return True
    else:
        return False
    
def dump_difficulty(difficulty):
    if difficulty[0].isnumeric() and difficulty[1].isnumeric():
        difficulty = [int(x) for x in difficulty]
        with open("settings.json", "w+") as file:
            settings = '{' + f'"block_id": "","algorithm_id": "","challenge_id": "","player_id": "","difficulty": {difficulty}' + '}'
            file.write(settings)
    else:
        sys.exit("Your difficulty parameters entered are not integers, please re enter them and try again")

def get_wasm_blob(algorithm_name, challenge_name):
    branch_name = f"{challenge_name}/{algorithm_name}"
    url = f"https://raw.githubusercontent.com/tig-foundation/tig-monorepo/{branch_name}/tig-algorithms/wasm/{branch_name}.wasm"
    response = requests.get(url)
    with open(f"wasm/{algorithm_name}.wasm", "wb") as file:
        file.write(response.content)
    print(f"Downloaded {branch_name}")