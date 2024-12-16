from common.structs import *
import requests
import json
import random

print("THIS IS AN EXAMPLE SCRIPT TO UPDATE YOUR MASTER CONFIG")
MASTER_IP = input("Enter Master IP: ")
MASTER_PORT = input("Enter Master Port: ")


# Get latest data
print("Fetching latest data from Master")
data = requests.get(f"http://{MASTER_IP}:{MASTER_PORT}/get-latest-data").json()
block = Block.from_dict(data["block"])
algorithms = {k: Algorithm.from_dict(v) for k, v in data["algorithms"].items()}
wasms = {k: Binary.from_dict(v) for k, v in data["wasms"].items()}
precommits = {k: Precommit.from_dict(v) for k, v in data["precommits"].items()}
benchmarks = {k: Benchmark.from_dict(v) for k, v in data["benchmarks"].items()}
proofs = {k: Proof.from_dict(v) for k, v in data["proofs"].items()}
frauds = {k: Fraud.from_dict(v) for k, v in data["frauds"].items()}
challenges = {k: Challenge.from_dict(v) for k, v in data["challenges"].items()}
difficulty_data = {k: [DifficultyData.from_dict(x) for x in v] for k, v in data["difficulty_data"].items()}

challenge_id_2_name = {c.id: c.details.name for c in challenges.values()}


# Calculate solutions, qualifiers and cutoff
solutions = {c_name: 0 for c_name in challenge_id_2_name.values()}
potential_qualifiers = {c_name: 0 for c_name in challenge_id_2_name.values()}

for benchmark_id in benchmarks:
    if benchmark_id in frauds:
        continue
    challenge_id = precommits[benchmark_id].settings.challenge_id
    c_name = challenge_id_2_name[challenge_id]
    difficulty = precommits[benchmark_id].settings.difficulty
    num_solutions = benchmarks[benchmark_id].details.num_solutions
    solutions[c_name] += num_solutions
    if difficulty in challenges[challenge_id].block_data.qualifier_difficulties:
        potential_qualifiers[c_name] += num_solutions
        
cutoff = int(min(solutions.values()) * block.config["opow"]["cutoff_multiplier"])

print("Stats (including non yet active benchmarks)")
print("#solutions: ", solutions)
print("#potential qualifiers: ", potential_qualifiers)
print("cutoff: ", cutoff)


# Calculate challenge weights
if cutoff == 0:
    weights = {c_name: 1.0 for c_name in challenge_id_2_name.values()}
elif cutoff < max(potential_qualifiers.values()):
    print("Adjusting weights to increase cutoff")
    m = max(potential_qualifiers.values())
    weights = {
        k: (m - v) / m + 0.1
        for k, v in potential_qualifiers.items()
    }
else:
    print("Adjusting weights to improve imbalance")
    m = max(solutions.values())
    weights = {
        k: (m - v) / m + 0.1
        for k, v in solutions.items()
    }
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}
print("Weights: ", weights)


# Selecting difficulties
selected_difficulties = {}
for c_id, stats in difficulty_data.items():
    solution_p = [
        (
            s.num_solutions / s.num_nonces,
            s.difficulty
        )
        for s in stats
    ]
    solution_p.sort(reverse=True)
    print(f"({challenges[c_id].details.name}) Top 10 difficulties by %solution: ", solution_p[:10])
    selection = set()
    # select 25 randomly offset difficulties around the top 10
    for _, difficulty in solution_p[:10]:
        selection.add(difficulty)
        for _ in range(25):
            selection.add((
                difficulty[0] + random.randint(-10, 10),
                difficulty[1] + random.randint(-10, 10)
            ))
    c_name = challenge_id_2_name[c_id]
    selected_difficulties[c_name] = list(selection)
    

# Update Master Config
config = requests.get(f"http://{MASTER_IP}:{MASTER_PORT}/get-config").json()
for c_name, c in config["precommit_manager_config"]["algo_selection"].items():
    c["weight"] = weights[c_name]
        
for c_name, c in config["difficulty_sampler_config"]["selected_difficulties"].items():
    c.clear()
    c.extend(selected_difficulties[c_name])

print("Updating your Master Config")
requests.post(f"http://{MASTER_IP}:{MASTER_PORT}/update-config", data=json.dumps(config), headers={"Content-Type": "application/json"})


print("Done")