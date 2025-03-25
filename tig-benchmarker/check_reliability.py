import requests

API_URL = "https://mainnet-api.tig.foundation"

print(f"Fetching block, challenges, and benchmarkers..")
block = requests.get(f"{API_URL}/get-block").json()["block"]
challenges = sorted(
    requests.get(f"{API_URL}/get-challenges?block_id={block['id']}").json()["challenges"],
    key=lambda x: x["id"]
)
opow = requests.get(f"{API_URL}/get-opow?block_id={block['id']}").json()["opow"]

print("Challenges:")
for i, c in enumerate(challenges):
    print(f"{i+1}) {c['details']['name']}")
c = challenges[int(input("Enter challenge index: ")) - 1]
print("")

for x in opow:
    x["reliability"] = x["block_data"]["solution_ratio_by_challenge"].get(c["id"], 0) / c["block_data"]["average_solution_ratio"]

opow = sorted(opow, key=lambda x: x["reliability"], reverse=True)

print(f"Benchmarkers Reliability for {c['details']['name']}:")
for i, x in enumerate(opow):
    print(f"{i+1}) {x['player_id']}: {x['reliability']:.4f}")

player = opow[int(input("Enter benchmarker index: ")) - 1]
print("")
print(f"Fetching benchmarks for {player['player_id']}..")
d = requests.get(f"{API_URL}/get-benchmarks?player_id={player['player_id']}&block_id={block['id']}").json()
precommits = {x["benchmark_id"]: x for x in d["precommits"]}
benchmarks = {x["id"]: x for x in d["benchmarks"]}
proofs = {x["benchmark_id"]: x for x in d["proofs"]}
frauds = {x["benchmark_id"]: x for x in d["frauds"]}

total_solutions = {}
total_nonces = {}
for b_id in proofs:
    p = precommits[b_id]
    b = benchmarks[b_id]
    if b_id in frauds or p["settings"]["challenge_id"] != c["id"]:
        continue
    k = f"difficulty {p['settings']['difficulty']}, algorithm_id: {p['settings']['algorithm_id']}"
    total_solutions[k] = total_solutions.get(k, 0) + b["details"]["num_solutions"]
    total_nonces[k] = total_nonces.get(k, 0) + p["details"]["num_nonces"]

reliability = {
    k: total_solutions[k] / total_nonces[k] / c["block_data"]["average_solution_ratio"]
    for k in total_solutions
}
reliability = sorted(reliability.items(), key=lambda x: x[1], reverse=True)
for k, r in reliability:
    print(f"{k}, num_solutions: {total_solutions[k]}, num_nonces: {total_nonces[k]}, reliability: {r:.4f}")