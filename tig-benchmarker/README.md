# tig-benchmarker

Benchmarker for TIG. Designed to run with a single master and multiple slaves distributed across servers.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Details](#configuration-details)
   * [.env file](#env-file)
   * [Master Config](#master-config)
   * [Tips](#tips)
4. [Hard Resetting](#hard-resetting)
5. [Finding your API Key](#finding-your-api-key)
6. [License](#license)

# Quick Start

1. Obtain a Testnet API Key ([instructions here](#finding-your-api-key))
   * Each new address on testnet gets 10 TIG balance
   * For more tokens, use testnet faucet at https://tigstats.com/faucet

2. Clone this repo (`blank_slate` branch)
    ```
    git clone https://github.com/tig-foundation/tig-monorepo -b blank_slate
    cd tig-monorepo/tig-benchmarker
    ```

3. Start a master 
   * If port 80 is in use, modify `UI_PORT` in your `.env` file
    ```
    docker-compose -f master.yml up
    ```

4. Start a slave (in a separate terminal)
    ```
    docker-compose -f slave.yml up slave satisfiability
    ```

5. Configure the master:
   * Visit `http://localhost/config` (or `http://localhost:<UI_PORT>/config` if `UI_PORT` was changed)
   * Paste and edit the following config:
     * Replace `player_id` with your wallet address
     * Replace `api_key` with your API key from step 1
     * Press **Save** after updating
    ```
    {
      "player_id": "0x0000000000000000000000000000000000000000",
      "api_key": "00000000000000000000000000000000",
      "api_url": "https://testnet-api.tig.foundation",
      "time_between_resubmissions": 60000,
      "max_concurrent_benchmarks": 4,
      "algo_selection": [
        {
          "algorithm_id": "c001_a004",
          "num_nonces": 40,
          "difficulty_range": [0, 0.5],
          "selected_difficulties": [],
          "weight": 1,
          "batch_size": 8
        }
      ],
      "time_before_batch_retry": 60000,
      "slaves": [
        {
          "name_regex": ".*",
          "algorithm_id_regex": ".*",
          "max_concurrent_batches": 1
        }
      ]
    }
    ```

6. Sit back and watch as benchmarks are submitted!
   * You can list available algorithms using:
   ```
   docker exec satisfiability list_algorithms --testnet
   ```

# Architecture Overview

* The **Master** defines benchmark scheduling strategy:
  * Algorithm selection
  * Difficulty selection
  * Batch sizing
  * Slave assignment
  ```
  docker-compose -f master.yml up
  ```

* A **Slave** benchmarks specific challenges
  ```
  docker-compose -f slave.yml up slave [challenge] .. [challenge]
  ```
  * CPU challenges include: `satisfiability`, `vehicle_routing`, and `knapsack`
  * GPU challenges (requires CUDA 12.6.3+) include: `vector_search`, and `hypergraph`
  * `slave/config.json` controls how many algorithms are ran concurrently

# Configuration Details

## `.env` file

Shared by both `master.yml` and `slave.yml`:

```
# Version of all benchmarker containers
VERSION=0.0.1
# Set to 1 to enable verbose logging
VERBOSE=1

POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=postgres
UI_PORT=80
DB_PORT=5432

# This is used by both master and slave
MASTER_PORT=5115
# This is used by slave to connect to master. Set to 172.17.0.1 if master and slave are running on same server
MASTER_IP=172.17.0.1

# Path to config file for slave. Mounts to /app/config.json inside slave container
SLAVE_CONFIG=./slave/config.json
# Directory for slave to download algorithms. Mounts to /app/algorithms inside slave containers
ALGORITHMS_DIR=./algorithms
# Directory for slave to store results. Mounts to /app/results inside slave containers
RESULTS_DIR=./results
# Seconds for results to live
TTL=300
# Name of the slave. Defaults to randomly generated name
SLAVE_NAME=
# How many worker threads to spawn in the slave container
NUM_WORKERS=8
```

Common variables to customise:
1. `VERBOSE=` (empty string for quieter logging)
2. `POSTGRES_PASSWORD`
3. `MASTER_IP` (set to `172.17.0.1` for slaves on same host as master)
4. `MASTER_PORT`
4. `SLAVE_NAME` (must be a unique name, or else master will assign duplicate batches)
5. `NUM_WORKERS` (number of worker threads on a slave)

## Master Config

The master config defines how benchmarking jobs are selected, scheduled, and distributed to slaves. This config can be edited via the master UI `/config` or via API `/update-config`.

```json
{
  "player_id": "0x0000000000000000000000000000000000000000",
  "api_key": "00000000000000000000000000000000",
  "api_url": "https://mainnet-api.tig.foundation",
  "time_between_resubmissions": 60000,
  "max_concurrent_benchmarks": 4,
  "algo_selection": [
    {
      "algorithm_id": "c001_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    },
    ...
  ],
  "time_before_batch_retry": 60000,
  "slaves": [
    {
      "name_regex": ".*",
      "algorithm_id_regex": ".*",
      "max_concurrent_batches": 1
    }
  ]
}
```

**Explanation:**
* `player_id`: Your wallet address (lowercase)
* `api_key`: See last section on how to obtain your API key
* `api_url`: mainnet (https://mainnet-api.tig.foundation) or testnet (https://testnet-api.tig.foundation)
* `time_between_resubmissions`: Time in milliseconds to wait before resubmitting an benchmark/proof which has not confirmed into a block
* `max_concurrent_benchmarks`: Maximum number of benchmarks that can be "in flight" at once (i.e., benchmarks where the proof has not been computed yet).
* `algo_selection`: list of algorithms that can be picked for benchmarking. Each entry has:
  * `algorithm_id`: id for the algorithm (e.g., c001_a001)
  * `num_nonces`: Number of instances to benchmark for this algorithm
  * `difficulty_range`: the bounds (0.0 = easiest, 1.0 = hardest) for a random difficulty sampling. Full range is `[0.0, 1.0]`
  * `selected_difficulties`: A list of difficulties `[[x1,y1], [x2, y2], ...]`. If any of the difficulties are in valid range, one will be randomly selected instead of sampling from the difficulty range
  * `weight`: Selection weight. An algorithm is chosen proportionally to `weight / total_weight`
  * `batch_size`: Number of nonces per batch. Must be a power of 2. For example, if num_nonces = 40 and batch_size = 8, the benchmark is split into 5 batches

## Tips

When benchmarking, you’ll likely want to run CPU and GPU algorithms on different servers. Below are examples for each case:

### Running a Slave with CPU Challenges

```bash
docker-compose -f slave.yml up slave satisfiability vehicle_routing knapsack
```

### Running a Slave with GPU Challenges
```
docker-compose -f slave.yml up slave vector_search hypergraph
```
> Note: GPU challenges require CUDA 12.6.3+ installed on the host.

### Configuring the Master to Route Batches

In your master config, you can route algorithms to different slaves based on their names and algorithm_ids. The example below routes CPU challenges to slaves with name starting with `cpu-`, and GPU challenges to slaves with name starting with `gpu-`:
```
"slaves": [
  {
    "name_regex": "cpu-.*",
    "algorithm_id_regex": "c00[123].*",
    "max_concurrent_batches": 1
  },
  {
    "name_regex": "gpu-.*",
    "algorithm_id_regex": "c00[45].*",
    "max_concurrent_batches": 1
  }
]
```

For this to work, names of slaves must match the regex by editing the `.env` file. Example:
```
SLAVE_NAME=cpu-server-01 
```

### Listing Available Algorithms

After launching a slave, you can list the available algorithms (and their ids) using:
```
docker exec <challenge_container_name> list_algorithms
```

Example output:
```
# docker exec satisfiability list_algorithms
...
id: c001_a056    name: sat_maximize_high    status: active with 0 merge points
id: c001_a057    name: sat_separate_vav     status: active with 0 merge points
id: c001_a058    name: sat_hybrid_vav       status: active with 0 merge points
id: c001_a059    name: sat_unified          status: merged @ round 62
id: c001_a060    name: sat_unified_opt      status: merged @ round 71
id: c001_a061    name: better_sat           status: merged @ round 65
id: c001_a062    name: even_better_sat      status: active with 0 merge points
```
This is helpful when updating the `algorithm_id_regex` field in your master config.

# Hard Resetting

To hard reset master:
1. Kill the services `docker-compose -f master.yml down`
2. Delete the database: `rm -rf db_data`

To hard reset slave:
1. Kill the services `docker-compose -f slave.yml down`
2. Delete the data: `rm -rf algorithms results`

# Finding your API Key

## Mainnet

1. Navigate to https://play.tig.foundation/
2. Connect your wallet
3. Your API key can be copied from the bottom left corner of the dashboard

## Testnet

1. Navigate to https://test.tig.foundation/
2. Connect your wallet
3. Your API key can be copied from the bottom left corner of the dashboard

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
