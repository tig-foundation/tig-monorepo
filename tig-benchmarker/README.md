# tig-benchmarker

Benchmarker for TIG. Expected setup is a single master and multiple slaves on different servers.

## Overview

Benchmarking in TIG works as follows:

1. Master submits a precommit to TIG protocol, with details of what they will benchmark:
   * `block_id`: start of the benchmark
   * `player_id`: address of the benchmarker
   * `challenge_id`: challenge to benchmark
   * `algorithm_id`: algorithm to benchmark
   * `difficulty`: difficulty of instances to randomly generate
   * `num_nonces`: number of instances to benchmark

2. TIG protocol confirms the precommit, and assigns it a random string

3. Master starts benchmarking:
   * polls TIG protocol for the confirmed precommit + random string
   * creates a benchmark job, splitting it into batches
   * slaves poll the master for batches to compute
   * slaves do the computation, and send results back to master

4. After benchmarking is finished, master submits benchmark to TIG protocol:
   * `solution_nonces`: list of nonces for which a solution was found
   * `merkle_root`: Merkle root of the tree constructed using results as leafs

5. TIG protocol confirms the benchmark, and randomly samples nonces requiring proof

6. Master prepares the proof:
   * polls TIG protocol for confirmed benchmark + sampled nonces
   * creates proof jobs
   * slaves poll the master for nonces requiring proof
   * slaves send Merkle branches back to master

7. Master submits proof to TIG protocol

8. TIG protocol confirms the proof, calculating the block from which the solutions will become "active" (eligible to earn rewards)
   * Verification is performed in parallel
   * Solutions will be inactive 120 blocks from when the benchmark started
   * The delay is determined by number of blocks between the start and when proof was confirmed
   * Each block, active solutions which qualify will earn rewards for the Benchmarker

# Starting Your Master

Simply run:

```
docker-compose up --build
```

This uses the `.env` file:

```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=postgres
UI_PORT=80
DB_PORT=5432
MASTER_PORT=5115
VERBOSE=
```

See last section on how to find your player_id & api_key.

**Notes:**
* Interaction with the master is via UI: `http://localhost`
    * If your UI port is not 80, then your UI is accessed via `http://localhost:<UI_PORT>`
    * If you are running on a server, then your UI is access via: `http://<SERVER_IP>`
    * Alternatively, you can [ssh port forward](https://www.ssh.com/academy/ssh/tunneling-example)
* The config of the master can be updated via the UI
* Recommend to run dockers in detached mode: `docker-compose up --detach`
* You can view the logs of each service individually: `docker-compose logs -f <service>`
    * There are 4 services: `db`, `master`, `ui`, `nginx`
* To query the database, recommend to use [pgAdmin](https://www.pgadmin.org/)

## Hard Resetting Your Master

1. Kill the services: `docker-compose down`
2. Delete the database: `rm -rf db_data`
3. Start your master

## Master Config

The master config defines how benchmarking jobs are selected, scheduled, and distributed to slaves. This config can be edited via the master UI or via API (`http://localhost:<MASTER_PORT>/update-config`).

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
  * `algorithm_id`: id for the algorithm (e.g., c001_a001). Tip: use [list_algorithms](../scripts/list_algorithms.py) script to get list of algorithm ids
  * `num_nonces`: Number of instances to benchmark for this algorithm
  * `difficulty_range`: the bounds (0.0 = easiest, 1.0 = hardest) for a random difficulty sampling. Full range is `[0.0, 1.0]`
  * `selected_difficulties`: A list of difficulties `[[x1,y1], [x2, y2], ...]`. If any of the difficulties are in valid range, one will be randomly selected instead of sampling from the difficulty range
  * `weight`: Selection weight. An algorithm is chosen proportionally to `weight / total_weight`
  * `batch_size`: Number of nonces per batch. Must be a power of 2. For example, if num_nonces = 40 and batch_size = 8, the benchmark is split into 5 batches

# Connecting Slaves

1. Run the appropiate [runtime docker image](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fruntime) for your slave. Available flavours are:
    * amd64 (x86_64 compatible)
    * aarch64
    * amd64-cuda12.6.3 (x86_64 compatible)
    * aarch64-cuda12.6.3
    ```
    # example
    docker run -it --gpus all ghcr.io/tig-foundation/tig-monorepo/runtime:0.0.1-amd64-cuda12.6.3
    ```

2. Run `slave.py`:
    ```
    # runtime docker container should start you in /app
    python3 slave.py --help
    ```

**Notes:**
* If your master is on a different server, add `--master <SERVER_IP>`
* Set a custom master port with `--port <MASTER_PORT>`
* To see all options, use `--help` 

## Slave Config

You can control execution limits via a JSON config:

```json
{
    "max_workers": 100,
    "algorithms": [
        {
            "id_regex": ".*",
            "cpu": 1.0,
            "gpu": 0.0
        }
    ]
}
```

**Explanation:**
* By default, `slave.py` uses all CPUs and all GPUs. To impose limits you should set the environment variables:
  * `CPU_VISIBLE_CORES`. e.g. you have 8 CPUs, to only expose the first and last: `export CPU_VISIBLE_CORES=0,7`
  * `CUDA_VISIBLE_DEVICES`. e.g. you have 4 GPUs, to only expose the first and second: `export CUDA_VISIBLE_DEVICES=0,1`
* `max_workers`: maximum concurrent tig-runtime processes.
* `algorithms`: rules for matching algorithms based on `id_regex`.
    * Regex matches algorithm ids (e.g., `c004_a[\d3]` matches all vector_search algorithms).
    * An algorithm + nonce only starts being processed if:
      ```
      TOTAL_USAGE["cpu"] + cpu <= len(VISIBLE_CPUS) and 
      TOTAL_USAGE["gpu"] + gpu <= len(VISIBLE_GPUS)
      ```
    * Total costs gets adjusted when processing starts and ends
      ```
      # when processing starts 
      TOTAL_USAGE["cpu"] += cpu
      TOTAL_USAGE["gpu"] += gpu

      # processing logic

      # when processing ends 
      TOTAL_USAGE["cpu"] -= cpu
      TOTAL_USAGE["gpu"] -= gpu
      ```

**Example:**

This example limits c001/c002/c003 to 2 concurrent instances per CPU. It also limits c004/c005 to 4 concurrent instances per GPU:

```json
{
    "max_workers": 10,
    "cpus": 4,
    "gpus": 2,
    "algorithms": [
        {
            "id_regex": "c00[123].*",
            "cpu_cost": 0.5,
            "gpu_cost": 0.0
        },
        {
            "id_regex": "c00[45].*",
            "cpu_cost": 0.0,
            "gpu_cost": 0.25
        }
    ]
}
```

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
