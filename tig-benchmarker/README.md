# tig-benchmarker

Benchmarker for TIG. Expected setup is a single master and multiple slaves on different servers.

# Starting Your Master

Simply run:

```
POSTGRES_USER=postgres \
POSTGRES_PASSWORD=mysecretpassword \
POSTGRES_DB=postgres \
UI_PORT=80 \
DB_PORT=5432 \
MASTER_PORT=5115 \
# set VERBOSE=1 for debug master logs
VERBOSE= \
docker-compose up --build
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

# Connecting Slaves

1. Compile `tig-worker`: 
    ```
    cargo build -p tig-worker --release
    ```
2. Run `slave.py`:
    ```
    # assume you are in tig-benchmarker folder
    python3 slave.py ../target/release/tig-worker
    ```

**Notes:**
* If your master is on a different server to your slave, you need to add the option `--master <SERVER_IP>`
* To set the number of workers (threads), use the option `--workers <NUM_WORKERS>`
* To use a different port, use the option `--port <MASTER_PORT>`
* To see all options, use `--help` 

## Hard Resetting Your Slave

1. Stop your slave
2. Remove the output folder (defaults to results): `rm -rf results`
3. Start your slave

# Optimising your Config

1. `difficulty_sampler_config` allows you to configure the difficulty your benchmarker selects for each challenge:
    * `selected_difficulties` expects a list of difficulties. If any of those difficulties are within the valid range, they will be picked
    * `difficulty_ranges` expects a range defined by 2 numbers: `[lower, higher]`. This is used to sample a random difficulty in the valid range. Examples:
        * `[0.0, 1.0]` samples the full range of valid difficulties
        * `[0.0, 0.1]` samples the easiest 10% of valid difficulties


2. `job_manager_config` allows you to set the `batch_size` for each challenge.
    * `batch_size` is the number of nonces that are part of a batch. Must be a power of 2
    * Recommend to pick a `batch_size` for your slave with lowest `num_workers` such that it takes a few seconds to compute (e.g. 5 seconds)
    * `batch_size` shouldn't be too small, or else network latency between master and slave will affect performance
    * Use `slave_manager_config` to support slaves with different `num_workers`

3. `precommit_manager_config` allows you to control your benchmarks:
    * `max_pending_benchmarks` is the maximum number of pending benchmarks. You dont want benchmarks to take too long, nor do you want your slaves to be idle too much

    * `num_nonces` is the number of nonces to compute per benchmark. See Discord #community-tools channel for stats to help pick 

    * `weight` affects how likely the challenge will be picked (weight of 0 will never be picked). Observe your cutoff and imbalance on the [benchmarker page](https://play.tig.foundation/benchmarker)

4. `slave_manager_config` allows you to control your slaves:
    * Slave names are matched to regexes (or else rejected). 
    * `max_concurrent_batches` is the max number of concurrent batches a slave can work on
    * `selected_challenges` is a whitelist of challenges for that slave. If you don't want a slave to benchmark a specific challenge, remove its entry from the list

# Finding your API Key

## Mainnet

1. Navigate to https://play.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Mainnet `player_id`
    * `api_key` is your Mainnet API key

## Testnet

1. Navigate to https://test.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Testnet `player_id`
    * `api_key` is your Testnet API key

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
