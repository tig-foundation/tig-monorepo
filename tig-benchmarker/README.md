# tig-benchmarker

Benchmarker for TIG. Expected setup is a single master and multiple slaves on different servers.

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

## Optimising your Master Config

See [docs.tig.foundation](https://docs.tig.foundation/benchmarking/benchmarker-config)

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
* If your master is on a different server to your slave, you need to add the option `--master <SERVER_IP>`
* To set the number of workers (threads), use the option `--workers <NUM_WORKERS>`
* To use a different port, use the option `--port <MASTER_PORT>`
* To see all options, use `--help` 

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
