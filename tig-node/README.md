## tig-node

`tig-node` consists of 2 components:

1. `postgres` - a local postgres database for storing TIG's blockchain data
2. `sychroniser` - a script that downloads all the historic blocks and polls the TIG API for new blocks, storing data in the postgres database

### Get Started

You will need `docker` and `docker-compose` installed. Run the following command:

```bash
docker-compose up --build
```

**Notes:**
* You can edit `.env` file to configure:
    1. postgres connection details (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`)
    2. the blockchain (`NODE_ENV`) can be set to either `mainnet` (default) or `testnet`
* The database is stored in `mainnet-data` (or `testnet-data`)
* To only start the database, run the command `docker-compose up postgres`
* Your database is accessible via port 5432 on localhost
    * Example: `PGPASSWORD=password psql -U postgres -h localhost`


## License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)