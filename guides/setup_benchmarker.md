# Setup Benchmarker

## Prerequisites
- python3.9+, pip
- git
- docker, docker-compose

# for amazon linux 2023

install dependencies
```
# dnf install python3 python3-devel python3-pip git docker postgresql-devel gcc
# curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose && chmod 555 /usr/local/bin/docker-compose
```

clone the repository
```
$ git clone https://github.com/tig-foundation/tig-monorepo.git
```

navigate to the benchmarker directory
```
$ cd tig-monorepo/tig-benchmarker
```

install python dependencies
```
$ python3 -m pip install -r master/requirements.txt
```

start up the benchmarker using docker compose
```
$ docker-compose up
```

to control and configure the benchmarker remotely, see [remote_pgadmin.md](remote_pgadmin.md) and [remote_webui.md](remote_webui.md)