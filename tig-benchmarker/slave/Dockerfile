FROM ubuntu:24.04

WORKDIR /app

RUN apt update && apt install -y curl python3 python3-pip
COPY slave/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --break-system-packages --no-cache-dir

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y docker-ce \
    && apt-get clean && rm -rf /var/lib/apt/lists/*