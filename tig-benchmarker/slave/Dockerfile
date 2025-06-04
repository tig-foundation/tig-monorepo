FROM python:3.13-slim

WORKDIR /app

COPY slave/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

RUN apt-get update && apt-get install -y docker.io \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY common common
COPY slave/main.py main.py
COPY slave/config.json config.json

CMD ["python", "main.py"]