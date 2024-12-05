FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p ./tig_benchmarker
COPY . ./tig_benchmarker