FROM python:3.13-slim

WORKDIR /app

COPY master/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY common common
COPY master/master master
COPY master/main.py main.py

CMD ["python", "main.py"]
