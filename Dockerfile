FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    cmake

RUN pip install -r requirements.txt


