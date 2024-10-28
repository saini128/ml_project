FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    cmake \
    nano \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install -r requirements.txt

