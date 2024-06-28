FROM python:3.10-slim AS builder

# Install gcc and other build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install your requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
