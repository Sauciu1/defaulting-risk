# Base image is Python 3.11
FROM python:3.11-slim-buster

# Set the working directory
WORKDIR /app


FROM python:3.11-slim

# Install dependencies for Poetry and your app
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry (official way)
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy only Poetry files first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Disable virtualenv creation and install deps
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy your full source code
COPY helpers/ /app/helpers

# Run the script
#CMD ["python", "./module_deployment.py"]


ENV HOST=0.0.0.0

EXPOSE 8989
