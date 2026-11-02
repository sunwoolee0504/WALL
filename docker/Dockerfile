# Build through docker/build.sh so the project root is the build context.
FROM python:3.10-slim-bookworm

ARG WALL_WHEEL=cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/wall-env/bin:$PATH \
    SC2PATH=/opt/StarCraftII

# Headless SC2 shared libraries plus Git for the pinned SMAC dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git curl unzip libglib2.0-0 libgomp1 libstdc++6 \
    libfontconfig1 libfreetype6 libx11-6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wall-setup
COPY requirements-optimized.txt ./
COPY scripts/install_optimized_env.sh scripts/check_environment.py ./scripts/
COPY src/envs/__init__.py ./src/envs/__init__.py
RUN WALL_PYTHON_BIN=/usr/local/bin/python bash scripts/install_optimized_env.sh "$WALL_WHEEL" /opt/wall-env

# Project and externally installed SC2 are mounted by run.sh at launch time.
WORKDIR /workspace/wall
CMD ["python", "scripts/check_environment.py"]
