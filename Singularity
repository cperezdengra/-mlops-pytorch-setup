Bootstrap: docker
From: python:3.10-slim

%labels
    Author cperezdengra
    Version v1.0
    Description MLOps con PyTorch + CI/CD + Singularity

%post
    apt-get update && apt-get install -y --no-install-recommends \
        git wget curl && \
        pip install --upgrade pip && \
        pip install torch torchvision && \
        pip install -r /app/requirements.txt && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

%files
    . /app

%environment
    export LC_ALL=C
    export PYTHONUNBUFFERED=1
    export PATH="/app:$PATH"

%runscript
    echo "Ejecutando entrenamiento..."
    python /app/model/train.py
