FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevents install confirmation prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney
ENV PATH="/opt/.venv/bin:$PATH"

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add deadsnakes PPA and install Python 3.12+ minimal build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv curl wget gnupg gcc g++ build-essential python3.12-dev && \
    apt-get purge -y software-properties-common && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 (not for system python version)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create and activate virtual environment
RUN python3.12 -m venv /opt/.venv && \
    ln -sf /opt/.venv/bin/python3.12 /opt/.venv/bin/python

# Install torch
RUN pip install torch==2.3.1+cu121 --find-links https://download.pytorch.org/whl/torch_stable.html

# Install Python deps
COPY requirements.in requirements.in 
RUN pip install --no-cache-dir pip-tools && \
    pip-compile requirements.in --output-file requirements.txt && \
    pip install -r requirements.txt && \
    pip uninstall -y pip-tools && \
    rm -rf ~/.cache ~/.local

# Set working directory and copy app
WORKDIR /app
COPY . .

RUN chmod +x train_all_levels.py
CMD ["python", "train_all_levels.py"]
