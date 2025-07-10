#!/bin/bash

# This is an installation script for new blackwell GPU architechture, we need to install the nightly version of PyTorch for CUDA to work if you are training on B200s or RTX 50 series

python -m venv .venv
. .venv/bin/activate
pip install pip-tools
pip-compile
pip install torch==2.8.0.dev20250603+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt # Do not pip-sync as it will uninstall the nightly build 
