#!/usr/bin/env bash
# Run inside an activated, dedicated Python 3.10 environment on CUDA 11.8 Linux.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
nvcc --version
python scripts/bootstrap_sources.py
python -m pip install --upgrade pip 'setuptools<70' wheel ninja
python -m pip install 'numpy==1.24.4' 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch'
python -m pip install -e third_party/neurad-studio 'dataclass-wizard<0.23'
# Current custom setup.py uses BUILD_CUDA (not BUILD_NO_CUDA). Zero selects lazy JIT.
BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e third_party/splatad
python -m pip install -e '.[test]'
mkdir -p outputs reports
python -m pip freeze > outputs/requirements.official.lock.txt
python -m splatad_drive doctor --check-upstream --output outputs/environment_official.json
python -m pytest -m cuda -v
