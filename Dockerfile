FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Use default answer for any questions asked by Debian tools
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS packages
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
    autoconf \
    clang-format \
    cmake \
    curl \
    git \
    libgoogle-glog-dev \
    libtool \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
# `-b`: run install in batch mode (without manual intervention)
# `-u`: update an existing installation
# `-p PREFIX`: install prefix
RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh >~/miniforge.sh \
    && /bin/bash ~/miniforge.sh -b -u -p /usr/local \
    && rm ~/miniforge.sh

# Switch to application directory
WORKDIR /diplomacy_cicero

# Update existing environment and install uv as the package manager
COPY environment.yaml .
RUN conda env update --file environment.yaml --prune \
    && pip install uv

# Use the conda-managed system Python (no venv in this image)
ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml .

# Install local pip packages
COPY thirdparty/ thirdparty/
# NOTE: Postman here links against pytorch for tensors, for this to work you may
# need to separately have installed cuda 12 on your own.
ENV Torch_DIR=/usr/local/lib/python3.10/site-packages/torch/share/cmake/Torch
RUN uv pip install --no-cache -e ./thirdparty/github/fairinternal/postman/nest/ \
    && ln -s /usr/local/cuda /usr/local/nvidia \
    && uv pip install --no-cache -e ./thirdparty/github/fairinternal/postman/postman/

# Install application requirements
COPY requirements-lock.txt .
RUN uv pip install --no-cache --force-reinstall "setuptools==68.2.2" \
    && uv pip install --no-cache --no-deps fairseq==0.12.2 \
    && uv pip install --no-cache --no-deps fairscale==0.4.2 \
    && uv pip install --no-cache --no-deps "parlai @ git+https://github.com/facebookresearch/ParlAI.git@5214f42a2058ef335f91f5afe66b2bd9ebfb2fbe" \
    && uv pip install --no-cache --no-deps -r requirements-lock.txt \
    && spacy download en_core_web_sm

# Install application itself
COPY conf/ conf/
COPY fairdiplomacy/ fairdiplomacy/
COPY fairdiplomacy_external/ fairdiplomacy_external/
COPY heyhi/ heyhi/
COPY parlai_diplomacy/ parlai_diplomacy/
COPY requirements.txt .
COPY setup.py .
COPY unit_tests/ unit_tests/
RUN uv pip install --no-cache --no-deps -e .

# Build application
COPY Makefile .
COPY dipcc/ dipcc/
RUN make

# Run unit tests
COPY slurm/ slurm/
RUN make test_fast

# Copy remaining files
COPY LICENSE.md .
COPY LICENSE_FOR_MODEL_WEIGHTS.txt .
COPY README.md .
COPY bin/ bin/
COPY run.py .

LABEL org.opencontainers.image.source=https://github.com/ALLAN-DIP/diplomacy_cicero
