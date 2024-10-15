FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Use default answer for any questions asked by Debian tools
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS packages
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
    autoconf=2.69-* \
    clang-format-8=1:8.0.1-* \
    cmake=3.16.3-* \
    curl=7.68.0-* \
    git=1:2.25.1-* \
    libgoogle-glog-dev=0.4.0-* \
    libtool=2.4.6-* \
    pkg-config=0.29.1-* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install conda
# `-b`: run install in batch mode (without manual intervention)
# `-u`: update an existing installation
# `-p PREFIX`: install prefix
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh >~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -u -p /usr/local \
    && rm ~/miniconda.sh

# Switch to application directory
WORKDIR /diplomacy_cicero

# Update existing environment
# `pip` needs to be updated separately to prevent version conflict
COPY environment-lock.yaml .
RUN conda env update --file environment-lock.yaml --prune \
    && pip install --no-cache-dir pip==24.0

# Install local pip packages
COPY thirdparty/ thirdparty/
# NOTE: Postman here links against pytorch for tensors, for this to work you may
# need to separately have installed cuda 11 on your own.
ENV Torch_DIR=/usr/local/lib/python3.7/site-packages/torch/share/cmake/Torch
RUN pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/nest/ \
    && ln -s /usr/local/cuda /usr/local/nvidia \
    && pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/postman/

# Install application requirements
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt \
    && spacy download en_core_web_sm

# Install application itself
COPY conf/ conf/
COPY fairdiplomacy/ fairdiplomacy/
COPY fairdiplomacy_external/ fairdiplomacy_external/
COPY heyhi/ heyhi/
COPY parlai_diplomacy/ parlai_diplomacy/
COPY pyproject.toml .
COPY requirements.txt .
COPY setup.py .
COPY unit_tests/ unit_tests/
RUN pip install --no-cache-dir -e .

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
