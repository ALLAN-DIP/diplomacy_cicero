FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.1.1-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.70 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
    autoconf=2.69-11.1 \
    build-essential=12.8ubuntu1.1 \
    bzip2=1.0.8-2 \
    ca-certificates=20230311ubuntu0.20.04.1 \
    clang-format-8=1:8.0.1-9 \
    cmake=3.16.3-1ubuntu1.20.04.1 \
    curl=7.68.0-1ubuntu2.23 \
    git=1:2.25.1-1ubuntu3.13 \
    libgoogle-glog-dev=0.4.0-1build1 \
    libtool=2.4.6-14 \
    pkg-config=0.29.1-0ubuntu4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install conda
# `-b`: run install in batch mode (without manual intervention)
# `-u`: update an existing installation
# `-p PREFIX`: install prefix
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh >~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -u -p /usr/local \
    && rm ~/miniconda.sh

# Create env?
COPY environment.yaml .
RUN conda env update --file environment.yaml --prune

# Install python requirements
RUN pip install --no-cache-dir pip==24.0

WORKDIR /diplomacy_cicero

# Local pip installs
COPY thirdparty/ thirdparty/
RUN pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/nest/

# NOTE: Postman here links against pytorch for tensors, for this to work you may
# need to separately have installed cuda 11 on your own.
RUN ln -s /usr/local/cuda /usr/local/nvidia
ENV Torch_DIR=/usr/local/lib/python3.7/site-packages/torch/share/cmake/Torch
RUN pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/postman/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN spacy download en_core_web_sm

COPY conf/ conf/
COPY fairdiplomacy/ fairdiplomacy/
COPY fairdiplomacy_external/ fairdiplomacy_external/
COPY heyhi/ heyhi/
COPY parlai_diplomacy/ parlai_diplomacy/
COPY pyproject.toml .
COPY setup.py .
COPY unit_tests/ unit_tests/
RUN pip install --no-cache-dir -e . -vv

# RUN scp wwongkam@ls6.tacc.utexas.edu:/corral/projects/DARPA-SHADE/Shared/UMD/best_model/pytorch_model.bin .

COPY Makefile .
COPY dipcc/ dipcc/

# Make
RUN make

COPY slurm/ slurm/

# Run unit tests
RUN make test_fast

COPY LICENSE.md .
COPY LICENSE_FOR_MODEL_WEIGHTS.txt .
COPY README.md .
COPY bin/ bin/
COPY run.py .

# TODO: Remove lines added for efficiency
RUN pip install -U --force-reinstall --no-deps 'diplomacy @ git+https://github.com/ALLAN-DIP/diplomacy.git@68b5e3343e77591e1a879b0d6074a0c8b7fede8e'
RUN pip install -U --force-reinstall --no-deps 'chiron_utils @ git+https://github.com/ALLAN-DIP/chiron-utils.git@9244c8a0b35c7bdf80e00df99e1a02548ed7e3f6'

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
    strace \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.source=https://github.com/ALLAN-DIP/diplomacy_cicero
