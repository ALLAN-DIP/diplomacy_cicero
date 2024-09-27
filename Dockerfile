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

WORKDIR /diplomacy_cicero

# Create env?
COPY environment.yaml .
RUN conda env update --file environment.yaml --prune

# Local pip installs
COPY thirdparty/ thirdparty/
RUN pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/nest/

# NOTE: Postman here links against pytorch for tensors, for this to work you may
# need to separately have installed cuda 11 on your own.
RUN ln -s /usr/local/cuda /usr/local/nvidia
ENV Torch_DIR=/usr/local/lib/python3.7/site-packages/torch/share/cmake/Torch
RUN pip install --no-cache-dir -e ./thirdparty/github/fairinternal/postman/postman/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && spacy download en_core_web_sm

COPY conf/ conf/
COPY fairdiplomacy/ fairdiplomacy/
COPY fairdiplomacy_external/ fairdiplomacy_external/
COPY heyhi/ heyhi/
COPY parlai_diplomacy/ parlai_diplomacy/
COPY pyproject.toml .
COPY setup.py .
COPY unit_tests/ unit_tests/
RUN pip install --no-cache-dir -e .

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
RUN pip install -U --no-cache-dir --force-reinstall --no-deps 'diplomacy @ git+https://github.com/ALLAN-DIP/diplomacy.git@1f6ce8803bfd35a3ebbcf9ded7325434f72d966a'
RUN pip install -U --no-cache-dir --force-reinstall --no-deps 'chiron_utils @ git+https://github.com/ALLAN-DIP/chiron-utils.git@216bb21cc1ac2f353de727e0ebaac31960471e95'

LABEL org.opencontainers.image.source=https://github.com/ALLAN-DIP/diplomacy_cicero
