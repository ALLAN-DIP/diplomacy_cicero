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
    autoconf \
    build-essential \
    bzip2 \
    ca-certificates \
    clang-format-8 \
    cmake \
    curl \
    git \
    libgoogle-glog-dev \
    libtool \
    pkg-config \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh >~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -u -p /usr/local \
    && rm ~/miniconda.sh

# Create env?

# Install pytorch, pybind11
# RUN conda install --yes pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
RUN conda install --yes pybind11

# Install go for boringssl in grpc
# We have some hacky patching code for protobuf that is not guaranteed
# to work on versions other than this.
RUN conda install --yes go protobuf==3.19.1

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

# Pinned `diplomacy` to `centaur-interface`
# Pinned `daidepp` to `main`
RUN pip install --no-cache-dir \
    git+https://github.com/ALLAN-DIP/diplomacy.git@027f42288c3c9ab3261ecd2961287ec319ddefdf \
    git+https://github.com/delaschwein/daidepp.git@859b99b4ac1cf6fc02f05b38d6bcd9781b47f97d \
    ujson==5.7.0

COPY . .
RUN pip install --no-cache-dir -e . -vv

# RUN scp wwongkam@ls6.tacc.utexas.edu:/corral/projects/DARPA-SHADE/Shared/UMD/best_model/pytorch_model.bin .

# Make
RUN make

# Run unit tests
RUN make test_fast
