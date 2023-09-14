FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.1.1-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.70 \
    DEBIAN_FRONTEND=noninteractive

# Hack due to Nvidia :/! Check if it's fixed ASAP
RUN rm /etc/apt/sources.list.d/cuda.list


#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04
#FROM ubuntu:20.04

RUN apt update && apt-get upgrade -y && apt-get install -y curl

RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime

RUN apt-get install -y wget bzip2 ca-certificates curl git build-essential build-essential \
clang-format-8 cmake autoconf libtool pkg-config libgoogle-glog-dev

# found we needed these too
#RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
##RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" | tee -a /etc/apt/sources.list
#RUN apt-get update
#RUN apt-get install --yes llvm-11 llvm-11-dev libedit-dev
#RUN apt-get install --yes libffi-dev 

# Apt installs

# Install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -u -p /usr/local

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
RUN pip install -U pip
#ENV LLVM_CONFIG=/usr/bin/llvm-config-11
#ENV CUDA_TOOLKIT_ROOT_DIR
#ENV CUDA_NVCC_EXECUTABLE
#ENV CUDA_INCLUDE_DIRS

# COPY . /diplomacy_cicero
WORKDIR /diplomacy_cicero

# don't forget to remove daidepp (the last line) and torch from requirements.txt 
COPY requirements.txt .
RUN pip install -r requirements.txt

# Local pip installs
COPY thirdparty/ thirdparty/
RUN pip install -e ./thirdparty/github/fairinternal/postman/nest/

# NOTE: Postman here links against pytorch for tensors, for this to work you may
# need to separately have installed cuda 11 on your own.
RUN ln -s /usr/local/cuda /usr/local/nvidia
ENV Torch_DIR=/usr/local/lib/python3.7/site-packages/torch/share/cmake/Torch
# COPY .git/ .git/
RUN pip install -e ./thirdparty/github/fairinternal/postman/postman/
COPY . .
RUN pip install -e . -vv
RUN pip install ujson
RUN pip install git+https://git@github.com/SHADE-AI/diplomacy.git@comm_state
RUN pip install git+https://github.com/SiddarGu/daidepp.git

# RUN scp wwongkam@ls6.tacc.utexas.edu:/corral/projects/DARPA-SHADE/Shared/UMD/best_model/pytorch_model.bin .

# Make
RUN make

# Run unit tests
RUN make test_fast
