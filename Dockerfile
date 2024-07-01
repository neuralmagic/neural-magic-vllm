# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version \
    && python3 -m pip --version

RUN apt-get update -y \
    && apt-get install -y python3-pip git curl sudo

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

# install development dependencies
COPY requirements-lint.txt requirements-lint.txt
COPY requirements-test.txt requirements-test.txt
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG PYTHON_VERSION=3

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

#################### EXTENSION Build IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS vllm-base
ARG CUDA_VERSION=12.4.1
WORKDIR /vllm-workspace

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# install nm-vllm wheel first, so that torch etc will be installed
ARG build_type="NIGHTLY"
ARG build_version="latest"
ENV INSTALL_TYPE=${build_type}
ENV INSTALL_VERSION=${build_version}
# UPSTREAM SYNC: Install nm-vllm with sparsity extras
# use nm pypi for now for testing
RUN --mount=type=bind,from=build \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TYPE}" = "NIGHTLY" ]; then \
        if [ "${INSTALL_VERSION}" = "latest" ]; then \
            pip install nm-vllm-nightly[sparse] --extra-index-url https://pypi.neuralmagic.com/simple; \
        else \
            pip install nm-vllm-nightly[sparse]==${INSTALL_VERSION} --extra-index-url https://pypi.neuralmagic.com/simple; \
        fi; \
    else \
        if [ "${INSTALL_VERSION}" = "latest" ]; then \
            pip install nm-vllm[sparse] --extra-index-url https://pypi.neuralmagic.com/simple; \
        else \
            pip install nm-vllm[sparse]==${INSTALL_VERSION} --extra-index-url https://pypi.neuralmagic.com/simple; \
        fi; \
    fi

#################### vLLM installation IMAGE ####################

#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

# check installed version
RUN pip freeze | grep -e nm-vllm -e nm-magic-wand

# doc requires source code
# we hide them inside `test_docs/` , so that this source code
# will not be imported by other tests
RUN mkdir test_docs
RUN mv docs test_docs/
RUN mv vllm test_docs/

#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
