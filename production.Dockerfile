FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

COPY src /opt/rfantibody/src
COPY scripts /opt/rfantibody/scripts
COPY include/USalign /opt/rfantibody/include/USalign
COPY include/SE3Transformer /opt/rfantibody/include/SE3Transformer
COPY poetry.lock /opt/rfantibody/poetry.lock
COPY pyproject.toml /opt/rfantibody/pyproject.toml
COPY README.md /opt/rfantibody/README.md

RUN apt-get update && apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.10 python3-pip pipx vim make wget

RUN alias "python"="python3.10"

RUN pip install poetry
RUN mkdir /opt/rfantibody/include/dgl
RUN wget -O /opt/rfantibody/include/dgl/dgl-2.4.0+cu118-cp310-cp310-manylinux1_x86_64.whl https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl

WORKDIR /opt/rfantibody/include/USalign
RUN make

WORKDIR /opt/rfantibody
RUN poetry install

ENTRYPOINT /bin/bash
