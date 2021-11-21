# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git cmake wget ffmpeg libsm6 libxext6 python3-opencv
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:${PATH}
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
WORKDIR /app
COPY . .
RUN jupyter nbextension enable --py widgetsnbextension




