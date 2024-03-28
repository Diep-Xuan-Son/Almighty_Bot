FROM python:3.9
FROM nvidia/cuda:12.0.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GRADIO_SERVER_PORT=8888
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
ENV TZ Asia/Vietnam

# Set the working directory
WORKDIR /mqbot

ADD . /mqbot
ADD pyproject.toml /mqbot/pyproject.toml
# ADD requirements.txt requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        # git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN python3 -m pip install pycocotools
# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install python dependencies
RUN python3 -m pip install -e .
# RUN pip --no-cache-dir install -r requirements.txt

CMD ["python3", "app.py"]


# docker build -t mqbot-gradio-web-api --load --rm .
# docker run -d -p 8888:8888 --name mqbot mqbot-gradio-web-api
