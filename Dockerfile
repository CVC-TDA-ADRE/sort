FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo
RUN ln -sv /usr/bin/python3 /usr/bin/python

WORKDIR /code
RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py && \
        rm get-pip.py

COPY requirements.txt /software/requirements.txt
RUN pip3 install -r /software/requirements.txt

