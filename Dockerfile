FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get update && apt-get install -y apt-utils build-essential curl git unzip autoconf \
    autogen libtool mlocate zlib1g-dev python3 wget bash-completion cmake-qt-gui pluma libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev python3-dev python3-numpy python3-pip libtbb2 libtbb-dev libjpeg-dev libpng-dev \
    libtiff-dev libjasper-dev libdc1394-22-dev libx11-dev && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN pip3 install tensorflow-gpu==2.0.0b1 pillow mxnet matplotlib==3.0.3 opencv-python==3.4.1.15


