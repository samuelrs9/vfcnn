FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get upgrade -y --allow-unauthenticated

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
        build-essential \
        curl \
        davix-dev \
        dcap-dev \
        fonts-freefont-ttf \
        g++ \
        gcc \
        gfal2 \
        gfortran \
        git \
        libafterimage-dev \
        libavahi-compat-libdnssd-dev \
        libfreetype6-dev \
        libftgl-dev \
        libgfal2-dev \
        libgif-dev \
        libgl2ps-dev \
        libglew-dev \
        libglu-dev \
        libgraphviz-dev \
        libgsl-dev \
        libjemalloc-dev \
        libjpeg-dev \
        libkrb5-dev \
        libldap2-dev \
        liblz4-dev \
        liblzma-dev \
        libpcre++-dev \
        libreadline-dev \
        libssl-dev \
        libxext-dev \
        libxft-dev \
        libxpm-dev \
        libz-dev \
        libzmq3-dev \
        locales \
        lsb-release \
        make \
        module-init-tools \
        pkg-config \
        rsync \
        srm-ifce-dev \
        unixodbc-dev \
        unzip \
        vim \
        wget \
        && \
    apt-get clean 

# Install Python 3.7 and pip
RUN apt-get update && apt-get install -y --allow-unauthenticated \
        python3.7 \
        python3.7-dev \
        python3-pip \
        && \
    apt-get clean

# Set Python 3.7 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install TensorFlow and dependencies
RUN pip install tensorflow-gpu==2.2.0 numpy==1.19.5 scipy ipython protobuf==3.19.6 matplotlib scikit-learn

# Install CMake
RUN pip install cmake

# Copy project files
COPY . /tf_kdtree/

# Set working directory
WORKDIR /tf_kdtree

# Fix TensorFlow library symlink and build
RUN python fix_missing_tflib.py && \
    cd src && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80" \
          -DPython_EXECUTABLE=$(which python) .. && \
    make -j$(nproc) && \
    cp libtf_nndistance.so /tf_kdtree/

#############################
