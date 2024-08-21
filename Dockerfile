FROM --platform=$BUILDPLATFORM python:3.11-alpine AS builder

RUN mkdir /notebooks
WORKDIR /notebooks

# Install system dependencies
RUN apk update && \
    apk --no-cache add \
    linux-headers \
    gcc \
    g++ \
    glpk \
    glpk-dev \
    openblas-dev \
    gfortran \
    pkgconfig \
    wget \
    make \
    patch

# Install IPOPT
RUN wget https://github.com/coin-or/Ipopt/archive/releases/3.14.4.tar.gz && \
    tar xvzf 3.14.4.tar.gz && \
    cd Ipopt-releases-3.14.4 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf Ipopt-releases-3.14.4 3.14.4.tar.gz

# Set environment variables for IPOPT
ENV IPOPT_DIR=/usr/local

# Copy requirements and install Python packages
COPY requirements.txt /notebooks/
RUN pip install --no-cache-dir -r requirements.txt

# Install Pyomo and its solver interfaces
RUN pip install --no-cache-dir pyomo

COPY . /notebooks/

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]