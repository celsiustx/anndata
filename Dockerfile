## Upstream dask image
ARG BASE=celsiustx/dask:e1ecfe21255f0e54275a4c4865fd013cb006e2bb
FROM $BASE

WORKDIR /opt/src

# For local development, swap these lines in for the clone+checkout blocks below
#COPY . anndata
#WORKDIR anndata
#RUN git clean -fdx

# Clone
RUN git clone --recurse-submodules https://github.com/celsiustx/anndata.git
WORKDIR anndata
RUN git remote add -f upstream https://github.com/theislab/anndata.git

# Checkout
ARG REF=origin/ctx
RUN git checkout $REF

## Build+Test Reqs
RUN pip install -r requirements.txt \
 && pip install -r requirements_tests.txt

## Install
RUN pip install -e .

RUN pytest -v


WORKDIR /
