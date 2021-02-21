## Upstream dask image
ARG BASE=celsiustx/dask:dc6cc91ab379f4bb7c7ba2e01b75505dd47fb987
FROM $BASE

WORKDIR /opt/src

# Clone
RUN git clone --recurse-submodules https://github.com/celsiustx/anndata.git
WORKDIR anndata
RUN git remote add -f upstream https://github.com/theislab/anndata.git

# Checkout
ARG REF=origin/ctx
RUN git checkout $REF

## Install
RUN pip install -e .[test] \
 && pip install psycopg2 scanpy

RUN pytest -v

WORKDIR /
