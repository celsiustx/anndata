## Upstream dask image
ARG BASE=celsiustx/dask:e1ecfe21255f0e54275a4c4865fd013cb006e2bb
FROM $BASE

WORKDIR /opt/src

# Clone
RUN git clone --recurse-submodules https://github.com/celsiustx/anndata.git
WORKDIR anndata
RUN git remote add -f upstream https://github.com/theislab/anndata.git

# Checkout
ARG REF=origin/ctx
RUN git checkout $REF

## Test
RUN pip install -r requirements.txt \
 && pip install -r requirements_tests.txt
RUN pytest -v

## Install
RUN pip install -e .

WORKDIR /
