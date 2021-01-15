## Upstream dask image
ARG BASE=celsiustx/dask:b1e2948a553d60f901a1e475a55dd1566d8eb7e7
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
 && pip install -r requirements_tests.txt \
 && pip install psycopg2 scanpy

## Install
RUN pip install -e .

RUN pytest -v

WORKDIR /
