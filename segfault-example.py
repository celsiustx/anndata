#!/usr/local/bin/python3.7
import json 
import numpy as np
import os
from scipy import sparse as scipy_sparse

import anndata
import dask
import scanpy

dask.config.set(scheduler='synchronous')
os.chdir("/mnt/ebs/moredata")  # Switch this to wherever you have the s3 data loaded.

# Get the .h5 file.
file1_10x_h5 = "CID003069-1.h5"
if not os.path.exists(file1_10x_h5):
    import subprocess
    subprocess.check_call("aws s3 cp s3://celsius-external-speedup/moredata/CID003069-1.h5 ." % file1_10x_h5, shell=True)

# Convert to .h5ad.
file2_h5ad = "CID003069-1.h5ad"
if not os.path.exists(file2_h5ad):
    an1_10x_nodask = scanpy.read_10x_h5(file1_10x_h5, gex_only=False)
    an1_10x_nodask.write_h5ad(file2_h5ad)

# Convert to .filtered.h5ad.
file3_h5ad_filtered = "CID003069-1.filtered.h5ad"
if not os.path.exists(file3_h5ad_filtered):
    an2_nodask = anndata.read_h5ad(file2_h5ad, as_sparse_fmt=scipy_sparse.csr_matrix)
    an2_nodask_full_rows = an2_nodask[np.where(an2_nodask.X.toarray().any(axis=1))]
    an2_nodask_full_rows_cols = an2_nodask_full_rows[np.where(an2_nodask_full_rows.X.toarray().any(axis=0))]
    an2_nodask_full_rows_cols.write(file3_h5ad_filtered)


# Sometimes the segfault happens inside of `load_dask_array`:
an3_dask = anndata.read_h5ad(file3_h5ad_filtered, backed=True, dask=True)

Xdask = an3_dask.X

# Sometimes it happens during compute():
Xreal = Xdask.compute()

# Sometimes we get this exception during compute():
"""
Traceback (most recent call last):
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/sigsegv_example.py", line 41, in <module>
    Xreal = Xdask.compute()
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/scanpy/anndata/dask/dask/base.py", line 166, in compute
    (result,) = compute(self, traverse=False, **kwargs)
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/scanpy/anndata/dask/dask/base.py", line 438, in compute
    return repack([f(r, *a) for r, (f, a) in zip(results, postcomputes)])
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/scanpy/anndata/dask/dask/base.py", line 438, in <listcomp>
    return repack([f(r, *a) for r, (f, a) in zip(results, postcomputes)])
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/scanpy/anndata/dask/dask/array/core.py", line 987, in finalize
    return concatenate3(results)
  File "/Users/ssmith/Dropbox/dev-celsius/scdb/scanpy/anndata/dask/dask/array/core.py", line 4377, in concatenate3
    result[idx] = arr
ValueError: setting an array element with a sequence.
"""
