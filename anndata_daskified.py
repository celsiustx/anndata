import warnings
import collections.abc as cabc
from collections import OrderedDict
from copy import deepcopy

from anndata._core.anndata import AnnData
from dask import dataframe as dd
from dask.array.backends import register_scipy_sparse
register_scipy_sparse()

from enum import Enum
from functools import reduce, singledispatch
from pathlib import Path
from os import PathLike
from typing import Any, Union, Optional  # Meta
from typing import Iterable, Sequence, Mapping, MutableMapping  # Generic ABCs
from typing import Tuple, List  # Generic

import h5py
from natsort import natsorted
import numpy as np
from numpy import ma
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical
from scipy import sparse
from scipy.sparse import issparse

from anndata._core.raw import Raw
from anndata._core.index import _normalize_indices, _subset, Index, Index1D, get_vector
from anndata._core.file_backing import AnnDataFileManager
from anndata._core.access import ElementRef
from anndata._core.aligned_mapping import (
    AxisArrays,
    AxisArraysView,
    PairwiseArrays,
    PairwiseArraysView,
    Layers,
    LayersView,
)
from anndata._core.views import (
    ArrayView,
    DictView,
    DataFrameView,
    as_view,
    _resolve_idxs,
)
from anndata._core.merge import merge_uns
from anndata._core.sparse_dataset import SparseDataset
from anndata import utils
from anndata.utils import convert_to_dict, ensure_df_homogeneous
from anndata.logging import anndata_logger as logger
from anndata.compat import (
    ZarrArray,
    ZappyArray,
    DaskArray,
    DaskDelayed,
    Literal,
    _slice_uns_sparse_matrices,
    _move_adj_mtx,
    _overloaded_uns,
    OverloadedDict,
)

####

def _init_as_view(self, adata_ref: "AnnData", oidx: Index, vidx: Index):
    from anndata._io.dask.utils import is_dask, daskify_call, daskify_method_call, \
        daskify_iloc, daskify_get_len_given_slice

    ### BEGIN COPIED FROM ORIGINAL
    if is_dask(adata_ref) or is_dask(oidx) or is_dask(vidx):
        use_dask = True

    if adata_ref.isbacked and adata_ref.is_view:
        raise ValueError(
            "Currently, you cannot index repeatedly into a backed AnnData, "
            "that is, you cannot make a view of a view."
        )
    self._is_view = True
    if isinstance(oidx, (int, np.integer)):
        oidx = slice(oidx, oidx + 1, 1)
    if isinstance(vidx, (int, np.integer)):
        vidx = slice(vidx, vidx + 1, 1)
    if adata_ref.is_view:
        prev_oidx, prev_vidx = adata_ref._oidx, adata_ref._vidx
        adata_ref = adata_ref._adata_ref
        oidx, vidx = _resolve_idxs((prev_oidx, prev_vidx), (oidx, vidx), adata_ref)
    self._adata_ref = adata_ref
    self._oidx = oidx
    self._vidx = vidx
    # the file is the same as of the reference object
    self.file = adata_ref.file
    ### END COPIED FROM ORIGINAL

    # views on attributes of adata_ref
    if (not is_dask(oidx)) and oidx == slice(None, None, None):
        # If we didn't slice obs, just return the original.
        n_obs = adata_ref.n_obs
        obs_sub = adata_ref.obs
    else:
        if is_dask(oidx) or is_dask(adata_ref.n_obs):
            n_obs = daskify_get_len_given_slice(oidx, adata_ref.n_obs)
        else:
            n_obs = len(range(*oidx.indices(adata_ref.n_obs)))
        obs_sub = daskify_iloc(adata_ref.obs, oidx)

    if (not is_dask(vidx)) and vidx == slice(None, None, None):
        # If we didnt' slice var, just return the original.
        var_sub = adata_ref.var
        n_vars = adata_ref.n_vars
    else:
        if is_dask(vidx) or is_dask(adata_ref.n_vars):
            n_vars = daskify_get_len_given_slice(vidx, adata_ref.n_vars)
        else:
            n_vars = len(range(*vidx.indices(adata_ref.n_vars)))
        var_sub = daskify_iloc(adata_ref.var, vidx)

    self._obsm = daskify_method_call(adata_ref.obsm, "iloc", oidx)
    self._varm = daskify_method_call(adata_ref.obsm, "iloc", oidx)
    self._layers = daskify_method_call(adata_ref.layers, "_view", self,
                                       (oidx, vidx))
    self._obsp = daskify_method_call(adata_ref.obsp, "_view", self, oidx)
    self._varp = daskify_method_call(adata_ref.varp, "_view", self, vidx)

    # Special case for old neighbors, backwards compat. Remove in anndata 0.8.
    uns_new1 = daskify_call(_slice_uns_sparse_matrices, adata_ref._uns,
                            self._oidx, adata_ref.n_obs)
    uns_new2 = daskify_method_call(self, "_remove_unused_categories",
                                   adata_ref.obs, obs_sub, uns_new1,
                                   inplace=False)
    uns_new = daskify_method_call(self, "_remove_unused_categories",
                                  adata_ref.var, var_sub, uns_new2,
                                  inplace=False)

    self._n_obs = n_obs
    self._n_vars = n_vars

    # set attributes
    def mk_dataframe_view(sub, ann, key):
        return DataFrameView(sub, view_args=(ann, key))

    def mk_dict_view(dat, ann):
        return DictView(dat, view_args=(ann, key))

    self._obs = daskify_call(mk_dataframe_view, obs_sub, self, "obs")
    self._var = daskify_call(mk_dataframe_view, var_sub, self, "var")
    self._uns = daskify_call(mk_dataframe_view, uns_new, self, "uns")

    ### BEGIN COPIED FROM ORIGINAL
    # set data
    if self.isbacked:
        self._X = None

    # set raw, easy, as it’s immutable anyways...
    if adata_ref._raw is not None:
        # slicing along variables axis is ignored
        self._raw = adata_ref.raw[oidx]
        self._raw._adata = self
    else:
        self._raw = None
    ### END COPIED FROM ORIGINAL

def X(self):
    from anndata._io.dask.hdf5.load_array import load_dask_array
    if self._X is None:
        if self.is_view:
            X = self._adata_ref.X[self._oidx, self._vidx]
        else:
            X = load_dask_array(path=self.file.filename, key='X',
                                format_str='csr', shape=self.shape)
            # NOTE: The original code has logic for when the backed X
            # comes from a Dataset below.  See the TODO below.
        self._X = X
    else:
        X = load_dask_array(path=self.file.filename, key='X',
                            format_str='csr', shape=self.shape)
    return self._X


class AnnDataDask(AnnData):
    def __init__(self, *args, **kwargs):
        self._dask = True
        super().__init__(*args, **kwargs)

    def _init_as_view(self, adata_ref: "AnnData", oidx: Index, vidx: Index):
        return _init_as_view(self, adata_ref, oidx, vidx)

    @property
    def X(self):
        return X(self)

    def __getitem__(self, index: Index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return self.__class__(self, oidx=oidx, vidx=vidx, asview=True)

