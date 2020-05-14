import warnings
import collections.abc as cabc
from collections import OrderedDict
from copy import deepcopy
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


def _init_as_actual(
    self,
    X=None,
    obs=None,
    var=None,
    uns=None,
    obsm=None,
    varm=None,
    varp=None,
    obsp=None,
    raw=None,
    layers=None,
    dtype="float32",
    shape=None,
    filename=None,
    filemode=None,
    fd=None,
    dask=False,
):
    from anndata._io.dask.utils import is_dask
    from anndata._core.anndata import _gen_dataframe

    # view attributes
    self._is_view = False
    self._adata_ref = None
    self._oidx = None
    self._vidx = None
    self._dask = dask

    # ----------------------------------------------------------------------
    # various ways of initializing the data
    # ----------------------------------------------------------------------

    # If X is a data frame, we store its indices for verification
    x_indices = []

    # init from file
    if filename is not None:
        self.file = AnnDataFileManager(self, filename, filemode, fd=fd)
    else:
        self.file = AnnDataFileManager(self, fd=fd)

        # init from AnnData
        if isinstance(X, AnnData):
            if any((obs, var, uns, obsm, varm, obsp, varp)):
                raise ValueError(
                    "If `X` is a dict no further arguments must be provided."
                )
            X, obs, var, uns, obsm, varm, obsp, varp, layers, raw = (
                X._X,
                X.obs,
                X.var,
                X.uns,
                X.obsm,
                X.varm,
                X.obsp,
                X.varp,
                X.layers,
                X.raw,
            )

        # init from DataFrame
        elif isinstance(X, pd.DataFrame):
            # to verify index matching, we wait until obs and var are DataFrames
            if obs is None:
                obs = pd.DataFrame(index=X.index)
            elif not isinstance(X.index, pd.RangeIndex):
                x_indices.append(("obs", "index", X.index))
            if var is None:
                var = pd.DataFrame(index=X.columns)
            elif not isinstance(X.columns, pd.RangeIndex):
                x_indices.append(("var", "columns", X.columns))
            X = ensure_df_homogeneous(X, "X")

    # ----------------------------------------------------------------------
    # actually process the data
    # ----------------------------------------------------------------------

    # check data type of X
    if X is not None:
        for s_type in StorageType:
            if isinstance(X, s_type.value):
                break
        else:
            class_names = ", ".join(c.__name__ for c in StorageType.classes())
            raise ValueError(
                f"`X` needs to be of one of {class_names}, not {type(X)}."
            )
        if shape is not None:
            raise ValueError("`shape` needs to be `None` if `X` is not `None`.")
        _check_2d_shape(X)
        # if type doesn’t match, a copy is made, otherwise, use a view
        if issparse(X) or isinstance(X, ma.MaskedArray):
            # TODO: maybe use view on data attribute of sparse matrix
            #       as in readwrite.read_10x_h5
            if X.dtype != np.dtype(dtype):
                X = X.astype(dtype)
        elif isinstance(X, ZarrArray):
            X = X.astype(dtype)
        elif is_dask(X):
            print(f'Pass Dask array: {X}')
            pass
        else:  # is np.ndarray or a subclass, convert to true np.ndarray
            X = np.array(X, dtype, copy=False)
        # data matrix and shape
        self._X = X
        self._n_obs, self._n_vars = self._X.shape
    else:
        self._X = None
        self._n_obs = len([] if obs is None else obs)
        self._n_vars = len([] if var is None else var)
        # check consistency with shape
        if shape is not None:
            if self._n_obs == 0:
                self._n_obs = shape[0]
            else:
                if self._n_obs != shape[0]:
                    raise ValueError("`shape` is inconsistent with `obs`")
            if self._n_vars == 0:
                self._n_vars = shape[1]
            else:
                if self._n_vars != shape[1]:
                    raise ValueError("`shape` is inconsistent with `var`")

    # annotations
    self._obs = _gen_dataframe(obs, self._n_obs, ["obs_names", "row_names"])
    self._var = _gen_dataframe(var, self._n_vars, ["var_names", "col_names"])

    # now we can verify if indices match!
    for attr_name, x_name, idx in x_indices:
        attr = getattr(self, attr_name)
        if isinstance(attr.index, pd.RangeIndex):
            attr.index = idx
        elif not idx.equals(attr.index):
            raise ValueError(f"Index of {attr_name} must match {x_name} of X.")

    # unstructured annotations
    self.uns = uns or OrderedDict()

    # TODO: Think about consequences of making obsm a group in hdf
    self._obsm = AxisArrays(self, 0, vals=convert_to_dict(obsm))
    self._varm = AxisArrays(self, 1, vals=convert_to_dict(varm))

    self._obsp = PairwiseArrays(self, 0, vals=convert_to_dict(obsp))
    self._varp = PairwiseArrays(self, 1, vals=convert_to_dict(varp))

    # Backwards compat for connectivities matrices in uns["neighbors"]
    _move_adj_mtx({"uns": self._uns, "obsp": self._obsp})

    self._check_dimensions()
    self._check_uniqueness()

    if self.filename:
        assert not isinstance(
            raw, Raw
        ), "got raw from other adata but also filename?"
        if {"raw", "raw.X"} & set(self.file):
            raw = dict(X=None, **raw)
    if not raw:
        self._raw = None
    elif isinstance(raw, cabc.Mapping):
        self._raw = Raw(self, **raw)
    else:  # is a Raw from another AnnData
        self._raw = Raw(self, raw._X, raw.var, raw.varm)

    # clean up old formats
    self._clean_up_old_format(uns)

    # layers
    self._layers = Layers(self, layers)

