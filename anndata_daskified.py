import warnings
import collections.abc as cabc
from collections import OrderedDict
from copy import deepcopy

import dask
from anndata._core.anndata import AnnData, ImplicitModificationWarning
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

from dask.array.backends import register_scipy_sparse
register_scipy_sparse()


class AnnDataDask(AnnData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_as_view(self, adata_ref: "AnnData", oidx: Index, vidx: Index):
        from anndata._io.dask.utils import daskify_call, daskify_method_call, \
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

    def _gen_repr(self, n_obs, n_vars) -> str:

        if self.isbacked:
            backed_at = f"backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"AnnData object with n_obs × n_vars = {n_obs} × {n_vars} {backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            obj = getattr(self, attr)
            if is_dask(obj):
                descr += f"\n    {attr}: {str(obj)}"
            elif hasattr(obj, 'keys'):
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
            else:
                from dask.dataframe import DataFrame
                if isinstance(obj, DataFrame):
                    descr += f"\n    {attr}: {str(obj.columns.tolist())[1:-1]}"
                else:
                    from sys import stderr
                    stderr.write(f'Unknown attr type {type(obj)}: {obj}\n')

        return descr

    @property
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

    @X.setter
    def X(self, value: Optional[Union[np.ndarray, sparse.spmatrix]]):
        raise NotImplementedError("The AnnDataDask.X is immutable!  "
                                  "To change attributes, make an updated copy() method...")

    def __getitem__(self, index: Index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return self.__class__(self, oidx=oidx, vidx=vidx, asview=True)

    def compute(self, *args, **kwargs):
        from anndata._io.dask.utils import compute_anndata
        return compute_anndata(self, *args, **kwargs)

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        from anndata._io.dask.utils import daskify_call
        import anndata._core
        uns_overloaded = daskify_call(anndata._core.anndata._overloaded_uns, self)
        if self.is_view:
            def uns_to_dictview(uns_, anndata):
                return DictView(uns_, view_args=(anndata, "uns"))
            uns_overloaded = daskify_call(uns_to_dictview, uns_overloaded, self)
        return uns_overloaded


    def copy(self, filename: Optional[PathLike] = None) -> "AnnData":
        """Full copy, optionally on disk."""
        from anndata._io.dask.utils import daskify_method_call
        if not self.isbacked:
            # TODO: We should _only_ have views in the dask=True case b/c otherwise backed = True.
            # See if this logic can simplify.
            # Possibly we
            if self.is_view:
                # TODO: How do I unambiguously check if this is a copy?
                # Subsetting this way means we don’t have to have a view type
                # defined for the matrix, which is needed for some of the
                # current distributed backend.
                X = _subset(self._adata_ref.X, (self._oidx, self._vidx)).copy()
            else:
                X = self.X.copy()
            # TODO: Figure out what case this is:
            if X is not None:
                dtype = X.dtype
                if (not (X.shape is self.shape)) and (is_dask(X.shape) or is_dask(self.shape)):
                    def x_reshape(X, Xshape, anndata_shape):
                        # This is a little ugly b/c we need the X.shape to compute
                        # before we work on the X that has it.
                        X.shape = Xshape
                        X.reshape(anndata_shape)
                    from anndata._io.dask.utils import daskify_call
                    X = daskify_call(x_reshape, X, X.shape, self.shape)
                elif X.shape != self.shape:
                    X = X.reshape(self.shape)
            else:
                dtype = "float32"
            return AnnData(
                X=X,
                obs=self.obs.copy(),
                var=self.var.copy(),
                # deepcopy on DictView does not work and is unnecessary
                # as uns was copied already before
                uns=self._uns.copy()
                if isinstance(self.uns, DictView)
                else deepcopy(self._uns),
                obsm=self.obsm.copy(),
                varm=self.varm.copy(),
                obsp=self.obsp.copy(),
                varp=self.varp.copy(),
                raw=self.raw.copy() if self.raw is not None else None,
                layers=self.layers.copy(),
                dtype=dtype,
            )
        else:
            # A normal backed/dask object
            from .._io import read_h5ad

            if filename is None:
                raise ValueError(
                    "To copy an AnnData object in backed mode, "
                    "pass a filename: `.copy(filename='myfilename.h5ad')`."
                )
            mode = self.file._filemode
            self.write(filename)
            return read_h5ad(filename, backed=mode, dask=True)

from anndata._core.anndata import _gen_dataframe

@_gen_dataframe.register(dd.DataFrame)
def _(anno, length, index_names):
    anno = anno.copy()
    if not is_string_dtype(anno.index):
        warnings.warn("Transforming to str index.", ImplicitModificationWarning)
        anno.index = anno.index.astype(str)
    return anno


def is_dask(obj) -> bool:
    return isinstance(obj, dask.base.DaskMethodsMixin)
