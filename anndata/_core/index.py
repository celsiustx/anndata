import collections.abc as cabc
from dask import base as dask_base
import dask.dataframe as dd
import dask.array as da
from functools import singledispatch
from itertools import repeat
from logging import getLogger
from typing import Union, Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse

from dask.dataframe.core import map_partitions

logger = getLogger(__file__)

Index1D = Union[slice, int, str, np.int64, np.ndarray]
Index = Union[Index1D, Tuple[Index1D, Index1D], spmatrix]


def _normalize_indices(
    index: Optional[Index], names0: pd.Index, names1: pd.Index
) -> Tuple[slice, slice]:
    # deal with tuples of length 1
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    # deal with pd.Series
    if isinstance(index, pd.Series):
        index: Index = index.values
    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError("AnnData can only be sliced in rows and columns.")
        # deal with pd.Series
        # TODO: The series should probably be aligned first
        # It seems this logic could be inside _normalize_index? -ssmith
        if isinstance(index[1], pd.Series):
            index = index[0], index[1].values
        if isinstance(index[0], pd.Series):
            index = index[0].values, index[1]
    # NOTE: This might be called unpack_indexer, since the axN is the indexer, and namesN is the index.
    ax0, ax1 = unpack_index(index)
    ax0n = _normalize_index(ax0, names0)
    ax1n = _normalize_index(ax1, names1)
    return ax0n, ax1n


def _normalize_index(
    indexer: Union[
        slice,
        np.integer,
        int,
        str,
        Sequence[Union[int, np.integer]],
        np.ndarray,
        pd.Index,
    ],
    index: pd.Index
) -> Union[slice, int, np.ndarray]:  # ndarray of int
    from anndata_dask import is_dask, daskify_call

    if not isinstance(index, pd.RangeIndex):
        assert (
            index.dtype != float and index.dtype != int
        ), "Don’t call _normalize_index with non-categorical/string names"

    # the following is insanely slow for sequences,
    # we replaced it using pandas below
    def name_idx(i):
        if isinstance(i, str):
            i = index.get_loc(i)
        return i

    if isinstance(indexer, pd.Series):
        if indexer.all():
            indexer = slice(0, len(indexer), 1)

    if isinstance(indexer, slice):
        start = name_idx(indexer.start)
        stop = name_idx(indexer.stop)
        # string slices can only be inclusive, so +1 in that case
        if isinstance(indexer.stop, str):
            stop = None if stop is None else stop + 1
        step = indexer.step
        return slice(start, stop, step)

    if isinstance(indexer, (np.integer, int)):
        return indexer
    elif isinstance(indexer, str):
        return index.get_loc(indexer)  # int
    elif isinstance(indexer, (pd.Series, Sequence, np.ndarray, pd.Index, spmatrix, np.matrix)):
        if hasattr(indexer, "shape") and (
            (indexer.shape == (index.shape[0], 1))
            or (indexer.shape == (1, index.shape[0]))
        ):
            if isinstance(indexer, spmatrix):
                indexer = indexer.toarray()
            indexer = np.ravel(indexer)
        if not isinstance(indexer, (np.ndarray, pd.Index)):
            indexer = np.array(indexer)
        if issubclass(indexer.dtype.type, (np.integer, np.floating)):
            return indexer  # Might not work for range indexes
        elif issubclass(indexer.dtype.type, np.bool_):
            if indexer.shape != index.shape:
                raise IndexError(
                    f"Boolean index does not match AnnData’s shape along this "
                    f"dimension. Boolean index has shape {indexer.shape} while "
                    f"AnnData index has shape {index.shape}."
                )
            positions = np.where(indexer)[0]
            return positions  # np.ndarray[int]
        else:  # indexer should be string array
            if not hasattr(index, "get_indexer"):
                pass
            positions = index.get_indexer(indexer)
            if np.any(positions < 0):
                not_found = indexer[positions < 0]
                raise KeyError(
                    f"Values {list(not_found)}, from {list(indexer)}, "
                    "are not valid obs/ var names or indices."
                )
            return positions  # np.ndarray[int]
    elif is_dask(indexer):

        return indexer
    elif is_dask(index):
        # NOTE: The index is the first arg b/c we are mapping, though _normalize_index
        # expects it to be the 2nd arg.
        def f(index, indexer):
            return _normalize_index(indexer, index.values)
        indexer.map_partitions(f, indexer, meta=index._meta)
    else:
        raise IndexError(f"Unknown indexer {indexer!r} of type {type(indexer)}")


def unpack_index(index: Index) -> Tuple[Index1D, Index1D]:
    if not isinstance(index, tuple):
        return index, slice(None)
    elif len(index) == 2:
        return index
    elif len(index) == 1:
        return index[0], slice(None)
    else:
        raise IndexError("invalid number of indices")


@singledispatch
def _subset(a: Union[np.ndarray, spmatrix, pd.DataFrame], subset_idx: Index):
    # Select as combination of indexes, not coordinates
    # Correcting for indexing behaviour of np.ndarray
    if all(isinstance(x, cabc.Iterable) for x in subset_idx):
        subset_idx = np.ix_(*subset_idx)
    return a[subset_idx]

@_subset.register(da.Array)
def _subset_dask_array(a: da.Array, idx: Index):
    from anndata_dask import daskify_call_return_array, daskify_calc_shape, daskify_call, is_dask
    new_shape = daskify_calc_shape(a.shape, idx)
    if any(is_dask(v) for v in new_shape):
        return daskify_call(_subset, a, idx)
    else:
        # This only works if the shape is fully computed.
        return daskify_call_return_array(_subset, a, idx,
                                         _dask_shape=new_shape,
                                         _dask_dtype=a.dtype,
                                         _dask_meta=a._meta)

@_subset.register(dask_base.DaskMethodsMixin)
def _subset_dask_general(a: dask_base.DaskMethodsMixin, subset_idx: Index):
    from anndata_dask import daskify_call
    return daskify_call(_subset, subset_idx)

@_subset.register(pd.DataFrame)
def _subset_df(df: pd.DataFrame, subset_idx: Index):
    return df.iloc[subset_idx]


def make_slice(idx, dimidx, n=2):
    mut = list(repeat(slice(None), n))
    mut[dimidx] = idx
    return tuple(mut)


def get_vector(adata, k, coldim, idxdim, layer=None):
    # adata could be self if Raw and AnnData shared a parent
    dims = ("obs", "var")
    col = getattr(adata, coldim).columns
    idx = getattr(adata, f"{idxdim}_names")

    in_col = k in col
    in_idx = k in idx

    if (in_col + in_idx) == 2:
        raise ValueError(
            f"Key {k} could be found in both .{idxdim}_names and .{coldim}.columns"
        )
    elif (in_col + in_idx) == 0:
        raise KeyError(
            f"Could not find key {k} in .{idxdim}_names or .{coldim}.columns."
        )
    elif in_col:
        return getattr(adata, coldim)[k].values
    elif in_idx:
        selected_dim = dims.index(idxdim)
        idx = adata._normalize_indices(make_slice(k, selected_dim))
        a = adata._get_X(layer=layer)[idx]
    if issparse(a):
        a = a.toarray()
    return np.ravel(a)
