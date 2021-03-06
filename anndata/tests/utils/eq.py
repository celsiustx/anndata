from dask.base import DaskMethodsMixin
from functools import singledispatch
import pandas as pd


@singledispatch
def normalize(o):
    '''Optionally convert an object to another, more canonical type (for
    equality-assertions)'''
    return o

from dask.array import Array
@normalize.register(Array)
def _(arr): return arr.compute()


from dask.dataframe import DataFrame as DDF
@normalize.register(DDF)
def _(ddf): return ddf.compute()


from dask import dataframe as dd
@normalize.register(dd.Series)
def _(dds): return dds.compute()


from anndata._io.h5ad import SparseDataset
@normalize.register(SparseDataset)
def _(sds): return sds.value


def base_eq(l, r):
    '''Custom equality operator'''
    assert type(l) == type(r), f'Mismatched types: {type(l)} vs. {type(r)}'
    result = (l == r)
    if not isinstance(result, bool) and not isinstance(result, np.bool_):
        raise NotImplementedError(f'l: {l} ({type(l)}), r {r} ({type(r)}), result: {result} ({type(result)})')
    assert result, result


import numpy as np
@singledispatch
def eq(l, r): base_eq(l, r)


from numpy import ndarray
from numpy.testing.utils import assert_array_equal
@eq.register(ndarray)
def _(l, r): assert_array_equal(l, r)


# from numpy import bool_
# @eq.register(bool_)
# def _(l, r): base_eq(l, r)


# from numpy import float32
# @eq.register(float32)
# def _(l, r): assert l == r

from scipy.sparse import spmatrix
@eq.register(spmatrix)
def _(l, r):
    if l.shape != r.shape:
        raise ValueError("Differing shapes: %s => %s" % (l.shape, r.shape))
    diff = (l != r)
    if hasattr(diff, "nnz"):
        assert(diff.nnz == 0)
    elif hasattr(l, "A") and hasattr(r, "A"):
        return False not in pd.Series((l.A != r.A).flatten()).value_counts()
    else:
        raise ValueError("Can't compare %s and %s!" % (l, r))


from pandas import DataFrame as DF
from pandas.testing import assert_frame_equal
@eq.register(DF)
def _(l, r):
    if isinstance(r, DaskMethodsMixin):
        r = r.compute()
    if l.index.names == ["_index"] and r.index.names == [None]:
        r.index.names = ["_index"]
    from dask.dataframe.utils import assert_eq
    assert_eq(l, r)
    # try:
    #     assert_frame_equal(l, r)
    # except AssertionError:
    #     raise AssertionError(f'{l} != {r}')


from pandas import Series
from pandas.testing import assert_series_equal
@eq.register(Series)
def _(l, r):
    if isinstance(r, DaskMethodsMixin):
        r = r.compute()
    assert_series_equal(l, r)


from anndata import AnnData
from anndata_dask import AnnDataDask
@eq.register(AnnData)
@eq.register(AnnDataDask)
def _(l: AnnData, r: AnnData):
    for k in ['X','obs','var',]:
        lv = getattr(l, k)
        rv = getattr(r, k)
        if isinstance(lv, DaskMethodsMixin):
            lv = lv.compute()
        if isinstance(rv, DaskMethodsMixin):
            rv = rv.compute()
        eq(lv, rv)
    # lc = l.compute()
    # from anndata.diff import diff_summary
    # differences = diff_summary(lc, r)
    # if differences != {}:
    #     raise Exception(f"Differences found!: {differences}")


def cmp(l, r):
    eq(
        normalize(l),
        normalize(r),
    )


def try_eq(l, r):
    try:
        eq(l, r)
    except AssertionError as e:
        return e
