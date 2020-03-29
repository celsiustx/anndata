from functools import singledispatch


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


import numpy as np
@singledispatch
def eq(l, r):
    '''Custom equality operator'''
    assert type(l) == type(r), f'Mismatched types: {type(l)} vs. {type(r)}'
    result = (l == r)
    if not isinstance(result, bool) and not isinstance(result, np.bool_):
        raise NotImplementedError(f'l: {l} ({type(l)}), r {r} ({type(r)}), result: {result} ({type(result)})')
    assert l == r


from numpy import float32
@eq.register(float32)
def _(l, r): assert l == r


from scipy.sparse import spmatrix
@eq.register(spmatrix)
def _(l, r): assert (l != r).nnz == 0


from pandas import DataFrame as DF
from pandas.testing import assert_frame_equal
@eq.register(DF)
def _(l, r): assert_frame_equal(l, r)


from pandas import Series
from pandas.testing import assert_series_equal
@eq.register(Series)
def _(l, r): assert_series_equal(l, r)


from anndata import AnnData
@eq.register(AnnData)
def _(l, r):
    for k in ['X','obs','var',]:
        eq(
            getattr(l, k),
            getattr(r, k)
        )


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
