from multipledispatch import dispatch


from scipy.sparse import spmatrix
@dispatch(spmatrix, spmatrix)
def eq(l, r): assert (l != r).nnz == 0


from anndata._io.h5ad import SparseDataset
from dask.array import Array
@dispatch(SparseDataset, Array)
def eq(l, r): eq(l.value, r.compute())


from pandas import DataFrame as DF
from dask.dataframe import DataFrame as DDF
from pandas.testing import assert_frame_equal
@dispatch(DF, DDF)
def eq(l, r): assert_frame_equal(l, r.compute())


@dispatch(spmatrix, Array)
def eq(l, r): eq(l, r.compute())


from pandas import Series
from pandas.testing import assert_series_equal
@dispatch(Series, Series)
def eq(l, r): assert_series_equal(l, r)


from dask import dataframe as dd
from pandas.testing import assert_series_equal
@dispatch(Series, dd.Series)
def eq(l, r): eq(l, r.compute())
