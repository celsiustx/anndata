from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from math import floor, sqrt
from numpy import array
from pandas import DataFrame as DF
from scipy import sparse

from anndata import AnnData, read_h5ad


@dataclass
class Obj:
    dict: Any
    default: Any = None

    def __getattr__(self, item):
        if item in self.dict:
            return self.dict[item]

        if self.default:
            return self.default[item]

        return self.dict[item]


def make_test_h5ad(R=100, C=200):
    def digits(n, b, empty_zero=False, significant_leading_zeros=True):
        '''Convert a number to an array of base-`b` digits, with a few toggle-able
        behaviors.

        The default parameters output an analogue to "spreadsheet-column" order, e.g. in
        base-26 you get the equivalent of: "A","B",…,"Z","AA","AB",… (but with arrays of
        integers ∈[0,26) instead of strings of letters).

        :param empty_zero: when True, start counting from an empty array at zero
                (otherwise, zero will be mapped to [0])
        :param significant_leading_zeros: when True, enumerate non-empty natural-number
                arrays containing the elements ∈[0,b)
        '''
        if significant_leading_zeros:
            if not empty_zero: return digits(n+1, b, empty_zero=True, significant_leading_zeros=significant_leading_zeros)
            bases = [1]
            while n >= bases[-1]:
                n -= bases[-1]
                bases.append(bases[-1]*b)
            n_digits = digits(n, b, empty_zero=True, significant_leading_zeros=False)
            return [0]*(len(bases)-1-len(n_digits)) + n_digits
        else:
            return digits(n // b, b, empty_zero=True, significant_leading_zeros=False) + [n%b] if n else [] if empty_zero else [0]

    def spreadsheet_column(idx):
        return ''.join([ chr(ord('A')+digit) for digit in digits(idx, 26) ])

    X = sparse.random(R, C, format="csc", density=0.1, random_state=123)

    obs = DF([
        {
            'label': f'row {r}',
            'idx²': r**2,
            'Prime': all([
                r % f != 0
                for f in range(2, floor(sqrt(r)))
            ]),
        }
        for r in range(R)
    ])

    var = DF([
        {
            'name': spreadsheet_column(c),
            'sqrt(idx)': sqrt(c),
        }
        for c in range(C)
    ])

    ad = AnnData(X=X, obs=obs, var=var)
    return ad


new_path = Path.cwd() / 'new.h5ad'
old_path = Path.cwd() / 'old.h5ad'  # written by running `make_test_h5ad` in AnnData 0.6.22


@pytest.mark.parametrize('dask', [True, False])
def test_cmp_new_old_h5ad(dask):
    R = 100
    C = 200

    ad = make_test_h5ad()
    def write(path, overwrite=False):
        if path.exists() and overwrite:
            path.unlink()
        if not path.exists():
            ad.write_h5ad(path)

    overwrite = False
    write(new_path, overwrite=overwrite)

    def compute(o): return o.compute() if dask else o

    def load_ad(path):
        ad = read_h5ad(path, backed='r', dask=dask)
        X = compute(ad.X)
        coo = X.tocoo() if dask else X.value.tocoo()
        rows, cols = coo.nonzero()
        nnz = list(zip(list(rows), list(cols)))
        return Obj(dict(ad=ad, nnz=nnz, obs=compute(ad.obs), var=compute(ad.var)), default=ad)

    old = load_ad(old_path)
    new = load_ad(new_path)

    print(old.nnz[:20])
    print(new.nnz[:20])
    assert old.nnz == new.nnz

    from pandas.testing import assert_frame_equal

    # old AnnData's set DFs' index.name to "index", new style leaves it None
    old.obs.index.name = None
    old.var.index.name = None

    print(old.obs.index)

    assert_frame_equal(old.obs, new.obs)
    assert_frame_equal(old.var, new.var)
    # with TemporaryDirectory() as dir:
    #     path = Path(dir) / 'tmp.h5ad'
    #     ad.write_h5ad(path)


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


@pytest.mark.parametrize('path', [old_path, new_path])
def test_dask_load(path):
    ad1 = read_h5ad(path, backed='r', dask=False)
    ad2 = read_h5ad(path, backed='r', dask= True)

    eq(ad1.X, ad2.X)
    eq(ad1.obs, ad2.obs)
    eq(ad1.var, ad2.var)

    # TODO: obsm, varm, obsp, varp, uns, layers, raw


from anndata._io.h5chunk import Chunk, Range


def test_pos():
    arr = array([ int(str(i)*2) for i in range(100) ])
    chunk = Chunk.whole_array(arr)
    ranges = chunk.ranges
    assert ranges == (Range(0, 0, 100, 100, 1),)

    R = 20
    C = 20
    arr = array([
        [ R*r+c for c in range(C) ]
        for r in range(R)
    ])
    chunk = Chunk.build((2, 3), ((4, 6), (12, 16)), (R, C))
    assert chunk == Chunk((
        Range(2,  4,  6, R, C),
        Range(3, 12, 16, C, 1),
    ))
