from anndata import AnnData
from math import floor, sqrt
from pandas import DataFrame as DF
from pathlib import Path
from numpy import array
import pytest
from scipy import sparse
from shutil import rmtree
from tempfile import TemporaryDirectory

from dataclasses import dataclass
from typing import Any


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


def test_load():
    R = 100
    C = 200

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
    new_path = Path.cwd() / 'new.h5ad'
    old_path = Path.cwd() / 'old.h5ad'
    def write(path, overwrite=False):
        if path.exists() and overwrite:
            path.unlink()
        if not path.exists():
            ad.write_h5ad(path)

    overwrite = False
    write(new_path, overwrite=overwrite)

    from anndata import read_h5ad

    def load_ad(path):
        ad = read_h5ad(path, backed='r', dask=True)
        X = ad.X.compute()
        coo = X.tocoo()
        rows, cols = coo.nonzero()
        nnz = list(zip(list(rows), list(cols)))
        return Obj(dict(ad=ad, nnz=nnz, obs=ad.obs, var=ad.var), default=ad)

    old = load_ad(old_path)
    new = load_ad(new_path)

    print(old.nnz[:20])
    print(new.nnz[:20])
    assert old.nnz == new.nnz

    from pandas.testing import assert_frame_equal
    assert_frame_equal(old.obs.compute(), new.obs.compute())
    assert_frame_equal(old.var.compute(), new.var.compute())

    # with TemporaryDirectory() as dir:
    #     path = Path(dir) / 'tmp.h5ad'
    #     ad.write_h5ad(path)



@pytest.mark.skip(reason="test data references a file in Ryan's home dir")
def test_dask():
    from anndata._io.register import register_numerics
    path = '/Users/ryan/c/celsius/notebooks/data/Fib.imputed.1k.legacy.h5ad'
    distributed = True
    if distributed:
        from dask.distributed import Client
        client = Client()
        print(f'client: {client}')

        from distributed import Worker, WorkerPlugin
        class MyPlugin(WorkerPlugin):
            def setup(self, worker: Worker):
                register_numerics()
                print(f'worker setup: {worker}')

        client.register_worker_plugin(MyPlugin())
    from anndata import read_h5ad
    ad = read_h5ad(path, backed='r', dask=True)
    print(ad.obs.head())
    from anndata._io.sql import write_sql
    db_url = 'postgres:///sc'
    write_sql(ad, 'test_dask_legacy', db_url, if_exists='replace', dask=True)


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
