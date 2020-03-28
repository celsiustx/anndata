from functools import singledispatch
from pathlib import Path

import pytest

from anndata import read_h5ad
from .utils.data import make_test_h5ad
from .utils.eq import eq
from .utils.obj import Obj

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


@pytest.mark.parametrize('path', [old_path, new_path])
def test_dask_load(path):
    ad1 = read_h5ad(path, backed='r', dask=False)
    ad2 = read_h5ad(path, backed='r', dask= True)

    @singledispatch
    def check(fn):
        if callable(fn):
            eq(fn(ad1), fn(ad2))
        else:
            raise NotImplementedError

    @check.register(tuple)
    def _(args):
        for arg in args:
            check(arg)

    @check.register(str)
    def _(k):
        eq(
            getattr(ad1, k),
            getattr(ad2, k)
        )

    check((
        'X','obs','var',
        # TODO: obsm, varm, obsp, varp, uns, layers, raw
    ))
