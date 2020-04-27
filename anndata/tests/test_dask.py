from functools import singledispatch
from pathlib import Path

import pytest

import anndata
from anndata import read_h5ad
from .utils.data import make_test_h5ad
from .utils.eq import cmp as eq
from .utils.obj import Obj

package_root = Path(anndata.__file__).parent.parent
new_path = package_root / 'new.h5ad'
old_path = package_root / 'old.h5ad'  # written by running `make_test_h5ad` in AnnData 0.6.22
assert(new_path.exists())
assert(old_path.exists())

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
    ad1 = read_h5ad(path, dask=False)
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

    # Basic support.
    check((
        lambda ad: ad.X * 2,

        lambda ad: ad.obs.Prime,
        lambda ad: ad.obs['Prime'],
        lambda ad: ad.obs[['Prime','idx²']],

        lambda ad: ad.var.name,
        lambda ad: ad.var['name'],
        lambda ad: ad.var[['sqrt(idx)','name']],

        lambda ad: ad.X[:],
        lambda ad: ad.X[:10],
        lambda ad: ad.X[:20,:20],
    ))

    # These work when we add deferred() around all .iloc calls and things that use them.
    check((
        lambda ad: ad[:10],
        lambda ad: ad[:10, :],
        lambda ad: ad[:10, :10],
        lambda ad: ad[:10, 0],

        lambda ad: ad[:, :10],
        lambda ad: ad[:10, :10],
        lambda ad: ad[0, :10],

        lambda ad: ad[10, 10]
    ))

    # These are known or believed to not work in Dask today
    TODO = [
        # iloc'ing row(s)/col(s) mostly does not work out, of the box:

        lambda ad: ad.obs.loc['2','Prime'],

        # .loc'ing ranges
        lambda ad: ad.obs.loc[:,:],
        lambda ad: ad.obs.loc[:,['Prime','label']],
        lambda ad: ad.obs.loc[:,'label'],

        # Integer ranges don't work in .loc because obs.index holds strs; this is true in non-dask mode, but seems broken
        lambda ad: ad.obs.loc[:10,:],
        lambda ad: ad.obs.loc[:10,['Prime','label']],
        lambda ad: ad.obs.loc[:10,'label'],

        # these work, but Dask returns a DF (to Pandas' Series)
        lambda ad: ad.obs.loc['10',['Prime','label']],
        lambda ad: ad.obs.loc['10',:],

        # works, but Dask returns a Series (to Pandas' scalar)
        lambda ad: ad.obs.loc['10','label'],
    ]
