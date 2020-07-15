import functools
from dataclasses import dataclass
from functools import singledispatch
import inspect
from pathlib import Path
from typing import Union

import pytest

import anndata
from anndata import read_h5ad
from .utils.data import make_test_h5ad
from .utils.eq import cmp as eq
from .utils.obj import Obj
from .._core.sparse_dataset import SparseDataset
from anndata._core.index import _normalize_index
from anndata_dask import is_dask
import pandas as pd

package_root = Path(anndata.__file__).parent.parent
new_path = package_root / 'new.h5ad'
old_path = package_root / 'old.h5ad'  # written by running `make_test_h5ad` in AnnData 0.6.22
assert (new_path.exists())
assert (old_path.exists())

import logging
logger = logging.getLogger(__file__)

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

    def compute(o):
        return o.compute() if dask else o

    def load_ad(path):
        ad = read_h5ad(path, backed='r', dask=dask)
        X = compute(ad.X)
        if isinstance(X, SparseDataset):
            X = X.value
        rows, cols = X.nonzero()
        nnz = list(zip(list(rows), list(cols)))
        return Obj(dict(ad=ad, nnz=nnz, obs=compute(ad.obs), var=compute(ad.var)),
                   default=ad)

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

    if old.obs.index.names == [None] and new.obs.index.names == ["_index"]:
        old.obs.index.names = ["_index"]

    if old.var.index.names == [None] and new.var.index.names == ["_index"]:
        old.var.index.names = ["_index"]

    assert_frame_equal(old.obs, new.obs)
    assert_frame_equal(old.var, new.var)
    # with TemporaryDirectory() as dir:
    #     path = Path(dir) / 'tmp.h5ad'
    #     ad.write_h5ad(path)


def filter_on_self_sum(ad):
    ad.obs["umi_counts"] = ad.X.sum(axis=1).A.flatten()
    adv = ad[ad.obs["umi_counts"] > 100.0]
    return adv

def group_sample(adata_list):
    if isinstance(adata_list[0], AnnDataDask):
        return anndata_dask.group_samples(adata_list)




@pytest.mark.parametrize('path', [old_path, new_path])
def test_dask_load(path):
    ad1 = read_h5ad(path, dask=False)
    ad2 = read_h5ad(path, backed='r', dask=True)

    @singledispatch
    def check(fn):
        import dask.base
        if callable(fn):
            with prevent_method_calls(dask.base.DaskMethodsMixin, "compute"):
                v_mem = fn(ad1)
                v_dask = fn(ad2)
            eq(v_mem, v_dask)
        else:
            raise NotImplementedError(f"Not callable!: {fn}")

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
        'X',
        'obs',
        'var',
        # TODO: obsm, varm, obsp, varp, uns, layers, raw
    ))

    # Basic support.
    check((
        lambda ad: ad.X * 2,

        lambda ad: ad.obs.Prime,
        lambda ad: ad.obs['Prime'],
        lambda ad: ad.obs[['Prime', 'idx²']],

        lambda ad: ad.var.name,
        lambda ad: ad.var['name'],
        lambda ad: ad.var[['sqrt(idx)', 'name']],

        lambda ad: ad.X[:],
        lambda ad: ad.X[:10],
        lambda ad: ad.X[:20, :20],
    ))

    # These possibly work with AnnDataDask modifications.
    # They go through AnnDataDask.__get_item__(ix)
    check((
        lambda ad: ad[:10],
        lambda ad: ad[:10, :10],

        lambda ad: ad[0, :10],
        lambda ad: ad[:10, 0],

        lambda ad: ad[:, :10],
        lambda ad: ad[:10, :],

        lambda ad: ad[10, 10],
    ))

    TODO1 = (
        filter_on_self_sum,
    )
    check(TODO1)

    # For these to work, we need to update how dask dataframes work,
    # or use a custom dataframe subclass with more features.
    # Either case will possibly use normalization like the AnnDataDask.__get_item__(),
    # since that method does successfully create indexes that will slice an obs or var.
    TODO2 = [
        # iloc'ing row(s)/col(s) mostly does not work out, of the box:
        lambda ad: ad.obs.loc['2', 'Prime'],

        # .loc'ing ranges
        lambda ad: ad.obs.loc[:, :],
        lambda ad: ad.obs.loc[:, ['Prime', 'label']],
        lambda ad: ad.obs.loc[:, 'label'],

        # Integer ranges don't work in .loc because obs.index holds strs; this is true in non-dask mode, but seems broken
        lambda ad: ad.obs.loc[:10, :],
        lambda ad: ad.obs.loc[:10, ['Prime', 'label']],
        lambda ad: ad.obs.loc[:10, 'label'],

        # these work, but Dask returns a DF (to Pandas' Series)
        lambda ad: ad.obs.loc['10', ['Prime', 'label']],
        lambda ad: ad.obs.loc['10', :],

        # works, but Dask returns a Series (to Pandas' scalar)
        lambda ad: ad.obs.loc['10', 'label'],
    ]


class PreventedMethodCallException(Exception):
    def __init__(self, cls: type, method_name: str,
                 args: Union[list, tuple], kwargs: dict, orig: callable):
        self.cls = cls
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.orig = orig


def prevent_method_calls(cls, method_name):
    from unittest.mock import patch
    orig = getattr(cls, method_name)
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        raise PreventedMethodCallException(cls=cls, method_name=method_name,
                                           args=args, kwargs=kwargs, orig=orig)
    ctx = patch.object(cls, method_name, wrapper)
    return ctx


@pytest.mark.parametrize('path', [old_path, new_path])
def test_load_without_compute(path):
    import dask.base

    with prevent_method_calls(dask.base.DaskMethodsMixin, 'compute') as c:
        # This should not call compute() on anything...
        ad = read_h5ad(path, backed='r', dask=True)

        # This should.
        with pytest.raises(PreventedMethodCallException):
            ad.obs.compute()


def verify_result_index_indexer(result, index, indexer):
    index_computed = index.compute() if is_dask(index) else index
    indexer_computed = indexer.compute() if is_dask(indexer) else indexer
    result_computed_expected = _normalize_index(indexer_computed, index_computed)
    result_computed = result.compute()
    mismatch = (result_computed != result_computed_expected)
    if hasattr(mismatch, "nnz"):
        assert (mismatch.nnz == 0)
    elif hasattr(mismatch, "value_counts"):
        vc = mismatch.value_counts()
        cnt = vc.get(True, 0)
        assert (cnt == 0)
    elif isinstance(mismatch, bool):
        assert (mismatch == False)
    else:
        try:
            count_true = pd.Series(mismatch).value_counts().get(True, 0)
            assert (count_true == 0)
        except Exception as e:
            assert (mismatch)
