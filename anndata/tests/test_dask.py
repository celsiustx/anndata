
from numpy import array
import pytest

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


@pytest.mark.skip(reason="calls methods on Pos that do not exist yet")
def test_pos():
    arr = array([ int(str(i)*2) for i in range(100) ])
    pos = Chunk.from_arr(arr, ((0, 100),))
    coords = pos.ranges
    assert coords == (Range(0, 0, 100, 100, 1),)

    R = 10
    C = 10
    arr = array([ [ R*r+c for c in range(C) ] for r in range(R) ])
    pos = Chunk.from_arr(arr, ((2, 5), (3, 7)))
    coords = pos.ranges
    assert coords == (Range(0, 2, 5, R, R), Range(1, 3, 7, C, 1))
