
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
