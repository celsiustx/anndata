from tempfile import NamedTemporaryFile

from h5py import File
from numpy import array, random
from numpy.testing import assert_equal
from scipy import sparse

from anndata._io.dask.hdf5.h5chunk import Chunk, Range
from anndata._io.dask.hdf5.load_array import load_dask_array


def test_chunk():
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


def test_dask_array_hdf5_load():
    M, N = 20, 100
    arr = random.random((M,N))

    m, n =  5,  10
    chunk_size = (m, n)
    with NamedTemporaryFile() as tmp:
        path = tmp.name
        key = 'arr'
        with File(path, 'w') as f:
            f[key] = arr

        da = load_dask_array(path=path, key=key, chunk_size=chunk_size)
        assert_equal(da.compute(), arr)

        chunk_ranges = [ len(dim) for dim in da.chunks ]
        assert chunk_ranges == [ M/m, N/n ]


# Dask `compute()`s that pool multiple sparse-matrix chunks require this, otherwise Dask
# will attempt to combine computed chunks' values into one ndarray, and raise a
# ValueError ("setting an array element with a sequence", in dask.array.core.concatenate3)
from dask.array.backends import register_scipy_sparse
register_scipy_sparse()


def test_dask_array_hdf5_load_sparse():
    M, N = 200, 1000
    density = 0.01
    format = 'csr'
    arr = sparse.random(M, N, density=density, format=format)

    m, n =  50,  100
    chunk_size = (m, n)

    with NamedTemporaryFile() as tmp:
        path = tmp.name
        key = 'arr'
        with File(path, 'w') as f:
            for k in [ 'data', 'indices', 'indptr', ]:
                f[f'{key}/{k}'] = getattr(arr, k)

            f[key].attrs['encoding-type'] = f'{format}_matrix'
            f[key].attrs['shape'] = (M, N)

        da = load_dask_array(path=path, key=key, chunk_size=chunk_size)
        result = da.compute(scheduler='single-threaded')
        assert (result != arr).nnz == 0

        chunk_ranges = [ len(dim) for dim in da.chunks ]
        assert chunk_ranges == [ M/m, N/n ]
