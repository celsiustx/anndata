#
# This module was added as part of adding dask support.
# Possibly consider upstreaming to dask vs. anndata?
#

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext

from dask.array import Array, from_array
from dask.array.core import normalize_chunks
from functools import partial, singledispatch
from h5py import File, Group
from numpy import array, cumprod, cumsum, empty, result_type, ix_
from typing import Collection, Tuple

from anndata._io.h5ad import SparseDataset

from .h5chunk import Chunk, H5Chunk, Range


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = result_type(*arrays)
    arr = empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


@singledispatch
def sparse_hdf5_group_to_backed_dataset(obj, **kwags):
    '''"to_array" adapter for `H5Chunk.arr()`: pass-through by default, but convert `h5py.Group`s to scipy.spmatrix's'''
    return obj


@sparse_hdf5_group_to_backed_dataset.register(Group)
def _(group, **kwargs): return SparseDataset(group, **kwargs).to_backed()


def make_chunk(ranges: Collection[Tuple[int,int]], block_info, shape, record_dtype, range_dtype, ndim, path, _name, to_array, to_array_kwargs):
    '''Given a `path` to an HDF5 file, and a list of [start,end) index-pairs (one per dimension), build an `H5Chunk`'''
    block_info = block_info[0]
    block_idxs = block_info['array-location']
    rank = len(block_idxs)
    arr = array(ranges)
    arr.dtype = range_dtype
    arr = arr.reshape((rank,))
    strides = list(reversed(cumprod([1] + list(reversed(shape[1:])))))
    pos = Chunk(tuple([
        Range(
            block_idx,
            start, end,
            shape[idx],
            stride,
        )
        for idx, (block_idx, (start, end), stride)
        in enumerate(zip(block_idxs, arr, strides))
    ]))
    return array(
        H5Chunk(
            path,
            _name,
            record_dtype,
            ndim,
            pos,
            to_array=partial(to_array, **to_array_kwargs)
        )
    ) \
    .reshape((1,)*rank)


def to_arr(chunk, rank):
    '''Expand a single-element `chunk` (containing a single `H5Chunk`) into its underlying elements'''
    return chunk[(0,) * rank].arr()


def load_dask_array(
    *,
    X=None,
    path=None, key=None,
    chunk_size ='auto',
    to_array=sparse_hdf5_group_to_backed_dataset,
    **to_array_kwargs
) -> Array:
    '''Load an HDF5 Dataset (or Group representing a sparse array) as a Dask Array.

    An existing HDF5 node `X` can be passed, xor a `path` (to an HDF5 file) and `key` (to load from within that file).

    First, a Dask Array of `H5Chunk` elements is created (one per Dask block), then those blocks are mapped over and
    expanded into their full underlying ndarrays (or scipy.spmatrix's).

    This function is useful and necessary because it roots the Dask graph of the returned Array in `H5Chunk`s, which
    are lazy/serializable pointers to the HDF5 ranges that will comprise each Dask block, allowing each block (which
    may execute on some arbitrary worker in a Dask cluster) to load its elements correctly (as opposed to naive
    approaches where an open `h5py.File` is bound into the task closure, which doesn't serialize).
    '''
    if X:
        ctx = nullcontext()
        path = X.file.filename
        key = X.name
    else:
        ctx = File(path, 'r')
        X = ctx[key]

    if isinstance(X, Group):
        X = SparseDataset(X, **to_array_kwargs)

    #print(f'Loading HDF5 tensor: {path}:{name}: {X}')

    with ctx:
        chunks = normalize_chunks(chunk_size, X.shape, dtype = X.dtype)
        rank = len(chunks)

        # For each dimension, convert chunk sizes into boundary-indices
        chunk_ends = [ cumsum(chunk) for chunk in chunks ]

        # For each dimension, convert chunk boundary-indices into [start,end) pairs
        range_dtype = [('start','<i8'),('end','<i8')]
        chunk_ranges = \
            [
                array(
                    list(
                        zip(
                            [0] + chunk_dim_ends[:-1].tolist(),
                            chunk_dim_ends,
                        )
                    ),
                    dtype=range_dtype
                )
                for chunk_dim_ends in chunk_ends
            ]

        # Ultimately, we want to build an array whose elements are `rank` [start,end)
        # index-pairs. Each element will precisely specify the range that a Dask Array
        # block should draw its elements from:
        full_dtype = [
            field
            for i in range(rank)
            for field in [
                (f'start_{i}','<i8'),
                (  f'end_{i}','<i8'),
            ]
        ]

        # The cartesian product of the `rank` lists of [start,end) index-pairs contains
        # one element per desired chunk, specifying that chunk's range along each axis:
        cp = cartesian_product(*chunk_ranges)
        cp.dtype = full_dtype
        cp = cp.reshape([len(c) for c in chunks])

        # Make a Dask Array with one element per block (representing the range of
        # elements that block should ultimately draw its elements from, in the form of
        # `rank` [start,end) index-pairs):
        da = from_array(cp, chunks=(1,)*rank)

        h5chunks = da.map_blocks(
            make_chunk,
            chunks=(1,)*rank,
            dtype=H5Chunk,
            shape=X.shape,
            record_dtype=X.dtype,
            range_dtype=range_dtype,
            ndim=X.ndim,
            path=path,
            _name=key,
            to_array=to_array,
            to_array_kwargs=to_array_kwargs,
        )

        # Expand each block (which each contain a single `H5Chunk`) into the corresponding array:
        arr = h5chunks.map_blocks(to_arr, chunks=chunks, dtype=X.dtype, rank=rank)

        # Intermittent SIGSEGV here.
        return arr
