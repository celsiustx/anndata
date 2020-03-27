#
# This module was added as part of adding dask support.
# Possibly consider upstreaming to dask vs. anndata?
#

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Union, Collection

from dask.array import from_array
from dask.array.core import normalize_chunks
from dask.dataframe import from_delayed
from dask.delayed import delayed
from h5py import Dataset, File, Group
from numpy import array, dtype, cumprod, cumsum, empty, result_type, ix_
from pandas import Categorical, DataFrame as DF
from scipy.sparse import spmatrix

from anndata._io.h5ad import SparseDataset


@dataclass
class Range:
    '''Info about a range along a specific axis in a tensor

    A `Range` has some awareness of other dimension `Range`s stored alongside it in a
    `Chunk` in that it stores a `stride` (the product of the size of all `Range`'s "to
    the right" of this one; useful when mapping between 1-D and N-D representations of a tensor)
    '''
    idx: int
    start: int
    end: int
    max: int
    stride: int

    @property
    def size(self): return self.end - self.start


@dataclass
class Chunk:
    '''N-Dimensional slice of a tensor in a rectilinear grid of similar "Chunks"

    Comprised of one `Range` for each dimension'''
    ranges: Tuple[Range, ...]

    @property
    def idx(self):
        '''"Linearized" index of the "start" corner of this `Chunk`'''
        return sum([range.stride * range.start for range in self.ranges])

    @staticmethod
    def whole_array(arr):
        '''Build a `Chunk` encompassing all elements in the input array'''
        shape = arr.shape
        rank = len(shape)
        return Chunk.build(
            (0,) * rank,
            [ (0, max) for max in shape ],
            shape,
        )

    @staticmethod
    def from_block_info(block_info):
        '''Build a `Chunk` from a Dask "block info"

        Dask passes a "block_info" parameter to the lambda passed to
        dask.array.Array.map_blocks; this converts that structure to a `Chunk`'''
        chunk_idxs = block_info['chunk-location']
        chunk_offsets = block_info['array-location']
        dim_maxs = block_info['shape']
        return Chunk.build(chunk_idxs, chunk_offsets, dim_maxs)

    @staticmethod
    def build(
        chunk_idxs: Collection[int],
        chunk_ranges: Collection[Tuple[int, int]],
        dim_maxs: Collection[int],
    ):
        '''Build a `Chunk` from some precursor values

        The three inputs should all be the same length (the "rank" of the containing
        tensor).

        :param chunk_idxs: coordinates of this `Chunk` in the tensor of `Chunk`s that
        comprise the tensor of which this `Chunk` is a member
        :param chunk_ranges: [start,end) tuples for each dimension, giving the ranges
        along each axis that this `Chunk`` spans.
        :param dim_maxs: the maximum size of each dimension in the containing tensor
        (independent of specific `Chunk`s' positions); used to compute "strides"
        '''
        assert len(chunk_idxs) == len(chunk_ranges)
        assert len(chunk_idxs) == len(dim_maxs)

        strides = list(
            reversed(
                cumprod(
                    [1] + \
                    list(
                        reversed([
                            max
                            for max
                            in dim_maxs[1:]
                        ])
                    )
                )
            )
        )
        return Chunk(tuple([
            Range(chunk_idx, start, end, max, stride)
            for chunk_idx, (start, end), max, stride
            in zip(
                chunk_idxs,
                chunk_ranges,
                dim_maxs,
                strides,
            )
        ]))


@dataclass
class H5Chunk:
    '''Lazy pointer to a chunk of an HDF5 dataset

    Opens the file and extracts the specified range'''
    file: Path
    path: str
    dtype: dtype
    ndim: int
    pos: Chunk
    to_array: Callable[[Union[Dataset,Group]], spmatrix] = None

    @property
    def shape(self):
        return tuple( range.size for range in self.ranges )

    @property
    def start(self):
        return tuple( range.start for range in self.ranges )

    @property
    def idx(self):
        return tuple( range.idx for range in self.ranges )

    @property
    def slice(self):
        return tuple( slice(range.start, range.end) for range in self.ranges )

    @property
    def ranges(self): return self.pos.ranges

    def arr(self):
        print(f'Opening {self.file} ({self.path}): {self.idx} ({self.slice})')
        with File(self.file, 'r') as f:
            arr = f[self.path]
            attrs = arr.attrs

            # Verify to_array exists if the HDF5 entry is a Group (and thus requires converting
            if isinstance(arr, Group):
                if not self.to_array:
                    raise Exception(f'Missing Group->spmatrix converter for {self.file}:{self.path}')

            if self.to_array:
                arr = self.to_array(arr)

            chunk = arr[self.slice]
            if 'sparse_format' in attrs:
                sparse_format = attrs['sparse_format']
                from anndata._core.sparse_dataset import get_memory_class
                mtx_class = get_memory_class(sparse_format)
                print(f'Converting chunk to sparse format: {sparse_format}')
                chunk = mtx_class(chunk)

            return chunk


def get_slice(path, name, start, end, index_col=None):
    '''Load rows [start,end) from HDF5 file `path` (group `name`) into a DataFrame'''
    with File(path, 'r') as f:
        obj = f[name]
        if isinstance(obj, Group):
            group = obj
            attrs = group.attrs
            assert 'column-order' in attrs
            columns = list(attrs['column-order'])
            def get_series(k):
                v = group[k]
                attrs = v.attrs
                if 'categories' in attrs.keys():
                    categories_ref = attrs['categories']
                    categories = group[categories_ref]
                    return Categorical.from_codes(v[start:end], categories)
                else:
                    return v[start:end]

            df = DF({ k: get_series(k) for k in columns })
        else:
            dataset = obj
            df = DF(dataset[start:end])

        if index_col is not None:
            if isinstance(index_col, str):
                df = df.set_index(index_col)
            elif callable(index_col):
                df = df.set_index(index_col(df))
            else:
                raise ValueError('Invalid index_col value: %s' % index_col)

        return df


def load_dataframe(*, dataset=None, group=None, path=None, name=None, chunk_size=2 ** 20, index_col=None,):
    obj = dataset or group
    if obj:
        ctx = nullcontext()
        path = obj.file.filename
        name = obj.name
    else:
        ctx = File(path, 'r')
        obj = ctx[name]
        if isinstance(obj, Group):
            group = obj
        else:
            dataset = obj

    with ctx:
        if group:
            cols = list(group.attrs['column-order'])
            #idx_key = group.attrs["_index"]  # TODO: use this / set index col correctly?
            itemsize = sum([ group[k].dtype.itemsize for k in cols ])
            [ (size,) ] = set([ group[k].shape for k in cols ])
        else:
            itemsize = dataset.dtype.itemsize
            (size,) = dataset.shape

        n_bytes = itemsize * size
        n_chunks = (n_bytes + chunk_size - 1) // chunk_size
        chunk_starts = [ (i * size // n_chunks) for i in range(n_chunks) ]
        chunk_slices = list(zip(chunk_starts, chunk_starts[1:] + [size]))

    chunks = [
        delayed(get_slice)(path, name, start, end, index_col)
        for start, end in chunk_slices
    ]

    ddf = from_delayed(chunks)
    return ddf


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = result_type(*arrays)
    arr = empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def sparse_hdf5_group_to_backed_dataset(obj):
    if isinstance(obj, Group):
        return SparseDataset(obj).to_backed()
    else:
        return obj


def make_chunk(ranges, block_info, shape, record_dtype, range_dtype, ndim, path, _name, to_array):
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
            to_array=to_array
        )
    ) \
    .reshape((1,)*rank)


def to_arr(c, rank): return c[(0,) * rank].arr()


def load_tensor(*, X=None, path=None, name=None, chunk_size = 'auto', to_array=sparse_hdf5_group_to_backed_dataset):
    if X:
        ctx = nullcontext()
        path = X.file.filename
        name = X.name
    else:
        ctx = File(path, 'r')
        X = ctx[name]

    print(f'Loading HDF5 tensor: {path}:{name}: {X}')

    with ctx:
        chunks = normalize_chunks(chunk_size, X.shape, dtype = X.dtype)
        rank = len(chunks)
        chunk_ends = [ cumsum(chunk) for chunk in chunks ]
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

        cp = cartesian_product(*chunk_ranges)

        full_dtype = [
            field
            for i in range(rank)
            for field in [
                (f'start_{i}','<i8'),
                (f'end_{i}','<i8')
            ]
        ]
        cp.dtype = full_dtype
        cp = cp.reshape([len(c) for c in chunks])

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
            _name='X',
            to_array=to_array,
        )

        arr = h5chunks.map_blocks(to_arr, chunks=chunks, dtype=X.dtype, rank=rank)

        return arr
