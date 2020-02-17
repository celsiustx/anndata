from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Union

from dask.array import from_array
from dask.array.core import normalize_chunks
from dask.dataframe import from_delayed
from dask.delayed import delayed
from h5py import Dataset, File, Group
from numpy import array, dtype, cumprod
from numpy import cumsum, empty, result_type, ix_
from pandas import Categorical, DataFrame as DF
from scipy.sparse import spmatrix

from anndata._io.h5ad import SparseDataset


@dataclass
class Coord:
    idx: int
    start: int
    end: int
    max: int
    stride: int

    @property
    def shape(self): return self.end - self.start


@dataclass
class Pos:
    coords: Tuple[Coord, ...]

    @property
    def idx(self):
        return sum([ coord.stride * coord.start for coord in self.coords ])

    @staticmethod
    def from_offset_shapes(offset_maxs: List[Tuple[Tuple[int, int], int]]):
        strides = list(
            reversed(
                cumprod(
                    [1] + \
                    list(
                        reversed([
                            max
                            for _, max
                            in offset_maxs[1:]
                        ])
                    )
                )
            )
        )

        return Pos(
            tuple([
                Coord(idx, start, end, max, stride)
                for (idx, ((start, end), max)), stride
                in zip(enumerate(offset_maxs), strides)
            ])
        )

    @staticmethod
    def from_offsets_shapes(offsets: Tuple[Tuple[int,int], ...], shape: Tuple[int, ...]):
        assert len(offsets) == len(shape)
        zipped = list(zip(offsets, shape))
        return Pos.from_offset_shapes(zipped)

    @staticmethod
    def from_arr(arr, offsets: Tuple[Tuple[int,int], ...]):
        return Pos.from_offsets_shapes(offsets, arr.shape)


@dataclass
class H5Chunk:
    '''Lazy pointer to a chunk of an HDF5 dataset

    Opens the file and extracts the specified range'''
    file: Path
    path: str
    dtype: dtype
    ndim: int
    pos: Pos
    to_array: Callable[[Union[Dataset,Group]], spmatrix] = None

    # def __init__(self):
    #     assert self.ndim == len(self.coords)

    @property
    def shape(self):
        return tuple( coord.shape for coord in self.coords )

    @property
    def start(self):
        return tuple( coord.start for coord in self.coords )

    @property
    def idx(self):
        return tuple( coord.idx for coord in self.coords )

    @property
    def slice(self):
        return tuple( slice(coord.start, coord.end) for coord in self.coords )

    @property
    def coords(self): return self.pos.coords

    def arr(self):
        print(f'Opening {self.file} ({self.path}): {self.idx} ({self.slice})')
        with File(self.file, 'r') as f:
            arr = f[self.path]

            # Verify to_array exists if the HDF5 entry is a Group (and thus requires converting
            if isinstance(arr, Group):
                if not self.to_array:
                    raise Exception(f'Missing Group->spmatrix converter for {self.file}:{self.path}')

            if self.to_array:
                arr = self.to_array(arr)

            chunk = arr[self.slice]

            return chunk


def get_slice(path, name, start, end):
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

            return DF({ k: get_series(k) for k in columns })
        else:
            dataset = obj
            return DF(dataset[start:end])


def load_dataframe(*, dataset=None, group=None, path=None, name=None, chunk_size=2 ** 20):
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
        delayed(get_slice)(path, name, start, end)
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


def make_chunk(ranges, block_info, shape, record_dtype, range_dtype, ndim, path, _name):
    block_info = block_info[0]
    block_idxs = block_info['array-location']
    rank = len(block_idxs)
    arr = array(ranges)
    arr.dtype = range_dtype
    arr = arr.reshape((rank,))
    strides = list(reversed(cumprod([1] + list(reversed(shape[1:])))))
    pos = Pos(tuple([
        Coord(
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
            to_array=lambda group: SparseDataset(group).to_backed() if isinstance(group, Group) else group
        )
    ) \
    .reshape((1,)*rank)


def to_arr(c, rank): return c[(0,)*rank].arr()


def load_tensor(*, X=None, path=None, name=None, chunk_size=2 ** 20, to_array=None):
    if X:
        ctx = nullcontext()
        path = X.file.filename
        name = X.name
    else:
        ctx = File(path, 'r')
        X = ctx[name]

    print(f'Loading HDF5 tensor: {path}:{name}: {X}')

    with ctx:
        chunks = normalize_chunks('auto', X.shape, dtype = X.dtype)
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
        )

        arr = h5chunks.map_blocks(to_arr, chunks=chunks, dtype=X.dtype, rank=rank)

        return arr
