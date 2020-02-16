from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Callable

from dask.array import Array
from dask.dataframe import from_delayed
from dask.delayed import delayed
from h5py import Dataset, File, Group
from numpy import array, dtype, ndarray
from pandas import Categorical, DataFrame as DF
from scipy.sparse import spmatrix


@dataclass
class Coord:
    idx: int
    offset: int
    shape: int
    stride: int

@dataclass
class Pos:
    coords: Tuple[Coord, ...]

    @property
    def idx(self):
        return sum([ coord.stride * coord.offset for coord in self.coords ])

    @staticmethod
    def from_offset_shapes(offsets):
        stride = 1
        strides = [stride]
        for i in range(len(offsets)-1, 0, -1):
            stride *= offsets[i][1]
            strides.append(stride)
        strides = reversed(strides)
        return Pos(
            tuple([
                Coord(idx, offset, shape, stride)
                for (idx, (offset, shape)), stride
                in zip(enumerate(offsets), strides)
            ])
        )

    @staticmethod
    def from_offsets_shapes(offsets, shapes):
        return Pos.from_offset_shapes(list(zip(offsets, shapes)))

    @staticmethod
    def from_arr(arr, offsets):
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

    def __init__(self):
        assert self.ndim == len(self.pos)

    @property
    def shape(self):
        return tuple( coord.shape for coord in self.pos )

    @property
    def offset(self):
        return tuple( coord.offset for coord in self.pos )

    @property
    def idx(self):
        return tuple( coord.idx for coord in self.pos )

    @property
    def slice(self):
        return tuple( slice(coord.offset, coord.shape) for coord in self.pos )

    def arr(self):
        print(f'Opening {self.file}: {self.idx} ({self.offset} + {self.shape})')
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
        group = f[name]
        attrs = group.attrs
        if 'column-order' in attrs:
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
            pass


def load_group(*, group=None, path=None, name=None, chunk_size=2**20):
    if group:
        ctx = nullcontext()
        path = group.file.filename
        name = group.name
    else:
        ctx = File(path, 'r')
        group = ctx[name]

    with ctx:
        cols = list(group.attrs['column-order'])
        idx_key = group.attrs["_index"]
        itemsize = sum([ group[k].dtype.itemsize for k in cols ])
        [ (size,) ] = set([ group[k].shape for k in cols ])
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

