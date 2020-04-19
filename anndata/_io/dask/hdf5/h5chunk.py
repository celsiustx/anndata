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

from h5py import Dataset, File, Group
from numpy import dtype, cumprod
from scipy.sparse import spmatrix


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
        #print(f'Opening {self.file} ({self.path}): {self.idx} ({self.slice})')
        with File(self.file, 'r') as f:
            arr = f[self.path]
            attrs = arr.attrs

            # Verify to_array exists if the HDF5 entry is a Group (and thus requires converting)
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


