from contextlib import nullcontext
from dask.dataframe import from_delayed
from dask.delayed import delayed
from dataclasses import dataclass
from numpy import unique
from pandas import Categorical, DataFrame as DF
from pathlib import Path
from typing import Tuple

from anndata._core.sparse_dataset import SparseDataset
from h5py import File, Group
from numpy import array, dtype


@dataclass
class H5Chunk:
    file: Path
    path: str
    dtype: dtype
    idx: Tuple[int,int]
    chunk: Tuple[Tuple[int,int],Tuple[int,int]]

    def nnz(self):
        print(f'Opening {self.file}: {self.idx} ({self.chunk})')
        with File(self.file, 'r') as f:
            group = f[self.path]
            assert isinstance(group, Group)
            dataset = SparseDataset(group)
            ((r,R),(c,C)) = self.chunk
            coo = dataset[r:R,c:C].tocoo()
            record_dtype = [
                ('r','i4'),
                ('c','i4'),
                ('v',self.dtype),
            ]
            arr = array(
                list(zip(r + coo.row, coo.col, coo.data)),
                dtype=record_dtype,
            )
            print(f'Closing {self.file}: {self.idx} ({self.chunk})')
            return arr


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

