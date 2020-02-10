from dataclasses import dataclass
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

