# Had to rename the test module to h5py_ so that it wouldn't conflict with the
# h5py import upon testing.

import numpy as np
import scipy.sparse as ss

from anndata import h5py


from anndata import AnnData
from math import floor, sqrt
from pandas import DataFrame as DF
from pathlib import Path
from numpy import array
import pytest
from scipy import sparse
from shutil import rmtree
from tempfile import TemporaryDirectory


def test_load():
    R = 100
    C = 200

    def digits(n, b, empty_zero=False, significant_leading_zeros=True):
        '''Convert a number to an array of base-`b` digits, with a few toggle-able
        behaviors.

        The default parameters output an analogue to "spreadsheet-column" order, e.g. in
        base-26 you get the equivalent of: "A","B",…,"Z","AA","AB",… (but with arrays of
        integers ∈[0,26) instead of strings of letters).

        :param empty_zero: when True, start counting from an empty array at zero
                (otherwise, zero will be mapped to [0])
        :param significant_leading_zeros: when True, enumerate non-empty natural-number
                arrays containing the elements ∈[0,b)
        '''
        if significant_leading_zeros:
            if not empty_zero: return digits(n+1, b, empty_zero=True, significant_leading_zeros=significant_leading_zeros)
            bases = [1]
            while n >= bases[-1]:
                n -= bases[-1]
                bases.append(bases[-1]*b)
            n_digits = digits(n, b, empty_zero=True, significant_leading_zeros=False)
            return [0]*(len(bases)-1-len(n_digits)) + n_digits
        else:
            return digits(n // b, b, empty_zero=True, significant_leading_zeros=False) + [n%b] if n else [] if empty_zero else [0]

    def spreadsheet_column(idx):
        return ''.join([ chr(ord('A')+digit) for digit in digits(idx, 26) ])

    X = sparse.random(R, C, format="csc", density=0.1, random_state=123)

    obs = DF([
        {
            'label': f'row {r}',
            'idx²': r**2,
            'Prime': all([
                r % f != 0
                for f in range(2, floor(sqrt(r)))
            ]),
        }
        for r in range(R)
    ])

    var = DF([
        {
            'name': spreadsheet_column(c),
            'sqrt(idx)': sqrt(c),
        }
        for c in range(C)
    ])

    ad = AnnData(X=X, obs=obs, var=var)
    new_path = Path.cwd() / 'new.h5ad'
    old_path = Path.cwd() / 'old.h5ad'
    def write(path, overwrite=False):
        if path.exists() and overwrite:
            path.unlink()
        if not path.exists():
            ad.write_h5ad(path)

    write(old_path, overwrite=True)

    # from anndata import read_h5ad
    #
    # def load_ad(path):
    #     ad = read_h5ad(path, backed='r', dask=True)
    #     X = ad.X.compute()
    #     coo = X.tocoo()
    #     rows, cols = coo.nonzero()
    #     nnz = list(zip(list(rows), list(cols)))
    #     return ad, nnz
    #
    # old_ad, old_nnz = load_ad(old_path)
    # new_ad, new_nnz = load_ad(new_path)
    #
    # print(old_nnz[:20])
    # print(new_nnz[:20])
    # assert old_nnz == new_nnz

    # with TemporaryDirectory() as dir:
    #     path = Path(dir) / 'tmp.h5ad'
    #     ad.write_h5ad(path)


def test_create_and_read_dataset(tmp_path):
    h5_path = tmp_path / 'test.h5'
    sparse_matrix = ss.csr_matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0],
    ], dtype=np.float64)
    with h5py.File(h5_path) as h5f:
        h5f.create_dataset('sparse/matrix', data=sparse_matrix)
    with h5py.File(h5_path) as h5f:
        assert (h5f['sparse']['matrix'][1:3] != sparse_matrix[1:3]).size == 0
        assert (h5f['sparse']['matrix'][2:] != sparse_matrix[2:]).size == 0
        assert (h5f['sparse']['matrix'][:2] != sparse_matrix[:2]).size == 0
        assert (h5f['sparse']['matrix'][-2:] != sparse_matrix[-2:]).size == 0
        assert (h5f['sparse']['matrix'][:-2] != sparse_matrix[:-2]).size == 0
        assert (h5f['sparse']['matrix'].value != sparse_matrix).size == 0


def test_create_dataset_from_dataset(tmp_path):
    from_h5_path = tmp_path / 'from.h5'
    to_h5_path = tmp_path / 'to.h5'
    sparse_matrix = ss.csr_matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0],
    ], dtype=np.float64)
    with h5py.File(from_h5_path) as from_h5f:
        from_dset = from_h5f.create_dataset('sparse/matrix', data=sparse_matrix)

        with h5py.File(to_h5_path) as to_h5f:
            to_h5f.create_dataset('sparse/matrix', data=from_dset)
            assert (to_h5f['sparse/matrix'].value != sparse_matrix).size == 0


def test_dataset_append(tmp_path):
    h5_path = tmp_path / 'test.h5'
    sparse_matrix = ss.csr_matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0],
    ], dtype=np.float64)
    to_append = ss.csr_matrix([
        [0, 1, 1],
        [1, 0, 0],
    ], dtype=np.float64)
    appended_matrix = ss.vstack((sparse_matrix, to_append))

    with h5py.File(h5_path) as h5f:
        h5f.create_dataset('matrix', data=sparse_matrix, chunks=(100000,))
        h5f['matrix'].append(to_append)
        assert (h5f['matrix'].value != appended_matrix).size == 0
