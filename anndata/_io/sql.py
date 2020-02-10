
from dask.array import Array, from_array
from dask.dataframe import DataFrame as DDF, read_sql_table, from_dask_array
from numpy import array, nan
from pandas import DataFrame as DF
from pathlib import Path
from scipy.sparse import spmatrix

from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset
from .h5chunk import H5Chunk


def read_sql(table_name_prefix, engine):
    obs = read_sql_table(f'{table_name_prefix}_obs', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    var = read_sql_table(f'{table_name_prefix}_var', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    X = None
    AnnData(
        X=X,
        obs=obs,
        var=var,
    )


def write_sql(ad, table_name_prefix, engine, if_exists=None, chunk_size=2**20):
    print(f'ad: {type(ad)}')
    obs = ad.obs
    var = ad.var
    X = ad.X

    #X_arr = from_array(X)

    if isinstance(X, SparseDataset):
        group = X.group
        file = group.file
        path = Path(file.filename)
        name = 'X'
        dtype = X.dtype
        itemsize = dtype.itemsize
        csc = X.to_backed()
        byte_size = csc.nnz * itemsize
        n_chunks = byte_size // chunk_size
        NR, NC = csc.shape
        CR = (NR + n_chunks - 1) // n_chunks
        chunk_offsets = list(range(0, NR, CR))
        chunk_offsets = list(zip(chunk_offsets, chunk_offsets[1:] + [NR]))
        chunks = [
            H5Chunk(
                path,
                name,
                dtype,
                (idx, 0),
                (
                    chunk_offset,
                    (0, NC)
                )
            )
            for idx, chunk_offset in enumerate(chunk_offsets)
        ]
        dask_chunks = from_array(chunks, chunks=(1,))
        record_dtype = [
            ('r','i4'),
            ('c','i4'),
            ('v',dtype),
        ]
        nnz = dask_chunks.map_blocks(lambda c: c[0].nnz(), chunks=(nan,), dtype=record_dtype)
        ddf = from_dask_array(nnz)
        print(f'Writing X')
        write_table(ddf, f'{table_name_prefix}_X', engine, if_exists='replace')
    else:
        raise Exception(f'Unexpected X matrix type {type(X)}: {X}')

    # obs = obs.reset_index()
    #obs.index obs.index.rename
    print(f'Writing obs')
    write_table(obs.reset_index(), f'{table_name_prefix}_obs', engine, if_exists=if_exists)
    print(f'Writing var')
    write_table(var.reset_index(), f'{table_name_prefix}_var', engine, if_exists=if_exists)


def write_table(df, table_name, engine, if_exists=None):
    if isinstance(df, DF):
        df.to_sql(table_name, engine, if_exists=if_exists)
    elif isinstance(df, DDF):
        from dask import compute, delayed
        ddf_to_sql = delayed(DF.to_sql)
        df._meta.to_sql(table_name, engine, if_exists=if_exists)
        res = [
            ddf_to_sql(d, table_name, engine, if_exists='append')
            for d in df.to_delayed()
        ]
        return compute(*res)
    else:
        raise Exception(f'Unrecognized DataFrame type {type(df)}:\n{df}')

def write_tensor(t, table_name, engine, if_exists=None):
    if isinstance(t, array):
        raise
    elif isinstance(t, Array):
        raise
    else:
        raise
