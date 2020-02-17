
from typing import Union, Tuple

from dask import compute, delayed
from dask.array import Array, from_array
from dask.dataframe import DataFrame as DDF, read_sql_table
from numpy import array, ndarray
from pandas import DataFrame as DF, Series, concat
from scipy.sparse import spmatrix


def df_to_sql(*args, db_url, **kwargs):
    return DF.to_sql(*args, con=db_url, **kwargs)

ddf_to_sql = delayed(df_to_sql)


from anndata import AnnData
from .h5chunk import Pos


# TODO: will overflow at 2**32, but performance is notably worse using Postgres `bigint` types, so leave as-is for now, until better optimizing can be done on a per-table basis
IDX_DTYPE = '<i4'


def read_sql(table_name_prefix, engine):
    '''WIP logic for loading an AnnData from SQL'''
    obs = read_sql_table(f'{table_name_prefix}_obs', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    var = read_sql_table(f'{table_name_prefix}_var', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    X = read_sql_table(f'{table_name_prefix}_X', engine, index_col='index', bytes_per_chunk=2**20).set_index('index')
    X = None
    AnnData(
        X=X,
        obs=obs,
        var=var,
    )
    raise NotImplementedError


def to_dataframe(arr, pos: Union[Pos, None]):
    if pos is None:
        pos = Pos.whole_array(arr)

    coords = pos.coords
    assert arr.ndim == len(coords), f'{arr.ndim} != {len(coords)}: {coords}'

    idx_offset = pos.idx

    if isinstance(arr, spmatrix):
        coo = arr.tocoo()
        (_, C) = coo.shape
        r_coord, c_coord = coords
        base_idx = pos.idx
        base_row = r_coord.start
        base_col = c_coord.start

        idx_dtype = [ ('idx',IDX_DTYPE) ]  # TODO: right-size integer index types?
        idxs_dtype = [
            (f'i{i}', IDX_DTYPE)
            for i in range(arr.ndim)
        ]

        if arr.dtype.fields:
            def make_record(r, c, v): return tuple([ base_idx + C*r + c, base_row + r, base_col + c,] + list(v))
            values_dtype = arr.dtype.descr
        else:
            def make_record(r, c, v): return ( base_idx + C*r + c, base_row + r, base_col + c, v )
            values_dtype = [ ('v', arr.dtype.str) ]

        dtype = idx_dtype + idxs_dtype + values_dtype
        print(f'dtype: {dtype}')
        arr = array(
            [
                make_record(r, c, v)
                for r, c, v in zip(coo.row, coo.col, coo.data)
            ],
            dtype=dtype,
        )
        df = DF(arr).set_index('idx')
        return df
    else:
        assert isinstance(arr, ndarray), f'Unexpected type {type(arr)}: {arr}'
        if arr.ndim == 1:
            (coord,) = coords
            assert len(arr) == (coord.end - coord.start), f'{len(arr)} != {coord.end} - {coord.start}'
            df = \
                concat(
                    [
                        Series(range(coord.start, coord.end), name='idx'),
                        Series(arr, name='v')
                    ],
                    axis=1,
                ) \
                .set_index('idx')
            return df

        def to_coo(idx, coords, idxs, arr):
            if arr.ndim == 0:
                values = \
                    [arr] \
                        if not arr.dtype.fields \
                        else list(arr)
                return tuple([idx] + idxs + values)

            coord = coords[0]
            coords = coords[1:]
            return [
                to_coo(
                    idx + i*coord.stride,
                    coords,
                    idxs + [ coord.start + i ],
                    row,
                )
                for i, row in enumerate(arr)
            ]

        dtype = arr.dtype
        idx_dtype = [ ('idx',IDX_DTYPE) ]
        idxs_dtype = [
            (f'i{i}', IDX_DTYPE)
            for i in range(arr.ndim)
        ]
        value_dtype = \
            [ ('v', dtype.str) ] \
                if not dtype.fields \
                else dtype.descr

        dtype = idx_dtype + idxs_dtype + value_dtype

        coo = to_coo(idx_offset, coords, [], arr)
        narr = array(coo, dtype=dtype)
        narr = narr.reshape((narr.size,))
        return DF(narr).set_index('idx')


def write_sql(ad, table_name_prefix, db_url, if_exists=None, dask=False):
    print(f'ad: {type(ad)}')
    obs = ad.obs
    var = ad.var
    X = ad.X

    print(f'Writing X')
    write_tensor(X, f'{table_name_prefix}/X', db_url, if_exists=if_exists, dask=dask)
    print(f'Writing obs')
    write_df(obs, f'{table_name_prefix}/obs', db_url, if_exists=if_exists)
    print(f'Writing var')
    write_df(var, f'{table_name_prefix}/var', db_url, if_exists=if_exists)


def write_df(df, table_name, db_url, if_exists=None):
    if isinstance(df, DF):
        df.to_sql(table_name, db_url, if_exists=if_exists)
    elif isinstance(df, DDF):
        df = df.reset_index()
        df._meta.to_sql(table_name, db_url, if_exists=if_exists)
        res = [
            ddf_to_sql(d, table_name, db_url=db_url, if_exists='append')
            for d in df.to_delayed()
        ]
        return compute(*res)
    else:
        raise Exception(f'Unrecognized DataFrame type {type(df)}:\n{df}')


def to_sql(block, block_info, table_name, db_url):
    print(f'block_info: {block_info}, block {type(block)}')
    block_info = block_info[0]
    df = to_dataframe(block, pos=Pos.from_block_info(block_info))
    df.to_sql(table_name, db_url, if_exists='append')
    return array(True).reshape((1,)*block.ndim)


def write_tensor(t, table_name, db_url, if_exists=None, dask=False):
    if isinstance(t, Array):
        dtype = t.dtype
        idx_dtype = [ ('idx', IDX_DTYPE) ]
        if t.ndim == 1:
            idxs_dtype = []
        else:
            idxs_dtype = [
                (f'i{i}', IDX_DTYPE)
                for i in range(t.ndim)
            ]
        value_dtype = \
            [ ('v', dtype.str) ] \
                if not dtype.fields \
                else dtype.descr

        dtype = idx_dtype + idxs_dtype + value_dtype

        meta = concat([ Series([], name=name, dtype=dt) for name, dt in dtype ], axis=1).set_index('idx')
        meta.to_sql(table_name, db_url, if_exists=if_exists)

        return t.map_blocks(to_sql, chunks=(1,)*t.ndim, dtype=bool, table_name=table_name, db_url=db_url).compute()
    elif dask:
        print(f'Wrapping tensor in dask: {t}')
        return write_tensor(from_array(t), table_name, db_url, if_exists)
    else:
        df = to_dataframe(t, pos=None)
        df.to_sql(table_name, db_url, if_exists=if_exists)
