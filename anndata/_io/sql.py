
from typing import Union, Tuple

from dask import compute, delayed
from dask.array import Array, from_array
from dask.dataframe import DataFrame as DDF, read_sql_table
from numpy import array, ndarray
from pandas import DataFrame as DF, Series, concat
from scipy.sparse import spmatrix

ddf_to_sql = delayed(DF.to_sql)

from anndata import AnnData
from .h5chunk import Pos

IDX_DTYPE = '<i8'

def read_sql(table_name_prefix, engine):
    obs = read_sql_table(f'{table_name_prefix}_obs', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    var = read_sql_table(f'{table_name_prefix}_var', engine, index_col='level_0', bytes_per_chunk=2**20).set_index('index')
    X = read_sql_table(f'{table_name_prefix}_X', engine, index_col='index', bytes_per_chunk=2**20).set_index('index')
    X = None
    AnnData(
        X=X,
        obs=obs,
        var=var,
    )


def to_dataframe(arr, pos: Union[Tuple[int, ...], Pos] = None):
    if pos is None:
        pos = Pos.from_arr(arr, [0]*arr.ndim)
    elif isinstance(pos, tuple):
        pos = Pos.from_arr(arr, pos)

    coords = pos.coords
    assert arr.ndim == len(coords), f'{arr.ndim} != {len(coords)}: {coords}'

    idx_offset = pos.idx

    if isinstance(arr, spmatrix):
        coo = arr.tocoo()
        (R, _) = coo.shape

        idx_dtype = [ ('idx',IDX_DTYPE) ]
        idxs_dtype = [
            (f'i{i}', IDX_DTYPE)
            for i in range(arr.ndim)
        ]

        if arr.dtype.fields:
            def make_record(r, c, v): return tuple([R*r + c, r, c,] + list(v))
            values_dtype = arr.dtype.descr
        else:
            def make_record(r, c, v): return (R*r + c, r, c, v)
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
            df = \
                concat(
                    [
                        Series(range(coord.offset, coord.offset + len(arr)), name='idx'),
                        Series(arr,name='v')
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
                    idxs + [ coord.offset + i ],
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


def write_sql(ad, table_name_prefix, engine, if_exists=None, dask=False):
    print(f'ad: {type(ad)}')
    obs = ad.obs
    var = ad.var
    X = ad.X

    print(f'Writing obs')
    write_table(obs, f'{table_name_prefix}/obs', engine, if_exists=if_exists)
    print(f'Writing var')
    write_table(var, f'{table_name_prefix}/var', engine, if_exists=if_exists)
    print(f'Writing X')
    write_tensor(X, f'{table_name_prefix}/X', engine, if_exists=if_exists, dask=dask)


def write_table(df, table_name, engine, if_exists=None):
    if isinstance(df, DF):
        df.to_sql(table_name, engine, if_exists=if_exists)
    elif isinstance(df, DDF):
        df = df.reset_index()
        df._meta.to_sql(table_name, engine, if_exists=if_exists)
        res = [
            ddf_to_sql(d, table_name, engine, if_exists='append')
            for d in df.to_delayed()
        ]
        return compute(*res)
    elif isinstance(df, Array):
        arr = df
        dtype = arr
        meta = concat([ Series([], name=name, dtype=dt) for name, dt in dtype ], axis=1)
    else:
        raise Exception(f'Unrecognized DataFrame type {type(df)}:\n{df}')


def write_tensor(t, table_name, engine, if_exists=None, dask=False):
    if isinstance(t, Array):
        dtype = t.dtype
        idx_dtype = [ ('idx',IDX_DTYPE) ]
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
        meta.to_sql(table_name, engine, if_exists=if_exists)

        def to_sql(block, block_info):
            print(f'block_info: {block_info}')
            offsets = [ offset for offset, _ in block_info[0]['array-location'] ]
            shape = block_info[0]['shape']
            df = to_dataframe(block, pos=Pos.from_offsets_shapes(offsets, shape))
            df.to_sql(table_name, engine, if_exists='append')
            return array(True).reshape((1,)*block.ndim)

        return t.map_blocks(to_sql, chunks=(1,)*t.ndim, dtype=bool).compute()
    elif dask:
        return write_tensor(from_array(t), table_name, engine, if_exists)
    else:
        df = to_dataframe(t)
        df.to_sql(table_name, engine, if_exists=if_exists)
