#
# This module was added as part of adding dask support.
# Possibly consider upstreaming to dask vs. anndata?
#

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext

from dask.dataframe import from_delayed
from dask.delayed import delayed
from h5py import File, Group
from pandas import Categorical, DataFrame as DF


def get_slice(path, name, start, end, columns, index_col=None):
    '''Load rows [start,end) from HDF5 file `path` (group `name`) into a DataFrame'''
    with File(path, 'r') as f:
        obj = f[name]

        if index_col and index_col not in columns:
            columns = [index_col] + columns

        if isinstance(obj, Group):
            group = obj
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
            df = DF(dataset[start:end])[columns]

        if index_col is not None:
            if isinstance(index_col, str):
                df = df.set_index(index_col)
            elif callable(index_col):
                df = df.set_index(index_col(df))
            else:
                raise ValueError('Invalid index_col value: %s' % index_col)

        return df


def load_dask_dataframe(
    *,
    dataset=None, group=None,
    path=None, key=None,
    chunk_size=2 ** 20,
    index_col=None, columns=None, require_columns=True,
):
    obj = dataset or group
    if obj:
        ctx = nullcontext()
        path = obj.file.filename
        key = obj.name
    else:
        ctx = File(path, 'r')
        obj = ctx[key]
        if isinstance(obj, Group):
            group = obj
        else:
            dataset = obj

    with ctx:
        if group:
            if not columns:
                if 'column-order' in group.attrs:
                    columns = list(group.attrs['column-order'])
                elif not require_columns:
                    columns = list(group.keys())
                else:
                    raise ValueError(f'Loading Dask Dataframe from {path}:{key}: column list required but not provided, and no "column-order" attribute found')
            #idx_key = group.attrs["_index"]  # TODO: use this / set index col correctly?
            itemsize = sum([ group[k].dtype.itemsize for k in columns ])
            [ (size,) ] = set([ group[k].shape for k in columns ])
        else:
            dtype = dataset.dtype
            all_cols = list(dtype.fields.keys())
            if columns:
                columns = [ col for col in all_cols if col in set(columns) ]
            else:
                columns = all_cols
            itemsize = dtype.itemsize
            (size,) = dataset.shape

        n_bytes = itemsize * size
        n_chunks = (n_bytes + chunk_size - 1) // chunk_size
        chunk_starts = [ (i * size // n_chunks) for i in range(n_chunks) ]
        chunk_slices = list(zip(chunk_starts, chunk_starts[1:] + [size]))

    chunks = [
        delayed(get_slice)(path, key, start, end, columns=columns, index_col=index_col)
        for start, end in chunk_slices
    ]

    ddf = from_delayed(chunks)
    return ddf
