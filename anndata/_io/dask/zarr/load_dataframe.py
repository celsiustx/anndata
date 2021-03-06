from contextlib import nullcontext

from dask import delayed
from dask.dataframe import from_delayed

from zarr import Group

# def get_slice(path, name, start, end, columns, index_col=None):
#     '''Load rows [start,end) from HDF5 file `path` (group `name`) into a DataFrame'''
#     with File(path, 'r') as f:
#         obj = f[name]
#
#         if index_col and \
#             isinstance(index_col, str) and \
#             index_col not in columns:
#             columns = [index_col] + columns
#
#         if isinstance(obj, Group):
#             group = obj
#             def get_series(k):
#                 v = group[k]
#                 attrs = v.attrs
#                 if 'categories' in attrs.keys():
#                     categories_ref = attrs['categories']
#                     categories = group[categories_ref]
#                     return Categorical.from_codes(v[start:end], categories)
#                 else:
#                     return v[start:end]
#
#             df: dd.DataFrame = DF({ k: get_series(k) for k in columns })
#         else:
#             dataset = obj
#             df: dd.DataFrame = DF(dataset[start:end])[columns]
#
#         if index_col is not None:
#             if isinstance(index_col, str):
#                 df = df.set_index(index_col)
#             elif callable(index_col):
#                 df = df.set_index(index_col(df))
#             else:
#                 raise ValueError('Invalid index_col value: %s' % index_col)
#
#         return df


def load_dask_dataframe(
    *,
    dataset=None, group=None,
    path=None, key=None,
    index_col=None, columns=None, require_columns=True,
):
    '''Load a Dask DataFrame from HDF5

    Pass in either:
    - an h5py.Dataset
    - an h5py.Group
    - a `path` to an HDF5 file (and `key` to a Dataset or Group inside that path)

    Datasets vs. Groups:
    - When loading a Dataset, a numpy recarray format is assumed.
    - When loading a Group, multiple equal-length Datasets (each corresponding to a
      Series) are assumed.

    Selecting columns:
    - If desired, select specific columns via the `columns` argument
    - If `columns` is omitted, a `column-order` group attr will be used to determine
      columns (and order)
    - If all Datasets in a Group should be loaded as Series, and the order doesn't
      matter, require_columns=False can be specified (normally this state will raise an
      error)
    '''
    if dataset and group:
        raise ValueError(f'Provide at most one of {"dataset","group"}')

    obj = dataset or group
    if obj:
        ctx = nullcontext()
        path = obj.store.path
        key = obj.name
    else:
        raise NotImplementedError
        # ctx = File(path, 'r')
        # obj = ctx[key]
        # if isinstance(obj, Group):
        #     group = obj
        # else:
        #     dataset = obj

    with ctx:
        if group:
            raise NotImplementedError
            # if not columns:
            #     if 'column-order' in group.attrs:
            #         columns = list(group.attrs['column-order'])
            #     elif not require_columns:
            #         columns = list(group.keys())
            #     else:
            #         raise ValueError(f'Loading Dask Dataframe from {path}:{key}: column list required but not provided, and no "column-order" attribute found')
            #
            # if index_col and index_col not in columns:
            #     columns = [index_col] + columns
            #
            # itemsize = sum([ group[k].dtype.itemsize for k in columns ])
            # [ (n_rows,) ] = set([ group[k].shape for k in columns ])
        else:
            import dask.array as da
            arr = da.from_zarr(dataset)
            import dask.dataframe as dd
            df = dd.from_array(arr)
            if index_col:
                if callable(index_col):
                    index_col = index_col(df)
                if index_col not in df.columns:
                    raise ValueError('index_col %s not found in columns %s' % (index_col, str(df.columns)))
                return df.set_index(index_col)
            return df
            # dtype = dataset.dtype
            # all_cols = list(dtype.fields.keys())
            # if columns:
            #     columns = [ col for col in all_cols if col in set(columns) ]
            # else:
            #     columns = all_cols
            # itemsize = dtype.itemsize
            # (n_rows,) = dataset.shape

        # n_bytes = itemsize * n_rows
        # n_chunks = (n_bytes + chunk_size - 1) // chunk_size
        # chunk_starts = [ (i * n_rows // n_chunks) for i in range(n_chunks) ]
        # chunk_slices = list(zip(chunk_starts, chunk_starts[1:] + [n_rows]))

    # chunks = [
    #     delayed(get_slice)(path, key, start, end, columns=columns, index_col=index_col)
    #     for start, end in chunk_slices
    # ]
    #
    # # The dask "meta" is a pandas dataframe with zero rows that captures the shape.
    # # By doing a single "slice" of zero rows we read the file a tiny amount,
    # # and this happens synchronously outside of the graph processing.
    # meta = get_slice(path, key, 0, 0, columns=columns, index_col=index_col)
    #
    # ddf = from_delayed(chunks, meta=meta)
    # ddf.partition_sizes = [ end-start for start, end in chunk_slices ]
    # return ddf
