
import functools
from logging import getLogger
from typing import Optional, List

import pandas as pd
import numpy as np

import dask
import dask.dataframe
import dask.array


logger = getLogger(__file__)


def is_dask(obj) -> bool:
    return isinstance(obj, dask.base.DaskMethodsMixin)


def daskify_call(f, *args, _dask_len=None, _dask_output_types=None, **kwargs):
    # Call a function with delayed() and do some checking around it.

    @functools.wraps(f)
    def inner(*args, **kwargs):
        retval = f(*args, **kwargs)
        if _dask_output_types is not None:
            if not isinstance(retval, _dask_output_types):
                logger.warning("Expected output type %s, got %s in %s!"
                               % (_dask_output_types, retval, f))
        if _dask_len is not None:
            length = len(retval)
            if length != _dask_len:
                logger.warning("Expected length %s, got %s on %s!"
                               % (_dask_len, length, retval))
        return retval

    # TODO: Set _len if possible.
    return dask.delayed(inner)(*args, **kwargs)


def daskify_method_call(obj, method_name, *args, _dask_obj_type=None, _dask_len=None, **kwargs):
    def call_method(obj_, method_name_, *args, **kwargs):
        if _dask_obj_type is not None:
            if not isinstance(obj_, _dask_obj_type):
                logger.warning("Expected object type %s, got %s in %s!"
                               % (_dask_obj_type, obj_, method_name_))
        bound_method = getattr(obj_, method_name_)
        return bound_method(*args, **kwargs)

    return daskify_call(call_method, obj, method_name, *args, _dask_len=_dask_len, **kwargs)

def daskify_calc_shape(old_shape, one_slice_per_dim):
    def get_new_len(dim_len, dim_slice):
        if dim_slice == slice(None, None, None):
            return dim_len
        if dim_slice.step == 1:
            # Do the common/simple version with math.
            stop = np.min(dim_slice.stop, dim_len)
            if stop >= dim_slice.start:
                return dim_slice.stop - dim_slice.start
            else:
                return
        else:
            # TODO: There is a way to calculate when the step is not 1.
            logger.warning("TODO: Verify slice calculation with stepping!")
            new_shape[dim] = range(*dim_slice.indices(dim_len))

    # TODO: Double check this especially with stepping.
    new_shape = []
    for dim in range(len(old_shape)):
        old_len = old_shape[dim]
        dim_slice = one_slice_per_dim[dim]
        if is_dask(old_len) or is_dask(dim_slice):
            new_shape.append(daskify_call(get_new_len, old_len, dim_slice, _dask_output_types=int))
        else:
            new_shape.append(get_new_len(old_len, dim_slice))
    return tuple(new_shape)


def daskify_call_return_array(f: callable, *args, _dask_shape, _dask_dtype, _dask_meta, **kwargs):
    return dask.array.from_delayed(
        daskify_call(f, *args,
                     _dask_len=None,
                     _dask_output_types=(list, np.array, pd.Series),
                     **kwargs),
        shape=_dask_shape,
        dtype=_dask_dtype,
        meta=_dask_meta
    )


def daskify_call_return_df(f: callable, *args, _dask_len=None, _dask_meta=None, **kwargs):
    return dask.dataframe.from_delayed(
        daskify_call(f, *args, _dask_len=None, _dask_output_types=pd.DataFrame, **kwargs),
        meta=_dask_meta,
        verify_meta=True
    )


def daskify_iloc(df, idx):
    def call_iloc(df_, idx_):
        return df_.iloc[idx_]
    meta = df._meta
    if meta is None:
        pass
    df = daskify_call_return_df(call_iloc, df, idx, _dask_meta=meta)
    return df


def daskify_get_len_given_slice(slc: slice, orig_len: int):
    def get_size(slc_, orig_len_):
        len(range(*slc_.indices(orig_len_)))
    return daskify_call(slc, orig_len)


def compute_anndata(an: "anndata.AnnData", *args, **kwargs):
    from anndata import AnnData

    def _compute_anndata(X, **raw_attr_value_pairs):
        # Construct an AnnData at a low-level,
        # swapping out each of the attributes specified.
        an = AnnData.__new__(AnnData)
        for key, value in raw_attr_value_pairs.items():
            setattr(an, key, value)
        an._X = X
        an._dask = False
        return an

    # Passing the attribute this way will automatically put them into
    # the graph in parallel:
    attribute_value_pairs = an.__dict__.copy()
    virtual = daskify_call(_compute_anndata, an.X, **attribute_value_pairs)
    real = virtual.compute(*args, **kwargs)
    return real
