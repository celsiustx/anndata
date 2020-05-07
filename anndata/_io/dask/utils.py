import dask
import dask.dataframe
import dask.array
from logging import getLogger
import pandas as pd
import numpy as np
from typing import Union, Optional

logger = getLogger(__file__)


def is_dask(obj) -> bool:
    return isinstance(obj, dask.base.DaskMethodsMixin)


def daskify_call(f, *args, _dask_len=None, _dask_output_types=None, **kwargs):
    # Call a function with delayed() and do some checking around it.
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


def daskify_method_call(obj, method_name, *args, __dask_obj_type=None, _dask_len=None, **kwargs):
    def call_method(obj_, method_name_, *args, **kwargs):
        if __dask_obj_type is not None:
            if not isinstance(o, __dask_obj_type):
                logger.warning("Expected object type %s, got %s in %s!"
                               % (__dask_obj_type, obj_, method_name_))
        bound_method = getattr(obj_, method_)
        return bound_method(*args, **kwargs)
    return daskify_call(call_method, obj, method_name,
                        *args,
                        _dask_len=_dask_len,
                        **kwargs)


def daskify_call_return_array(f: callable, *args, _dask_len=None, **kwargs):
    return dask.array.from_delayed(
        daskify_call(f, *args, _dask_len=None, __dask_output_types=(list, np.array, pd.Series), **kwargs)
    )


def daskify_call_return_df(f: callable, *args, _dask_len=None, **kwargs):
    return dask.dataframe.from_delayed(
        daskify_call(f, *args, _dask_len=None, _dask_output_types=pd.DataFrame, **kwargs)
    )

def daskify_method_call_return_df(obj, method_name, *args, _dask_len=None, **kwargs):
    return daskify_call_return_df(obj, method_name,
                                  *args,
                                  _dask_len=_dask_len,
                                  **kwargs)

def daskify_iloc(df, idx):
    def call_iloc(df, idx):
        return df.iloc[idx]
    return daskify_call_return_df(call_iloc, df, idx)

def daskify_get_size(idx, n):
    def get_size(idx, n):
        len(range(*idx.indices(n)))
    return daskify_call(idx, n)
