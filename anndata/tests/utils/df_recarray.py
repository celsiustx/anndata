import h5py
import numpy as np
from pandas.api.types import is_string_dtype, is_categorical

def df_to_records_fixed_width(df, var_len_str=True):
    '''Convert a DataFrame to a Numpy "record array" (copied from AnnData 0.6.22)'''
    uns = {}  # unstructured dictionary for storing categories
    names = ['_index']
    if is_string_dtype(df.index):
        if var_len_str:
            index = df.index.values.astype(h5py.special_dtype(vlen=str))
        else:
            max_len_index = 0 if 0 in df.shape else df.index.map(len).max()
            index = df.index.values.astype('S{}'.format(max_len_index))
    else:
        index = df.index.values
    arrays = [index]
    for k in df.columns:
        names.append(k)
        if is_string_dtype(df[k]) and not is_categorical(df[k]):
            if var_len_str:
                arrays.append(df[k].values.astype(h5py.special_dtype(vlen=str)))
            else:
                lengths = df[k].map(len)
                if is_categorical(lengths): lengths = lengths.cat.as_ordered()
                arrays.append(df[k].values.astype('S{}'.format(lengths.max())))
        elif is_categorical(df[k]):
            uns[k + '_categories'] = df[k].cat.categories
            arrays.append(df[k].cat.codes)
        else:
            arrays.append(df[k].values)
    formats = [v.dtype for v in arrays]
    return np.rec.fromarrays(
        arrays,
        dtype={'names': names, 'formats': formats}), uns
