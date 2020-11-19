
#
# The module contains AnnDataDask, a subclass of AnnData.
#
# It also contains the daskify_* utility functions used elsewhere in this package,
# and the is_dask() function.
#
# The eventual home of this class is TBD, but the current strategy is:
# - AnnData has no dask-specific logic except error messages.
# - AnnDataDask has all of the dask overrides.
# - Other functions and objects may or may not have dask-awareness internally.
#   ^^ Ideally we remove this too, but it is trickier.
#
from copy import deepcopy
import functools
from functools import singledispatch
from os import PathLike
from typing import Union, Optional  # Meta
from typing import MutableMapping, Tuple, List  # Generic ABCs
import warnings

import numpy as np
from numpy import dtype, nan, ndarray
import pandas as pd
from pandas.api.types import is_string_dtype
from scipy import sparse

import dask
from dask import delayed
import dask.dataframe as dd
from dask.dataframe import DataFrame, Series
import dask.array as da
from dask.array import Array
from dask.array import from_delayed, concatenate
from dask.array.backends import register_scipy_sparse

from anndata._core.anndata import _gen_dataframe
from anndata._core.anndata import AnnData, ImplicitModificationWarning
from anndata._io.dask.hdf5.load_array import load_dask_array
from anndata._core.index import _subset, Index, unpack_index, _normalize_indices
from anndata._core.aligned_mapping import (
    AxisArrays,
    PairwiseArrays,
)
from anndata._core.views import (
    DictView,
    DataFrameView,
    _resolve_idxs,
)
from anndata.utils import convert_to_dict
from anndata.logging import anndata_logger as logger

register_scipy_sparse()


def is_dask(obj) -> bool:
    return isinstance(obj, dask.base.DaskMethodsMixin)


def normalize_slice(idxr, size):
    m = idxr.start or 0
    M = idxr.stop if idxr.stop is not None else size
    Δ = idxr.step or 1
    if m < 0: m += size
    if M < 0: M += size
    m, M = min(m, M), max(m, M)
    m = np.clip(m, 0, size)
    M = np.clip(M, 0, size)
    if Δ < 0:
        Δ = -Δ
        m += (M-m)%Δ

    return m, M, Δ


def bools_to_ints(bools: Union[pd.Series, ndarray]):
    '''Convert a bool-Series to an int-Series representing the indices where True was found.

    - reset_index twice to get a column of auto-incrementing ints (the initial index may be e.g. strings)
    - restore the original index
    - slice using the original booleans (along the original index)
    - take the first column (the pre-slice auto-incrementing integers)
    '''
    assert bools.dtype == dtype(bool)
    sliced = \
        bools \
            .reset_index(drop=True) \
            .reset_index() \
            .set_index(bools.index) \
            [bools]
    return sliced[sliced.columns[0]]


def maybe_bools_to_ints(slicer):
    if isinstance(slicer, (pd.Series, ndarray)) and slicer.dtype == dtype(bool):
        return bools_to_ints(slicer)
    else:
        return slicer


@singledispatch
def partition_idxr(idxr, partition_sizes):
    '''Given an indexer and list of partition sizes, return a map from [partition idx]
    to [indexer containing elements corresponding to that partition].

    The latter can be an in-memory or Dask object; in either case, overlapping
    partitions can be zipped and the indexer applied to each partition's elements.
    '''
    raise NotImplementedError('%s: %s' % (type(idxr), idxr))

@partition_idxr.register(slice)
@partition_idxr.register(range)
def _(idxr, partition_sizes):
    partition_slices = {}
    if not partition_sizes:
        return partition_slices
    partition_ends = np.cumsum(partition_sizes).tolist()
    size = partition_ends[-1]
    partition_idx_ranges = zip(
        [0] + partition_ends,
        partition_ends,
    )

    m, M, Δ = normalize_slice(idxr, size)

    for partition_idx, (start, end) in enumerate(partition_idx_ranges):
        if start < M and end > m:
            if start >= m:
                first = (start - m) % Δ
            else:
                first = m
            last = min(end, M) - start
            partition_slices[partition_idx] = slice(first, last, Δ)

    return partition_slices

@partition_idxr.register(int)
@partition_idxr.register(np.integer)
def _(idxr, partition_sizes):
    return partition_idxr(range(idxr, idxr+1), partition_sizes)

@partition_idxr.register(list)
@partition_idxr.register(tuple)
def _(idxr, partition_sizes):
    '''Break a list of integer indices into sets that can be applied within partitions.

    Example:
    - idxr: [2,3,5,8,13,21,34,55]
    - partition_sizes: [10,20,30])
    - return: { 0: [2,3,5,8], 1: [3,11], 2: [4,25] }

    (Note that the 13, 21, 34, and 55 are converted to the relative offsets within
    partitions 1 and 2 (which start at indices 10 and 30, resp.)

    TODO: factor with similar block in dataframe iloc?

    :param idxr: integer indices
    :param partition_sizes:
    :return: map from [ partition index ] to [ relative indices on that partition
        corresponding to its intersection with `idxr` ]
    '''
    idxr = tuple(idxr)
    partition_idx_lists = {}
    if not partition_sizes:
        return partition_idx_lists
    cur_partition_idxs = []
    idx_pos = 0
    num_idxs = len(idxr)
    partition_ends = np.cumsum(partition_sizes).tolist()
    size = partition_ends[-1]
    npartitions = len(partition_ends)
    partition_idx = 0
    cur_partition_end = partition_ends[0]
    while idx_pos < num_idxs:
        idx = idxr[idx_pos]
        if idx < 0: idx += size
        while idx >= cur_partition_end:
            if cur_partition_idxs:
                partition_idx_lists[partition_idx] = cur_partition_idxs
                cur_partition_idxs = []
            partition_idx += 1
            if partition_idx == npartitions:
                break
            cur_partition_end = partition_ends[partition_idx]
        cur_partition_idxs.append(idx)
        idx_pos += 1
    if cur_partition_idxs:
        partition_idx_lists[partition_idx] = cur_partition_idxs

    return partition_idx_lists
@partition_idxr.register(pd.Series)
def _(idxr, partition_sizes):
    return partition_idxr(idxr.values, partition_sizes)

@partition_idxr.register(ndarray)
def _(idxr, partition_sizes):
    if idxr.dtype == dtype(int):
        return partition_idxr(idxr.tolist(), partition_sizes)
    elif idxr.dtype == dtype(bool):
        size = sum(partition_sizes)
        if len(idxr) != size:
            raise ValueError(
                'Slicing with bool Series of size %d but partition sizes sum to %d (%s)' % (
                    len(idxr), size, partition_sizes
                )
            )
        return partition_idxr(bools_to_ints(idxr), partition_sizes)
    else:
        raise NotImplementedError

@partition_idxr.register(Series)
def _(idxr, partition_sizes):
    if idxr.dtype == dtype(int):
        # TODO: is this case doable?
        # if not idxr.known_divisions:
        #     raise ValueError("Can't slice with a Dask series of ints with unknown divisions: %s" % idxr)
        # divisions = tuple(zip(idxr.divisions, idxr.divisions[1:]))
        raise NotImplementedError
    elif idxr.dtype == dtype(bool):
        # TODO: Check that we can ignore None partition_sizes.  Possibly unsafe.
        if idxr.partition_sizes is not None and idxr.partition_sizes != partition_sizes:
            raise ValueError(
                "Bool dask.dataframe.Series partition_sizes don't match slicee's: %s vs %s; %s" % (
                    idxr.partition_sizes, partition_sizes, idxr
                )
            )
        return {
            idx: idxr.partitions[idx]
            for idx in range(idxr.npartitions)
        }
    else:
        raise NotImplementedError('%s: %s' % (type(idxr), idxr))


def slice_block(X, oidx, vidx):
    # Have to avoid directly slicing w e.g. bool-Series bc spmatrix implements that
    # incorrectly (False and True just get casted to integer 0 and 1, and you get a
    # bunch of copies of row 0 and 1, instead of just the rows corresponding to `True`
    # values in the indexer)
    oidx = maybe_bools_to_ints(oidx)
    vidx = maybe_bools_to_ints(vidx)
    sliced = X[oidx, vidx]
    return sliced

slice_block = delayed(slice_block)


class AnnDataDask(AnnData):
    def _init_as_actual(
        self,
        X=None,
        obs=None,
        var=None,
        uns=None,
        obsm=None,
        varm=None,
        varp=None,
        obsp=None,
        raw=None,
        layers=None,
        dtype="float32",
        shape=None,
        filename=None,
        filemode=None,
        fd=None
    ):
        if X is None:
            if self.file is None:
                raise ValueError("Expected either a filename or X argument!")
            X = load_dask_array(path=self.file.filename, key='X',
                                chunk_size=(self._n_obs, "auto"),
                                format_str='csr', shape=self.shape)
        super()._init_as_actual(X, obs, var, uns, obsm, varm, varp, obsp, raw, layers, dtype, shape, filename, filemode, fd)

    @classmethod
    def slice_df(cls, df, idx: Index):
        #slice_type = None  # int, str, bool, None
        if isinstance(idx, (int, np.integer)):
            idx = np.array([idx])
            slice_type = int
        elif isinstance(idx, slice):
            if idx == slice(None):
                slice_type = None
            else:
                # start = idx.start or 0
                # stop = idx.stop
                # m, M = min(idx.start, idx.stop)
                # idx = range(idx.start, idx.stop, idx.step)
                slice_type = int
        elif isinstance(idx, range):
            slice_type = int
        elif isinstance(idx, (list, set)):
            if not idx:
                slice_type = None
            else:
                i = idx[0]
                if isinstance(i, int):
                    slice_type = int
                elif isinstance(i, str):
                    slice_type = str
                elif isinstance(i, bool):
                    slice_type = bool
                else:
                    raise ValueError('Unsupported slice type: list/set of %s; %s' % (type(i), str(idx[:100])))
        elif isinstance(idx, (np.ndarray, pd.Series, dd.Series, da.Array)):
            if idx.dtype == np.dtype('int64'):
                slice_type = int
            elif idx.dtype == np.dtype('O'):
                slice_type = str
            elif idx.dtype == np.dtype('bool'):
                slice_type = bool
            else:
                raise ValueError('Unsupported slice type: ndarray of %s; %s' % (str(idx.dtype), str(idx[:100])))
        else:
            raise ValueError('Unsupported slice type: %s; %s' % (str(type(idx)), str(idx)))

        if slice_type is int:
            return df.iloc[idx]
        elif slice_type is str:
            return df.loc[idx]
        elif slice_type is bool:
            return df[idx]
        else:
            assert slice_type is None
            return df

    def __getstate__(self):
        if self.is_view:
            raise Exception("AnnDataDask should not have be a view, since it uses dask for view logic.")
        else:
            return self.file.filename

    def __setstate__(self, filename):
        from anndata import read_h5ad
        self2 = read_h5ad(filename, backed="r", dask=True)
        for k, v in self2.__dict__.items():
            setattr(self, k, v)

    def _init_as_view(self, adata_ref: "AnnData", oidx: Index, vidx: Index):
        # NOTE: This method has a large chunk at the beginning and end that is
        # copied from the parent _init_as_view.  Unless we refactor the parent
        # those will need to be kept in sync.
        obs = self.slice_df(adata_ref.obs, oidx)
        self._obs = obs
        var = self.slice_df(adata_ref.var, vidx)
        self._var = var
        X: dask.array.Array = adata_ref.X

        row_slice_map = partition_idxr(oidx, X.chunks[0])
        col_slice_map = partition_idxr(vidx, X.chunks[1])

        def slice_size(idxr):
            if is_dask(idxr):
                return nan
            elif isinstance(idxr, pd.Series) and idxr.dtype == dtype(bool):
                return idxr.sum()
            elif isinstance(idxr, slice):
                return (idxr.stop - idxr.start) // idxr.step
            else:
                return len(idxr)

        def make_block(r, c):
            row_slice = row_slice_map.get(r, [])
            col_slice = col_slice_map.get(c, [])
            n_rows = slice_size(row_slice)
            n_cols = slice_size(col_slice)
            return from_delayed(
                slice_block(
                    X.blocks[r, c],
                    row_slice,
                    col_slice,
                ),
                # TODO: when the {row,col}_slice_map values are
                #  in-memory (not Dask objects), we can len()
                #  them and fill in the post-slice array-block
                #  shape here. One caveat: we have not converted
                #  bool-slicers to ints yet, so len() on those
                #  will not be what you want here; bools.sum()
                #  is probably right
                shape=(n_rows, n_cols),
                dtype=X.dtype,
                meta=X._meta,
                name='%s-sliced-%s-%s' % (X._name, r, c),
            )

        chunks = X.chunks
        R, C = len(chunks[0]), len(chunks[1])
        X = \
            concatenate(
                [
                    concatenate(
                        [
                            make_block(r, c)
                            for c in range(C)
                        ],
                        axis=1,
                        allow_unknown_chunksizes=True,
                    )
                    for r in range(R)
                ],
                allow_unknown_chunksizes=True,
            )

        self._X = X
        # TODO: {{obs,var}{m,p},uns}

        # Bunt on uns for now, and get test cases.
        self._uns = adata_ref._uns.copy()

    def _create_axis_arrays(self, axis, vals_raw):
        if is_dask(self.obs) or is_dask(self.var):
            # This is a little ugly.  We have a child object that reads cousin attributes
            # on the parent: obs_names and var_names.
            # If obs_names or var_names are delayed, we don't want to
            # wait for the whole AnnData to compute,
            # because we will have a circularity problem.  Plus it may never.
            def mk_axis_arrays(obs, var, vals_raw):
                safe_copy = self._raw_copy()
                safe_copy._obs = obs
                safe_copy._var = var
                return AxisArrays(safe_copy, axis, vals=convert_to_dict(vals_raw))
            return daskify_call(mk_axis_arrays, self.obs, self.var, vals_raw)
        elif is_dask(vals_raw):
            return daskify_call(AxisArrays, self.obs_names, self.var_names, vals_raw)
        else:
            return AxisArrays(self, axis, vals=convert_to_dict(vals_raw))

    def _create_pairwise_arrays(self, axis, vals_raw):
        return PairwiseArrays(self, axis, vals=convert_to_dict(vals_raw))

    def _gen_repr(self, n_obs, n_vars) -> str:

        if self.isbacked:
            backed_at = f"backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"AnnData object with n_obs × n_vars = {n_obs} × {n_vars} {backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            obj = getattr(self, attr)
            if is_dask(obj):
                descr += f"\n    {attr}: {str(obj)}"
            elif hasattr(obj, 'keys'):
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
            else:
                if isinstance(obj, DataFrame):
                    descr += f"\n    {attr}: {str(obj.columns.tolist())[1:-1]}"
                else:
                    from sys import stderr
                    stderr.write(f'Unknown attr type {type(obj)}: {obj}\n')

        return descr

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value: Optional[Union[np.ndarray, sparse.spmatrix]]):
        raise NotImplementedError("The AnnDataDask.X is immutable!  "
                                  "To change attributes, make an updated copy() method...")

    def _normalize_indices(self, index: Optional[Index]) -> Tuple[slice, slice]:
        if isinstance(index, tuple):
            if len(index) == 1:
                oix = index[0]
                vix = slice(None)
            elif len(index) == 2:
                oix, vix = index
            else:
                raise ValueError("Slice must be on 1 or 2 dimensions, not %d: %s" % (len(index), index))
        else:
            oix = index
            vix = slice(None)
        return oix, vix
        #return _normalize_indices(index, self.obs_names, self.var_names)

    def __getitem__(self, index: Index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return self.__class__(self, oidx=oidx, vidx=vidx, asview=True)

    def to_dask_delayed(self, *args, _debug:bool=False, **kwargs):
        if self.is_view:
            def _compute_anndata_view(adata_ref, oidx, vidx):
                return adata_ref.__class__(adata_ref, oidx=oidx, vidx=vidx, asview=True)
            virtual = daskify_call(_compute_anndata_view, self._adata_ref.to_dask_delayed(), self._oidx, self._vidx)
            return virtual

        def _compute_anndata(X, **raw_attr_value_pairs):
            # Construct an AnnData at a low-level,
            # swapping out each of the attributes specified.
            an = AnnData.__new__(AnnData)
            for key, value in raw_attr_value_pairs.items():
                setattr(an, key, value)
            if hasattr(X, "tocsr"):
                X = X.tocsr()
            an._X = X
            an._dask = False
            return an

        # Passing the attribute this way will automatically put them into
        # the graph in parallel:
        attribute_value_pairs = self.__dict__.copy()

        if _debug:
            # Rather than compute everything in dask,
            # iterate through the attributes and compute each individually.
            for key, value in (attribute_value_pairs.items()):
                if hasattr(value, "compute"):
                    try:
                        value_computed = value.compute()
                        attribute_value_pairs[key] = value_computed
                    except Exception as e:
                        # Break here to examine failures.
                        logger.error("Error computing %s: %s" % (key, e))
                        value_computed = value.compute()
                        pass

        virtual = daskify_call(_compute_anndata, self.X, **attribute_value_pairs)
        return virtual

    def compute(self, *args, _debug: bool = False, **kwargs):
        return AnnData(
            X=self.X.compute(),
            obs=self.obs.compute(),
            var=self.var.compute(),
        )
        # virtual = self.to_dask_delayed(*args, _debug=_debug, **kwargs)
        # real = virtual.compute(*args, **kwargs)
        # return real

    # NOTE: We override the property accessor but not set/delete.
    # The .uns is immutable on AnnDataDask.  Use .copy_with_changes(...)
    # to make an alternate AnnDataDask with an updated uns.
    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        return super().uns

        """
        import anndata._core
        if self.is_view:
            def uns_overload_and_dictview(uns1):
                self_safe_copy = self._raw_copy()
                setattr(self_safe_copy, "_uns", uns1)
                try:
                    uns2 = anndata._core.anndata._overloaded_uns(self)
                    return DictView(uns2, view_args=(self_safe_copy, "uns"))
                except Exception as e:
                    logger.error("Error calculating uns on a view!", exc_info=e)
                    raise e
            uns = daskify_call(uns_overload_and_dictview, self._uns)
        else:
            uns = daskify_call(anndata._core.anndata._overloaded_uns, self)
        return uns
        """

    def _check_dimensions(self, key=None):
        # These checks can't occur until the data is vivified.
        pass

    def copy(self, filename: Optional[PathLike] = None) -> "AnnData":
        """Full copy, optionally on disk."""
        if not self.isbacked:
            # TODO: We should _only_ have views in the dask=True case b/c otherwise backed = True.
            # See if this logic can simplify.
            # Possibly we
            if self.is_view:
                # TODO: How do I unambiguously check if this is a copy?
                # Subsetting this way means we don’t have to have a view type
                # defined for the matrix, which is needed for some of the
                # current distributed backend.
                X = _subset(self._adata_ref.X, (self._oidx, self._vidx)).copy()
            else:
                X = self.X.copy()
            # TODO: Figure out what case this is:
            if X is not None:
                dtype = X.dtype
                if (not (X.shape is self.shape)) and (is_dask(X.shape) or is_dask(self.shape)):
                    def x_reshape(X, Xshape, anndata_shape):
                        # This is a little ugly b/c we need the X.shape to compute
                        # before we work on the X that has it.
                        X.shape = Xshape
                        X.reshape(anndata_shape)
                    X = daskify_call(x_reshape, X, X.shape, self.shape)
                elif X.shape != self.shape:
                    X = X.reshape(self.shape)
            else:
                dtype = "float32"
            return AnnData(
                X=X,
                obs=self.obs.copy(),
                var=self.var.copy(),
                # deepcopy on DictView does not work and is unnecessary
                # as uns was copied already before
                uns=self._uns.copy()
                if isinstance(self.uns, DictView)
                else deepcopy(self._uns),
                obsm=self.obsm.copy(),
                varm=self.varm.copy(),
                obsp=self.obsp.copy(),
                varp=self.varp.copy(),
                raw=self.raw.copy() if self.raw is not None else None,
                layers=self.layers.copy(),
                dtype=dtype,
            )
        else:
            # A normal backed/dask object
            from anndata._io import read_h5ad

            if filename is None:
                raise ValueError(
                    "To copy an AnnData object in backed mode, "
                    "pass a filename: `.copy(filename='myfilename.h5ad')`."
                )
            mode = self.file._filemode
            self.write(filename)
            return read_h5ad(filename, backed=mode, dask=True)

    def _raw_copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        for attr, val in self.__dict__.items():
            setattr(new, attr, val)
        return new

    def copy_with_changes(self: "AnnDataDask", **kwargs):
        kw = dict(
            X=self.X,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            obsm=self.obsm,
            varm=self.varm,
            obsp=self.obsp,
            varp=self.varp,
            layers=self.layers,
            raw=self.raw,
            dtype=self._dtype,
            shape=self.shape
        )
        kw.update(**kwargs)
        return self.__class__(**kw)


@_gen_dataframe.register(DataFrame)
def _(anno, length, index_names):
    anno = anno.copy()
    if not is_string_dtype(anno.index):
        warnings.warn("Transforming to str index.", ImplicitModificationWarning)
        anno.index = anno.index.astype(str)
    return anno


def daskify_iloc(df, idx):
    # Now works in dask.
    return df.iloc[idx]
    """
    def call_iloc(df_, idx_):
        return df_.iloc[idx_]
    meta = df._meta
    if meta is None:
        pass
    df = daskify_call_return_df(call_iloc, df, idx, _dask_meta=meta)
    return df
    """


def daskify_get_len_given_index(index: slice, orig_len: int):
    if isinstance(index, Series):
        if index.dtype == np.dtype(bool):
            # Size will depend on actual values in each partition; unknown at build time
            pass
        else:
            # If this is a Series of scalars that each select a row, the number of
            # output rows will match the size of the Series.
            if hasattr(index, "_len"):
                return getattr(index, "_len")

    def get_size(index_, orig_len_):
        if isinstance(index_, slice):
            len(range(*index_.indices(orig_len_)))
        elif isinstance(index_, pd.Series):
            if index_.dtype == np.dtype(bool):
                return index_.sum()
            else:
                return index_.size

    return daskify_call(get_size, index, orig_len)

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
    if is_dask(obj):
        return daskify_call(call_method, obj, method_name, *args, _dask_len=_dask_len, **kwargs)
    else:
        return getattr(obj, method_name)(*args, **kwargs)


def daskify_calc_shape(old_shape, one_slice_per_dim):
    def get_new_len(dim_len, dim_slice):
        if isinstance(dim_slice, slice):
            start = 0 if dim_slice.start is None else dim_slice.start
            stop = dim_len if dim_slice.stop is None else dim_slice.stop
            step = 1 if dim_slice.step is None else dim_slice.step
            if start < 0:
                start += dim_len
                assert start >= 0
            if stop < 0:
                stop += dim_len
                assert stop >= 0
            if stop < start:
                m, M = stop, start
                start, stop = m, M
                step = -step
            stop = min(stop, dim_len)
            return (stop - start) // abs(step)
        elif isinstance(dim_slice, np.ndarray):
            if dim_slice.dtype == np.dtype(bool):
                assert dim_slice.shape == (dim_len,)
                return dim_slice.sum()
            else:
                return len(dim_slice)
        else:
            raise Exception('Unexpected slice: %s' % str(dim_slice))

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
    if "_dask_output_types" not in kwargs:
        kwargs["_dask_output_types"] = (list, np.array, pd.Series)
    if "_dask_len" not in kwargs:
        kwargs["_dask_len"] = None
    return dask.array.from_delayed(
        daskify_call(f, *args, **kwargs),
        shape=_dask_shape,
        dtype=_dask_dtype,
        meta=_dask_meta,
    )


def daskify_call_return_df(f: callable, *args, _dask_len=None, _dask_meta=None, _dask_output_types=pd.DataFrame, **kwargs):
    return dd.from_delayed(
        daskify_call(f, *args, _dask_len=_dask_len, _dask_output_types=_dask_output_types, **kwargs),
        meta=_dask_meta,
        verify_meta=True
    )
