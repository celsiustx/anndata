from typing import List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse

import anndata
from anndata._core.aligned_mapping import AxisArrays, AxisArraysView, LayersView, \
    PairwiseArraysView
from anndata_dask import is_dask

DIFF_PARTS = ['X', 'obs', 'var', 'uns', 'obsm', 'varm', 'layers', 'raw',
              'shape', 'obsp', 'varp']


def diff_summary(a: anndata.AnnData, b: anndata.AnnData, select_parts: Optional[List[str]] = None):
    """
    Emit a dictionary of differences between two AnnData objects.
    This is meant to be human readable, for debugging.  The values are summary text
    strings when there is a difference.

    Note that this will compute() any dask elements examined.

    :param a: AnnData
    :param b: AnnData
    :param select_parts: Optional[List[str]] Limit the diff to specific attributes.
    :return: dict A dictionary keyed by attribute with differences, containing a text description.
    """
    changes = {}

    if select_parts is None:
        diff_parts = DIFF_PARTS
    else:
        diff_parts = select_parts

    for part in diff_parts:
        aa = getattr(a, part)
        bb = getattr(b, part)

        if is_dask(aa):
            try:
                aa = aa.compute()
            except Exception as e:
                changes[part] = "%s does not compute for A %s!: %s" % (part, aa, e)
                continue
        if is_dask(bb):
            try:
                bb = bb.compute()
            except Exception as e:
                changes[part] = "%s does not compute for B %s!: %s" % (part,bb,  e)
                continue

        if aa is None or bb is None:
            if aa is None and bb is None:
                pass
            else:
                changes[part] = "%s => %s" % (aa, bb)
            continue

        # The class used for X may be different but still hold identical data.
        if isinstance(aa, anndata._core.sparse_dataset.SparseDataset):
            aa = aa.value
        if isinstance(bb, anndata._core.sparse_dataset.SparseDataset):
            bb = bb.value

        if issparse(aa) and issparse(bb):
            cnt = (aa != bb).nnz
            if cnt != 0:
                changes[part] = "count of differences between sparse arrays: %s" % cnt
        elif type(aa) != type(bb):
            changes[part] = "class mismatch: %s => %s" % (aa.__class__, bb.__class__)
        elif isinstance(aa, pd.DataFrame):
            delta = diff_df(aa, bb)
            if delta is not None:
                changes[part] = str(delta) # Let pandas make a nice string.
        elif isinstance(aa, (list, tuple, AxisArrays, AxisArraysView, LayersView,
                             PairwiseArraysView)):
            # Default in most cases hopefully.
            if aa != bb:
                changes[part] = "differ: %s => %s" % (aa, bb)
        else:
            aa_str = _simplify_for_diff(aa)
            bb_str = _simplify_for_diff(bb)
            if aa_str != bb_str:
                changes[part] = "differ: %s => %s" % (aa_str, bb_str)
    return changes


def _simplify_for_diff(value):
    import anndata
    import json
    import pandas._libs.index
    from collections import OrderedDict
    from anndata._core.aligned_mapping import AxisArrays
    from anndata.compat._overloaded_dict import OverloadedDict

    if isinstance(value, OverloadedDict):
        return {k:v for k, v in value.items()}
    elif isinstance(value, (anndata._core.aligned_mapping.LayersBase,
                            anndata._core.aligned_mapping.PairwiseArrays,
                            dict,
                            OrderedDict)):
        value2 = {}
        dct = value if isinstance(value, dict) else value.__dict__
        for k, v in dct.items():
            if (isinstance(v, (anndata.AnnData, pd._libs.index.ObjectEngine)) or
                    (v is value) or
                    (v.__class__ == object)):
                continue
            try:
                v2 = _simplify_for_diff(v)
            except RecursionError as e:
                raise e
            value2[k] = v2
        value_str = json.dumps(value2)
        return value_str
    elif not hasattr(value, "__dict__"):
        return value
    else:
        raise ValueError("Cannot simplify %s" % value)


def diff_df(a: pd.DataFrame, b: pd.DataFrame):
    difference_locations = np.where(a != b)
    if len(difference_locations[0]) == 0 and len(difference_locations[1]) == 0:
        return None
    else:
        ne: pd.DataFrame = (a != b)
        ne_stacked = ne.stack()
        changed = ne_stacked[ne_stacked]
        changed_from = b.values[difference_locations]
        changed_to = a.values[difference_locations]
        delta = pd.DataFrame({'from': changed_from, 'to': changed_to},
                             index=changed.index)
        return delta
