from math import floor, sqrt
from pandas import DataFrame as DF
from scipy import sparse

from anndata import AnnData


def digits(n, b, empty_zero=False, significant_leading_zeros=True):
    '''Convert a number to an array of base-`b` digits, with a few toggle-able
    behaviors.

    The default parameters output an analogue to "spreadsheet-column" order, e.g. in
    base-26 you get the equivalent of: "A","B",…,"Z","AA","AB",… (but with arrays of
    integers ∈[0,26) instead of strings of letters).

    :param empty_zero: when True, start counting from an empty array at zero
            (otherwise, zero will be mapped to [0])
    :param significant_leading_zeros: when True, enumerate non-empty natural-number
            arrays containing the elements ∈[0,b)
    '''
    if significant_leading_zeros:
        if not empty_zero: return digits(n+1, b, empty_zero=True, significant_leading_zeros=significant_leading_zeros)
        bases = [1]
        while n >= bases[-1]:
            n -= bases[-1]
            bases.append(bases[-1]*b)
        n_digits = digits(n, b, empty_zero=True, significant_leading_zeros=False)
        return [0]*(len(bases)-1-len(n_digits)) + n_digits
    else:
        return digits(n // b, b, empty_zero=True, significant_leading_zeros=False) + [n%b] if n else [] if empty_zero else [0]


def spreadsheet_column(idx):
    return ''.join([ chr(ord('A')+digit) for digit in digits(idx, 26) ])


def make_obs(r, R=None):
    if R is None:
        r, R = 0, r
    return DF([
        {
            'label': f'row {i}',
            'idx²': i**2,
            'Prime': i >= 2 and all([
                i % f != 0
                for f in range(2, floor(sqrt(i)))
            ]),
        }
        for i in range(r, R)
    ])


def make_var(c, C=None):
    if C is None:
        c, C = 0, c
    return DF([
        {
            'name': spreadsheet_column(i),
            'sqrt(idx)': sqrt(i),
        }
        for i in range(c, C)
    ])


def make_test_h5ad(R=100, C=200):
    if isinstance(R, int): R = (0, R)
    (r, R) = R
    if isinstance(C, int): C = (0, C)
    (c, C) = C
    M, N = R-r, C-c
    X = sparse.random(M, N, format="csc", density=0.1, random_state=123)
    obs = make_obs(r, R)
    var = make_var(c, C)
    ad = AnnData(X=X, obs=obs, var=var)
    return ad
