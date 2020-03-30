from math import floor, sqrt
from pandas import DataFrame as DF
from scipy import sparse

from anndata import AnnData

def make_test_h5ad(R=100, C=200):
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

    X = sparse.random(R, C, format="csc", density=0.1, random_state=123)

    obs = DF([
        {
            'label': f'row {r}',
            'idx²': r**2,
            'Prime': all([
                r % f != 0
                for f in range(2, floor(sqrt(r)))
            ]),
        }
        for r in range(R)
    ])

    var = DF([
        {
            'name': spreadsheet_column(c),
            'sqrt(idx)': sqrt(c),
        }
        for c in range(C)
    ])

    ad = AnnData(X=X, obs=obs, var=var)
    return ad
