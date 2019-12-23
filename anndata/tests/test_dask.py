
def test_dask():
    path = '/Users/ryan/c/celsius/notebooks/data/Fib.imputed.h5ad'
    from anndata import read_h5ad
    ad = read_h5ad(
        path,
        dask=True
    )
    #print(ad.obs.head())
