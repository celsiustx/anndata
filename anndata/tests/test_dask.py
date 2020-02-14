
def test_dask():
    path = '/Users/ryan/c/celsius/notebooks/data/Fib.imputed.1k.h5ad'
    from anndata import read_h5ad
    ad = read_h5ad(path, backed=True, dask=True)
    print(ad.obs.head())
    from anndata._io.sql import write_sql
    from sqlalchemy import create_engine
    engine = create_engine('postgres:///sc')
    write_sql(ad, 'test_dask', engine, if_exists='replace')


#def test_write_dask():
