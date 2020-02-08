
def test_dask():
    path = '/Users/ryan/c/celsius/notebooks/data/Fib.imputed.h5ad'
    from anndata import read_h5ad
    ad = read_h5ad(
        path,
        dask=True
    )
    print(ad.obs.head())
    from anndata._io.sql import write_sql
    from sqlalchemy import create_engine
    engine = create_engine('postgres:///sc')
    write_sql(ad, 'test', engine, if_exists='replace')


#def test_write_dask():
