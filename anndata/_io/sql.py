
from pandas import DataFrame as DF
from dask.dataframe import DataFrame as DDF

def read_sql(ad, table_name_prefix, engine):
    pass


def write_sql(ad, table_name_prefix, engine, if_exists=None):
    print(f'ad: {type(ad)}')
    obs = ad.obs
    var = ad.var
    X = ad.X

    from dask.array import from_array
    X_arr = from_array(X)

    write_table(obs, f'{table_name_prefix}_obs', engine, if_exists=if_exists)
    write_table(var, f'{table_name_prefix}_var', engine, if_exists=if_exists)


def write_table(df, table_name, engine, if_exists=None):
    if isinstance(df, DF):
        df.to_sql(table_name, engine, if_exists=if_exists)
    elif isinstance(df, DDF):
        from dask import compute, delayed
        ddf_to_sql = delayed(DF.to_sql)
        df._meta.to_sql(table_name, engine, if_exists=if_exists)
        res = [ ddf_to_sql(d, table_name, engine, if_exists='append') for d in df.to_delayed() ]
        compute(*res)
    else:
        raise Exception(f'Unrecognized DataFrame type {type(df)}:\n{df}')
