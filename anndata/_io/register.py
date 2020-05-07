import numpy as np
from psycopg2.extensions import adapt, register_adapter

def adapt_float(f):
    return adapt(float(f))

def adapt_int(i):
    return adapt(int(i))

def register_numerics():
    register_adapter(np.int32, adapt_int)
    register_adapter(np.int64, adapt_int)
    register_adapter(np.float32, adapt_float)
    register_adapter(np.float64, adapt_float)
