from numpy import array

from anndata._io.h5chunk import Chunk, Range


def test_pos():
    arr = array([ int(str(i)*2) for i in range(100) ])
    chunk = Chunk.whole_array(arr)
    ranges = chunk.ranges
    assert ranges == (Range(0, 0, 100, 100, 1),)

    R = 20
    C = 20
    arr = array([
        [ R*r+c for c in range(C) ]
        for r in range(R)
    ])
    chunk = Chunk.build((2, 3), ((4, 6), (12, 16)), (R, C))
    assert chunk == Chunk((
        Range(2,  4,  6, R, C),
        Range(3, 12, 16, C, 1),
    ))
