

import numpy

npy = numpy.arange(2048, dtype=numpy.int32)
npy2 = numpy.repeat([2], 2048).astype(numpy.int32)

import thunklib
a = thunklib.thunk(npy)
c = thunklib.thunk(npy2)
print(a)
b = a * c
print(b)
print(b)
print(a)
print(c)