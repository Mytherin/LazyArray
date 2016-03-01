
import thunklib
import numpy
a = numpy.arange(5, dtype=numpy.int32)
b = thunklib.thunk(a)
c = b * b
print(type(c))
print(c)
