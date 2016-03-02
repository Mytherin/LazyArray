
import thunklib
import numpy
a = numpy.arange(5, dtype=numpy.int32)
b = thunklib.thunk(a)
print(numpy.sqrt(b))
