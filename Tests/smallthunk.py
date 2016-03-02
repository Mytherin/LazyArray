
import thunklib
import numpy
a = numpy.arange(5, dtype=numpy.double)
b = thunklib.thunk(a)
c = b * b 
print(type(c))
print(c)
