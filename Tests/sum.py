
import numpy

a = numpy.arange(10000000)
print(a.sum())

import thunklib
a = thunklib.thunk(a)
print(a.sum())
