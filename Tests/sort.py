
import numpy
import thunklib


numpy.random.seed(137)
a = numpy.random.randint(0, 100, size=(200000,)).astype(numpy.float64)
b = thunklib.thunk(a * 2)

b.sort()
c = b

numpy.random.seed(137)
a = numpy.random.randint(0, 100, size=(200000,)).astype(numpy.float64) * 2
a.sort()

if numpy.array_equal(a, c.asnumpyarray()):
    print("Success!")
else:
    print("Failure! Different arrays.")
    print("NumPy", a)
    print("Thunk", c.asnumpyarray())