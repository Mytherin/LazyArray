
import nplazy
import numpy
a = numpy.arange(5)
b = nplazy.lazyarray(a)
c = b + b
d = c * 2
print(type(d))
print(d)