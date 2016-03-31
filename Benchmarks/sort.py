
import numpy
import time

element_count = 100000000

array_list = []

numpy.random.seed(137)
a = numpy.random.randint(0,100,element_count)

start = time.time()
d =  numpy.sort(a * a * a)
end = time.time()

array_list.append(d)

print("NumPy Evaluation", end - start)

import thunklib

numpy.random.seed(137)
a = thunklib.thunk(numpy.random.randint(0,100,element_count))


start = time.time()
d = a * a * a
d.sort()
d.evaluate()
end = time.time()

array_list.append(d.asnumpyarray())

print("Pipelined Evaluation", end - start)


numpy.random.seed(137)
a = thunklib.thunk(numpy.random.randint(0,100,element_count))


start = time.time()
d =  (a * a * a).asnumpyarray()
d.sort()
end = time.time()

print("Unary Sort, Pipelined Multiplication", end - start)