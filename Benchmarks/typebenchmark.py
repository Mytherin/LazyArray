
import numpy
import time
import thunklib

element_count = 100000000

numpy.random.seed(137)
a = numpy.random.randint(0,100,element_count).astype(numpy.int32)
b = numpy.random.randint(0,100,element_count).astype(numpy.float32)

start = time.time()
d = a * b
end = time.time()

print("NumPy Evaluation", end - start)


numpy.random.seed(137)

a = thunklib.thunk(numpy.random.randint(0,100,element_count).astype(numpy.int32))
b = thunklib.thunk(numpy.random.randint(0,100,element_count).astype(numpy.float32))

start = time.time()
d = a * b
d.evaluate()
end = time.time()

print("Pipelined Evaluation", end - start)