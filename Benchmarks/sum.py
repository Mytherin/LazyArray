
import thunklib
import numpy
import time

element_count = 100000000

numpy.random.seed(137)
a = thunklib.thunk(numpy.random.randint(0,100,element_count))


start = time.time()
d =  numpy.sum((a * a * a * a * a).asnumpyarray())
end = time.time()

print("NumPy Evaluation", end - start)

numpy.random.seed(137)
a = thunklib.thunk(numpy.random.randint(0,100,element_count))


start = time.time()
d = (a * a * a * a * a).sum()
d.evaluate()
end = time.time()

print("Pipelined Evaluation", end - start)