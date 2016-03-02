
import numpy
import nplazy
import time

element_count = 10000000

numpy.random.seed(137)
a = nplazy.lazyarray(numpy.random.randint(0,100,element_count))

start = time.time()
d =  a * a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a
d.materialize()
end = time.time()

print("Lazy Evaluation (No Intermediates)", end - start)

numpy.random.seed(137)
a = numpy.random.randint(0,100,element_count)


start = time.time()
d =  a * a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a
end = time.time()

print("Eager Evaluation (Many Intermediates)", end - start)

import thunklib

numpy.random.seed(137)
a = thunklib.thunk(numpy.random.randint(0,100,element_count))


start = time.time()
d = a * a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a* a * a
d.evaluate()
end = time.time()

print("Pipelined Evaluation", end - start)