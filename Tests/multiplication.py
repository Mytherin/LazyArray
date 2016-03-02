
import thunklib
import numpy

dtype_list = [ numpy.dtype('int8'), numpy.dtype('int16'), numpy.dtype('int32'), numpy.dtype('int64'), numpy.dtype('uint8'), numpy.dtype('uint16'), numpy.dtype('uint32'), numpy.dtype('uint64'), numpy.dtype('float32'), numpy.dtype('float64'), numpy.dtype('complex64') ]

for dtype in dtype_list:
	for dtype2 in dtype_list:
		print(dtype)
		print(dtype2)
		a = numpy.arange(5, dtype=dtype)
		b = numpy.arange(5, dtype=dtype2)
		print(a * b)
		a = thunklib.thunk(a)
		b = thunklib.thunk(b)
		print(a * b)

