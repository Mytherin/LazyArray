
import thunklib
import numpy

dtype_list = [ numpy.dtype('int8'), numpy.dtype('int16'), numpy.dtype('int32'), numpy.dtype('int64'), numpy.dtype('uint8'), numpy.dtype('uint16'), numpy.dtype('uint32'), numpy.dtype('uint64'), numpy.dtype('float32'), numpy.dtype('float64'), numpy.dtype('complex64') ]

errors = 0
tests = 0

for dtype in dtype_list:
    for dtype2 in dtype_list:
        a = numpy.arange(5, dtype=dtype)
        b = numpy.arange(5, dtype=dtype2)
        numpy_result = (a * b).__str__()
        a = thunklib.thunk(a)
        b = thunklib.thunk(b)
        thunk_result = (a * b).__str__()
        if thunk_result != numpy_result:
            errors += 1
            print("Failed test %s*%s" % (dtype, dtype2))
        tests += 1


print("\nFailed %d/%d tests." % (errors, tests) if errors > 0 else "Success!")


