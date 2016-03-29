from distutils.core import setup, Extension

import os
os.system('mkdir generated')
os.system('python gencode.py')

package_name = "thunklib" #if you change this you also need to change the init function in thunkpackage.c

generated_sources = ['generated/operations.c', 'generated/thunk_lazy_functions.c', 'generated/thunkops_pipeline.c']
generated_headers = ['generated/operations.h', 'generated/thunk_lazy_functions.h', 'generated/thunkops_pipeline.h']


from os import environ

debug = False
import sys
for v in sys.argv:
    if 'debug' in v:
        debug = True
        sys.argv.remove(v)
        break

if debug:
    environ['CFLAGS'] = (environ['CFLAGS'] if 'CFLAGS' in environ else '') + '-Wall -Wno-unused-function -O0 -g'
else:
    environ['CFLAGS'] = (environ['CFLAGS'] if 'CFLAGS' in environ else '') + '-Wno-unused-function'

import numpy


setup(
    name=package_name,
    version='1.0',
    description='Lazy NumPy Arrays.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        include_dirs=[numpy.get_include()],
        depends=['blockmask.h', 'thunk.h', 'thunktypes.h', 'thunkops.h', 'thunk_config.h', 'thunk_sort.h', 'thunktypes.h', 'ufunc_pipeline.h'] + generated_headers,
        sources=['blockmask.c', 'thunk.c', 'thunkpackage.c', 'thunk_as_number.c', 'thunkops.c', 'thunk_binarypipeline.c', 'thunk_sort.c', 'thunk_aggregationpipeline.c', 'thunk_unarypipeline.c', 'thunk_binaryfunction.c', 'thunk_unaryfunction.c', 'thunk_methods.c', 'ufunc_pipeline.c'] + generated_sources
        )])

