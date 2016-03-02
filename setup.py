from distutils.core import setup, Extension

package_name = "thunklib" #if you change this you also need to change the init function in thunkpackage.c

setup(
    name=package_name,
    version='1.0',
    description='Lazy NumPy Arrays.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        depends=['blockmask.h', 'thunk.h', 'thunktypes.h', 'thunkops.h', 'thunk_config.h', 'thunktypes.h', 'generated/thunkops_binarypipeline.h'],
        sources=['blockmask.c', 'thunk.c', 'thunkpackage.c', 'thunk_as_number.c', 'thunkops.c', 'generated/thunkops_binarypipeline.c', 'thunkops_unarypipeline.c', 'thunk_binarypipeline.c', 'thunk_unarypipeline.c', 'thunk_binaryfunction.c', 'thunk_unaryfunction.c', 'generated/thunktypes.c'
        ]
        )])

