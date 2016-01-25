from distutils.core import setup, Extension

package_name = "nplazy" #if you change this you also need to change the init function in nplazy.c

setup(
    name=package_name,
    version='1.0',
    description='Lazy NumPy Arrays.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        sources=['nplazy.c', 'lazyarray.c'])])

