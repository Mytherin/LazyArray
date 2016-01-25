
#include "lazyarray.h"

static char module_docstring[] =
    "This module provides lazy NumPy arrays.";
static char lazyarray_docstring[] =
    "nplazy.lazyarray(array) => Creates a lazy array from a numpy array.";

static PyMethodDef module_methods[] = {
    {"lazyarray", PyLazyArray_FromArray, METH_O, lazyarray_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnplazy(void);
PyMODINIT_FUNC initnplazy(void)
{   
    PyLazyArray_Init();

    //initialize module
    PyObject *m = Py_InitModule3("nplazy", module_methods, module_docstring);
    if (m == NULL)
        return;
}
