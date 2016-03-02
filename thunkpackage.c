
#include "thunk.h"
#include "initializers.h"

static char module_docstring[] =
    "This module provides THUNKS.";
static char thunk_docstring[] =
    "thunk.thunk(array) => Creates a thunk array from a numpy array.";

static PyMethodDef module_methods[] = {
    {"thunk", PyThunk_FromArray, METH_O, thunk_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initthunklib(void);
PyMODINIT_FUNC initthunklib(void)
{   
    PyThunk_Init();
    initialize_thunk_as_number();
    initialize_thunk_methods();
    
    //initialize module
    PyObject *m = Py_InitModule3("thunklib", module_methods, module_docstring);
    if (m == NULL)
        return;
}
