
#include "thunk.h"
#include "thunktypes.h"

#include "generated/thunk_lazy_functions.h"


static PyObject *
_thunk_evaluate(PyThunkObject *self, PyObject *args) {
    (void) args;
	if (PyThunk_Evaluate(self) == NULL) {
		return NULL;
	}
    PyThunk_Evaluate(self);
    Py_RETURN_NONE;
}

static PyObject *
_thunk_isevaluated(PyThunkObject *self, PyObject *args) {
    (void) args;
    if (PyThunk_IsEvaluated(self)) {
    	Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
_thunk_asarray(PyThunkObject *self, PyObject *args) {
    (void) args;
    PyObject *arr = PyThunk_AsArray((PyObject*) self);
    Py_INCREF(arr);
    return arr;
}

struct PyMethodDef thunk_methods[] = {
    {"sqrt", (PyCFunction)thunk_lazysqrt, METH_NOARGS,"sqrt() => "},
    {"sum", (PyCFunction)thunk_lazysum, METH_NOARGS, "sum() => "},
    {"sort", (PyCFunction)thunk_lazysort, METH_NOARGS, "sort() =>"},
    {"evaluate", (PyCFunction)_thunk_evaluate, METH_NOARGS,"evaluate() => "},
    {"isevaluated", (PyCFunction)_thunk_isevaluated, METH_NOARGS,"isevaluated() => "},
    {"asnumpyarray", (PyCFunction)_thunk_asarray, METH_NOARGS,"asnumpyarray() => "},
    {NULL}  /* Sentinel */
};

void initialize_thunk_methods(void) {
    import_array();
}