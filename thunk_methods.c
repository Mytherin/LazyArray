
#include "thunk.h"
#include "thunktypes.h"

/*
static ssize_t PyObject_Cardinality(PyObject *v) {
	if (PyThunk_CheckExact(v)) {
		return PyThunk_Cardinality(v);
	}
	return 1;
}

static int PyObject_ctype(PyObject* v) {
	if (PyThunk_CheckExact(v)) {
		return PyThunk_Type(v);
	}
	return NPY_INT32;
}

#define thunk_unary_pipeline_function(function)                                                       \
static PyObject*                                                                                      \
thunk_lazy##function(PyObject *self, PyObject *unused) {                                              \
	(void) unused;                                                                                    \
	ssize_t cardinality = PyObject_Cardinality(self);                                                 \
	int type = FloatUnaryTypeResolution(PyObject_ctype(self));                                        \
	PyObject *op = PyThunkUnaryPipeline_FromFunction(pipeline_##function, self);                      \
	return PyThunk_FromExactOperation(op, cardinality, type);                                         \
}

thunk_unary_pipeline_function(sqrt)
*/

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

struct PyMethodDef thunk_methods[] = {
    /*{"sqrt", (PyCFunction)thunk_lazysqrt, METH_NOARGS,"sqrt() => "},*/
    {"evaluate", (PyCFunction)_thunk_evaluate, METH_NOARGS,"evaluate() => "},
    {"isevaluated", (PyCFunction)_thunk_isevaluated, METH_NOARGS,"isevaluated() => "},
    {NULL}  /* Sentinel */
};

void initialize_thunk_methods(void) {
    import_array();
}