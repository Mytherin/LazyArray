
#include "thunk.h"
#include "thunktypes.h"
#include "initializers.h"

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

#define thunk_operator(operator)                                                                      \
static PyObject*                                                                                      \
thunk_lazy##operator(PyObject *v, PyObject *w) {                                                      \
	ssize_t left_cardinality = PyObject_Cardinality(v);                                               \
	ssize_t right_cardinality = PyObject_Cardinality(w);                                              \
	ssize_t cardinality;                                                                              \
	if (left_cardinality != right_cardinality && left_cardinality > 1 && right_cardinality > 1) {     \
        PyErr_SetString(PyExc_TypeError, "Invalid cardinalities.");                                   \
        return NULL;                                                                                  \
	}                                                                                                 \
	cardinality = max(left_cardinality, right_cardinality);                                           \
	int type = StandardBinaryTypeResolution(PyObject_ctype(v), PyObject_ctype(w));                    \
	PyObject *op = PyThunkBinaryPipeline_FromFunction(pipeline_##operator, v, w, PyArray_Type.tp_as_number->nb_##operator); \
	return PyThunk_FromExactOperation(op, cardinality, type);                                         \
}

thunk_operator(multiply)
thunk_operator(add)
thunk_operator(subtract)
thunk_operator(divide)


PyNumberMethods thunk_as_number = {
    (binaryfunc)thunk_lazyadd,   /*nb_add*/
    (binaryfunc)thunk_lazysubtract,         /*nb_subtract*/
    (binaryfunc)thunk_lazymultiply,         /*nb_multiply*/
    (binaryfunc)thunk_lazydivide,         /*nb_divide*/
    0,         /*nb_remainder*/
    0,         /*nb_divmod*/
    0,         /*nb_power*/
    0,         /*nb_negative*/
    0,         /*nb_positive*/
    0,         /*nb_absolute*/
    0,         /*nb_nonzero*/
    0,         /*nb_invert*/
    0,         /*nb_lshift*/
    0,         /*nb_rshift*/
    0,         /*nb_and*/
    0,         /*nb_xor*/
    0,         /*nb_or*/
    0,         /*nb_coerce*/
    0,         /*nb_int*/
    0,         /*nb_long*/
    0,         /*nb_float*/
    0,         /*nb_oct*/
    0,         /*nb_hex*/
    0,                           /*nb_inplace_add*/
    0,                           /*nb_inplace_subtract*/
    0,                           /*nb_inplace_multiply*/
    0,                           /*nb_inplace_divide*/
    0,                           /*nb_inplace_remainder*/
    0,                           /*nb_inplace_power*/
    0,                           /*nb_inplace_lshift*/
    0,                           /*nb_inplace_rshift*/
    0,                           /*nb_inplace_and*/
    0,                           /*nb_inplace_xor*/
    0,                           /*nb_inplace_or*/
    0,         /* nb_floor_divide */
    0, /* nb_true_divide */
    0,                           /* nb_inplace_floor_divide */
    0,                           /* nb_inplace_true_divide */
    0,          /* nb_index */
};

void initialize_thunk_as_number(void) {
    import_array();
}