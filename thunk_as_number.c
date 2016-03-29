
#include "generated/thunk_lazy_functions.h"
#include "thunk.h"

static PyObject *
_thunk_lazypower_wrapper(PyThunkObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo)) {
    return thunk_lazypower((PyObject*) a1, o2);
}

static int
_thunk_nonzero(PyThunkObject *mp) {
    npy_intp n;

    n = PyThunk_Cardinality(mp);
    if (n == 1) {
        PyThunk_Evaluate(mp);
        return PyArray_DESCR(mp->storage)->f->nonzero(PyArray_DATA(mp->storage), mp->storage);
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array " \
                        "with more than one element is ambiguous. " \
                        "Use a.any() or a.all()");
        return -1;
    }
}

PyNumberMethods thunk_as_number = {
    (binaryfunc)thunk_lazyadd,   /*nb_add*/
    (binaryfunc)thunk_lazysubtract,         /*nb_subtract*/
    (binaryfunc)thunk_lazymultiply,         /*nb_multiply*/
    (binaryfunc)thunk_lazydivide,         /*nb_divide*/
    (binaryfunc)thunk_lazyremainder,         /*nb_remainder*/
    0,         /*nb_divmod*/
    (ternaryfunc)_thunk_lazypower_wrapper,         /*nb_power*/
    (unaryfunc)thunk_lazynegative,         /*nb_negative*/
    0,         /*nb_positive*/
    (unaryfunc)thunk_lazyabsolute,         /*nb_absolute*/
    (inquiry)_thunk_nonzero,         /*nb_nonzero*/
    (unaryfunc)thunk_lazyinvert,         /*nb_invert*/
    (binaryfunc)thunk_lazyleft_shift,         /*nb_lshift*/
    (binaryfunc)thunk_lazyright_shift,         /*nb_rshift*/
    (binaryfunc)thunk_lazybitwise_and,         /*nb_and*/
    (binaryfunc)thunk_lazybitwise_xor,         /*nb_xor*/
    (binaryfunc)thunk_lazybitwise_or,         /*nb_or*/
    0,         /*nb_coerce*/
    (unaryfunc)thunk_lazyint,         /*nb_int*/
    (unaryfunc)thunk_lazylong,         /*nb_long*/
    (unaryfunc)thunk_lazyfloat,         /*nb_float*/
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
    (binaryfunc)thunk_lazyfloor_divide,         /* nb_floor_divide */
    (binaryfunc)thunk_lazytrue_divide, /* nb_true_divide */
    0,                           /* nb_inplace_floor_divide */
    0,                           /* nb_inplace_true_divide */
    0,          /* nb_index */
};
