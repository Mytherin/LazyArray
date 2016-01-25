
#include "lazyarray.h"

void PyLazyArray_Init() {
    if (PyType_Ready(&PyLazyArray_Type) < 0)
        return;
    import_array();
}

static PyObject* 
PyLazyArray_IsMaterialized(PyObject *lz, PyObject* args) {
    (void)args;
    if (PyLazyArray_ISMATERIALIZED(lz)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}
PyObject *PyLazyArray_AsArray(PyObject *self, PyObject *args) {
    (void)args;
    PyLazyArray_MATERIALIZE(self);
    PyObject *arr = (PyObject*)PyLazyArray_ASNUMPY(self);
    Py_INCREF(arr);
    return arr;
}

PyObject *PyLazyArray_FromArray(PyObject *unused, PyObject *input) {
    register PyLazyArrayObject *array;
    (void) unused;
    if (!PyArray_CheckExact(input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array as parameter.");
        return NULL;
    }
    array = (PyLazyArrayObject *)PyObject_MALLOC(sizeof(PyLazyArrayObject));
    if (array == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)array, &PyLazyArray_Type);
    array->nparr = (PyArrayObject*)PyArray_FromAny(input, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);
    array->materialized = true;
    array->operation = NULL;
    return (PyObject*)array;
}

PyObject *PyLazyArray_FromOperation(PyLazyFunctionOperation* operation) {
    register PyLazyArrayObject *array;
    array = (PyLazyArrayObject *)PyObject_MALLOC(sizeof(PyLazyArrayObject));
    if (array == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)array, &PyLazyArray_Type);
    array->nparr = NULL;
    array->materialized = false;
    array->operation = operation;
    return (PyObject*)array;
}

bool 
PyLazyArray_Materialize(PyLazyArrayObject *self) {
    if (self->materialized) return;
    PyObject *ret = (self->operation->function(self->operation->a, self->operation->b));
    if (ret == NULL) return false;
    self->nparr = (PyArrayObject*)ret;
    self->materialized = true;
    return true;
}

static PyObject *
_lazyarray_materialize(PyLazyArrayObject *self, PyObject *args) {
    (void) args;
    if (self->materialized) {
        Py_RETURN_NONE;
    }
    if (self->operation == NULL) {
        PyErr_SetString(PyExc_TypeError, "The lazy array is not materialized, and we don't know how to materialize it.");
        return NULL;
    }
    PyLazyArray_Materialize(self);
    Py_INCREF(self);
    return (PyObject*) self;
}

static PyMethodDef lazyarray_methods[] = {
    {"materialize", (PyCFunction)_lazyarray_materialize, METH_NOARGS,"materialize() => "},
    {"ismaterialized", (PyCFunction)PyLazyArray_IsMaterialized, METH_NOARGS,"ismaterialized() => "},
    {"asnumpyarray", (PyCFunction)PyLazyArray_AsArray, METH_NOARGS,"asnumpyarray() => "},
    {NULL}  /* Sentinel */
};


#define LAZYBINARYARRAYFUNC(tpe)                                                                                                                     \
static PyObject *                                                                                                                                    \
lazyarray_##tpe(PyObject *v, PyObject *w) {                                                                                                          \
    if (PyLazyArray_CheckExact(v)) {                                                                                                                 \
        PyLazyArray_MATERIALIZE(v);                                                                                                                  \
        v = (PyObject*)PyLazyArray_ASNUMPY(v);                                                                                                       \
    }                                                                                                                                                \
    if (PyLazyArray_CheckExact(w)) {                                                                                                                 \
        PyLazyArray_MATERIALIZE(w);                                                                                                                  \
        w = (PyObject*)PyLazyArray_ASNUMPY(w);                                                                                                       \
    }                                                                                                                                                \
    return PyArray_Type.tp_as_number->nb_##tpe(v, w);                                                                                                \
}                                                                                                                                                    \
static PyObject *                                                                                                                                    \
lazyarray_lazy##tpe(PyLazyArrayObject *v, PyLazyArrayObject *w) {                                                                                    \
    PyLazyFunctionOperation *op = (PyLazyFunctionOperation*)PyLazyFunction_FromFunction((PyCFunction)lazyarray_##tpe, (PyObject*)v, (PyObject*)w);   \
    return PyLazyArray_FromOperation(op);                                                                                                            \
}

LAZYBINARYARRAYFUNC(add)
LAZYBINARYARRAYFUNC(subtract)
LAZYBINARYARRAYFUNC(multiply)
LAZYBINARYARRAYFUNC(divide)


static PyObject *
lazyarray_str(PyLazyArrayObject *self)
{
    PyLazyArray_MATERIALIZE(self);
    assert(self->materialized);
    return PyArray_Type.tp_str((PyObject*)PyLazyArray_ASNUMPY(self));
}

static PyObject *
lazyarray_repr(PyLazyArrayObject *self)
{
    PyLazyArray_MATERIALIZE(self);
    assert(self->materialized);
    return PyArray_Type.tp_repr((PyObject*)PyLazyArray_ASNUMPY(self));
}

static PyNumberMethods lazyarray_as_number = {
    (binaryfunc)lazyarray_lazyadd,   /*nb_add*/
    (binaryfunc)lazyarray_lazysubtract,         /*nb_subtract*/
    (binaryfunc)lazyarray_lazymultiply,         /*nb_multiply*/
    (binaryfunc)lazyarray_lazydivide,         /*nb_divide*/
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

PyTypeObject PyLazyArray_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "ndlazyarray",
    sizeof(PyLazyArrayObject),
    0,
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    (reprfunc)lazyarray_repr,                   /* tp_repr */
    &lazyarray_as_number,                       /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)lazyarray_str,                    /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
     | Py_TPFLAGS_CHECKTYPES
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    "Lazy NumPy Array.",                        /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    lazyarray_methods,                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    PyType_GenericNew,                          /* tp_new */
    PyObject_Del,                               /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0, 
    0,
    0
};

PyObject*
PyLazyFunction_FromFunction(PyCFunction function, PyObject *a, PyObject *b) {
    PyLazyFunctionOperation *op = PyObject_MALLOC(sizeof(PyLazyFunctionOperation));
    if (op == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)op, &PyLazyFunctionOperation_Type);
    op->function = function;
    Py_XINCREF(a);
    op->a = a;
    Py_XINCREF(b);
    op->b = b;
    return (PyObject*)op;
}

static void
PyLazyFunctionOperation_dealloc(PyLazyFunctionOperation* self)
{
    Py_XDECREF(self->a);
    Py_XDECREF(self->b);
    self->ob_type->tp_free((PyObject*)self);
}

PyTypeObject PyLazyFunctionOperation_Type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "lazyarray.binaryfunction",
    sizeof(PyLazyFunctionOperation),
    0,
    (destructor)PyLazyFunctionOperation_dealloc,/* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "Lazy binary function.",                    /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    PyType_GenericNew,                          /* tp_new */
    PyObject_Del,                               /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0, 
    0,
    0
};
