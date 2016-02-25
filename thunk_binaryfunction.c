
#include "thunkops.h"

PyObject*
PyThunkBinaryFunction_FromFunction(BinaryFunction function, PyObject *left, PyObject *right) {
    PyThunkOperation_BinaryFunction *op = PyObject_MALLOC(sizeof(PyThunkOperation_BinaryFunction));
    if (op == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)op, &PyThunkBinaryFunction_Type);
    op->function = function;
    Py_XINCREF(left);
    op->left = left;
    Py_XINCREF(right);
    op->right = right;
    return (PyObject*)op;
}

static void
PyThunkBinaryFunction_dealloc(PyThunkOperation_BinaryFunction* self)
{
    Py_XDECREF(self->left);
    Py_XDECREF(self->right);
    self->ob_type->tp_free((PyObject*)self);
}

PyTypeObject PyThunkBinaryFunction_Type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "thunk.binaryFunction",
    sizeof(PyThunkOperation_BinaryFunction),
    0,
    (destructor)PyThunkBinaryFunction_dealloc,   /* tp_dealloc */
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
    "Binary Function function.",                 /* tp_doc */
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
