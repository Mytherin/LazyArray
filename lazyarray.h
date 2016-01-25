
#ifndef Py_LAZYARRAYOBJECT_H
#define Py_LAZYARRAYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

// Python library
#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code,
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>

#include <stdbool.h>

typedef struct {
    PyObject_HEAD
    PyCFunction function;
    PyObject *a;
    PyObject *b;
} PyLazyFunctionOperation;

PyAPI_DATA(PyTypeObject) PyLazyFunctionOperation_Type;

PyObject *PyLazyFunction_FromFunction(PyCFunction, PyObject *, PyObject *);

typedef struct {
    PyObject_VAR_HEAD
    // underlying numpy array
    PyArrayObject *nparr;
    // flag that indicates whether or not the array is materialized
    bool materialized;
    // operation that materializes the array
    PyLazyFunctionOperation *operation;
} PyLazyArrayObject;

PyAPI_DATA(PyTypeObject) PyLazyArray_Type;


#define PyLazyArray_Check(op) ((op)->ob_type == &PyLazyArray_Type)
#define PyLazyArray_CheckExact(op) ((op)->ob_type == &PyLazyArray_Type)

#define PyLazyArray_ASNUMPY(a) ((PyLazyArrayObject*)a)->nparr

bool PyLazyArray_Materialize(PyLazyArrayObject *a);
#define PyLazyArray_MATERIALIZE(a) if (!((PyLazyArrayObject*)a)->materialized) { if (!PyLazyArray_Materialize((PyLazyArrayObject*) a)) return NULL; }
#define PyLazyArray_ISMATERIALIZED(a) (((PyLazyArrayObject*)a)->materialized)

PyObject *PyLazyArray_FromArray(PyObject *, PyObject*);
PyObject *PyLazyArray_FromOperation(PyLazyFunctionOperation* operation);

PyObject *PyLazyArray_AsArray(PyObject *, PyObject*);

void PyLazyArray_Init();

#ifdef __cplusplus
}
#endif
#endif /* !Py_LAZYARRAYOBJECT_H */