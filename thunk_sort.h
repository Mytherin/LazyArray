
#ifndef Py_THUNK_SORT_H
#define Py_THUNK_SORT_H

#include "thunkops.h"

PyObject *PyArrayObject_Merge(PyArrayObject *a, PyArrayObject *b, PyArray_Descr *out_type);
PyObject *PyArray_MergeArrays(PyObject *self, PyObject *args);

#endif