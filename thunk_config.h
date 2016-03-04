
#ifndef Py_THUNK_CONFIG_H
#define Py_THUNK_CONFIG_H

// Python library
#include <Python.h>
// Boolean type
#include <stdbool.h>

#define THUNK_CARDINALITY_EXACT    0x0001
#define THUNK_CARDINALITY_MAXIMUM  0x0002
#define THUNK_CARDINALITY_APPROX   0x0004

#define max(x,y) (x > y ? x : y)
#define min(x,y) (x < y ? x : y)

#endif