


#ifndef Py_UFUNC_PIPELINE_H
#define Py_UFUNC_PIPELINE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "initializers.h"

#include "thunk_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <numpy/ufuncobject.h>

int PyUFunc_ResolveTypes(PyUFuncObject *ufunc, PyArrayObject **op, PyArray_Descr **out_type);
int PyUFunc_PipelinedFunction(PyUFuncObject *ufunc, PyArrayObject **args, size_t start, size_t end);

#ifdef __cplusplus
}
#endif
#endif /*Py_UFUNC_PIPELINE_H*/
