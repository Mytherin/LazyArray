

#ifndef Py_THUNKOPS_H
#define Py_THUNKOPS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "thunk_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#define BLOCK_SIZE 1024

typedef void (*UnaryPipelineFunction)(void *storage, void *a, size_t start, size_t end, int storage_type, int a_type);
typedef void (*BinaryPipelineFunction)(void *storage, void *a, void *b, size_t start, size_t end, int storage_type, int a_type, int b_type);
typedef void (*UnaryFunction)(void *storage, void *a);
typedef void (*BinaryFunction)(void *storage, void *a, void *b);

typedef struct {
	PyObject_HEAD
} PyThunkOperation;

PyAPI_DATA(PyTypeObject) PyThunkOperation_Type;

typedef struct {
	PyObject_HEAD
	UnaryPipelineFunction function;
	PyObject *left;
} PyThunkOperation_UnaryPipeline;

PyAPI_DATA(PyTypeObject) PyThunkUnaryPipeline_Type;

typedef struct {
	PyObject_HEAD
	BinaryPipelineFunction function;
	PyObject *left;
	PyObject *right;
} PyThunkOperation_BinaryPipeline;

PyAPI_DATA(PyTypeObject) PyThunkBinaryPipeline_Type;

typedef struct {
	PyObject_HEAD
	UnaryFunction function;
	PyObject *left;
} PyThunkOperation_UnaryFunction;

PyAPI_DATA(PyTypeObject) PyThunkUnaryFunction_Type;

typedef struct {
	PyObject_HEAD
	BinaryFunction function;
	PyObject *left;
	PyObject *right;
} PyThunkOperation_BinaryFunction;

PyAPI_DATA(PyTypeObject) PyThunkBinaryFunction_Type;

#define PyThunkOperation_Check(op) ((op)->ob_type == &PyThunkOperation_Type)

#define PyThunkUnaryPipeline_CheckExact(op) ((op)->ob_type == &PyThunkUnaryPipeline_Type)
#define PyThunkBinaryPipeline_CheckExact(op) ((op)->ob_type == &PyThunkBinaryPipeline_Type)
#define PyThunkUnaryFunction_CheckExact(op) ((op)->ob_type == &PyThunkUnaryFunction_Type)
#define PyThunkBinaryFunction_CheckExact(op) ((op)->ob_type == &PyThunkBinaryFunction_Type)

PyObject* PyThunkUnaryPipeline_FromFunction(UnaryPipelineFunction function, PyObject *left);
PyObject* PyThunkBinaryPipeline_FromFunction(BinaryPipelineFunction function, PyObject *left, PyObject *right, PyCFunction base_function);

#include "generated/thunkops_binarypipeline.h"

void pipeline_sqrt(void *storage, void *a, size_t start, size_t end, int storage_type, int a_type);

#ifdef __cplusplus
}
#endif
#endif /*Py_THUNK_H*/
