

#ifndef Py_THUNKOPS_H
#define Py_THUNKOPS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "thunk_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <numpy/ufuncobject.h>

extern size_t BLOCK_SIZE;

typedef void (*UnaryPipelineFunction)(PyArrayObject **, size_t start, size_t end);
typedef void (*BinaryPipelineFunction)(PyArrayObject **, size_t start, size_t end);
typedef void (*UnaryFunction)(PyArrayObject **);
typedef void (*BinaryFunction)(PyArrayObject **);

#define PyThunkOps_HEAD           \
	PyObject_HEAD              \
	PyCFunction base_function;


typedef struct {
	PyThunkOps_HEAD
} PyThunkOperation;

PyAPI_DATA(PyTypeObject) PyThunkOperation_Type;

typedef struct {
	PyThunkOps_HEAD
	UnaryPipelineFunction function;
	PyObject *left;
} PyThunkOperation_UnaryPipeline;

PyAPI_DATA(PyTypeObject) PyThunkUnaryPipeline_Type;

typedef struct {
	PyThunkOps_HEAD
	BinaryPipelineFunction function;
	PyObject *left;
	PyObject *right;
} PyThunkOperation_BinaryPipeline;

PyAPI_DATA(PyTypeObject) PyThunkBinaryPipeline_Type;

typedef struct {
	PyThunkOps_HEAD
	UnaryFunction function;
	PyObject *left;
} PyThunkOperation_UnaryFunction;

PyAPI_DATA(PyTypeObject) PyThunkUnaryFunction_Type;

typedef struct {
	PyThunkOps_HEAD
	BinaryFunction function;
	PyObject *left;
	PyObject *right;
} PyThunkOperation_BinaryFunction;

PyAPI_DATA(PyTypeObject) PyThunkBinaryFunction_Type;

typedef struct {
	PyThunkOps_HEAD
	PyUFuncObject *function;
	PyObject *left;
} PyThunkOperation_AggregationPipeline;

PyAPI_DATA(PyTypeObject) PyThunkAggregationPipeline_Type;

int generic_binary_cardinality_resolver(size_t left_cardinality, size_t right_cardinality, ssize_t *cardinality, ssize_t *cardinality_type);
int generic_unary_cardinality_resolver(size_t left_cardinality, ssize_t *cardinality, ssize_t *cardinality_type);

#define PyThunkOperation_Check(op) ((op)->ob_type == &PyThunkOperation_Type)

#define PyThunkUnaryPipeline_CheckExact(op) ((op)->ob_type == &PyThunkUnaryPipeline_Type)
#define PyThunkBinaryPipeline_CheckExact(op) ((op)->ob_type == &PyThunkBinaryPipeline_Type)
#define PyThunkUnaryFunction_CheckExact(op) ((op)->ob_type == &PyThunkUnaryFunction_Type)
#define PyThunkBinaryFunction_CheckExact(op) ((op)->ob_type == &PyThunkBinaryFunction_Type)

PyObject* PyThunkUnaryPipeline_FromFunction(UnaryPipelineFunction function, PyObject *left);
PyObject* PyThunkBinaryPipeline_FromFunction(BinaryPipelineFunction function, PyObject *left, PyObject *right);

#ifdef __cplusplus
}
#endif
#endif /*Py_THUNK_H*/
