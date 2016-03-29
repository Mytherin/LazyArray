
#ifndef Py_THUNK_H
#define Py_THUNK_H
#ifdef __cplusplus
extern "C" {
#endif

#include "blockmask.h"
#include "thunkops.h"


#define PyThunk_HEAD     \
	PyObject_HEAD        \
	ssize_t cardinality; \
	int type;            \
	size_t options;

typedef struct {
	PyThunk_HEAD
    // underlying numpy array that stores the data
    PyArrayObject *storage;
    // flag that indicates whether or not the array is evaluated
    bool evaluated;
    // the operation that can be called to materialitze the array
	PyThunkOperation *operation;
	// contains information on which blocks are and are not evaluated
	PyBlockMask *blockmask;
} PyThunkObject;

PyAPI_DATA(PyTypeObject) PyThunk_Type;

#define PyThunk_Check(op) ((op)->ob_type == &PyThunk_Type)
#define PyThunk_CheckExact(op) ((op)->ob_type == &PyThunk_Type)


// Evaluate the entire thunk
PyObject* PyThunk_Evaluate(PyThunkObject *thunk);
// Evaluate the specified block of the thunk
PyObject* PyThunk_EvaluateBlock(PyThunkObject *thunk, size_t block);
// Returns true if the thunk has been evaluated, and false otherwise
bool PyThunk_IsEvaluated(PyThunkObject *thunk);
// Returns true if the specified block of the thunk has been evaluated, and false otherwise
bool PyThunk_IsEvaluatedBlock(PyThunkObject *thunk, size_t block);
// If the thunk is fully evaluated, finalize the evaluation (destroy the operation object and blockmask)
void PyThunk_FinalizeEvaluation(PyThunkObject *thunk);

#define PyThunk_DATA(obj) PyArray_DATA(((PyThunkObject*)obj)->storage)
#define PyThunk_GetData(obj) PyThunk_DATA(obj)
#define PyThunk_Cardinality(obj) ((PyThunkObject*)obj)->cardinality
#define PyThunk_Type(obj) ((PyThunkObject*)obj)->type

#define PyThunk_MatchingStorage(thunk, other) (PyThunk_CheckExact(other) && other->ob_refcnt == 1 && ((PyThunkObject*)other)->type == thunk->type && ((PyThunkObject*)other)->cardinality == thunk->cardinality)

PyArrayObject* PyArrayObject_Block(PyArrayObject *array, size_t start, size_t end);

PyObject *PyThunk_AsArray(PyObject*);
PyObject *PyThunk_AsUnevaluatedArray(PyObject* thunk);
PyObject *PyThunk_AsTypeArray(PyObject *thunk);
PyObject *PyThunk_FromArray(PyObject *, PyObject*);
PyObject *PyThunk_FromOperation(PyObject *operation, ssize_t cardinality, int cardinality_type, int type);
PyObject *PyThunk_Copy(PyThunkObject *original);
void PyThunk_FromOperation_Inplace(PyThunkObject *thunk, PyObject *operation, ssize_t cardinality, int cardinality_type, int type);
void PyThunk_Init(void);

PyNumberMethods thunk_as_number;
extern struct PyMethodDef thunk_methods[];

#ifdef __cplusplus
}
#endif
#endif /*Py_THUNK_H*/
