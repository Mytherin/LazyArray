
#ifndef Py_THUNK_H
#define Py_THUNK_H
#ifdef __cplusplus
extern "C" {
#endif

#include "blockmask.h"
#include "thunkops.h"

#define THUNK_CARDINALITY_EXACT    0x0001
#define THUNK_CARDINALITY_MAXIMUM  0x0002
#define THUNK_CARDINALITY_APPROX   0x0004

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
    // the operation that can be called to materialize the array
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

#define PyThunk_DATA(obj) PyArray_DATA(((PyThunkObject*)obj)->storage)
#define PyThunk_GetData(obj) PyThunk_DATA(obj)
#define PyThunk_Cardinality(obj) ((PyThunkObject*)obj)->cardinality
#define PyThunk_Type(obj) ((PyThunkObject*)obj)->type

PyObject *PyThunk_AsArray(PyObject*);
PyObject *PyThunk_FromArray(PyObject *, PyObject*);
PyObject *PyThunk_FromExactOperation(PyObject *operation, ssize_t cardinality, int type);

void PyThunk_Init(void);

PyNumberMethods thunk_as_number;
extern struct PyMethodDef thunk_methods[];

#ifdef __cplusplus
}
#endif
#endif /*Py_THUNK_H*/
