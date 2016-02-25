
#include "thunk.h"

void PyThunk_Evaluate(PyThunkObject *thunk) {
	if (PyThunk_IsEvaluated(thunk)) {
		return;
	}

	if (PyThunkUnaryPipeline_CheckExact(thunk->operation) || PyThunkBinaryPipeline_CheckExact(thunk->operation)) {
		for(size_t i = 0; i < PyBlockMask_BlockCount(thunk->blockmask); i++) {
			PyThunk_EvaluateBlock(thunk, i);
		}
	} else if (PyThunkUnaryFunction_CheckExact(thunk->operation)) {
		PyThunkOperation_UnaryFunction *operation = (PyThunkOperation_UnaryFunction*)thunk->operation;
		UnaryFunction function = (UnaryFunction)(operation->function);
		if (thunk->storage == NULL) {
			// no storage, have to obtain storage from somewhere
			if (operation->left->ob_refcnt == 1 && ((PyThunkObject*)operation->left)->type == thunk->type) {
				// the referenced object has only one reference (from here)
				// this means it will get destroyed after this operation
				// since it has the same type, we can use its storage directly
				thunk->storage = ((PyThunkObject*)operation->left)->storage;
			} else {
				// we have to create storage for the operation
				thunk->storage = (PyArrayObject*)PyArray_EMPTY(1, (npy_intp[1]) { thunk->cardinality }, thunk->type, 0);
			}
		}
		function(PyThunk_GetData(thunk), PyThunk_GetData(operation->left));
	} else if (PyThunkBinaryFunction_CheckExact(thunk->operation)) {
		PyThunkOperation_BinaryFunction *operation = (PyThunkOperation_BinaryFunction*)thunk->operation;
		BinaryFunction function = (BinaryFunction)(operation->function);
		if (thunk->storage == NULL) {
			// no storage, have to obtain storage from somewhere
			if (operation->left->ob_refcnt == 1 && ((PyThunkObject*)operation->left)->type == thunk->type) {
				thunk->storage = ((PyThunkObject*)operation->left)->storage;
			} else if (operation->right->ob_refcnt == 1 && ((PyThunkObject*)operation->right)->type == thunk->type) {
				thunk->storage = ((PyThunkObject*)operation->right)->storage;
			} else {
				thunk->storage = (PyArrayObject*)PyArray_EMPTY(1, (npy_intp[1]) { thunk->cardinality }, thunk->type, 0);
			}
		}
		function(PyThunk_GetData(thunk), PyThunk_GetData(operation->left), PyThunk_GetData(operation->right));
	}
}

void 
PyThunk_EvaluateBlock(PyThunkObject *thunk, size_t block) {
	if (PyThunk_IsEvaluated(thunk) || PyThunk_IsEvaluatedBlock(thunk, block)) {
		return;
	}
	if (PyThunkUnaryPipeline_CheckExact(thunk->operation)) {
		PyThunkOperation_UnaryPipeline *operation = (PyThunkOperation_UnaryPipeline*)thunk->operation;
		UnaryPipelineFunction function = (UnaryPipelineFunction)(operation->function);
		PyThunk_EvaluateBlock(operation->left, block);
		if (thunk->storage == NULL) {
			// no storage, have to obtain storage from somewhere
			if (operation->left->ob_refcnt == 1 && ((PyThunkObject*)operation->left)->type == thunk->type) {
				// the referenced object has only one reference (from here)
				// this means it will get destroyed after this operation
				// since it has the same type, we can use its storage directly
				thunk->storage = ((PyThunkObject*)operation->left)->storage;
			} else {
				// we have to create storage for the operation
				thunk->storage = (PyArrayObject*)PyArray_EMPTY(1, (npy_intp[1]) { thunk->cardinality }, thunk->type, 0);
			}
		}
		function(PyThunk_GetData(thunk), PyThunk_GetData(operation->left), block, PyThunk_Type(thunk), PyThunk_Type(operation->left));
		PyBlockMask_SetBlock(thunk->blockmask, block);
	} else if (PyThunkBinaryPipeline_CheckExact(thunk->operation)) {
		PyThunkOperation_BinaryPipeline *operation = (PyThunkOperation_BinaryPipeline*)thunk->operation;
		BinaryPipelineFunction function = (BinaryPipelineFunction)(operation->function);
		PyThunk_EvaluateBlock(operation->left, block);
		PyThunk_EvaluateBlock(operation->right, block);
		if (thunk->storage == NULL) {
			// no storage, have to obtain storage from somewhere
			if (operation->left->ob_refcnt == 1 && ((PyThunkObject*)operation->left)->type == thunk->type) {
				thunk->storage = ((PyThunkObject*)operation->left)->storage;
			} else if (operation->right->ob_refcnt == 1 && ((PyThunkObject*)operation->right)->type == thunk->type) {
				thunk->storage = ((PyThunkObject*)operation->right)->storage;
			} else {
				thunk->storage = (PyArrayObject*)PyArray_EMPTY(1, (npy_intp[1]) { thunk->cardinality }, thunk->type, 0);
			}
		}
		function(PyThunk_GetData(thunk), PyThunk_GetData(operation->left), PyThunk_GetData(operation->right), block, PyThunk_Type(thunk), PyThunk_Type(operation->left), PyThunk_Type(operation->right));
		PyBlockMask_SetBlock(thunk->blockmask, block);
	} else {
		PyThunk_Evaluate(thunk);
	}
}

PyObject*
PyThunk_FromExactOperation(PyObject *operation, ssize_t cardinality, int type) {
	register PyThunkObject *thunk;

    thunk = (PyThunkObject *)PyObject_MALLOC(sizeof(PyThunkObject));
    if (thunk == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)thunk, &PyThunk_Type);
    thunk->storage = NULL;
    thunk->evaluated = false;
    thunk->operation = (PyThunkOperation*)operation;
    thunk->cardinality = cardinality;
    thunk->type = type;
    thunk->options = THUNK_CARDINALITY_EXACT;
    thunk->blockmask = PyBlockMask_FromBlocks(cardinality / BLOCK_SIZE + 1);
    return (PyObject*)thunk;
}

PyObject*
PyThunk_FromArray(PyObject *unused, PyObject *input) {
    register PyThunkObject *thunk;
    (void) unused;
    if (!PyArray_CheckExact(input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array as parameter.");
        return NULL;
    }
    thunk = (PyThunkObject *)PyObject_MALLOC(sizeof(PyThunkObject));
    if (thunk == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)thunk, &PyThunk_Type);
    thunk->storage = (PyArrayObject*) PyArray_FromAny(input, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);
    thunk->evaluated = true;
    thunk->operation = NULL;
    thunk->cardinality =  PyArray_SIZE(thunk->storage);
    thunk->type = PyArray_TYPE(thunk->storage);
    thunk->options = THUNK_CARDINALITY_EXACT;
    return (PyObject*)thunk;
}


bool
PyThunk_IsEvaluated(PyThunkObject* thunk) {
	return thunk->evaluated || PyBlockMask_Evaluated(thunk->blockmask);
}

bool 
PyThunk_IsEvaluatedBlock(PyThunkObject *thunk, size_t block) {
	return thunk->evaluated || PyBlockMask_CheckBlock(thunk->blockmask, block);
}

void PyThunk_Init() {
    if (PyType_Ready(&PyThunk_Type) < 0)
        return;
    import_array();
}


static PyObject *
thunk_str(PyThunkObject *self)
{
    PyThunk_Evaluate(self);
    return PyArray_Type.tp_str((PyObject*)self->storage);
}


PyTypeObject PyThunk_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "thunk",
    sizeof(PyThunkObject),
    0,
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    (reprfunc)0,                   /* tp_repr */
    &thunk_as_number,                       /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)thunk_str,                    /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
     | Py_TPFLAGS_CHECKTYPES
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    "Thunk.",                        /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                          /* tp_methods */
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
