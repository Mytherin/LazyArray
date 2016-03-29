
#include "thunk.h"
#include "ufunc_pipeline.h"
#include "_ufunc_static_functions_clone.h"


/* copy from numpy code */
int PyReduceFunc_ResolveTypes(PyUFuncObject *ufunc, PyArrayObject *arr, PyArray_Descr **out_dtype)
{
    int i, retcode;
    PyArrayObject *op[3] = {arr, arr, NULL};
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};

    *out_dtype = NULL;


    /* Use the type resolution function to find our loop */
    retcode = ufunc->type_resolver(ufunc, NPY_UNSAFE_CASTING, op, NULL, dtypes);
    if (retcode == -1) {
        return -1;
    }
    else if (retcode == -2) {
        return -1;
    }

    /*
     * The first two type should be equivalent. Because of how
     * reduce has historically behaved in NumPy, the return type
     * could be different, and it is the return type on which the
     * reduction occurs.
     */
    if (!PyArray_EquivTypes(dtypes[0], dtypes[1])) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(dtypes[i]);
        }
        return -1;
    }

    Py_DECREF(dtypes[0]);
    Py_DECREF(dtypes[1]);
    *out_dtype = dtypes[2];

    return 0;
}

int PyReduceFunc_Execute(PyUFuncObject *ufunc, PyArrayObject *array, PyObject **out) {
    PyCFunctionWithKeywords reduce = (PyCFunctionWithKeywords) PyUFunc_Type.tp_methods[0].ml_meth;

    *out = reduce((PyObject*)ufunc, PyTuple_Pack(1, (PyObject*)array), NULL);
    if (*out == NULL) {
        return -1;
    }
    return 0;
}

int PyReduceFunc_ExecuteBlock(PyUFuncObject *ufunc, PyArrayObject *array, PyObject **out, size_t start, size_t end) {
    int retval = 0;

    PyArrayObject *block = PyArrayObject_Block(array, start, end);
    if (block == NULL) {
        return -1;
    }

    retval = PyReduceFunc_Execute(ufunc, block, out);

    Py_XDECREF(block);
    return retval;
}

int PyUFunc_ResolveTypes(PyUFuncObject *ufunc, PyArrayObject **op, PyArray_Descr **out_type) {
    int retval = 0;

    PyArray_Descr *dtypes[NPY_MAXARGS];
    for (size_t i = 0; i < NPY_MAXARGS; ++i) {
        dtypes[i] = NULL;
    }

    retval = ufunc->type_resolver(ufunc, NPY_DEFAULT_ASSIGN_CASTING, op, NULL, dtypes);
    if (retval < 0) {
        goto fail;
    }

    *out_type = dtypes[0];
fail:
    return retval;    
};

int PyUFunc_PipelinedFunction(PyUFuncObject *ufunc, PyArrayObject **args, size_t start, size_t end) {
    int retval = 0;

    int i, nin = ufunc->nin, nout = ufunc->nout;
    int nop = nin + nout;

    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyArrayObject *op[NPY_MAXARGS];
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    int trivial_loop_ok = 1;
    for (i = 0; i < NPY_MAXARGS; ++i) {
        op[i] = NULL;
        dtypes[i] = NULL;
    }

    retval = ufunc->type_resolver(ufunc, casting, args, NULL, dtypes);
    if (retval < 0) {
        goto fail;
    }

    for(i = 0; i < nop; ++i) {
        size_t size = PyArray_SIZE(args[i]);
        if (size > 1) {
            npy_intp elements[1] = { end - start };
            Py_XINCREF(PyArray_DESCR(args[i]));
            op[i] = (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(args[i]), 1, elements, NULL, PyArray_ElementPointer(args[i], start), NPY_ARRAY_CARRAY | !NPY_ARRAY_OWNDATA, NULL);
        } else {
            Py_XINCREF(args[i]);
            op[i] = args[i];
        }
        if (PyArray_TYPE(args[i]) != dtypes[i]->type_num) {
            assert(i < nin); // Casting makes no sense for output arrays, only for input arrays. If we have to cast the output array something went wrong
            PyArrayObject *converted = (PyArrayObject*) PyArray_CastToType(op[i], dtypes[i], 0);
            Py_XDECREF(op[i]);
            op[i] = converted;
        }
    }

    retval = execute_legacy_ufunc_loop(ufunc, trivial_loop_ok, op, dtypes, order);

    for(i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
    }

fail:
    return retval;
}

void initialize_ufunc_pipeline(void) {
    import_array();
    import_umath();
}