
#include "ufunc_pipeline.h"
#include "_ufunc_static_functions_clone.h"

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

    npy_intp elements[1] = { end - start };
    for(i = 0; i < nop; ++i) {
        Py_XINCREF(PyArray_DESCR(args[i]));
        op[i] = (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(args[i]), 1, elements, NULL, (void*)(PyArray_BYTES(args[i]) + start * PyArray_DESCR(args[i])->elsize), NPY_ARRAY_CARRAY | !NPY_ARRAY_OWNDATA, NULL);
        if (PyArray_TYPE(args[i]) != dtypes[i]->type_num) {
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
}