
#include "thunk_sort.h"
#include "initializers.h"

#include "generated/thunk_lazy_functions.h"
#include "generated/thunkops_pipeline.h"

#include "thunk.h"

static int
DOUBLE_merge_arrays(npy_double *left, npy_double *right, size_t left_size, size_t right_size, npy_double *result) {
    npy_double *left_end = left + left_size;
    npy_double *right_end = right + right_size;
    // first merge the two arrays until either of the arrays runs out of values
    while (left < left_end && right < right_end) {
        if (*left < *right) {
            *result++ = *left++;
        }
        else {
            *result++ = *right++;
        }
    }
    // copy the remaining values from the array that has not run out of values
    if (right < right_end) {
        left = right;
        left_end = right_end;
    } 
    memcpy(result, left, sizeof(npy_double) * (left_end - left));
    return 1;
}

int
merge_arrays(void *left, void *right, size_t left_size, size_t right_size, void *result, int typenum) {
    switch(typenum) {
        case NPY_DOUBLE:
            return DOUBLE_merge_arrays((npy_double*) left, (npy_double*) right, left_size, right_size, (npy_double*) result);
        default:
            return -1;
    }
}

PyObject*
PyArrayObject_Merge(PyArrayObject *a, PyArrayObject *b, PyArray_Descr *out_type) {
    if (out_type->type_num != NPY_DOUBLE) {
        // only double supported now
        return NULL;
    }
    Py_INCREF(a);
    Py_INCREF(b);
    if (PyArray_DESCR(a)->type_num != NPY_DOUBLE) {
        PyArrayObject *converted = (PyArrayObject*) PyArray_CastToType(a, out_type, 0);
        Py_DECREF(a);
        a = converted;
    }
    if (PyArray_DESCR(b)->type_num != NPY_DOUBLE) {
        PyArrayObject *converted = (PyArrayObject*) PyArray_CastToType(b, out_type, 0);
        Py_DECREF(b);
        b = converted;
    }

    size_t left_size = PyArray_SIZE(a);
    size_t right_size = PyArray_SIZE(b);

    npy_intp elements[1] = { left_size + right_size };
    Py_INCREF(out_type);
    PyObject *result = PyArray_Empty(1, elements, out_type, 0);

    DOUBLE_merge_arrays(PyArray_DATA(a), PyArray_DATA(b), left_size, right_size, PyArray_DATA((PyArrayObject*)result));
    Py_DECREF(a);
    Py_DECREF(b);
    return result;
}

PyObject *PyArray_MergeArrays(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    if (PyArg_ParseTuple(args, "OO", &a, &b) < 0) {
        PyErr_BadArgument();
        return NULL;
    }
    if (!PyArray_CheckExact(a)) {
        PyErr_BadArgument();
        return NULL;
    }
    if (!PyArray_CheckExact(b)) {
        PyErr_BadArgument();
        return NULL;
    }
    return PyArrayObject_Merge((PyArrayObject*) a, (PyArrayObject*) b, PyArray_DescrFromType(NPY_DOUBLE));
}


void pipeline_blocksort(PyArrayObject **args, size_t start, size_t end) {
    void *inptr = PyArray_DATA(args[0]);
    void *outptr = PyArray_DATA(args[1]);
    PyArray_Descr *descr = PyArray_DESCR(args[1]);
    // the sort function we are calling is an in-place sort, so if the input and output data is different we first have to copy the data to the new location
    if (inptr != outptr) {
        // we only sort the current block, so we only copy the current block of data as well
        PyArrayObject *in = PyArrayObject_Block(args[0], start, end);
        PyArrayObject *out = PyArrayObject_Block(args[1], start, end);
        PyArray_CopyInto(out, in);
        Py_DECREF(in);
        Py_DECREF(out);
    }
    // call the actual sort function
    descr->f->sort[0]((void*)(PyArray_BYTES(args[1]) + start * descr->elsize), end - start, NULL);
}

void recursive_merge(char *inptr, size_t block_size, size_t total_size, size_t elsize, int typenum) {
    if (block_size > total_size) return;
    // recursive merge sort, first merge all blocks with size block_size together, resulting in sorted blocks of size (block_size * 2)
    // then recursively continue to merge those blocks together again
    void *temp_result = malloc(block_size * 2 * elsize);
    for(size_t el = 0; el + block_size < total_size; el += block_size * 2) {
        size_t right_size = min(block_size, total_size - (el + block_size));
        merge_arrays(inptr + (el * elsize), inptr + ((el + block_size) * elsize), block_size, right_size, temp_result, typenum);
        memcpy(inptr + (el * elsize), temp_result, (block_size + right_size) * elsize);
    }
    free(temp_result);
    recursive_merge(inptr, block_size * 2, total_size, elsize, typenum);
}

void unary_mergesort(PyArrayObject **args) {
    if (PyArray_DATA(args[0]) != PyArray_DATA(args[1])) {
        // we do an in-place merge, so if the input and output data is different we first copy the data
        PyArray_CopyInto(args[1], args[0]);
    }
    // each of the blocks (should be) sorted here, so now we can merge them together recursively
    recursive_merge(PyArray_BYTES(args[1]), BLOCK_SIZE, PyArray_SIZE(args[1]), PyArray_DESCR(args[1])->elsize, PyArray_DESCR(args[1])->type_num);
}

PyObject *thunk_lazysort(PyObject *v, PyObject *unused) {
    PyArrayObject *args[NPY_MAXARGS];
    PyArray_Descr *types[NPY_MAXARGS];
    ssize_t cardinality, cardinality_type;
    (void) unused;
    for(size_t i = 0; i < NPY_MAXARGS; i++) {
        types[i] = NULL;
        args[i] = NULL;
    }
    // sort is an in-place operation, so we can't just return a new thunk as we do with other functions
    // instead, we do the following:
    // create a copy of the original thunk "v" (stored in 'copy')
    // create the thunk responsible for sorting v
    // --> sorting happens in two steps
    // --> first pipeline sort, this sorts the individual blocks in a pipeline fashion
    // --> then merge sort, this is a unary function that merges the sorted blocks to create a fully sorted array
    // after creating these two thunks, we assign the final thunk to 'v' again, so 'v' is updated to reflect the sort status
    args[0] = (PyArrayObject*) PyThunk_AsTypeArray(v);
    types[0] = PyArray_DESCR(args[0]);
    sqrt_resolve_cardinality(PyThunk_Cardinality(v), &cardinality, &cardinality_type);

    PyObject *copy = (PyObject*) PyThunk_Copy((PyThunkObject*) v);
    PyObject *blocksort = PyThunkUnaryPipeline_FromFunction(pipeline_blocksort, copy);
    PyObject *initial_thunk = PyThunk_FromOperation(blocksort, cardinality, cardinality_type, types[0]->type_num);
    PyObject *merge = PyThunkUnaryFunction_FromFunction(unary_mergesort, initial_thunk);
    PyThunk_FromOperation_Inplace((PyThunkObject*) v, merge, cardinality, cardinality_type, types[0]->type_num);
    Py_RETURN_NONE;
}

void initialize_sort(void) {
    import_array();
}