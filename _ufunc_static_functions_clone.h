

#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) ( \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (PyArray_FLAGS(arr1)&(NPY_ARRAY_C_CONTIGUOUS| \
                                      NPY_ARRAY_F_CONTIGUOUS)) & \
                                (PyArray_FLAGS(arr2)&(NPY_ARRAY_C_CONTIGUOUS| \
                                              NPY_ARRAY_F_CONTIGUOUS)) \
                        )

#define PyArray_TRIVIALLY_ITERABLE(arr) ( \
                    PyArray_NDIM(arr) <= 1 || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS) \
                    )
#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr); \
                    data = PyArray_BYTES(arr); \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                    ((PyArray_NDIM(arr) == 1) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_ITEMSIZE(arr)));


#define PyArray_TRIVIALLY_ITERABLE_PAIR(arr1, arr2) (\
                    PyArray_TRIVIALLY_ITERABLE(arr1) && \
                        (PyArray_NDIM(arr2) == 0 || \
                         PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) || \
                         (PyArray_NDIM(arr1) == 0 && \
                             PyArray_TRIVIALLY_ITERABLE(arr2) \
                         ) \
                        ) \
                    )
#define PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(arr1, arr2, \
                                        count, \
                                        data1, data2, \
                                        stride1, stride2) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    stride1 = (size1 == 1 ? 0 : ((PyArray_NDIM(arr1) == 1) ? \
                                                PyArray_STRIDE(arr1, 0) : \
                                                PyArray_ITEMSIZE(arr1))); \
                    stride2 = (size2 == 1 ? 0 : ((PyArray_NDIM(arr2) == 1) ? \
                                                PyArray_STRIDE(arr2, 0) : \
                                                PyArray_ITEMSIZE(arr2))); \
                }

#define PyArray_TRIVIALLY_ITERABLE_TRIPLE(arr1, arr2, arr3) (\
                PyArray_TRIVIALLY_ITERABLE(arr1) && \
                    ((PyArray_NDIM(arr2) == 0 && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_NDIM(arr1) == 0 && \
                        PyArray_TRIVIALLY_ITERABLE(arr2) && \
                            (PyArray_NDIM(arr3) == 0 || \
                                PyArray_EQUIVALENTLY_ITERABLE(arr2, arr3) \
                            ) \
                     ) \
                    ) \
                )

#define PyArray_PREPARE_TRIVIAL_TRIPLE_ITERATION(arr1, arr2, arr3, \
                                        count, \
                                        data1, data2, data3, \
                                        stride1, stride2, stride3) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    npy_intp size3 = PyArray_SIZE(arr3); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    count = ((size3 > count) || size3 == 0) ? size3 : count; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    data3 = PyArray_BYTES(arr3); \
                    stride1 = (size1 == 1 ? 0 : ((PyArray_NDIM(arr1) == 1) ? \
                                                PyArray_STRIDE(arr1, 0) : \
                                                PyArray_ITEMSIZE(arr1))); \
                    stride2 = (size2 == 1 ? 0 : ((PyArray_NDIM(arr2) == 1) ? \
                                                PyArray_STRIDE(arr2, 0) : \
                                                PyArray_ITEMSIZE(arr2))); \
                    stride3 = (size3 == 1 ? 0 : ((PyArray_NDIM(arr3) == 1) ? \
                                                PyArray_STRIDE(arr3, 0) : \
                                                PyArray_ITEMSIZE(arr3))); \
                }


// static int
// check_for_trivial_loop(PyUFuncObject *ufunc,
//                         PyArrayObject **op,
//                         PyArray_Descr **dtype,
//                         npy_intp buffersize)
// {
//     npy_intp i, nin = ufunc->nin, nop = nin + ufunc->nout;

//     for (i = 0; i < nop; ++i) {
//         /*
//          * If the dtype doesn't match, or the array isn't aligned,
//          * indicate that the trivial loop can't be done.
//          */
//         if (op[i] != NULL &&
//                 (!PyArray_ISALIGNED(op[i]) ||
//                 !PyArray_EquivTypes(dtype[i], PyArray_DESCR(op[i]))
//                                         )) {
//             /*
//              * If op[j] is a scalar or small one dimensional
//              * array input, make a copy to keep the opportunity
//              * for a trivial loop.
//              */
//             if (i < nin && (PyArray_NDIM(op[i]) == 0 ||
//                     (PyArray_NDIM(op[i]) == 1 &&
//                      PyArray_DIM(op[i],0) <= buffersize))) {
//                 PyArrayObject *tmp;
//                 Py_INCREF(dtype[i]);
//                 tmp = (PyArrayObject *)
//                             PyArray_CastToType(op[i], dtype[i], 0);
//                 if (tmp == NULL) {
//                     return -1;
//                 }
//                 Py_DECREF(op[i]);
//                 op[i] = tmp;
//             }
//             else {
//                 return 0;
//             }
//         }
//     }

//     return 1;
// }

static void
trivial_two_operand_loop(PyArrayObject **op,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    char *data[2];
    npy_intp count[2], stride[2];
    int needs_api;
    NPY_BEGIN_THREADS_DEF;

    needs_api = PyDataType_REFCHK(PyArray_DESCR(op[0])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[1]));

    PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(op[0], op[1],
                                            count[0],
                                            data[0], data[1],
                                            stride[0], stride[1]);
    count[1] = count[0];

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(count[0]);
    }

    innerloop(data, count, stride, innerloopdata);

    NPY_END_THREADS;
}

static void
trivial_three_operand_loop(PyArrayObject **op,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    char *data[3];
    npy_intp count[3], stride[3];
    int needs_api;
    NPY_BEGIN_THREADS_DEF;

    needs_api = PyDataType_REFCHK(PyArray_DESCR(op[0])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[1])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[2]));

    PyArray_PREPARE_TRIVIAL_TRIPLE_ITERATION(op[0], op[1], op[2],
                                            count[0],
                                            data[0], data[1], data[2],
                                            stride[0], stride[1], stride[2]);
    count[1] = count[0];
    count[2] = count[0];

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(count[0]);
    }

    innerloop(data, count, stride, innerloopdata);

    NPY_END_THREADS;
}

static int
execute_legacy_ufunc_loop(PyUFuncObject *ufunc,
                    int trivial_loop_ok,
                    PyArrayObject **op,
                    PyArray_Descr **dtypes,
                    NPY_ORDER order)
{
    npy_intp nin = ufunc->nin, nout = ufunc->nout;
    PyUFuncGenericFunction innerloop;
    void *innerloopdata;
    int needs_api = 0;

    if (ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                    &innerloop, &innerloopdata, &needs_api) < 0) {
        return -1;
    }

    /* First check for the trivial cases that don't need an iterator */
    if (trivial_loop_ok) {
        if (nin == 1 && nout == 1) {
            if (op[1] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE(op[0])) {
                Py_INCREF(dtypes[1]);
                op[1] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                             dtypes[1],
                             PyArray_NDIM(op[0]),
                             PyArray_DIMS(op[0]),
                             NULL, NULL,
                             PyArray_ISFORTRAN(op[0]) ?
                                            NPY_ARRAY_F_CONTIGUOUS : 0,
                             NULL);
                if (op[1] == NULL) {
                    return -1;
                }

                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[1] != NULL &&
                        PyArray_NDIM(op[1]) >= PyArray_NDIM(op[0]) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1])) {

                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
        else if (nin == 2 && nout == 1) {
            if (op[2] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1])) {
                PyArrayObject *tmp;
                /*
                 * Have to choose the input with more dimensions to clone, as
                 * one of them could be a scalar.
                 */
                if (PyArray_NDIM(op[0]) >= PyArray_NDIM(op[1])) {
                    tmp = op[0];
                }
                else {
                    tmp = op[1];
                }
                Py_INCREF(dtypes[2]);
                op[2] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                 dtypes[2],
                                 PyArray_NDIM(tmp),
                                 PyArray_DIMS(tmp),
                                 NULL, NULL,
                                 PyArray_ISFORTRAN(tmp) ?
                                                NPY_ARRAY_F_CONTIGUOUS : 0,
                                 NULL);
                if (op[2] == NULL) {
                    return -1;
                }


                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[2] != NULL &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[0]) &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[1]) &&
                    PyArray_TRIVIALLY_ITERABLE_TRIPLE(op[0], op[1], op[2])) {

                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
    }

    return 0;
}
