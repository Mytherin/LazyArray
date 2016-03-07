

f = open('generated/thunkops_pipeline.c', 'w+')
f.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#include "../thunkops.h"
#include "thunkops_pipeline.h"
#include "../ufunc_pipeline.h"

""")

f2 = open('generated/thunkops_pipeline.h', 'w+')
f2.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#ifndef Py_THUNKOPS_PIPELINE_H
#define Py_THUNKOPS_PIPELINE_H

#include "operations.h"
""")

f3 = open('generated/operations.h', 'w+')
f3.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#ifndef Py_OPERATIONS_H
#define Py_OPERATIONS_H

#include "../thunk_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

void initialize_operators(void);
""")

f4 = open('generated/operations.c', 'w+')
f4.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#include "operations.h"
void initialize_operators(void) {
	import_array();
	PyObject *numpy_module = PyImport_ImportModule("numpy");
	PyObject *dd = PyModule_GetDict(numpy_module);
""")

variables = []

f5 = open('generated/thunk_lazy_functions.h', 'w+')
f5.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#ifndef Py_THUNK_AS_NUMBER_H
#define Py_THUNK_AS_NUMBER_H

#include "operations.h"

""")


f6 = open('generated/thunk_lazy_functions.c', 'w+')
f6.write("""
/* THIS FILE IS GENERATED BY gencode.py */
/* DO NOT EDIT THIS FILE MANUALLY */

#include "thunk_lazy_functions.h"
#include "thunkops_pipeline.h"
#include "../thunkops.h"
#include "../thunk.h"
""")


def generate_binary_pipeline_operator(operatorname):
	f.write('\n/* %s */\n'% operatorname)
	f2.write('\n/* %s */\n'% operatorname)
	f2.write('void pipeline_%s(PyArrayObject **args, size_t start, size_t end);\n' % operatorname)
	f.write("""
void pipeline_%s(PyArrayObject **args, size_t start, size_t end) {
	PyUFunc_PipelinedFunction(%s_ufunc, args, start, end);
}
""" % (operatorname, operatorname))
	f2.write('int %s_resolve_types(PyArrayObject **args, PyArray_Descr **out_types);\n' % operatorname)
	f.write("""
int %s_resolve_types(PyArrayObject **args, PyArray_Descr **out_types) {
	return PyUFunc_ResolveTypes(%s_ufunc, args, out_types);
}
""" % (operatorname, operatorname))
	f2.write('int %s_resolve_cardinality(size_t left_cardinality, size_t right_cardinality, ssize_t *cardinality, ssize_t *cardinality_type);\n' % operatorname)
	f.write("""
int %s_resolve_cardinality(size_t left_cardinality, size_t right_cardinality, ssize_t *cardinality, ssize_t *cardinality_type) {
	return generic_binary_cardinality_resolver(left_cardinality, right_cardinality, cardinality, cardinality_type);
}
""" % operatorname)
	f3.write('extern PyUFuncObject *%s_ufunc;\n' % operatorname)
	variables.append('PyUFuncObject *%s_ufunc = NULL;\n' % operatorname)
	f4.write('\t%s_ufunc = (PyUFuncObject*) PyDict_GetItemString(dd, "%s");\n' % (operatorname,operatorname))
	f4.write('\tif (%s_ufunc == NULL) { printf("Failed to load %s.\\n"); }\n' % (operatorname, operatorname))
	f5.write('PyObject *thunk_lazy%s(PyObject *v, PyObject *w);\n' % operatorname)
	f6.write("""
PyObject *thunk_lazy%s(PyObject *v, PyObject *w) {
	PyArrayObject *args[NPY_MAXARGS];
	PyArray_Descr *types[NPY_MAXARGS];
	ssize_t cardinality, cardinality_type;
	for(size_t i = 0; i < NPY_MAXARGS; i++) {
		types[i] = NULL;
		args[i] = NULL;
	}
	args[0] = (PyArrayObject*) PyThunk_AsTypeArray(v);
	args[1] = (PyArrayObject*) PyThunk_AsTypeArray(w);
	%s_resolve_types(args, types);
	%s_resolve_cardinality(PyThunk_Cardinality(v), PyThunk_Cardinality(w), &cardinality, &cardinality_type);
	PyObject *op = PyThunkBinaryPipeline_FromFunction(pipeline_%s, v, w);
	return PyThunk_FromOperation(op, cardinality, cardinality_type, types[0]->type_num);
}
""" % (operatorname, operatorname, operatorname, operatorname))


f3.write('\n/* binary operations */\n')
f4.write('\n\t/* binary operations */\n')

# todo: inplace functions -> 'inplace_add', 'inplace_subtract', 'inplace_multiply', 'inplace_divide', 'inplace_remainder', 'inplace_power', 'inplace_lshift', 'inplace_rshift', 'inplace_and', 'inplace_xor', 'inplace_or','inplace_floor_divide', 'inplace_true_divide'
# todo: divmod: 'divmod' (not a ufunc)

binary_operations = ['multiply', 'add', 'subtract', 'divide', 'remainder', 'power', 'left_shift', 'right_shift', 'bitwise_and', 'bitwise_xor', 'bitwise_or', 'floor_divide', 'true_divide']

for op in binary_operations:
	generate_binary_pipeline_operator(op)


def generate_unary_pipeline_operator(operatorname):
	f.write('\n/* %s */\n'% operatorname)
	f2.write('\n/* %s */\n'% operatorname)
	f2.write('void pipeline_%s(PyArrayObject **args, size_t start, size_t end);\n' % operatorname)
	f.write("""
void pipeline_%s(PyArrayObject **args, size_t start, size_t end) {
	PyUFunc_PipelinedFunction(%s_ufunc, args, start, end);
}
""" % (operatorname, operatorname))
	f2.write('int %s_resolve_types(PyArrayObject **args, PyArray_Descr **out_types);\n' % operatorname)
	f.write("""
int %s_resolve_types(PyArrayObject **args, PyArray_Descr **out_types) {
	return PyUFunc_ResolveTypes(%s_ufunc, args, out_types);
}
""" % (operatorname, operatorname))
	f2.write('int %s_resolve_cardinality(size_t left_cardinality, ssize_t *cardinality, ssize_t *cardinality_type);\n' % operatorname)
	f.write("""
int %s_resolve_cardinality(size_t left_cardinality, ssize_t *cardinality, ssize_t *cardinality_type) {
	return generic_unary_cardinality_resolver(left_cardinality, cardinality, cardinality_type);
}
""" % operatorname)
	f3.write('extern PyUFuncObject *%s_ufunc;\n' % operatorname)
	variables.append('PyUFuncObject *%s_ufunc = NULL;\n' % operatorname)
	f4.write('\t%s_ufunc = (PyUFuncObject*) PyDict_GetItemString(dd, "%s");\n' % (operatorname,operatorname))
	f4.write('\tif (%s_ufunc == NULL) { printf("Failed to load %s.\\n"); }\n' % (operatorname, operatorname))
	f5.write('PyObject *thunk_lazy%s(PyObject *v, PyObject *w);\n' % operatorname)
	f6.write("""
PyObject *thunk_lazy%s(PyObject *v, PyObject *unused) {
	PyArrayObject *args[NPY_MAXARGS];
	PyArray_Descr *types[NPY_MAXARGS];
	ssize_t cardinality, cardinality_type;
	(void) unused;
	for(size_t i = 0; i < NPY_MAXARGS; i++) {
		types[i] = NULL;
		args[i] = NULL;
	}
	args[0] = (PyArrayObject*) PyThunk_AsTypeArray(v);
	%s_resolve_types(args, types);
	%s_resolve_cardinality(PyThunk_Cardinality(v), &cardinality, &cardinality_type);
	PyObject *op = PyThunkUnaryPipeline_FromFunction(pipeline_%s, v);
	return PyThunk_FromOperation(op, cardinality, cardinality_type, types[0]->type_num);
}
""" % (operatorname, operatorname, operatorname, operatorname))

#todo: positive, hex, oct

single_operations = ['negative', 'absolute', 'nonzero', 'invert', 'int', 'long', 'float', 'sqrt']


for op in single_operations:
	generate_unary_pipeline_operator(op)


f5.write('\n#endif\n')
f4.write('}\n')
for variable in variables:
	f4.write(variable)
f3.write('\n#endif\n')
f2.write('\n#endif\n')

