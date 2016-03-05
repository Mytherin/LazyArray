
#include "thunkops.h"
#include "initializers.h"

size_t BLOCK_SIZE = 10000;

int generic_binary_cardinality_resolver(PyArrayObject **args, ssize_t *cardinality, ssize_t *cardinality_type) {
    ssize_t left_cardinality = PyArray_SIZE(args[0]);
    ssize_t right_cardinality = PyArray_SIZE(args[1]);
    if (left_cardinality == 1 || right_cardinality == 1 || left_cardinality == right_cardinality) {
        *cardinality_type = THUNK_CARDINALITY_EXACT;
        *cardinality = max(left_cardinality, right_cardinality);
        return 1;
    }
    return -1;
}

void initialize_thunkops(void) {
    import_array();
}