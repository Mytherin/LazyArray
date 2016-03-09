
#include "thunkops.h"
#include "initializers.h"

size_t BLOCK_SIZE = 10000;

int generic_binary_cardinality_resolver(size_t left_cardinality, size_t right_cardinality, ssize_t *cardinality, ssize_t *cardinality_type) {
    if (left_cardinality == 1 || right_cardinality == 1 || left_cardinality == right_cardinality) {
        *cardinality_type = THUNK_CARDINALITY_EXACT;
        *cardinality = max(left_cardinality, right_cardinality);
        return 1;
    }
    return -1;
}

int generic_unary_cardinality_resolver(size_t left_cardinality, ssize_t *cardinality, ssize_t *cardinality_type) {
    *cardinality = left_cardinality;
    *cardinality_type = THUNK_CARDINALITY_EXACT;
    return 1;
}

void initialize_thunkops(void) {
    import_array();
}
