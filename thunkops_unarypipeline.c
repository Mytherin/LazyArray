
#include "thunkops.h"

static void 
pipeline_sqrt_double_int(double *storage, int *a, size_t start, size_t end) {
    for(size_t i = start; i < end; i++) {
        storage[i] = sqrt(a[i]);
    }
}


void 
pipeline_sqrt(void *storage, void *a, size_t start, size_t end, int storage_type, int a_type) {
    switch(storage_type) {
        case NPY_FLOAT64:
            switch(a_type) {
                case NPY_INT32:
                    pipeline_sqrt_double_int((double*) storage, (int*) a, start, end);
                    break;
            }
            break;
    }
}