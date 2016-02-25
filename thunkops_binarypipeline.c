
#include "thunkops.h"


static void 
pipeline_multiplication_int_int_int(int *storage, int *a, int *b, size_t start, size_t end) {
    for(size_t i = start; i < end; i++) {
        storage[i] = a[i] * b[i];
    }
}

void 
pipeline_multiplication(void *storage, void *a, void *b, size_t block, int storage_type, int a_type, int b_type) {
    size_t start = block * BLOCK_SIZE;
    size_t end = (block + 1) * BLOCK_SIZE;
    switch(storage_type) {
        case NPY_INT32:
            switch(a_type) {
                case NPY_INT32:
                    switch (b_type) {
                        case NPY_INT32:
                            pipeline_multiplication_int_int_int((int*) storage, (int*) a, (int*) b, start, end);
                            break;
                    }
                    break;
            }
            break;
    }
}