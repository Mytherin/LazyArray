
#include "blockmask.h"

#define PyBlockMask_SIZE (sizeof(PyBlockMask) - sizeof(unsigned int))

PyBlockMask *PyBlockMask_FromBlocks(size_t block_count) {
	size_t offsets = block_count / 32 + 1;
	PyBlockMask *mask = (PyBlockMask*) malloc(PyBlockMask_SIZE + sizeof(unsigned int) * offsets);
	memset(mask, 0, PyBlockMask_SIZE + sizeof(unsigned int) * offsets);
	mask->block_count = block_count;
	return mask;
}

void PyBlockMask_SetBlock(PyBlockMask *mask, size_t block) {
	size_t offset = block / 32;
	size_t block_number = block - (offset * 32);
	mask->blocks[offset] |= 1 << block_number;
	mask->evaluated++;
}

bool PyBlockMask_CheckBlock(PyBlockMask *mask, size_t block) {
	size_t offset = block / 32;
	size_t block_number = block - (offset * 32);
	return mask->blocks[offset] & (1 << block_number);
}

bool PyBlockMask_Evaluated(PyBlockMask *mask) {
	return mask == NULL ? false : mask->evaluated == mask->block_count;
}