
#ifndef Py_BLOCKMASK_H
#define Py_BLOCKMASK_H

#include "thunk_config.h"

typedef struct {
	// amount of blocks that the block mask supports
	size_t block_count;
	// amount of blocks that have been evaluated
	size_t evaluated;
	// flag that indicates which blocks are and are not evaluated
	unsigned int blocks[1];
} PyBlockMask;

// Create a block mask from the specified amount of blocks
PyBlockMask *PyBlockMask_FromBlocks(size_t block_count);
// Evaluates a specified block
void PyBlockMask_SetBlock(PyBlockMask *mask, size_t block);
// Returns true if a specific block has been evaluated, and false otherwise
bool PyBlockMask_CheckBlock(PyBlockMask *mask, size_t block);
// Returns true if all blocks have been evaluated, and false otherwise
bool PyBlockMask_Evaluated(PyBlockMask *mask);
// Returns the amount of blocks the block mask can store
#define PyBlockMask_BlockCount(obj) ((PyBlockMask*)obj)->block_count

#endif