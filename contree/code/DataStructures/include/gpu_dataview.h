#ifndef GPU_DATAVIEW_H
#define GPU_DATAVIEW_H

#include "gpu_structs.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Forward declare for C++ compiler
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

/**
 * Splits a parent GPUDataview into left and right children on the GPU.
 * * @param parent The parent view to split.
 * @param left   The struct to populate for the left child.
 * @param right  The struct to populate for the right child.
 * @param split_feat_idx The feature index used for splitting.
 * @param threshold      The value threshold for splitting.
 * @param child_depth    The depth of the children (used for memory pooling).
 * @param stream         Optional CUDA stream.
 */
void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int child_depth, cudaStream_t stream = 0);

#endif