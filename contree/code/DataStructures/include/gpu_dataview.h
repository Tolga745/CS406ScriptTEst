#ifndef GPU_DATAVIEW_H
#define GPU_DATAVIEW_H

#include "gpu_structs.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int child_depth, cudaStream_t stream = 0);

#endif