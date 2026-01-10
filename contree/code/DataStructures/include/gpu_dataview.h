#ifndef GPU_DATAVIEW_H
#define GPU_DATAVIEW_H

#include "gpu_structs.h"

#ifdef __CUDACC__
// In CUDA files (.cu), we have access to the runtime headers
#include <cuda_runtime.h>
#else
// In C++ files (.cpp), we forward declare to avoid including huge CUDA headers
// cudaStream_t is a pointer to an opaque struct CUstream_st
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, cudaStream_t stream = 0);

void split_gpu_dataview_preallocated(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int* d_row_map_buffer, cudaStream_t stream = 0);

#endif