#ifndef GPU_DATAVIEW_H
#define GPU_DATAVIEW_H

#include "gpu_structs.h"

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, cudaStream_t stream = 0);

#endif