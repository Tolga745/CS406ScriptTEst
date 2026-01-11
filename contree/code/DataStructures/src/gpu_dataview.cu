#include "gpu_dataview.h"
#include "gpu_solver.cuh"
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>

// Predicate to decide direction based on the Row Map
// The input is a tuple: <value, label, original_index, unique_index> (or subset)
// We only check original_index against the map.
struct RowSidePredicate {
    const int* row_map;
    RowSidePredicate(const int* map) : row_map(map) {}

    template <typename Tuple>
    __device__ bool operator()(const Tuple& t) const {
        // We assume the Original Index is the 3rd element (index 2) of the tuple
        int original_idx = thrust::get<2>(t);
        // Map: 0 = Left, 1 = Right
        // Partition keeps elements where predicate is true in the first part (Left)
        return row_map[original_idx] == 0; 
    }
};



/__global__ void partition_features_kernel(
    const float* __restrict__ src_values,
    const int* __restrict__ src_labels,
    const int* __restrict__ src_indices,
    const int* __restrict__ row_map,
    float* dst_l_values, int* dst_l_labels, int* dst_l_indices,
    float* dst_r_values, int* dst_r_labels, int* dst_r_indices,
    int num_instances,
    int num_features
) {
    // Grid Y handles Features
    int f = blockIdx.y; 
    if (f >= num_features) return;

    // Offsets for this feature
    size_t src_offset = (size_t)f * num_instances;
    
}

// Corrected Kernel Definition
__global__ void partition_all_features_kernel(
    const float* __restrict__ src_val_base, const int* __restrict__ src_lbl_base, const int* __restrict__ src_idx_base,
    const int* __restrict__ row_map,
    float* dst_l_val_base, int* dst_l_lbl_base, int* dst_l_idx_base,
    float* dst_r_val_base, int* dst_r_lbl_base, int* dst_r_idx_base,
    int num_instances, int num_left, int num_right
) {
    int f = blockIdx.x; // 1D Grid of Features
    
    // Pointers for this feature
    const float* s_val = src_val_base + (size_t)f * num_instances;
    const int* s_lbl = src_lbl_base + (size_t)f * num_instances;
    const int* s_idx = src_idx_base + (size_t)f * num_instances;

    float* l_val = dst_l_val_base + (size_t)f * num_left;
    int* l_lbl = dst_l_lbl_base + (size_t)f * num_left;
    int* l_idx = dst_l_idx_base + (size_t)f * num_left;

    float* r_val = dst_r_val_base + (size_t)f * num_right;
    int* r_lbl = dst_r_lbl_base + (size_t)f * num_right;
    int* r_idx = dst_r_idx_base + (size_t)f * num_right;

    // Block-wise Scan & Scatter
    __shared__ int left_offset_base;
    __shared__ int right_offset_base;

    if (threadIdx.x == 0) { left_offset_base = 0; right_offset_base = 0; }
    __syncthreads();

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    for (int i = 0; i < num_instances; i += bdim) {
        int idx = i + tid;
        bool active = (idx < num_instances);
        
        int row_id = active ? s_idx[idx] : 0;
        int side = active ? row_map[row_id] : 1; // 0=Left, 1=Right. Default to Right if inactive (dump)
        bool is_left = active && (side == 0);
        bool is_right = active && (side == 1);

        // Exclusive Scan for this block
        // Naive shared mem scan for simplicity (Block size 256 is small)
        __shared__ int scan_l[256];
        __shared__ int scan_r[256];
        scan_l[tid] = is_left ? 1 : 0;
        scan_r[tid] = is_right ? 1 : 0;
        __syncthreads();

        // Hillis-Steele
        for (int offset = 1; offset < bdim; offset *= 2) {
            int val_l = 0, val_r = 0;
            if (tid >= offset) { val_l = scan_l[tid - offset]; val_r = scan_r[tid - offset]; }
            __syncthreads();
            if (tid >= offset) { scan_l[tid] += val_l; scan_r[tid] += val_r; }
            __syncthreads();
        }

        // Convert inclusive to exclusive and get total
        int total_l = scan_l[bdim-1];
        int total_r = scan_r[bdim-1];
        int my_idx_l = scan_l[tid] - (is_left ? 1 : 0);
        int my_idx_r = scan_r[tid] - (is_right ? 1 : 0);

        // Scatter
        if (is_left) {
            int dst_pos = left_offset_base + my_idx_l;
            l_val[dst_pos] = s_val[idx];
            l_lbl[dst_pos] = s_lbl[idx];
            l_idx[dst_pos] = s_idx[idx];
        }
        if (is_right) {
            int dst_pos = right_offset_base + my_idx_r;
            r_val[dst_pos] = s_val[idx];
            r_lbl[dst_pos] = s_lbl[idx];
            r_idx[dst_pos] = s_idx[idx];
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            left_offset_base += total_l;
            right_offset_base += total_r;
        }
        __syncthreads();
    }
}

// Generate Map Kernel (Kept same, just optimized formatting)
__global__ void mark_split_indices_kernel(const float* feature_values, const int* original_indices, int* row_to_side_map, int num_instances, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instances) {
        row_to_side_map[original_indices[idx]] = (feature_values[idx] < threshold) ? 0 : 1;
    }
}

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int child_depth, cudaStream_t stream) {
    left.num_features = parent.num_features; left.num_classes = parent.num_classes;
    right.num_features = parent.num_features; right.num_classes = parent.num_classes;

    // 1. Memory Management (Pool)
    bool use_pool = (child_depth < recursion_buffers.size());
    if (use_pool) {
        auto& buffer = recursion_buffers[child_depth];
        left.d_values = buffer.d_values; left.d_labels = buffer.d_labels; left.d_row_indices = buffer.d_row_indices;
        size_t left_sz = (size_t)left.num_instances * left.num_features;
        right.d_values = buffer.d_values + left_sz; right.d_labels = buffer.d_labels + left_sz; right.d_row_indices = buffer.d_row_indices + left_sz;
        left.owns_memory = false; right.owns_memory = false;
    } else {
        // Fallback for extreme depths (should be rare)
        size_t l_sz = (size_t)left.num_instances * left.num_features;
        size_t r_sz = (size_t)right.num_instances * right.num_features;
        cudaMalloc(&left.d_values, l_sz * sizeof(float)); cudaMalloc(&left.d_labels, l_sz * sizeof(int)); cudaMalloc(&left.d_row_indices, l_sz * sizeof(int));
        cudaMalloc(&right.d_values, r_sz * sizeof(float)); cudaMalloc(&right.d_labels, r_sz * sizeof(int)); cudaMalloc(&right.d_row_indices, r_sz * sizeof(int));
        left.owns_memory = true; right.owns_memory = true;
    }

    int* d_row_map_ptr = use_pool ? d_global_row_map : nullptr;
    if (!d_row_map_ptr) cudaMalloc(&d_row_map_ptr, parent.num_instances * sizeof(int));

    // 2. Mark Split (Assignment Map)
    int offset = split_feat_idx * parent.num_instances;
    int blockSize = 256;
    int gridSize = (parent.num_instances + blockSize - 1) / blockSize;
    
    mark_split_indices_kernel<<<gridSize, blockSize, 0, stream>>>(
        parent.d_values + offset, parent.d_row_indices + offset, d_row_map_ptr, parent.num_instances, threshold
    );

    // 3. Parallel Partition (One kernel call for ALL features)
    // This replaces the loop + Thrust calls
    partition_all_features_kernel<<<parent.num_features, 256, 0, stream>>>(
        parent.d_values, parent.d_labels, parent.d_row_indices,
        d_row_map_ptr,
        left.d_values, left.d_labels, left.d_row_indices,
        right.d_values, right.d_labels, right.d_row_indices,
        parent.num_instances, left.num_instances, right.num_instances
    );

    if (!use_pool) cudaFree(d_row_map_ptr);
}