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

// Kernel to generate the Left/Right map based on threshold
__global__ void mark_split_indices_kernel(
    const float* feature_values,
    const int* original_indices,
    int* row_to_side_map, 
    int num_instances,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instances) {
        int row_id = original_indices[idx];
        // If value < threshold, go Left (0), else Right (1)
        row_to_side_map[row_id] = (feature_values[idx] < threshold) ? 0 : 1;
    }
}

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int child_depth, cudaStream_t stream) {
    left.num_features = parent.num_features; left.num_classes = parent.num_classes;
    right.num_features = parent.num_features; right.num_classes = parent.num_classes;

    // --- OPTIMIZATION START ---
    bool use_pool = (child_depth < recursion_buffers.size());

    static int debug_counter = 0;
    if (debug_counter++ % 1000 == 0) {
        if (use_pool) {
            std::cout << "[GPU] Using Pool for depth " << child_depth << std::endl;
        } else {
            std::cout << "[GPU] !!! FALLBACK TO MALLOC (Pool Miss) !!! Depth: " << child_depth << " PoolSize: " << recursion_buffers.size() << std::endl;
        }
    }

    if (use_pool) {
        auto& buffer = recursion_buffers[child_depth];
        
        // Point Left to the start of the pre-allocated buffer
        left.d_values      = buffer.d_values;
        left.d_labels      = buffer.d_labels;
        left.d_row_indices = buffer.d_row_indices;
        left.owns_memory   = false;

        // Point Right to the memory immediately after Left's data
        size_t left_total_size = (size_t)left.num_instances * left.num_features;
        right.d_values      = buffer.d_values + left_total_size;
        right.d_labels      = buffer.d_labels + left_total_size;
        right.d_row_indices = buffer.d_row_indices + left_total_size;
        right.owns_memory   = false;
    } else {
        // Fallback (Slow)
        size_t left_elem = (size_t)left.num_instances * left.num_features;
        size_t right_elem = (size_t)right.num_instances * right.num_features;
        cudaMalloc(&left.d_values, left_elem * sizeof(float));
        cudaMalloc(&left.d_labels, left_elem * sizeof(int));
        cudaMalloc(&left.d_row_indices, left_elem * sizeof(int));
        cudaMalloc(&right.d_values, right_elem * sizeof(float));
        cudaMalloc(&right.d_labels, right_elem * sizeof(int));
        cudaMalloc(&right.d_row_indices, right_elem * sizeof(int));
        left.owns_memory = true;
        right.owns_memory = true;
    }
    
    int* d_row_map_ptr = use_pool ? d_global_row_map : nullptr;
    if (!d_row_map_ptr) cudaMalloc(&d_row_map_ptr, parent.num_instances * sizeof(int));

    // --- OPTIMIZATION END ---

    int offset = split_feat_idx * parent.num_instances;
    int blockSize = 256;
    int gridSize = (parent.num_instances + blockSize - 1) / blockSize;
    
    mark_split_indices_kernel<<<gridSize, blockSize, 0, stream>>>(
        parent.d_values + offset,
        parent.d_row_indices + offset,
        d_row_map_ptr,
        parent.num_instances,
        threshold
    );

    for (int f = 0; f < parent.num_features; f++) {
        size_t p_offset = (size_t)f * parent.num_instances;
        size_t l_offset = (size_t)f * left.num_instances;
        size_t r_offset = (size_t)f * right.num_instances;

        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(parent.d_values + p_offset),
            thrust::device_pointer_cast(parent.d_labels + p_offset),
            thrust::device_pointer_cast(parent.d_row_indices + p_offset)
        ));

        auto zip_out_left = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(left.d_values + l_offset),
            thrust::device_pointer_cast(left.d_labels + l_offset),
            thrust::device_pointer_cast(left.d_row_indices + l_offset)
        ));
        
        auto zip_out_right = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(right.d_values + r_offset),
            thrust::device_pointer_cast(right.d_labels + r_offset),
            thrust::device_pointer_cast(right.d_row_indices + r_offset)
        ));

        thrust::stable_partition_copy(
            thrust::cuda::par.on(stream),
            zip_in, zip_in + parent.num_instances,
            zip_out_left,
            zip_out_right,
            RowSidePredicate(d_row_map_ptr)
        );
    }

    if (!use_pool) cudaFree(d_row_map_ptr);
}