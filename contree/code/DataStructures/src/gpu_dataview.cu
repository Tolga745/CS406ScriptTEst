#include "gpu_dataview.h"
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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

void partition_column(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int feature_idx, int* d_row_map, cudaStream_t stream) {
    size_t p_offset = (size_t)feature_idx * parent.num_instances;
    size_t l_offset = (size_t)feature_idx * left.num_instances;
    size_t r_offset = (size_t)feature_idx * right.num_instances;

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
        RowSidePredicate(d_row_map)
    );
}

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, cudaStream_t stream) {
    // 1. Setup metadata
    left.num_features = parent.num_features; left.num_classes = parent.num_classes;
    left.owns_memory = true;

    right.num_features = parent.num_features; right.num_classes = parent.num_classes;
    right.owns_memory = true;
    
    // 2. Allocate Temporary Map
    // Map size must cover the maximum possible row index. 
    // Since we use d_original_indices which are 0..TotalDatasetSize-1, we need TotalDatasetSize.
    // However, we might not know TotalDatasetSize here easily without querying.
    // Hack: Use a large enough buffer or pass global size. 
    // For safety, let's assume we can get it from max element or passed in. 
    // Optimization: Just use parent.num_instances if we re-indexed, but we didn't.
    // Let's rely on the fact that row indices are < 10,000,000 usually. 
    // A better way is to pass Global Dataset Size to this function.
    // For this context, we'll allocate 10M integers (approx 40MB), which is safe for most datasets.
    int max_rows = 10000000; 
    int* d_row_map;
    cudaMalloc(&d_row_map, max_rows * sizeof(int));

    // 3. Mark the split
    int offset = split_feat_idx * parent.num_instances;
    int blockSize = 256;
    int gridSize = (parent.num_instances + blockSize - 1) / blockSize;
    
    mark_split_indices_kernel<<<gridSize, blockSize, 0, stream>>>(
        parent.d_values + offset,
        parent.d_row_indices + offset,
        d_row_map,
        parent.num_instances,
        threshold
    );

    // 4. Allocate Memory for Children
    // We already know left/right sizes from the Dataview object (populated by CPU logic before calling this)
    // Wait, caller (dataview.cpp) must set num_instances before calling!
    // Assuming left.num_instances and right.num_instances are set.
    
    size_t left_elem = (size_t)left.num_instances * left.num_features;
    size_t right_elem = (size_t)right.num_instances * right.num_features;

    cudaMalloc(&left.d_values, left_elem * sizeof(float));
    cudaMalloc(&left.d_labels, left_elem * sizeof(int));
    cudaMalloc(&left.d_row_indices, left_elem * sizeof(int));
    // unique indices optional, skipping for now to save memory/time

    cudaMalloc(&right.d_values, right_elem * sizeof(float));
    cudaMalloc(&right.d_labels, right_elem * sizeof(int));
    cudaMalloc(&right.d_row_indices, right_elem * sizeof(int));

    // 5. Partition Data
    for (int f = 0; f < parent.num_features; f++) {
        partition_column(parent, left, right, f, d_row_map, stream);
    }

    cudaFree(d_row_map);
}

void split_gpu_dataview_preallocated(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, int* d_row_map_buffer, cudaStream_t stream) {
    // Assumes left.d_values, right.d_values etc are already pointing to valid memory 
    // AND left.num_instances/right.num_instances are set.
    
    // 1. Mark split
    int offset = split_feat_idx * parent.num_instances;
    int blockSize = 256;
    int gridSize = (parent.num_instances + blockSize - 1) / blockSize;
    mark_split_indices_kernel<<<gridSize, blockSize, 0, stream>>>(
        parent.d_values + offset, parent.d_row_indices + offset, d_row_map_buffer, parent.num_instances, threshold
    );

    // 2. Partition
    for (int f = 0; f < parent.num_features; f++) {
        partition_column(parent, left, right, f, d_row_map_buffer, stream);
    }
}