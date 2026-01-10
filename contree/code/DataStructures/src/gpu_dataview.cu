// code/DataStructures/src/gpu_dataview.cu
#include "gpu_dataview.h"
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

// Kernel to mark which rows go to the Left child (1) or Right child (0)
// We only need to run this on the split feature column.
__global__ void mark_split_indices_kernel(
    const float* feature_values,
    const int* original_indices,
    int* row_to_side_map, // Output: Size = Max Original Index + 1 (or use hash map if sparse)
    int num_instances,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instances) {
        int row_id = original_indices[idx];
        // If value < threshold, it goes Left (1), else Right (0)
        row_to_side_map[row_id] = (feature_values[idx] < threshold) ? 1 : 0;
    }
}

// Custom predicate for Thrust to decide direction based on row ID
struct RowSidePredicate {
    const int* row_map;
    RowSidePredicate(const int* map) : row_map(map) {}

    __device__ bool operator()(const int& original_index) const {
        return row_map[original_index] == 1; // 1 means Left
    }
};

// Use a tuple zip iterator or custom kernel to copy all properties (val, label, etc)
// For simplicity, we can just run copy_if 4 times or use a struct.
// A custom kernel is often faster than 4 Thrust calls.

__global__ void partition_feature_column_kernel(
    const float* parent_val, const int* parent_lbl, const int* parent_ind, const int* parent_uniq,
    float* left_val, int* left_lbl, int* left_ind, int* left_uniq,
    float* right_val, int* right_lbl, int* right_ind, int* right_uniq,
    const int* row_to_side_map,
    int num_instances,
    int* left_count_out // Atomic counter for safety (optional if known)
) {
    // This requires a "Scan" operation to be parallel, which is what thrust::partition does.
    // Writing a raw kernel for stable partition is complex. 
    // It is HIGHLY recommended to use thrust::stable_partition_copy or copy_if.
}

void split_gpu_dataview(const GPUDataview& parent, GPUDataview& left, GPUDataview& right, int split_feat_idx, float threshold, cudaStream_t stream) {
    // 1. Setup metadata
    left.num_features = parent.num_features; left.num_classes = parent.num_classes;
    right.num_features = parent.num_features; right.num_classes = parent.num_classes;

    // 2. Determine split mask (Row ID -> Left/Right)
    // We need a map covering the range of original indices. 
    // Assuming original_indices are 0..TotalDatasetSize
    // Ideally pass `total_dataset_size` or find max.
    // For now, let's assume we allocate a temporary buffer for the map.
    int* d_row_map;
    int max_rows = 100000; // FIX: Should come from global config or parent
    cudaMalloc(&d_row_map, max_rows * sizeof(int));

    // Launch kernel on the specific feature column that determines the split
    int offset = split_feat_idx * parent.num_instances;
    int blockSize = 256;
    int gridSize = (parent.num_instances + blockSize - 1) / blockSize;
    
    mark_split_indices_kernel<<<gridSize, blockSize, 0, stream>>>(
        parent.d_values + offset,
        parent.d_original_indices + offset,
        d_row_map,
        parent.num_instances,
        threshold
    );

    // 3. Count Left/Right sizes (needed for allocation)
    // We can do this by counting the split feature column
    thrust::device_ptr<int> dev_map_ptr(d_row_map);
    // Note: We need to count valid entries in the map, but the map is sparse.
    // Better: Count how many items in the split feature column went left.
    // Or just count the predicate on the split feature.
    
    // Simplification: We assume we know the split point index from the CPU logic!
    // The CPU `Dataview::split_data_points` knows `split_point` (int). 
    // `left.num_instances = split_point;`
    // `right.num_instances = parent.num_instances - split_point;`
    // pass these counts into this function to save a reduction!
    
    // ALLOCATE MEMORY FOR CHILDREN
    size_t left_size = left.num_instances;
    size_t right_size = right.num_instances;
    
    cudaMalloc(&left.d_values, left_size * left.num_features * sizeof(float));
    // ... allocate other left/right buffers ...

    // 4. Partition every feature column
    for (int f = 0; f < parent.num_features; f++) {
        int p_offset = f * parent.num_instances;
        int l_offset = f * left_size;
        int r_offset = f * right_size;

        // Use Thrust to copy to Left/Right based on the d_row_map
        // We define a Zip Iterator to move Value+Label+Ind+Unique together?
        // Or just move indices and look up? 
        // Keeping it physical (copying values) is best for memory coalescence in solver.

        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(parent.d_values + p_offset),
            thrust::device_pointer_cast(parent.d_labels + p_offset),
            thrust::device_pointer_cast(parent.d_original_indices + p_offset),
            thrust::device_pointer_cast(parent.d_unique_indices + p_offset)
        ));

        auto zip_out_left = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(left.d_values + l_offset),
            thrust::device_pointer_cast(left.d_labels + l_offset),
            thrust::device_pointer_cast(left.d_original_indices + l_offset),
            thrust::device_pointer_cast(left.d_unique_indices + l_offset)
        ));
        
        auto zip_out_right = thrust::make_zip_iterator(thrust::make_tuple(
             // ... similar for right ...
        ));

        // STABLE_PARTITION_COPY is crucial to preserve sorting!
        thrust::stable_partition_copy(
            thrust::cuda::par.on(stream),
            zip_in, zip_in + parent.num_instances,
            zip_out_left,
            zip_out_right,
            RowSidePredicate(d_row_map) // Predicate checks d_original_indices part of tuple?
            // Wait, predicate needs the Original Index.
            // Thrust predicate receives the value of the iterator.
            // Our iterator is a tuple.
        );
    }
    
    cudaFree(d_row_map);
}