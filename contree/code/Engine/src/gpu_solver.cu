#include "gpu_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include <omp.h>
#include <cstring> // For memset

// --- CONFIGURATION ---
#define MAX_CLASSES 32 
#define MAX_COUNTERS (MAX_CLASSES * 2)
#define NUM_STREAMS 32 // Pool size for streams

GPUDataset global_gpu_dataset;

// --- PER-STREAM BUFFERS (Fixes Race Condition) ---
cudaStream_t g_streams[NUM_STREAMS];
int* g_d_assignment_buffers[NUM_STREAMS];      // GPU Buffers
int* g_h_pinned_assignment_buffers[NUM_STREAMS]; // CPU Pinned Buffers

// ---------------------------------------------------------
// KERNEL: Generate Assignments (Fast Path)
// ---------------------------------------------------------
__global__ void generate_assignments_kernel(
    const float* __restrict__ values,
    const int* __restrict__ original_indices,
    const int* __restrict__ feature_offsets,
    int* __restrict__ active_mask,
    int feature_index,
    float threshold,
    int count
) {
    int start_idx = feature_offsets[feature_index];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int global_idx = start_idx + tid;
        int orig_idx = original_indices[global_idx];
        active_mask[orig_idx] = (values[global_idx] < threshold) ? 0 : 1;
    }
}

__device__ int calculate_misclassification(int* counts, int num_classes, int total_count, int& best_label_out) {
    int max_freq = 0;
    int best_label = 0;
    for(int c = 0; c < num_classes; ++c) {
        if(counts[c] > max_freq) {
            max_freq = counts[c];
            best_label = c;
        }
    }
    best_label_out = best_label;
    return total_count - max_freq;
}

__global__ void compute_splits_kernel(
    const float* __restrict__ values,       // GPUDataview.d_values
    const int* __restrict__ labels,         // GPUDataview.d_labels
    const int* __restrict__ row_indices,    // GPUDataview.d_data_point_indices (0..N-1)
    const int* __restrict__ assignment_map, // The split assignments (0 for Left, 1 for Right)
    int num_features,
    int num_instances,                      // Size of the current GPUDataview
    int num_classes,
    
    // Outputs (Size: num_features)
    int* best_scores_left, float* best_thresholds_left, int* best_labels_left_L, int* best_labels_left_R,
    int* best_child_scores_left_L, int* best_child_scores_left_R,
    int* leaf_scores_left, int* leaf_labels_left,

    int* best_scores_right, float* best_thresholds_right, int* best_labels_right_L, int* best_labels_right_R,
    int* best_child_scores_right_L, int* best_child_scores_right_R,
    int* leaf_scores_right, int* leaf_labels_right
) {
    int feature_idx = blockIdx.x;
    if (feature_idx >= num_features) return;

    extern __shared__ int shared_counts[];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    
    // CHANGE 1: Calculate offsets based on compact GPUDataview structure
    int start_idx = feature_idx * num_instances;
    // int end_idx = start_idx + num_instances; // Not strictly needed, we use count
    int count = num_instances;

    int chunk_size = (count + bdim - 1) / bdim;
    int my_start = tid * chunk_size;
    int my_end = min(my_start + chunk_size, count);

    int num_counters_per_thread = num_classes * 2;
    // Initialize shared memory
    for (int i = 0; i < num_counters_per_thread; ++i) shared_counts[tid * num_counters_per_thread + i] = 0;
    __syncthreads();

    // PASS 1: Parallel Counting
    for (int i = my_start; i < my_end; ++i) {
        int global_idx = start_idx + i;
        
        // CHANGE 2: Use row_indices to look up the assignment map
        // row_indices maps the sorted column position (i) to the dense row ID (0..N-1)
        int row_id = row_indices[global_idx]; 
        int assignment = assignment_map[row_id]; 
        
        int label = labels[global_idx];

        if (assignment == 0) shared_counts[tid * num_counters_per_thread + label]++; 
        else if (assignment == 1) shared_counts[tid * num_counters_per_thread + num_classes + label]++; 
    }
    __syncthreads();

    // Hillis-Steele Scan (Unchanged logic)
    for (int offset = 1; offset < bdim; offset *= 2) {
        int neighbor_vals[MAX_COUNTERS]; 
        bool has_neighbor = (tid >= offset);
        if (has_neighbor) {
            for (int c = 0; c < num_counters_per_thread; ++c) neighbor_vals[c] = shared_counts[(tid - offset) * num_counters_per_thread + c];
        }
        __syncthreads(); 
        if (has_neighbor) {
            for (int c = 0; c < num_counters_per_thread; ++c) shared_counts[tid * num_counters_per_thread + c] += neighbor_vals[c];
        }
        __syncthreads();
    }

    int my_starting_counts[MAX_COUNTERS]; 
    if (tid > 0) {
        for (int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = shared_counts[(tid - 1) * num_counters_per_thread + c];
    } else {
        for(int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = 0;
    }
    
    __shared__ int total_counts[MAX_COUNTERS]; 
    if (tid == bdim - 1) {
        for (int c = 0; c < num_counters_per_thread; ++c) total_counts[c] = shared_counts[tid * num_counters_per_thread + c];
    }
    __syncthreads();

    // Leaf Scores (Thread 0) - Unchanged logic
    if (tid == 0) {
        int total_L_size = 0; int total_R_size = 0;
        for(int c=0; c<num_classes; ++c) {
            total_L_size += total_counts[c]; total_R_size += total_counts[num_classes + c];
        }
        int leaf_lbl_L, leaf_lbl_R;
        int leaf_err_L = calculate_misclassification(&total_counts[0], num_classes, total_L_size, leaf_lbl_L);
        int leaf_err_R = calculate_misclassification(&total_counts[num_classes], num_classes, total_R_size, leaf_lbl_R);
        leaf_scores_left[feature_idx] = leaf_err_L; leaf_labels_left[feature_idx] = leaf_lbl_L;
        leaf_scores_right[feature_idx] = leaf_err_R; leaf_labels_right[feature_idx] = leaf_lbl_R;
    }
    __syncthreads();

    // PASS 2: Split Finding - Unchanged logic but uses updated offsets
    int local_best_score_L = 99999999; float local_best_thresh_L = 0.0f;
    int local_best_lL_L=0, local_best_lR_L=0, local_best_cL_L=-1, local_best_cR_L=-1;
    int local_best_score_R = 99999999; float local_best_thresh_R = 0.0f;
    int local_best_lL_R=0, local_best_lR_R=0, local_best_cL_R=-1, local_best_cR_R=-1;

    int curr_counts_L[MAX_CLASSES]; int curr_counts_R[MAX_CLASSES]; 
    for(int c=0; c<num_classes; ++c) {
        curr_counts_L[c] = my_starting_counts[c]; curr_counts_R[c] = my_starting_counts[num_classes + c];
    }
    int curr_L_size = 0; for(int c=0; c<num_classes; ++c) curr_L_size += curr_counts_L[c];
    int curr_R_size = 0; for(int c=0; c<num_classes; ++c) curr_R_size += curr_counts_R[c];
    int total_L_size = 0; for(int c=0; c<num_classes; ++c) total_L_size += total_counts[c];
    int total_R_size = 0; for(int c=0; c<num_classes; ++c) total_R_size += total_counts[num_classes+c];

    float prev_value = -999999.0f;
    if (my_start > 0) prev_value = values[start_idx + my_start - 1];
    else if (count > 0 && my_start == 0) prev_value = values[start_idx];

    for (int i = my_start; i < my_end; ++i) {
        int global_idx = start_idx + i;
        float val = values[global_idx];
        
        // CHANGE 3: Use row_indices again for assignment lookup
        int row_id = row_indices[global_idx];
        int assignment = assignment_map[row_id];
        
        int label = labels[global_idx];
        bool value_changed = (val > prev_value + 1e-6f);
        
        if (value_changed && i > 0) { 
            float threshold = (prev_value + val) * 0.5f;
            if (curr_L_size > 0 && curr_L_size < total_L_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) { int rem = total_counts[c] - curr_counts_L[c]; if(rem > max_rem) { max_rem = rem; rem_lbl = c; } }
                int score_R = (total_L_size - curr_L_size) - max_rem;
                int total_score = score_L + score_R;
                if (total_score < local_best_score_L) { local_best_score_L = total_score; local_best_thresh_L = threshold; local_best_lL_L = l_lbl; local_best_lR_L = rem_lbl; local_best_cL_L = score_L; local_best_cR_L = score_R; }
            }
            if (curr_R_size > 0 && curr_R_size < total_R_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_R, num_classes, curr_R_size, l_lbl);
                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) { int rem = total_counts[num_classes + c] - curr_counts_R[c]; if(rem > max_rem) { max_rem = rem; rem_lbl = c; } }
                int score_R = (total_R_size - curr_R_size) - max_rem;
                int total_score = score_L + score_R;
                if (total_score < local_best_score_R) { local_best_score_R = total_score; local_best_thresh_R = threshold; local_best_lL_R = l_lbl; local_best_lR_R = rem_lbl; local_best_cL_R = score_L; local_best_cR_R = score_R; }
            }
        }
        if (assignment == 0) { curr_counts_L[label]++; curr_L_size++; } 
        else if (assignment == 1) { curr_counts_R[label]++; curr_R_size++; }
        prev_value = val;
    }

    // Reduction - Unchanged logic
    __syncthreads(); shared_counts[tid] = local_best_score_L; __syncthreads();
    if (tid == 0) {
        int best_score = leaf_scores_left[feature_idx]; int best_t = -1;
        for(int t=0; t<bdim; ++t) { if (shared_counts[t] < best_score) { best_score = shared_counts[t]; best_t = t; } }
        best_scores_left[feature_idx] = best_score; shared_counts[0] = best_t;
    }
    __syncthreads();
    if (shared_counts[0] != -1 && tid == shared_counts[0]) {
        best_thresholds_left[feature_idx] = local_best_thresh_L; best_labels_left_L[feature_idx] = local_best_lL_L; best_labels_left_R[feature_idx] = local_best_lR_L; best_child_scores_left_L[feature_idx] = local_best_cL_L; best_child_scores_left_R[feature_idx] = local_best_cR_L;
    }
    __syncthreads(); shared_counts[tid] = local_best_score_R; __syncthreads();
    if (tid == 0) {
        int best_score = leaf_scores_right[feature_idx]; int best_t = -1;
        for(int t=0; t<bdim; ++t) { if (shared_counts[t] < best_score) { best_score = shared_counts[t]; best_t = t; } }
        best_scores_right[feature_idx] = best_score; shared_counts[0] = best_t;
    }
    __syncthreads();
    if (shared_counts[0] != -1 && tid == shared_counts[0]) {
        best_thresholds_right[feature_idx] = local_best_thresh_R; best_labels_right_L[feature_idx] = local_best_lL_R; best_labels_right_R[feature_idx] = local_best_lR_R; best_child_scores_right_L[feature_idx] = local_best_cL_R; best_child_scores_right_R[feature_idx] = local_best_cR_R;
    }
}

// ---------------------------------------------------------
// LAUNCHER 1: Fast Path (GPU Generation)
// ---------------------------------------------------------
void launch_specialized_solver_kernel_gpu_gen(
    int feature_index, float threshold, int upper_bound,
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
) {
    if (global_gpu_dataset.num_classes > MAX_CLASSES) { std::cerr << "ERR: Class limit exceeded" << std::endl; exit(1); }
    
    int thread_id = 0;
    #ifdef _OPENMP
        thread_id = omp_get_thread_num();
    #endif
    int stream_idx = thread_id % NUM_STREAMS;
    cudaStream_t stream = g_streams[stream_idx];
    int* d_assignment_buffer = g_d_assignment_buffers[stream_idx]; // Use Private Buffer

    int num_instances = global_gpu_dataset.num_instances;
    int blockSize = 256;
    int gridSize = (num_instances + blockSize - 1) / blockSize;

    generate_assignments_kernel<<<gridSize, blockSize, 0, stream>>>(
        global_gpu_dataset.d_values, global_gpu_dataset.d_original_indices, global_gpu_dataset.d_feature_offsets, 
        d_assignment_buffer, feature_index, threshold, num_instances
    );

    int num_feats = global_gpu_dataset.num_features;
    size_t shared_mem_size = (256 * global_gpu_dataset.num_classes * 2) * sizeof(int);

    compute_splits_kernel<<<num_feats, 256, shared_mem_size, stream>>>(
        global_gpu_dataset.d_values, global_gpu_dataset.d_labels, global_gpu_dataset.d_original_indices, global_gpu_dataset.d_feature_offsets, 
        d_assignment_buffer,
        num_feats, global_gpu_dataset.num_instances, global_gpu_dataset.num_classes,
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,
        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );
    
    // Copy Results...
    size_t int_bytes = num_feats * sizeof(int); size_t float_bytes = num_feats * sizeof(float);
    cudaMemcpyAsync(h_best_scores_left, global_gpu_dataset.d_score_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_thresholds_left, global_gpu_dataset.d_thresh_L, float_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_left_L, global_gpu_dataset.d_lbl_L_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_left_R, global_gpu_dataset.d_lbl_L_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_left_L, global_gpu_dataset.d_cscore_L_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_left_R, global_gpu_dataset.d_cscore_L_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_scores_left, global_gpu_dataset.d_leaf_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_labels_left, global_gpu_dataset.d_leaflbl_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_scores_right, global_gpu_dataset.d_score_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_thresholds_right, global_gpu_dataset.d_thresh_R, float_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_right_L, global_gpu_dataset.d_lbl_R_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_right_R, global_gpu_dataset.d_lbl_R_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_right_L, global_gpu_dataset.d_cscore_R_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_right_R, global_gpu_dataset.d_cscore_R_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_scores_right, global_gpu_dataset.d_leaf_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_labels_right, global_gpu_dataset.d_leaflbl_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------
// LAUNCHER 2: Slow Path (FIXED SCATTER + RACE CONDITION)
// ---------------------------------------------------------
void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
) {
    if (global_gpu_dataset.num_classes > MAX_CLASSES) { std::cerr << "ERR: Class limit exceeded" << std::endl; exit(1); }

    int thread_id = 0;
    #ifdef _OPENMP
        thread_id = omp_get_thread_num();
    #endif
    int stream_idx = thread_id % NUM_STREAMS;
    cudaStream_t stream = g_streams[stream_idx];
    
    // Use PRIVATE buffers for this stream (prevents race conditions)
    int* d_assignment_buffer = g_d_assignment_buffers[stream_idx];
    int* h_pinned_buffer = g_h_pinned_assignment_buffers[stream_idx];
    int num_instances = global_gpu_dataset.num_instances;

    // 1. Reset Host Buffer (Must be -1 for inactive indices)
    std::memset(h_pinned_buffer, -1, num_instances * sizeof(int));

    // 2. SCATTER assignments to correct positions (FIXED BUG)
    // The kernel expects active_mask[original_index] to be 0 or 1.
    for(size_t i=0; i<active_indices.size(); ++i) {
        h_pinned_buffer[active_indices[i]] = split_assignment[i];
    }

    // 3. Copy valid buffer to GPU
    cudaMemcpyAsync(d_assignment_buffer, h_pinned_buffer, num_instances * sizeof(int), cudaMemcpyHostToDevice, stream);

    int num_feats = global_gpu_dataset.num_features;
    size_t shared_mem_size = (256 * global_gpu_dataset.num_classes * 2) * sizeof(int);

    compute_splits_kernel<<<num_feats, 256, shared_mem_size, stream>>>(
        global_gpu_dataset.d_values, global_gpu_dataset.d_labels, global_gpu_dataset.d_original_indices, global_gpu_dataset.d_feature_offsets, 
        d_assignment_buffer, // Pass stream-specific buffer
        num_feats, global_gpu_dataset.num_instances, global_gpu_dataset.num_classes,
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,
        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );

    // Retrieve results... (same as above)
    size_t int_bytes = num_feats * sizeof(int); size_t float_bytes = num_feats * sizeof(float);
    cudaMemcpyAsync(h_best_scores_left, global_gpu_dataset.d_score_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_thresholds_left, global_gpu_dataset.d_thresh_L, float_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_left_L, global_gpu_dataset.d_lbl_L_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_left_R, global_gpu_dataset.d_lbl_L_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_left_L, global_gpu_dataset.d_cscore_L_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_left_R, global_gpu_dataset.d_cscore_L_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_scores_left, global_gpu_dataset.d_leaf_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_labels_left, global_gpu_dataset.d_leaflbl_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_scores_right, global_gpu_dataset.d_score_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_thresholds_right, global_gpu_dataset.d_thresh_R, float_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_right_L, global_gpu_dataset.d_lbl_R_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_labels_right_R, global_gpu_dataset.d_lbl_R_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_right_L, global_gpu_dataset.d_cscore_R_L, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_best_child_scores_right_R, global_gpu_dataset.d_cscore_R_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_scores_right, global_gpu_dataset.d_leaf_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_leaf_labels_right, global_gpu_dataset.d_leaflbl_R, int_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    num_features = cpu_dataset.get_features_size();
    num_instances = cpu_dataset.get_instance_number();
    
    // Initialize Streams and Buffers
    for(int i=0; i<NUM_STREAMS; ++i) {
        cudaStreamCreate(&g_streams[i]);
        cudaMalloc(&g_d_assignment_buffers[i], num_instances * sizeof(int));
        cudaMallocHost(&g_h_pinned_assignment_buffers[i], num_instances * sizeof(int));
    }

    // Flatten Data (Standard logic...)
    int max_label = 0;
    std::vector<float> h_vals; std::vector<int> h_labels;
    std::vector<int> h_indices; std::vector<int> h_offsets;
    h_offsets.push_back(0); int current_offset = 0;

    const auto& features_data = cpu_dataset.get_features_data();
    for (const auto& feature_vec : features_data) {
        for (const auto& elem : feature_vec) {
            h_vals.push_back(elem.value); h_labels.push_back(elem.label); h_indices.push_back(elem.data_point_index);
            if (elem.label > max_label) max_label = elem.label;
        }
        current_offset += feature_vec.size(); h_offsets.push_back(current_offset);
    }
    num_classes = max_label + 1;
    
    cudaMalloc(&d_values, h_vals.size() * sizeof(float));
    cudaMalloc(&d_labels, h_labels.size() * sizeof(int));
    cudaMalloc(&d_original_indices, h_indices.size() * sizeof(int));
    cudaMalloc(&d_feature_offsets, h_offsets.size() * sizeof(int));

    cudaMemcpy(d_values, h_vals.data(), h_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_indices, h_indices.data(), h_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    size_t int_bytes = num_features * sizeof(int); size_t float_bytes = num_features * sizeof(float);
    cudaMalloc(&d_score_L, int_bytes); cudaMalloc(&d_thresh_L, float_bytes);
    cudaMalloc(&d_lbl_L_L, int_bytes); cudaMalloc(&d_lbl_L_R, int_bytes); 
    cudaMalloc(&d_cscore_L_L, int_bytes); cudaMalloc(&d_cscore_L_R, int_bytes);
    cudaMalloc(&d_leaf_L, int_bytes); cudaMalloc(&d_leaflbl_L, int_bytes);
    cudaMalloc(&d_score_R, int_bytes); cudaMalloc(&d_thresh_R, float_bytes);
    cudaMalloc(&d_lbl_R_L, int_bytes); cudaMalloc(&d_lbl_R_R, int_bytes); 
    cudaMalloc(&d_cscore_R_L, int_bytes); cudaMalloc(&d_cscore_R_R, int_bytes);
    cudaMalloc(&d_leaf_R, int_bytes); cudaMalloc(&d_leaflbl_R, int_bytes);
}

void GPUDataset::free() {
    cudaFree(d_values); cudaFree(d_labels); cudaFree(d_original_indices); cudaFree(d_feature_offsets);
    cudaFree(d_score_L); cudaFree(d_thresh_L); cudaFree(d_lbl_L_L); cudaFree(d_lbl_L_R); cudaFree(d_cscore_L_L); cudaFree(d_cscore_L_R); cudaFree(d_leaf_L); cudaFree(d_leaflbl_L);
    cudaFree(d_score_R); cudaFree(d_thresh_R); cudaFree(d_lbl_R_L); cudaFree(d_lbl_R_R); cudaFree(d_cscore_R_L); cudaFree(d_cscore_R_R); cudaFree(d_leaf_R); cudaFree(d_leaflbl_R);

    // Free Streams and Buffers
    for(int i=0; i<NUM_STREAMS; ++i) {
        cudaStreamDestroy(g_streams[i]);
        cudaFree(g_d_assignment_buffers[i]);
        cudaFreeHost(g_h_pinned_assignment_buffers[i]);
    }
}