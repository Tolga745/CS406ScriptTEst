#include "gpu_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm> // For std::max

// --- CONSTANTS ---
#define MAX_CLASSES 32 
#define MAX_COUNTERS (MAX_CLASSES * 2)

// --- GLOBAL DEFINITION ---
GPUDataset global_gpu_dataset;

// --- GPUDataset IMPLEMENTATION ---

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    this->num_features = cpu_dataset.get_features_size();
    this->num_instances = cpu_dataset.get_instance_number();
    
    // Calculate total elements
    size_t total_elements = (size_t)this->num_features * this->num_instances;
    
    // Allocate Host Memory for flattening
    float* h_values;
    int* h_labels;
    int* h_indices;
    // We scan for max label to set num_classes
    int max_label = 0;

    cudaMallocHost(&h_values, total_elements * sizeof(float));
    cudaMallocHost(&h_labels, total_elements * sizeof(int));
    cudaMallocHost(&h_indices, total_elements * sizeof(int));

    // Flatten Data (Feature-Major Order)
    size_t global_idx = 0;
    const auto& feature_data = cpu_dataset.get_features_data();

    for (int f = 0; f < this->num_features; f++) {
        const auto& current_feat_vec = feature_data[f];
        for (int i = 0; i < this->num_instances; i++) {
            const auto& element = current_feat_vec[i];
            
            h_values[global_idx] = element.value;
            h_labels[global_idx] = element.label;
            h_indices[global_idx] = element.data_point_index;
            
            if (element.label > max_label) max_label = element.label;

            global_idx++;
        }
    }
    this->num_classes = max_label + 1;

    // Allocate GPU Memory (Data)
    cudaMalloc(&d_values, total_elements * sizeof(float));
    cudaMalloc(&d_labels, total_elements * sizeof(int));
    cudaMalloc(&d_original_indices, total_elements * sizeof(int));
    d_feature_offsets = nullptr; // Unused in current kernel

    // Copy to GPU
    cudaMemcpy(d_values, h_values, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_indices, h_indices, total_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate GPU Memory (Buffers)
    // d_assignment_buffer needs to handle the largest possible instance count (Root)
    cudaMalloc(&d_assignment_buffer, this->num_instances * sizeof(int));
    
    size_t int_bytes_feats = this->num_features * sizeof(int);
    size_t float_bytes_feats = this->num_features * sizeof(float);

    cudaMalloc(&d_score_L, int_bytes_feats); cudaMalloc(&d_score_R, int_bytes_feats);
    cudaMalloc(&d_thresh_L, float_bytes_feats); cudaMalloc(&d_thresh_R, float_bytes_feats);
    cudaMalloc(&d_lbl_L_L, int_bytes_feats); cudaMalloc(&d_lbl_L_R, int_bytes_feats);
    cudaMalloc(&d_lbl_R_L, int_bytes_feats); cudaMalloc(&d_lbl_R_R, int_bytes_feats);
    cudaMalloc(&d_cscore_L_L, int_bytes_feats); cudaMalloc(&d_cscore_L_R, int_bytes_feats);
    cudaMalloc(&d_cscore_R_L, int_bytes_feats); cudaMalloc(&d_cscore_R_R, int_bytes_feats);
    cudaMalloc(&d_leaf_L, int_bytes_feats); cudaMalloc(&d_leaf_R, int_bytes_feats);
    cudaMalloc(&d_leaflbl_L, int_bytes_feats); cudaMalloc(&d_leaflbl_R, int_bytes_feats);

    // Free Host Memory
    cudaFreeHost(h_values);
    cudaFreeHost(h_labels);
    cudaFreeHost(h_indices);
}

void GPUDataset::free() {
    if (d_values) cudaFree(d_values);
    if (d_labels) cudaFree(d_labels);
    if (d_original_indices) cudaFree(d_original_indices);
    if (d_assignment_buffer) cudaFree(d_assignment_buffer);
    
    if (d_score_L) cudaFree(d_score_L); if (d_score_R) cudaFree(d_score_R);
    if (d_thresh_L) cudaFree(d_thresh_L); if (d_thresh_R) cudaFree(d_thresh_R);
    if (d_lbl_L_L) cudaFree(d_lbl_L_L); if (d_lbl_L_R) cudaFree(d_lbl_L_R);
    if (d_lbl_R_L) cudaFree(d_lbl_R_L); if (d_lbl_R_R) cudaFree(d_lbl_R_R);
    if (d_cscore_L_L) cudaFree(d_cscore_L_L); if (d_cscore_L_R) cudaFree(d_cscore_L_R);
    if (d_cscore_R_L) cudaFree(d_cscore_R_L); if (d_cscore_R_R) cudaFree(d_cscore_R_R);
    if (d_leaf_L) cudaFree(d_leaf_L); if (d_leaf_R) cudaFree(d_leaf_R);
    if (d_leaflbl_L) cudaFree(d_leaflbl_L); if (d_leaflbl_R) cudaFree(d_leaflbl_R);

    d_values = nullptr;
}

// --- KERNELS ---

// Kernel 1: Generate the Assignment Map (0=Left, 1=Right)
// UPDATED: Now uses row_indices to write to the correct Original ID location
__global__ void generate_assignment_map_kernel(
    const float* feature_column,
    const int* row_indices,    // NEW: Needed to map sorted position to original row ID
    int* assignment_map,
    int num_instances,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instances) {
        int row_id = row_indices[idx];
        // If value < threshold, go Left (0), else Right (1)
        assignment_map[row_id] = (feature_column[idx] < threshold) ? 0 : 1;
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

// Kernel 2: Compute Best Splits
__global__ void compute_splits_kernel(
    const float* __restrict__ values,
    const int* __restrict__ labels,
    const int* __restrict__ row_indices,
    const int* __restrict__ assignment_map,
    int num_features,
    int num_instances,
    int num_classes,
    
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
    
    int start_idx = feature_idx * num_instances;
    int count = num_instances;

    int chunk_size = (count + bdim - 1) / bdim;
    int my_start = tid * chunk_size;
    int my_end = min(my_start + chunk_size, count);

    int num_counters_per_thread = num_classes * 2;
    for (int i = 0; i < num_counters_per_thread; ++i) shared_counts[tid * num_counters_per_thread + i] = 0;
    __syncthreads();

    // PASS 1: Parallel Counting
    for (int i = my_start; i < my_end; ++i) {
        int global_idx = start_idx + i;
        
        int row_id = row_indices[global_idx]; 
        int assignment = assignment_map[row_id]; 
        
        int label = labels[global_idx];

        if (assignment == 0) shared_counts[tid * num_counters_per_thread + label]++; 
        else if (assignment == 1) shared_counts[tid * num_counters_per_thread + num_classes + label]++; 
    }
    __syncthreads();

    // Hillis-Steele Scan (Inclusive)
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

    // Prepare starting counts (Exclusive Scan) for Pass 2
    int my_starting_counts[MAX_COUNTERS]; 
    if (tid > 0) {
        for (int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = shared_counts[(tid - 1) * num_counters_per_thread + c];
    } else {
        for(int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = 0;
    }
    
    // Total Counts are in the last thread's shared memory
    __shared__ int total_counts[MAX_COUNTERS]; 
    if (tid == bdim - 1) {
        for (int c = 0; c < num_counters_per_thread; ++c) total_counts[c] = shared_counts[tid * num_counters_per_thread + c];
    }
    __syncthreads();

    // Leaf Scores (Only Thread 0 needs to compute this once per feature)
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

    // PASS 2: Split Finding
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
        
        int row_id = row_indices[global_idx];
        int assignment = assignment_map[row_id];
        
        int label = labels[global_idx];
        bool value_changed = (val > prev_value + 1e-6f);
        
        if (value_changed && i > 0) { 
            float threshold = (prev_value + val) * 0.5f;
            // Left Child Split Evaluation
            if (curr_L_size > 0 && curr_L_size < total_L_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) { int rem = total_counts[c] - curr_counts_L[c]; if(rem > max_rem) { max_rem = rem; rem_lbl = c; } }
                int score_R = (total_L_size - curr_L_size) - max_rem;
                int total_score = score_L + score_R;
                if (total_score < local_best_score_L) { local_best_score_L = total_score; local_best_thresh_L = threshold; local_best_lL_L = l_lbl; local_best_lR_L = rem_lbl; local_best_cL_L = score_L; local_best_cR_L = score_R; }
            }
            // Right Child Split Evaluation
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

    // Parallel Reduction for Best Scores
    // Reduce Left Score
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
    
    // Reduce Right Score
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

// --- UNIFIED LAUNCHER ---
void run_specialized_solver_gpu(
    const GPUDataview& dataview,
    int split_feature_index,
    float split_threshold,
    int upper_bound,
    
    // Outputs
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
) {
    if (dataview.num_classes > MAX_CLASSES) { std::cerr << \"ERR: Class limit exceeded\" << std::endl; exit(1); }

    // PERFORMANCE FIX: Use Pre-allocated Buffers from global_gpu_dataset
    // We assume global_gpu_dataset is initialized and large enough.
    // The buffers in global_gpu_dataset are size `num_features` (of the whole dataset).
    // The dataview might have the same number of features (vertical partition not supported yet), so indices match.

    int* d_assignment_map = global_gpu_dataset.d_assignment_buffer; 

    // 2. Generate Assignments (Left/Right) for the specific split being evaluated
    int block = 256;
    int grid = (dataview.num_instances + block - 1) / block;
    
    // Calculate pointer to the specific feature column for the split
    float* split_feature_col = dataview.d_values + (size_t)split_feature_index * dataview.num_instances;
    
    // CORRECTNESS FIX: Pass row_indices to map sorted position -> original ID
    int* split_feature_row_indices = dataview.d_row_indices + (size_t)split_feature_index * dataview.num_instances;

    generate_assignment_map_kernel<<<grid, block>>>(
        split_feature_col,
        split_feature_row_indices, // Fix
        d_assignment_map,
        dataview.num_instances,
        split_threshold
    );

    // 3. Run Solver
    size_t int_bytes = dataview.num_features * sizeof(int); 
    size_t float_bytes = dataview.num_features * sizeof(float);
    size_t shared_mem = (256 * dataview.num_classes * 2) * sizeof(int);

    // Use Global Buffers (No cudaMalloc here!)
    compute_splits_kernel<<<dataview.num_features, 256, shared_mem>>>(
        dataview.d_values,
        dataview.d_labels,
        dataview.d_row_indices,
        d_assignment_map,
        dataview.num_features,
        dataview.num_instances,
        dataview.num_classes,
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,
        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );

    // 4. Copy Back
    cudaMemcpy(h_best_scores_left, global_gpu_dataset.d_score_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_thresholds_left, global_gpu_dataset.d_thresh_L, float_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_left_L, global_gpu_dataset.d_lbl_L_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_left_R, global_gpu_dataset.d_lbl_L_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_child_scores_left_L, global_gpu_dataset.d_cscore_L_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_child_scores_left_R, global_gpu_dataset.d_cscore_L_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_scores_left, global_gpu_dataset.d_leaf_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_labels_left, global_gpu_dataset.d_leaflbl_L, int_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_best_scores_right, global_gpu_dataset.d_score_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_thresholds_right, global_gpu_dataset.d_thresh_R, float_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_right_L, global_gpu_dataset.d_lbl_R_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_right_R, global_gpu_dataset.d_lbl_R_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_child_scores_right_L, global_gpu_dataset.d_cscore_R_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_child_scores_right_R, global_gpu_dataset.d_cscore_R_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_scores_right, global_gpu_dataset.d_leaf_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_labels_right, global_gpu_dataset.d_leaflbl_R, int_bytes, cudaMemcpyDeviceToHost);
}