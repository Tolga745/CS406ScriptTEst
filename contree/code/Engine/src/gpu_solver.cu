#include "gpu_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>
#include <cassert>

// --- CONFIGURATION ---
// Support up to 32 classes (32 * 2 counters = 64 integers per thread)
// If you have a dataset with >32 classes, increase this number.
#define MAX_CLASSES 32 
#define MAX_COUNTERS (MAX_CLASSES * 2)

GPUDataset global_gpu_dataset;

// Global Pinned Memory Pointer
int* g_pinned_assignment_buffer = nullptr;

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
    const float* __restrict__ values,
    const int* __restrict__ labels,
    const int* __restrict__ original_indices,
    const int* __restrict__ feature_offsets,
    const int* __restrict__ active_mask, 
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

    // Shared Memory: Stores counters for every thread
    extern __shared__ int shared_counts[];
    
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    
    int start_idx = feature_offsets[feature_idx];
    int end_idx = feature_offsets[feature_idx + 1];
    int count = end_idx - start_idx;

    int chunk_size = (count + bdim - 1) / bdim;
    int my_start = tid * chunk_size;
    int my_end = min(my_start + chunk_size, count);

    int num_counters_per_thread = num_classes * 2;
    
    // Safety check (only needs to be checked once, usually done on host, but good for debug)
    // if (num_counters_per_thread > MAX_COUNTERS) return; 

    // 1. Initialize Shared Memory
    for (int i = 0; i < num_counters_per_thread; ++i) {
        shared_counts[tid * num_counters_per_thread + i] = 0;
    }
    __syncthreads();

    // ---------------------------------------------------------
    // PASS 1: Parallel Counting
    // ---------------------------------------------------------
    for (int i = my_start; i < my_end; ++i) {
        int global_idx = start_idx + i;
        int orig_idx = original_indices[global_idx];
        int assignment = active_mask[orig_idx]; 
        int label = labels[global_idx];

        if (assignment == 0) {
            shared_counts[tid * num_counters_per_thread + label]++; // Left
        } else if (assignment == 1) {
            shared_counts[tid * num_counters_per_thread + num_classes + label]++; // Right
        }
    }
    __syncthreads();

    // ---------------------------------------------------------
    // OPTIMIZATION: Parallel Prefix Sum (Hillis-Steele Scan)
    // ---------------------------------------------------------
    // We use a register array 'neighbor_vals' to avoid Read-After-Write hazards.
    
    for (int offset = 1; offset < bdim; offset *= 2) {
        int neighbor_vals[MAX_COUNTERS]; // FIXED: Increased size
        bool has_neighbor = (tid >= offset);
        
        // Read neighbor values
        if (has_neighbor) {
            for (int c = 0; c < num_counters_per_thread; ++c) {
                neighbor_vals[c] = shared_counts[(tid - offset) * num_counters_per_thread + c];
            }
        }
        __syncthreads(); 

        // Add neighbor values
        if (has_neighbor) {
            for (int c = 0; c < num_counters_per_thread; ++c) {
                shared_counts[tid * num_counters_per_thread + c] += neighbor_vals[c];
            }
        }
        __syncthreads();
    }

    // Recover Exclusive Sum for "Starting Counts"
    int my_starting_counts[MAX_COUNTERS]; // FIXED: Increased size
    if (tid > 0) {
        for (int c = 0; c < num_counters_per_thread; ++c) {
            my_starting_counts[c] = shared_counts[(tid - 1) * num_counters_per_thread + c];
        }
    } else {
        for(int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = 0;
    }
    
    // Calculate TOTALS (The last thread holds the sum of everyone)
    __shared__ int total_counts[MAX_COUNTERS]; // FIXED: Increased size
    if (tid == bdim - 1) {
        for (int c = 0; c < num_counters_per_thread; ++c) {
            total_counts[c] = shared_counts[tid * num_counters_per_thread + c];
        }
    }
    __syncthreads();

    // ---------------------------------------------------------
    // CALCULATE LEAF SCORES (Thread 0)
    // ---------------------------------------------------------
    if (tid == 0) {
        int total_L_size = 0;
        int total_R_size = 0;
        for(int c=0; c<num_classes; ++c) {
            total_L_size += total_counts[c];
            total_R_size += total_counts[num_classes + c];
        }

        int leaf_lbl_L, leaf_lbl_R;
        int leaf_err_L = calculate_misclassification(&total_counts[0], num_classes, total_L_size, leaf_lbl_L);
        int leaf_err_R = calculate_misclassification(&total_counts[num_classes], num_classes, total_R_size, leaf_lbl_R);
        
        leaf_scores_left[feature_idx] = leaf_err_L; leaf_labels_left[feature_idx] = leaf_lbl_L;
        leaf_scores_right[feature_idx] = leaf_err_R; leaf_labels_right[feature_idx] = leaf_lbl_R;
    }
    __syncthreads();

    // ---------------------------------------------------------
    // PASS 2: Split Finding
    // ---------------------------------------------------------
    
    int local_best_score_L = 99999999;
    float local_best_thresh_L = 0.0f;
    int local_best_lL_L=0, local_best_lR_L=0, local_best_cL_L=-1, local_best_cR_L=-1;

    int local_best_score_R = 99999999;
    float local_best_thresh_R = 0.0f;
    int local_best_lL_R=0, local_best_lR_R=0, local_best_cL_R=-1, local_best_cR_R=-1;

    // Local running counters
    int curr_counts_L[MAX_CLASSES]; // FIXED
    int curr_counts_R[MAX_CLASSES]; // FIXED
    for(int c=0; c<num_classes; ++c) {
        curr_counts_L[c] = my_starting_counts[c];
        curr_counts_R[c] = my_starting_counts[num_classes + c];
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
        int orig_idx = original_indices[global_idx];
        int assignment = active_mask[orig_idx];
        int label = labels[global_idx];

        bool value_changed = (val > prev_value + 1e-6f);
        
        if (value_changed && i > 0) { 
            float threshold = (prev_value + val) * 0.5f;

            // --- LEFT Node Check ---
            if (curr_L_size > 0 && curr_L_size < total_L_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                
                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) {
                    int rem = total_counts[c] - curr_counts_L[c];
                    if(rem > max_rem) { max_rem = rem; rem_lbl = c; }
                }
                int score_R = (total_L_size - curr_L_size) - max_rem;
                r_lbl = rem_lbl;

                int total_score = score_L + score_R;
                if (total_score < local_best_score_L) {
                    local_best_score_L = total_score;
                    local_best_thresh_L = threshold;
                    local_best_lL_L = l_lbl; local_best_lR_L = r_lbl;
                    local_best_cL_L = score_L; local_best_cR_L = score_R;
                }
            }
            
            // --- RIGHT Node Check ---
            if (curr_R_size > 0 && curr_R_size < total_R_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_R, num_classes, curr_R_size, l_lbl);

                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) {
                    int rem = total_counts[num_classes + c] - curr_counts_R[c];
                    if(rem > max_rem) { max_rem = rem; rem_lbl = c; }
                }
                int score_R = (total_R_size - curr_R_size) - max_rem;
                r_lbl = rem_lbl;

                int total_score = score_L + score_R;
                if (total_score < local_best_score_R) {
                    local_best_score_R = total_score;
                    local_best_thresh_R = threshold;
                    local_best_lL_R = l_lbl; local_best_lR_R = r_lbl;
                    local_best_cL_R = score_L; local_best_cR_R = score_R;
                }
            }
        }

        // Update Counters
        if (assignment == 0) {
            curr_counts_L[label]++; curr_L_size++;
        } else if (assignment == 1) {
            curr_counts_R[label]++; curr_R_size++;
        }
        prev_value = val;
    }

    // ---------------------------------------------------------
    // REDUCTION
    // ---------------------------------------------------------
    
    // REDUCE LEFT
    __syncthreads();
    shared_counts[tid] = local_best_score_L;
    __syncthreads();
    
    if (tid == 0) {
        int best_score = leaf_scores_left[feature_idx];
        int best_t = -1;
        for(int t=0; t<bdim; ++t) {
            if (shared_counts[t] < best_score) {
                best_score = shared_counts[t];
                best_t = t;
            }
        }
        best_scores_left[feature_idx] = best_score;
        shared_counts[0] = (best_t != -1) ? best_t : -1;
    }
    __syncthreads();
    
    int winner_L = shared_counts[0];
    if (winner_L != -1 && tid == winner_L) {
        best_thresholds_left[feature_idx] = local_best_thresh_L;
        best_labels_left_L[feature_idx] = local_best_lL_L;
        best_labels_left_R[feature_idx] = local_best_lR_L;
        best_child_scores_left_L[feature_idx] = local_best_cL_L;
        best_child_scores_left_R[feature_idx] = local_best_cR_L;
    }
    
    // REDUCE RIGHT
    __syncthreads();
    shared_counts[tid] = local_best_score_R;
    __syncthreads();
    
    if (tid == 0) {
        int best_score = leaf_scores_right[feature_idx];
        int best_t = -1;
        for(int t=0; t<bdim; ++t) {
            if (shared_counts[t] < best_score) {
                best_score = shared_counts[t];
                best_t = t;
            }
        }
        best_scores_right[feature_idx] = best_score;
        shared_counts[0] = best_t;
    }
    __syncthreads();
    
    int winner_R = shared_counts[0];
    if (winner_R != -1 && tid == winner_R) {
        best_thresholds_right[feature_idx] = local_best_thresh_R;
        best_labels_right_L[feature_idx] = local_best_lL_R;
        best_labels_right_R[feature_idx] = local_best_lR_R;
        best_child_scores_right_L[feature_idx] = local_best_cL_R;
        best_child_scores_right_R[feature_idx] = local_best_cR_R;
    }
}

void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
) {
    if (global_gpu_dataset.num_classes > MAX_CLASSES) {
        std::cerr << "ERROR: Dataset has " << global_gpu_dataset.num_classes 
                  << " classes, but MAX_CLASSES is " << MAX_CLASSES 
                  << ". Increase MAX_CLASSES in gpu_solver.cu!" << std::endl;
        exit(1);
    }

    // Write directly to Pinned Memory
    for(size_t i=0; i<active_indices.size(); ++i) {
        g_pinned_assignment_buffer[active_indices[i]] = split_assignment[i];
    }
    
    cudaMemset(global_gpu_dataset.d_assignment_buffer, -1, global_gpu_dataset.num_instances * sizeof(int));
    cudaMemcpy(global_gpu_dataset.d_assignment_buffer, g_pinned_assignment_buffer, global_gpu_dataset.num_instances * sizeof(int), cudaMemcpyHostToDevice);

    int num_feats = global_gpu_dataset.num_features;
    size_t int_bytes = num_feats * sizeof(int);
    size_t float_bytes = num_feats * sizeof(float);
    
    int block_size = 256;
    size_t shared_mem_size = (block_size * global_gpu_dataset.num_classes * 2) * sizeof(int);

    compute_splits_kernel<<<num_feats, block_size, shared_mem_size>>>(
        global_gpu_dataset.d_values, global_gpu_dataset.d_labels, global_gpu_dataset.d_original_indices, global_gpu_dataset.d_feature_offsets, 
        global_gpu_dataset.d_assignment_buffer,
        num_feats, global_gpu_dataset.num_instances, global_gpu_dataset.num_classes,
        
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, 
        global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,

        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, 
        global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );
    cudaDeviceSynchronize();

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

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    num_features = cpu_dataset.get_features_size();
    num_instances = cpu_dataset.get_instance_number();
    
    // Flatten Data
    int max_label = 0;
    std::vector<float> h_vals;
    std::vector<int> h_labels;
    std::vector<int> h_indices;
    std::vector<int> h_offsets;
    h_offsets.push_back(0);
    int current_offset = 0;

    const auto& features_data = cpu_dataset.get_features_data();
    for (const auto& feature_vec : features_data) {
        for (const auto& elem : feature_vec) {
            h_vals.push_back(elem.value);
            h_labels.push_back(elem.label);
            h_indices.push_back(elem.data_point_index);
            if (elem.label > max_label) max_label = elem.label;
        }
        current_offset += feature_vec.size();
        h_offsets.push_back(current_offset);
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

    // --- NEW: Allocate Workspace ---
    size_t int_bytes = num_features * sizeof(int);
    size_t float_bytes = num_features * sizeof(float);

    cudaMalloc(&d_assignment_buffer, num_instances * sizeof(int));
    
    // OPTIMIZATION: Allocate Pinned Memory for Host Buffer
    cudaMallocHost(&g_pinned_assignment_buffer, num_instances * sizeof(int));

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
    
    // Free workspace
    cudaFree(d_assignment_buffer);
    if (g_pinned_assignment_buffer) cudaFreeHost(g_pinned_assignment_buffer);
    
    cudaFree(d_score_L); cudaFree(d_thresh_L); cudaFree(d_lbl_L_L); cudaFree(d_lbl_L_R); cudaFree(d_cscore_L_L); cudaFree(d_cscore_L_R); cudaFree(d_leaf_L); cudaFree(d_leaflbl_L);
    cudaFree(d_score_R); cudaFree(d_thresh_R); cudaFree(d_lbl_R_L); cudaFree(d_lbl_R_R); cudaFree(d_cscore_R_L); cudaFree(d_cscore_R_R); cudaFree(d_leaf_R); cudaFree(d_leaflbl_R);
}