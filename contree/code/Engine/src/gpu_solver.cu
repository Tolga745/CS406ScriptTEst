#include "gpu_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

GPUDataset global_gpu_dataset;

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

    // Shared Memory Layout: [BlockDim * Num_Classes * 2]
    // Stores the private counts for each thread.
    // After Pass 1, we perform a Prefix Sum in-place on this array.
    extern __shared__ int shared_counts[];
    
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    
    int start_idx = feature_offsets[feature_idx];
    int end_idx = feature_offsets[feature_idx + 1];
    int count = end_idx - start_idx;

    // Calculate chunk size for this thread (Grid-Stride style, but we need deterministic chunks for Scan)
    // Simple Chunking: Each thread gets a contiguous block
    int chunk_size = (count + bdim - 1) / bdim;
    int my_start = tid * chunk_size;
    int my_end = min(my_start + chunk_size, count);

    // 1. Initialize Shared Memory
    int num_counters_per_thread = num_classes * 2;
    for (int i = 0; i < num_counters_per_thread; ++i) {
        shared_counts[tid * num_counters_per_thread + i] = 0;
    }
    __syncthreads();

    // ---------------------------------------------------------
    // PASS 1: Parallel Counting (Each thread counts its chunk)
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
    // PREFIX SUM (Scan) on Shared Memory
    // ---------------------------------------------------------
    // We need to know the running total of counts BEFORE this thread's chunk.
    // We do a naive prefix sum over threads (O(Threads^2) but Threads=256, so very fast)
    // Each thread 'tid' sums up counts from thread 0 to tid-1.
    
    // Create local copy of "Starting Counts" for this thread
    // (We use a register array if num_classes is small, assuming max 10 classes for now)
    // Dynamic allocation not possible in registers, so we use a small buffer or re-read shared.
    // For general safety, let's just modify shared memory in place cautiously or use a temporary buffer.
    
    // To allow fully parallel scan, we can just let every thread read the others.
    // Simple iterative scan:
    int my_starting_counts[20]; // Hardcoded max classes = 10 (Left+Right) to avoid dynamic arrays
    for(int c=0; c<num_counters_per_thread; ++c) my_starting_counts[c] = 0;

    if (tid > 0) {
        for (int t = 0; t < tid; ++t) {
            for (int c = 0; c < num_counters_per_thread; ++c) {
                my_starting_counts[c] += shared_counts[t * num_counters_per_thread + c];
            }
        }
    }
    
    // Calculate TOTALS (last thread finishes the sum)
    __shared__ int total_counts[20];
    if (tid == bdim - 1) {
        for (int c = 0; c < num_counters_per_thread; ++c) {
            total_counts[c] = my_starting_counts[c] + shared_counts[tid * num_counters_per_thread + c];
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
    // PASS 2: Parallel Scan & Split Finding
    // ---------------------------------------------------------
    // Each thread initializes its running counters with 'my_starting_counts'
    // Then scans its own chunk.
    
    int local_best_score_L = 99999999;
    float local_best_thresh_L = 0.0f;
    int local_best_lL_L=0, local_best_lR_L=0, local_best_cL_L=-1, local_best_cR_L=-1;

    int local_best_score_R = 99999999;
    float local_best_thresh_R = 0.0f;
    int local_best_lL_R=0, local_best_lR_R=0, local_best_cL_R=-1, local_best_cR_R=-1;

    // Local running counters
    int curr_counts_L[10];
    int curr_counts_R[10];
    for(int c=0; c<num_classes; ++c) {
        curr_counts_L[c] = my_starting_counts[c];
        curr_counts_R[c] = my_starting_counts[num_classes + c];
    }
    
    int curr_L_size = 0; for(int c=0; c<num_classes; ++c) curr_L_size += curr_counts_L[c];
    int curr_R_size = 0; for(int c=0; c<num_classes; ++c) curr_R_size += curr_counts_R[c];
    
    int total_L_size = 0; for(int c=0; c<num_classes; ++c) total_L_size += total_counts[c];
    int total_R_size = 0; for(int c=0; c<num_classes; ++c) total_R_size += total_counts[num_classes+c];

    // Previous Value logic: If we are at start of chunk, check the global array for prev value
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
        
        // Only check split if value changed AND we are not at the absolute start
        if (value_changed && i > 0) { 
            float threshold = (prev_value + val) * 0.5f;

            // --- LEFT Node Check ---
            if (curr_L_size > 0 && curr_L_size < total_L_size) {
                int l_lbl, r_lbl;
                int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                
                // Remainder logic
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
    // REDUCTION: Find Best Split Across All Threads
    // ---------------------------------------------------------
    // Store local bests in shared memory to reduce
    // Re-use shared_counts buffer (it's large enough)
    // Structure: [tid] -> score
    
    // REDUCE LEFT
    __syncthreads();
    shared_counts[tid] = local_best_score_L;
    __syncthreads();
    
    // Simple reduction by Thread 0
    if (tid == 0) {
        int best_score = leaf_scores_left[feature_idx]; // Start with Leaf Score
        int best_t = -1;
        
        for(int t=0; t<bdim; ++t) {
            if (shared_counts[t] < best_score) {
                best_score = shared_counts[t];
                best_t = t;
            }
        }
        
        best_scores_left[feature_idx] = best_score;
        if (best_t != -1) {
            // Wait, Thread 0 needs the details (threshold, labels) from Thread best_t
            // We need a way to communicate that.
            // Shared memory is limited.
            // Let's iterate again? Or store best index in shared?
            shared_counts[0] = best_t; // Broadcast winner
        } else {
            shared_counts[0] = -1;
        }
    }
    __syncthreads();
    
    int winner_L = shared_counts[0];
    if (winner_L != -1 && tid == winner_L) {
        // Winner writes to global memory
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
    cudaMemset(global_gpu_dataset.d_assignment_buffer, -1, global_gpu_dataset.num_instances * sizeof(int));
    std::vector<int> full_assignment(global_gpu_dataset.num_instances, -1);
    for(size_t i=0; i<active_indices.size(); ++i) full_assignment[active_indices[i]] = split_assignment[i];
    cudaMemcpy(global_gpu_dataset.d_assignment_buffer, full_assignment.data(), global_gpu_dataset.num_instances * sizeof(int), cudaMemcpyHostToDevice);

    int num_feats = global_gpu_dataset.num_features;
    size_t int_bytes = num_feats * sizeof(int);
    size_t float_bytes = num_feats * sizeof(float);
    
    // Dynamic Shared Memory Size
    // Need: blockDim * num_classes * 2 (counters)
    // We assume max 10 classes for the stack arrays inside kernel, but dynamic mem size needs to match.
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

// (GPUDataset::initialize and free remain the same as previous step, verify they are present in your file)

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    // ... (Keep previous initialization logic for d_values, d_labels, etc.) ...
    num_features = cpu_dataset.get_features_size();
    num_instances = cpu_dataset.get_instance_number();
    
    // ... [Insert previous logic for finding max_label and flattening data here] ...
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
    cudaFree(d_score_L); cudaFree(d_thresh_L); cudaFree(d_lbl_L_L); cudaFree(d_lbl_L_R); cudaFree(d_cscore_L_L); cudaFree(d_cscore_L_R); cudaFree(d_leaf_L); cudaFree(d_leaflbl_L);
    cudaFree(d_score_R); cudaFree(d_thresh_R); cudaFree(d_lbl_R_L); cudaFree(d_lbl_R_R); cudaFree(d_cscore_R_L); cudaFree(d_cscore_R_R); cudaFree(d_leaf_R); cudaFree(d_leaflbl_R);
}