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
    
    // Output pointers...
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
    int* total_counts_L = &shared_counts[0];
    int* total_counts_R = &shared_counts[num_classes];
    int* curr_counts_L  = &shared_counts[2 * num_classes];
    int* curr_counts_R  = &shared_counts[3 * num_classes];

    // 1. Initialize Shared Memory in Parallel
    for (int i = threadIdx.x; i < 4 * num_classes; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    int start_idx = feature_offsets[feature_idx];
    int end_idx = feature_offsets[feature_idx + 1];
    int count = end_idx - start_idx;

    // ---------------------------------------------------------
    // OPTIMIZED PASS 1: Parallel Counting
    // ---------------------------------------------------------
    // All threads process the data in a "Grid Stride" pattern
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int global_idx = start_idx + i;
        int orig_idx = original_indices[global_idx];
        int assignment = active_mask[orig_idx]; 
        int label = labels[global_idx];

        if (assignment == 0) {
            atomicAdd(&total_counts_L[label], 1);
        } else if (assignment == 1) {
            atomicAdd(&total_counts_R[label], 1);
        }
    }
    __syncthreads(); // Wait for all threads to finish counting

    // ---------------------------------------------------------
    // INTERMEDIATE: Calculate Total Sizes (One thread or reduction)
    // ---------------------------------------------------------
    __shared__ int total_L_size;
    __shared__ int total_R_size;
    if (threadIdx.x == 0) {
        total_L_size = 0;
        total_R_size = 0;
        for(int c=0; c<num_classes; ++c) {
            total_L_size += total_counts_L[c];
            total_R_size += total_counts_R[c];
        }

        // Calculate Leaf Scores (Parent Node Errors)
        int leaf_lbl_L, leaf_lbl_R;
        int leaf_err_L = calculate_misclassification(total_counts_L, num_classes, total_L_size, leaf_lbl_L);
        int leaf_err_R = calculate_misclassification(total_counts_R, num_classes, total_R_size, leaf_lbl_R);
        
        // Write outputs
        leaf_scores_left[feature_idx] = leaf_err_L;
        leaf_labels_left[feature_idx] = leaf_lbl_L;
        leaf_scores_right[feature_idx] = leaf_err_R;
        leaf_labels_right[feature_idx] = leaf_lbl_R;

        best_scores_left[feature_idx] = leaf_err_L; 
        best_scores_right[feature_idx] = leaf_err_R;
        
        // Defaults
        best_child_scores_left_L[feature_idx] = -1; best_child_scores_left_R[feature_idx] = -1;
        best_child_scores_right_L[feature_idx] = -1; best_child_scores_right_R[feature_idx] = -1;
    }
    __syncthreads();

    // ---------------------------------------------------------
    // PASS 2: Find Best Split
    // ---------------------------------------------------------
    // NOTE: Fully parallelizing Pass 2 requires a Parallel Scan (Prefix Sum).
    // For now, we keep this part serialized (Thread 0 only) to ensure correctness.
    // However, since Pass 1 (memory intensive) is now parallel, it will be faster.
    
    if (threadIdx.x == 0) {
        int curr_L_size = 0;
        int curr_R_size = 0;
        float prev_value = -999999.0f;
        if(count > 0) prev_value = values[start_idx];

        for (int i = 0; i < count; i++) {
            int global_idx = start_idx + i;
            float val = values[global_idx];
            int orig_idx = original_indices[global_idx];
            int assignment = active_mask[orig_idx];
            int label = labels[global_idx];

            bool value_changed = (val > prev_value + 1e-6f);
            
            if (value_changed) {
                float threshold = (prev_value + val) * 0.5f;

                // LEFT Node Split Check
                if (curr_L_size > 0 && curr_L_size < total_L_size) {
                    int l_lbl, r_lbl;
                    int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                    
                    int max_rem = 0; int rem_lbl = 0;
                    for(int c=0; c<num_classes; ++c) {
                        int rem = total_counts_L[c] - curr_counts_L[c];
                        if(rem > max_rem) { max_rem = rem; rem_lbl = c; }
                    }
                    int score_R = (total_L_size - curr_L_size) - max_rem;
                    r_lbl = rem_lbl;

                    if ((score_L + score_R) < best_scores_left[feature_idx]) {
                        best_scores_left[feature_idx] = score_L + score_R;
                        best_thresholds_left[feature_idx] = threshold;
                        best_labels_left_L[feature_idx] = l_lbl;
                        best_labels_left_R[feature_idx] = r_lbl;
                        best_child_scores_left_L[feature_idx] = score_L;
                        best_child_scores_left_R[feature_idx] = score_R;
                    }
                }

                // RIGHT Node Split Check
                if (curr_R_size > 0 && curr_R_size < total_R_size) {
                    int l_lbl, r_lbl;
                    int score_L = calculate_misclassification(curr_counts_R, num_classes, curr_R_size, l_lbl);

                    int max_rem = 0; int rem_lbl = 0;
                    for(int c=0; c<num_classes; ++c) {
                        int rem = total_counts_R[c] - curr_counts_R[c];
                        if(rem > max_rem) { max_rem = rem; rem_lbl = c; }
                    }
                    int score_R = (total_R_size - curr_R_size) - max_rem;
                    r_lbl = rem_lbl;

                    if ((score_L + score_R) < best_scores_right[feature_idx]) {
                        best_scores_right[feature_idx] = score_L + score_R;
                        best_thresholds_right[feature_idx] = threshold;
                        best_labels_right_L[feature_idx] = l_lbl;
                        best_labels_right_R[feature_idx] = r_lbl;
                        best_child_scores_right_L[feature_idx] = score_L;
                        best_child_scores_right_R[feature_idx] = score_R;
                    }
                }
            }

            if (assignment == 0) {
                curr_counts_L[label]++; curr_L_size++;
            } else if (assignment == 1) {
                curr_counts_R[label]++; curr_R_size++;
            }
            prev_value = val;
        }
    }
}

void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
) {
    // 1. Prepare Active Mask
    
    cudaMemset(global_gpu_dataset.d_assignment_buffer, -1, global_gpu_dataset.num_instances * sizeof(int));

    // Create the CPU vector (This is still a CPU bottleneck, but we can't remove it easily yet)
    std::vector<int> full_assignment(global_gpu_dataset.num_instances, -1);
    for(size_t i=0; i<active_indices.size(); ++i) full_assignment[active_indices[i]] = split_assignment[i];
    
    // Copy to pre-allocated buffer
    cudaMemcpy(global_gpu_dataset.d_assignment_buffer, full_assignment.data(), global_gpu_dataset.num_instances * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Launch Kernel using PRE-ALLOCATED buffers
    int num_feats = global_gpu_dataset.num_features;
    size_t int_bytes = num_feats * sizeof(int);
    size_t float_bytes = num_feats * sizeof(float);
    size_t shared_mem_size = 4 * global_gpu_dataset.num_classes * sizeof(int);

    compute_splits_kernel<<<num_feats, 1, shared_mem_size>>>(
        global_gpu_dataset.d_values, global_gpu_dataset.d_labels, global_gpu_dataset.d_original_indices, global_gpu_dataset.d_feature_offsets, 
        global_gpu_dataset.d_assignment_buffer, // <--- Using persistent buffer
        num_feats, global_gpu_dataset.num_instances, global_gpu_dataset.num_classes,
        
        // Output buffers (Persistent)
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, 
        global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,

        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, 
        global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );
    cudaDeviceSynchronize();

    // 3. Copy Results Back
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