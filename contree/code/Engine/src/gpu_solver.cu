#include "gpu_solver.cuh"
#include "dataview.h" // code2's dataview header
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// --- CONSTANTS ---
// Limit on number of classes for shared memory usage. 
// 32 classes * 2 counters * 256 threads * 4 bytes = ~64KB shared mem.
#define MAX_CLASSES 32 
#define MAX_COUNTERS (MAX_CLASSES * 2)

// --- GLOBAL DEFINITION ---
GPUDataset global_gpu_dataset;

std::vector<GPURecursionBuffer> recursion_buffers;
int* d_global_row_map = nullptr;

void GPURecursionBuffer::allocate(size_t total_elements) {
    cudaMalloc(&d_values, total_elements * sizeof(float));
    cudaMalloc(&d_labels, total_elements * sizeof(int));
    cudaMalloc(&d_row_indices, total_elements * sizeof(int));
}

void GPURecursionBuffer::free() {
    if (d_values) cudaFree(d_values);
    if (d_labels) cudaFree(d_labels);
    if (d_row_indices) cudaFree(d_row_indices);
    d_values = nullptr;
}

void allocate_recursion_buffers(int max_depth, int num_instances, int num_features) {
    free_recursion_buffers();

    // Allocate global scratch map
    cudaMalloc(&d_global_row_map, num_instances * sizeof(int));

    // Allocate buffers for depths 0 to max_depth
    size_t total_elements = (size_t)num_instances * num_features;
    recursion_buffers.resize(max_depth + 1);

    for(int i = 0; i <= max_depth; ++i) {
        recursion_buffers[i].allocate(total_elements);
    }
}

void free_recursion_buffers() {
    for(auto& buf : recursion_buffers) {
        buf.free();
    }
    recursion_buffers.clear();
    if (d_global_row_map) {
        cudaFree(d_global_row_map);
        d_global_row_map = nullptr;
    }
}

// --- GPUDataset IMPLEMENTATION ---

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    this->num_features = cpu_dataset.get_features_size();
    this->num_instances = cpu_dataset.get_instance_number();
    
    size_t total_elements = (size_t)this->num_features * this->num_instances;
    
    // Allocate Host Memory for flattening
    float* h_values;
    int* h_labels;
    int* h_indices;
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
    d_feature_offsets = nullptr; 

    // Copy to GPU
    cudaMemcpy(d_values, h_values, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_indices, h_indices, total_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate GPU Memory (Buffers)
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

__global__ void generate_assignment_map_kernel(
    const float* feature_column,
    const int* row_indices, 
    int* assignment_map,
    int num_instances,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instances) {
        int row_id = row_indices[idx];
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

// Kernel: Compute Best Splits using Shared Memory Reduction
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

    // Prepare starting counts (Exclusive Scan)
    int my_starting_counts[MAX_COUNTERS]; 
    if (tid > 0) {
        for (int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = shared_counts[(tid - 1) * num_counters_per_thread + c];
    } else {
        for(int c = 0; c < num_counters_per_thread; ++c) my_starting_counts[c] = 0;
    }
    
    // Total Counts
    __shared__ int total_counts[MAX_COUNTERS]; 
    if (tid == bdim - 1) {
        for (int c = 0; c < num_counters_per_thread; ++c) total_counts[c] = shared_counts[tid * num_counters_per_thread + c];
    }
    __syncthreads();

    // Leaf Scores (Thread 0)
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

    float prev_value = -1e30f; 
    bool has_prev = false;

    if (my_start > 0) {
        prev_value = values[start_idx + my_start - 1];
        has_prev = true;
    } else if (count > 0 && my_start == 0) {
        prev_value = values[start_idx];
        has_prev = true;
    }

    for (int i = my_start; i < my_end; ++i) {
        int global_idx = start_idx + i;
        float val = values[global_idx];
        
        int row_id = row_indices[global_idx];
        int assignment = assignment_map[row_id];
        int label = labels[global_idx];

        bool value_changed = has_prev && (val > prev_value);
        
        if (value_changed && i > 0) { 
            float threshold = (prev_value + val) * 0.5f;
            // Left Child
            if (curr_L_size > 0 && curr_L_size < total_L_size) {
                int l_lbl;
                int score_L = calculate_misclassification(curr_counts_L, num_classes, curr_L_size, l_lbl);
                int max_rem = 0; int rem_lbl = 0;
                for(int c=0; c<num_classes; ++c) { int rem = total_counts[c] - curr_counts_L[c]; if(rem > max_rem) { max_rem = rem; rem_lbl = c; } }
                int score_R = (total_L_size - curr_L_size) - max_rem;
                int total_score = score_L + score_R;
                if (total_score < local_best_score_L) { local_best_score_L = total_score; local_best_thresh_L = threshold; local_best_lL_L = l_lbl; local_best_lR_L = rem_lbl; local_best_cL_L = score_L; local_best_cR_L = score_R; }
            }
            // Right Child
            if (curr_R_size > 0 && curr_R_size < total_R_size) {
                int l_lbl;
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

void prepare_gpu_view(const Dataview& cpu_view, GPUDataview& gpu_view) {
    if (gpu_view.d_values != nullptr) return;

    // Use recursion_buffers[0] as scratch space/storage for the root view
    // Ensure allocate_recursion_buffers was called before this!
    if (recursion_buffers.empty()) {
        std::cerr << "Error: Recursion buffers not allocated." << std::endl;
        return;
    }
    auto& buffer = recursion_buffers[0];
    
    int num_instances = cpu_view.get_dataset_size();
    int num_features = cpu_view.get_feature_number();
    size_t total_elements = (size_t)num_instances * num_features;

    std::vector<float> h_val(total_elements);
    std::vector<int> h_lbl(total_elements);
    std::vector<int> h_idx(total_elements);

    size_t global_idx = 0;
    for (int f = 0; f < num_features; f++) {
        const auto& feat_vec = cpu_view.get_sorted_dataset_feature(f);
        for (const auto& elem : feat_vec) {
            h_val[global_idx] = elem.value;
            h_lbl[global_idx] = elem.label;
            h_idx[global_idx] = elem.data_point_index;
            global_idx++;
        }
    }

    cudaMemcpy(buffer.d_values, h_val.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer.d_labels, h_lbl.data(), total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer.d_row_indices, h_idx.data(), total_elements * sizeof(int), cudaMemcpyHostToDevice);

    gpu_view.d_values = buffer.d_values;
    gpu_view.d_labels = buffer.d_labels;
    gpu_view.d_row_indices = buffer.d_row_indices;
    gpu_view.num_instances = num_instances;
    gpu_view.num_features = num_features;
    gpu_view.num_classes = cpu_view.get_class_number();
    gpu_view.owns_memory = false; 
}

void run_specialized_solver_gpu(
    const GPUDataview& active_view,
    int split_feature_index,
    float split_threshold,
    int upper_bound,
    
    // Outputs
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right,
    bool fetch_full_results
) {
    if (active_view.num_classes > MAX_CLASSES) { std::cerr << "ERR: Class limit exceeded" << std::endl; exit(1); }

    int* d_assignment_map = global_gpu_dataset.d_assignment_buffer; 
    
    int block = 256;
    int grid = (active_view.num_instances + block - 1) / block;

    if (split_feature_index == -1) {
        // Root case: Set map to 0
        cudaMemset(d_assignment_map, 0, active_view.num_instances * sizeof(int));
    } else {
        float* split_feature_col = active_view.d_values + (size_t)split_feature_index * active_view.num_instances;
        int* split_feature_row_indices = active_view.d_row_indices + (size_t)split_feature_index * active_view.num_instances;

        generate_assignment_map_kernel<<<grid, block>>>(
            split_feature_col,
            split_feature_row_indices, 
            d_assignment_map,
            active_view.num_instances,
            split_threshold
        );
    }

    size_t shared_mem = (256 * active_view.num_classes * 2) * sizeof(int);

    compute_splits_kernel<<<active_view.num_features, 256, shared_mem>>>(
        active_view.d_values,
        active_view.d_labels,
        active_view.d_row_indices,
        d_assignment_map,
        active_view.num_features,
        active_view.num_instances,
        active_view.num_classes,
        global_gpu_dataset.d_score_L, global_gpu_dataset.d_thresh_L, global_gpu_dataset.d_lbl_L_L, global_gpu_dataset.d_lbl_L_R, global_gpu_dataset.d_cscore_L_L, global_gpu_dataset.d_cscore_L_R, global_gpu_dataset.d_leaf_L, global_gpu_dataset.d_leaflbl_L,
        global_gpu_dataset.d_score_R, global_gpu_dataset.d_thresh_R, global_gpu_dataset.d_lbl_R_L, global_gpu_dataset.d_lbl_R_R, global_gpu_dataset.d_cscore_R_L, global_gpu_dataset.d_cscore_R_R, global_gpu_dataset.d_leaf_R, global_gpu_dataset.d_leaflbl_R
    );

    // 4. Copy Back
    size_t int_bytes = active_view.num_features * sizeof(int); 
    size_t float_bytes = active_view.num_features * sizeof(float);

    cudaMemcpy(h_best_scores_left, global_gpu_dataset.d_score_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_scores_left, global_gpu_dataset.d_leaf_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_scores_right, global_gpu_dataset.d_score_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leaf_scores_right, global_gpu_dataset.d_leaf_R, int_bytes, cudaMemcpyDeviceToHost);

    if (fetch_full_results) {
        cudaMemcpy(h_best_thresholds_left, global_gpu_dataset.d_thresh_L, float_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_labels_left_L, global_gpu_dataset.d_lbl_L_L, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_labels_left_R, global_gpu_dataset.d_lbl_L_R, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_child_scores_left_L, global_gpu_dataset.d_cscore_L_L, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_child_scores_left_R, global_gpu_dataset.d_cscore_L_R, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_leaf_labels_left, global_gpu_dataset.d_leaflbl_L, int_bytes, cudaMemcpyDeviceToHost);

        cudaMemcpy(h_best_thresholds_right, global_gpu_dataset.d_thresh_R, float_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_labels_right_L, global_gpu_dataset.d_lbl_R_L, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_labels_right_R, global_gpu_dataset.d_lbl_R_R, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_child_scores_right_L, global_gpu_dataset.d_cscore_R_L, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_child_scores_right_R, global_gpu_dataset.d_cscore_R_R, int_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_leaf_labels_right, global_gpu_dataset.d_leaflbl_R, int_bytes, cudaMemcpyDeviceToHost);
    }
}