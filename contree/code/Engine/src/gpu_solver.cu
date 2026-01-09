#include "gpu_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

GPUDataset global_gpu_dataset;

// Kernel updated to accept all output arrays
__global__ void compute_splits_kernel(
    const float* __restrict__ values,
    const int* __restrict__ labels,
    const int* __restrict__ original_indices,
    const int* __restrict__ feature_offsets,
    const int* __restrict__ active_mask, 
    int num_features,
    int num_instances,
    int num_classes,
    
    // Output buffers on Device
    int* best_scores_left,
    float* best_thresholds_left,
    int* best_labels_left_L,
    int* best_labels_left_R,
    
    int* best_scores_right,
    float* best_thresholds_right,
    int* best_labels_right_L,
    int* best_labels_right_R
) {
    int feature_idx = blockIdx.x;
    if (feature_idx >= num_features) return;

    // Placeholder logic: You must implement the actual scanning logic here later.
    // For now, we just set default values so it runs without crashing.
    if (threadIdx.x == 0) {
        best_scores_left[feature_idx] = 999999; 
        best_thresholds_left[feature_idx] = 0.0f;
        best_labels_left_L[feature_idx] = 0;
        best_labels_left_R[feature_idx] = 0;

        best_scores_right[feature_idx] = 999999;
        best_thresholds_right[feature_idx] = 0.0f;
        best_labels_right_L[feature_idx] = 0;
        best_labels_right_R[feature_idx] = 0;
    }
}

void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    
    // Left Child Outputs (Host Pointers)
    int* h_best_scores_left,
    float* h_best_thresholds_left,
    int* h_best_labels_left_L,
    int* h_best_labels_left_R,
    
    // Right Child Outputs (Host Pointers)
    int* h_best_scores_right,
    float* h_best_thresholds_right,
    int* h_best_labels_right_L,
    int* h_best_labels_right_R
) {
    // 1. Prepare Active Mask on GPU
    int* d_assignment;
    cudaMalloc(&d_assignment, global_gpu_dataset.num_instances * sizeof(int));
    cudaMemset(d_assignment, -1, global_gpu_dataset.num_instances * sizeof(int));
    
    std::vector<int> full_assignment(global_gpu_dataset.num_instances, -1);
    for(size_t i=0; i<active_indices.size(); ++i) {
        full_assignment[active_indices[i]] = split_assignment[i];
    }
    cudaMemcpy(d_assignment, full_assignment.data(), global_gpu_dataset.num_instances * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Allocate Device Output Arrays
    // We need 8 arrays on the GPU (Score, Threshold, LabelL, LabelR) * 2 (Left/Right)
    int num_feats = global_gpu_dataset.num_features;
    size_t int_bytes = num_feats * sizeof(int);
    size_t float_bytes = num_feats * sizeof(float);

    int *d_score_L, *d_lbl_L_L, *d_lbl_L_R;
    float *d_thresh_L;
    int *d_score_R, *d_lbl_R_L, *d_lbl_R_R;
    float *d_thresh_R;

    cudaMalloc(&d_score_L, int_bytes);
    cudaMalloc(&d_thresh_L, float_bytes);
    cudaMalloc(&d_lbl_L_L, int_bytes);
    cudaMalloc(&d_lbl_L_R, int_bytes);

    cudaMalloc(&d_score_R, int_bytes);
    cudaMalloc(&d_thresh_R, float_bytes);
    cudaMalloc(&d_lbl_R_L, int_bytes);
    cudaMalloc(&d_lbl_R_R, int_bytes);

    // 3. Launch Kernel
    compute_splits_kernel<<<num_feats, 256>>>(
        global_gpu_dataset.d_values,
        global_gpu_dataset.d_labels,
        global_gpu_dataset.d_original_indices,
        global_gpu_dataset.d_feature_offsets,
        d_assignment,
        num_feats,
        global_gpu_dataset.num_instances,
        global_gpu_dataset.num_classes,
        // Device Pointers
        d_score_L, d_thresh_L, d_lbl_L_L, d_lbl_L_R,
        d_score_R, d_thresh_R, d_lbl_R_L, d_lbl_R_R
    );
    cudaDeviceSynchronize();

    // 4. Copy Results Back to Host
    cudaMemcpy(h_best_scores_left, d_score_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_thresholds_left, d_thresh_L, float_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_left_L, d_lbl_L_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_left_R, d_lbl_L_R, int_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_best_scores_right, d_score_R, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_thresholds_right, d_thresh_R, float_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_right_L, d_lbl_R_L, int_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_labels_right_R, d_lbl_R_R, int_bytes, cudaMemcpyDeviceToHost);

    // 5. Cleanup
    cudaFree(d_assignment);
    cudaFree(d_score_L); cudaFree(d_thresh_L); cudaFree(d_lbl_L_L); cudaFree(d_lbl_L_R);
    cudaFree(d_score_R); cudaFree(d_thresh_R); cudaFree(d_lbl_R_L); cudaFree(d_lbl_R_R);
}

void GPUDataset::initialize(const Dataset& cpu_dataset) {
    // Implementation needed (flattening logic)
}
void GPUDataset::free() {
    // Implementation needed (cudaFree)
}