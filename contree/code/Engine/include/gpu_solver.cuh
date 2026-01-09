#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <vector>
#include "dataset.h"

struct GPUDataset {
    // Permanent Data (Read Only)
    float* d_values;
    int* d_labels;
    int* d_original_indices;
    int* d_feature_offsets;
    int num_features;
    int num_instances;
    int num_classes;

    
    int* d_assignment_buffer;   // Size: num_instances
    
    // Output Buffers (Size: num_features)
    int *d_score_L, *d_lbl_L_L, *d_lbl_L_R, *d_cscore_L_L, *d_cscore_L_R, *d_leaf_L, *d_leaflbl_L;
    float *d_thresh_L;
    int *d_score_R, *d_lbl_R_L, *d_lbl_R_R, *d_cscore_R_L, *d_cscore_R_R, *d_leaf_R, *d_leaflbl_R;

    void initialize(const Dataset& cpu_dataset);
    void free();
};

extern GPUDataset global_gpu_dataset;

void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    
    // (Output pointers remain the same)
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right
);

#endif