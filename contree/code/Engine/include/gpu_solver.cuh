#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <vector>
#include "dataset.h"

struct GPUDataset {
    float* d_values;
    int* d_labels;
    int* d_original_indices;
    int* d_feature_offsets;
    int num_features;
    int num_instances;
    int num_classes;

    void initialize(const Dataset& cpu_dataset);
    void free();
};

extern GPUDataset global_gpu_dataset;

void launch_specialized_solver_kernel(
    const std::vector<int>& active_indices,
    const std::vector<int>& split_assignment,
    int upper_bound,
    
    // Left Node Outputs
    int* h_best_scores_left,
    float* h_best_thresholds_left,
    int* h_best_labels_left_L,
    int* h_best_labels_left_R,
    int* h_best_child_scores_left_L, 
    int* h_best_child_scores_left_R, 
    int* h_leaf_scores_left,
    int* h_leaf_labels_left,         
    
    // Right Node Outputs
    int* h_best_scores_right,
    float* h_best_thresholds_right,
    int* h_best_labels_right_L,
    int* h_best_labels_right_R,
    int* h_best_child_scores_right_L, 
    int* h_best_child_scores_right_R, 
    int* h_leaf_scores_right,
    int* h_leaf_labels_right          
);

#endif