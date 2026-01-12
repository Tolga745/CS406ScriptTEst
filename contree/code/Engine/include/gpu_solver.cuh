#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <vector>
#include "dataset.h"
#include "gpu_structs.h"

// Forward declaration
class Dataview;

struct GPURecursionBuffer {
    float* d_values = nullptr;
    int* d_labels = nullptr;
    int* d_row_indices = nullptr;

    void allocate(size_t total_elements);
    void free();
};

extern std::vector<GPURecursionBuffer> recursion_buffers;
extern int* d_global_row_map; // Shared scratch buffer

void allocate_recursion_buffers(int max_depth, int num_instances, int num_features);
void free_recursion_buffers();

struct GPUDataset {
    // Permanent Data (Read Only)
    float* d_values;
    int* d_labels;
    int* d_original_indices;
    int* d_feature_offsets;
    int num_features;
    int num_instances;
    int num_classes;
    size_t total_elements;
    
    int* d_assignment_buffer;   // Size: num_instances
    
    // Output Buffers (Size: num_features)
    // These hold the results of the split evaluation for all features in parallel
    int *d_score_L, *d_lbl_L_L, *d_lbl_L_R, *d_cscore_L_L, *d_cscore_L_R, *d_leaf_L, *d_leaflbl_L;
    float *d_thresh_L;
    int *d_score_R, *d_lbl_R_L, *d_lbl_R_R, *d_cscore_R_L, *d_cscore_R_R, *d_leaf_R, *d_leaflbl_R;
    float *d_thresh_R;
    
    void initialize(const Dataset& cpu_dataset);
    void free();
};

// Global instance to hold the GPU copy of the dataset
extern GPUDataset global_gpu_dataset;

// Helper to move a CPU Dataview to GPU (used for the root node or fallbacks)
void prepare_gpu_view(const Dataview& cpu_view, GPUDataview& gpu_view);

/**
 * Main entry point for the GPU-based split evaluation.
 * Calculates the best split for every feature in the active_view.
 */
void run_specialized_solver_gpu(
    const GPUDataview& active_view,
    int split_feature_index,
    float split_threshold,
    int upper_bound,
    
    // Output pointers (Host pointers where results will be copied)
    int* h_best_scores_left, float* h_best_thresholds_left, int* h_best_labels_left_L, int* h_best_labels_left_R, int* h_best_child_scores_left_L, int* h_best_child_scores_left_R, int* h_leaf_scores_left, int* h_leaf_labels_left,
    int* h_best_scores_right, float* h_best_thresholds_right, int* h_best_labels_right_L, int* h_best_labels_right_R, int* h_best_child_scores_right_L, int* h_best_child_scores_right_R, int* h_leaf_scores_right, int* h_leaf_labels_right,
    bool fetch_full_results = true
);

#endif