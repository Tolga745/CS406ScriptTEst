// code/DataStructures/include/gpu_structs.h
#pragma once

// Note: We do NOT include cuda_runtime.h here to keep it C++ compatible.
// The pointers (float*, int*) are standard types.

struct GpuDataset {
    // Global dataset (Read Only)
    float* values = nullptr;
    int* unique_value_indices = nullptr;
    int* data_point_indices = nullptr;
    int* labels = nullptr;

    int num_features = 0;
    int num_instances = 0;
    size_t total_elements = 0; 

    // Declaration only - implementation moves to .cu file
    void free();
};

struct GPUDataview {
    // Subset specific data (physically contiguous)
    float* d_values = nullptr;        // Size: num_features * num_instances
    int* d_labels = nullptr;          // Size: num_features * num_instances
    int* d_row_indices = nullptr;     // Size: num_features * num_instances
    int* d_unique_indices = nullptr;  // Size: num_features * num_instances
    
    int num_instances = 0;
    int num_features = 0;
    int num_classes = 0;

    // Declaration only - implementation moves to .cu file
    void free();
};