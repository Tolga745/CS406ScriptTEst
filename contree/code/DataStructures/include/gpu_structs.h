// code/DataStructures/include/gpu_structs.h
#pragma once
#include <cuda_runtime.h>

struct GpuDataset {
    // Global dataset (Read Only)
    float* values;
    int* unique_value_indices;
    int* data_point_indices;
    int* labels;

    int num_features;
    int num_instances;
    size_t total_elements; 

    void free() {
        if (values) cudaFree(values);
        if (unique_value_indices) cudaFree(unique_value_indices);
        if (data_point_indices) cudaFree(data_point_indices);
        if (labels) cudaFree(labels);
        values = nullptr;
    }
};

struct GPUDataview {
    // Subset specific data (physically contiguous)
    float* d_values = nullptr;        // Size: num_features * num_instances
    int* d_labels = nullptr;          // Size: num_features * num_instances
    int* d_row_indices = nullptr;     // Size: num_features * num_instances (Maps sorted index -> row ID 0..N)
    int* d_unique_indices = nullptr;  // Size: num_features * num_instances
    
    int num_instances = 0;
    int num_features = 0;
    int num_classes = 0;

    void free() {
        if (d_values) cudaFree(d_values);
        if (d_labels) cudaFree(d_labels);
        if (d_row_indices) cudaFree(d_row_indices);
        if (d_unique_indices) cudaFree(d_unique_indices);
        d_values = nullptr;
        d_labels = nullptr;
        d_row_indices = nullptr;
        d_unique_indices = nullptr;
    }
};