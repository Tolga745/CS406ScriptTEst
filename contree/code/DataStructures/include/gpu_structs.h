// gpu_structs.h
#pragma once

struct GpuDataset {
    // These pointers will live on the GPU (Device Memory)
    float* values;
    int* unique_value_indices;
    int* data_point_indices;
    int* labels;

    int num_features;
    int num_instances;
    size_t total_elements; // num_features * num_instances

    // Helper to free memory
    void free() {
        if (values) cudaFree(values);
        if (unique_value_indices) cudaFree(unique_value_indices);
        if (data_point_indices) cudaFree(data_point_indices);
        if (labels) cudaFree(labels);
    }
};