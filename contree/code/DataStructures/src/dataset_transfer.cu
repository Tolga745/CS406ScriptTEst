// code/DataStructures/src/dataset_transfer.cu
#include "dataset.h"
#include "gpu_structs.h"
#include <vector>
#include <cuda_runtime.h>

// --- Existing Upload Function ---
GpuDataset UploadDatasetToGPU(const Dataset& cpu_dataset) {
    GpuDataset gpu_data;
    gpu_data.num_features = cpu_dataset.get_features_size();
    gpu_data.num_instances = cpu_dataset.get_instance_number();
    gpu_data.total_elements = (size_t)gpu_data.num_features * gpu_data.num_instances;

    // 1. Allocate Temporary Host Memory
    float* h_values;
    int* h_unique;
    int* h_indices;
    int* h_labels;

    cudaMallocHost(&h_values, gpu_data.total_elements * sizeof(float));
    cudaMallocHost(&h_unique, gpu_data.total_elements * sizeof(int));
    cudaMallocHost(&h_indices, gpu_data.total_elements * sizeof(int));
    cudaMallocHost(&h_labels, gpu_data.total_elements * sizeof(int));

    // 2. Flatten the Data
    size_t global_idx = 0;
    const auto& feature_data = cpu_dataset.get_features_data();
    
    for (int f = 0; f < gpu_data.num_features; f++) {
        const auto& current_feat_vec = feature_data[f];
        for (int i = 0; i < gpu_data.num_instances; i++) {
            const auto& element = current_feat_vec[i];
            
            h_values[global_idx] = element.value;
            h_unique[global_idx] = element.unique_value_index;
            h_indices[global_idx] = element.data_point_index;
            h_labels[global_idx]  = element.label;
            
            global_idx++;
        }
    }

    // 3. Allocate Memory on GPU
    cudaMalloc(&gpu_data.values, gpu_data.total_elements * sizeof(float));
    cudaMalloc(&gpu_data.unique_value_indices, gpu_data.total_elements * sizeof(int));
    cudaMalloc(&gpu_data.data_point_indices, gpu_data.total_elements * sizeof(int));
    cudaMalloc(&gpu_data.labels, gpu_data.total_elements * sizeof(int));

    // 4. Copy to GPU
    cudaMemcpy(gpu_data.values, h_values, gpu_data.total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data.unique_value_indices, h_unique, gpu_data.total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data.data_point_indices, h_indices, gpu_data.total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data.labels, h_labels, gpu_data.total_elements * sizeof(int), cudaMemcpyHostToDevice);

    // 5. Clean up Host Memory
    cudaFreeHost(h_values);
    cudaFreeHost(h_unique);
    cudaFreeHost(h_indices);
    cudaFreeHost(h_labels);

    return gpu_data;
}

// --- NEW IMPLEMENTATIONS ---

void GpuDataset::free() {
    if (values) cudaFree(values);
    if (unique_value_indices) cudaFree(unique_value_indices);
    if (data_point_indices) cudaFree(data_point_indices);
    if (labels) cudaFree(labels);
    
    values = nullptr;
    unique_value_indices = nullptr;
    data_point_indices = nullptr;
    labels = nullptr;
}

void GPUDataview::free() {
    if (!owns_memory) return; 

    if (d_values) cudaFree(d_values);
    if (d_labels) cudaFree(d_labels);
    if (d_row_indices) cudaFree(d_row_indices);
    if (d_unique_indices) cudaFree(d_unique_indices);
    
    d_values = nullptr;
    d_labels = nullptr;
    d_row_indices = nullptr;
    d_unique_indices = nullptr;
}