#include "dataset.h"
#include <numeric>
#include <algorithm> // For std::iota if needed elsewhere
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

int Dataset::get_instance_number() const {
    return (int) feature_data[0].size();
}

const std::vector<std::vector<Dataset::FeatureElement>>& Dataset::get_features_data() const {
    return feature_data;
}

void Dataset::add_feature_index_pair(int feature_index, int data_point_index, float value, int label) {
    if(data_point_index == 0) {
        feature_data.emplace_back();
    }
    feature_data[feature_index].push_back({value, -1, data_point_index, label});
}

const std::vector<Dataset::FeatureElement>& Dataset::get_feature(int index) const {
    return feature_data[index];
}

int Dataset::get_features_size() const {
    return (int) feature_data.size();
}

// ---------------------------------------------------------
// MODIFIED: Parallel Sorting on GPU
// ---------------------------------------------------------
void Dataset::sort_feature_values() {
    // Iterate over each feature column
    for (auto &feature_vec : feature_data) {
        
        // 1. Move data to GPU (Thrust handles allocation and copy)
        // FeatureElement is a simple struct, so it copies bit-wise.
        thrust::device_vector<FeatureElement> d_vec = feature_vec;

        // 2. Sort on GPU
        // We use a lambda marked with __device__ for the comparison
        thrust::sort(d_vec.begin(), d_vec.end(), 
            [] __device__ (const FeatureElement& a, const FeatureElement& b) {
                return a.value < b.value;
            }
        );

        // 3. Move sorted data back to CPU
        thrust::copy(d_vec.begin(), d_vec.end(), feature_vec.begin());
    }
}

void Dataset::compute_unique_value_indices() {
    std::vector<size_t> idx(get_instance_number());
    for (auto& cur_feature_data : feature_data) {
        std::iota(idx.begin(), idx.end(), 0);
        // Note: Since we already sorted 'cur_feature_data' by value in sort_feature_values(),
        // we might not need to sort 'idx' again if the logic allows. 
        // However, keeping original logic for safety:
        std::sort(idx.begin(), idx.end(),
            [&cur_feature_data](size_t i1, size_t i2) {return cur_feature_data[i1].value < cur_feature_data[i2].value;});
            
        double prev = -1.0f;
        bool first = true;
        int cur_unique_value_index = -1;
        for (size_t ix : idx) {
            auto& cur_feature_element = cur_feature_data[ix];
            if (first || cur_feature_element.value - prev >= EPSILON) cur_unique_value_index++;
            cur_feature_element.unique_value_index = cur_unique_value_index;
            prev = cur_feature_element.value;
            first = false;
        }
    }
}