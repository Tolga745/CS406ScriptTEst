#include "dataset.h"
#include <numeric>
#include <algorithm>
#include <omp.h> // Include OpenMP

int Dataset::get_instance_number() const {
    // Safety check for empty dataset
    if (feature_data.empty()) return 0;
    return (int) feature_data[0].size();
}

const std::vector<std::vector<Dataset::FeatureElement>>& Dataset::get_features_data() const {
    return feature_data;
}

void Dataset::add_feature_index_pair(int feature_index, int data_point_index, float value, int label) {
    // This is typically called during file parsing (serial), so no locks needed unless parsing is parallel.
    if(data_point_index == 0) {
        feature_data.emplace_back();
    }
    // Note: If you ever parallelize the file reader, this needs a lock. 
    // Assuming serial loading for now.
    feature_data[feature_index].push_back({value, -1, data_point_index, label});
}

const std::vector<Dataset::FeatureElement>& Dataset::get_feature(int index) const {
    return feature_data[index];
}

int Dataset::get_features_size() const {
    return (int) feature_data.size();
}

void Dataset::sort_feature_values() {
    // PARALLELIZATION: Independent sorting of each feature column
    // This provides massive speedup during the pre-processing phase.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < feature_data.size(); ++i) {
        std::sort(feature_data[i].begin(), feature_data[i].end(),
                  [](const FeatureElement& first, const FeatureElement& second) {
                      return first.value < second.value;
                  }
        );
    }
}

void Dataset::compute_unique_value_indices() {
    int n_features = (int)feature_data.size();
    
    // PARALLELIZATION: Calculating unique indices is independent per feature
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_features; ++i) {
        // Direct access to the feature vector for this thread
        auto& cur_feature_data = feature_data[i];
        
        // Local vector for indices
        std::vector<size_t> idx(cur_feature_data.size());
        std::iota(idx.begin(), idx.end(), 0);
        
        // Sort indices based on value
        std::sort(idx.begin(), idx.end(),
            [&cur_feature_data](size_t i1, size_t i2) {
                return cur_feature_data[i1].value < cur_feature_data[i2].value;
            }
        );

        double prev = -1.0f;
        bool first = true;
        int cur_unique_value_index = -1;
        
        // Assign unique IDs
        for (size_t ix : idx) {
            auto& cur_feature_element = cur_feature_data[ix];
            if (first || cur_feature_element.value - prev >= EPSILON) cur_unique_value_index++;
            cur_feature_element.unique_value_index = cur_unique_value_index;
            prev = cur_feature_element.value;
            first = false;
        }
    }
}