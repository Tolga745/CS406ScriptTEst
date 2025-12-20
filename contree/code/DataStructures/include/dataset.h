#ifndef DATASET_H
#define DATASET_H

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "configuration.h"

class Dataset {
public:
    struct FeatureElement {
        float value;
        int unique_value_index;
        int data_point_index;
        int label;
    };

    struct DataPoint {
        int data_point_index;
        int label;
        std::vector<float> feature_values;
    };

    int get_instance_number() const;

    const std::vector<std::vector<FeatureElement>>& get_features_data() const;
    const std::vector<FeatureElement>& get_feature(int index) const;

    int get_features_size() const;

    void add_feature_index_pair(int feature_index, int data_point_index, float value, int label);

    void sort_feature_values();

    void compute_unique_value_indices();

    // feature data, indexed by [feature, data_ix]
    std::vector<std::vector<Dataset::FeatureElement>> feature_data;
};

#endif //DATASET_H
