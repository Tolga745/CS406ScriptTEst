#include "dataview.h"
#include <algorithm>
#include <omp.h> // Include OpenMP

// Constructor
Dataview::Dataview(Dataset* sorted_dataset, Dataset* unsorted_dataset, int class_number, const bool sort_by_gini_index) 
    : unsorted_dataset(unsorted_dataset), label_frequency(class_number, 0), class_number(class_number), sort_by_gini_index(sort_by_gini_index) {
    
    int num_features = sorted_dataset->get_features_size();
    feature_data.resize(num_features); 
    possible_split_indices.resize(num_features); 
    gini_values.resize(num_features); 

    // Handle Feature 0 separately (often small distinct logic in legacy code, but kept serial for safety or moved to parallel if identical)
    // Keeping your original logic for feature 0 as serial to preserve exact behavior of `label_frequency` accumulation
    const auto& first_feature = sorted_dataset->get_feature(0);
    feature_data[0].resize(first_feature.size());
    
    int last_unique_index = -1;
    for (int feature_element_idx = 0; feature_element_idx < first_feature.size(); feature_element_idx++) {
        feature_data[0][feature_element_idx] = std::move(first_feature[feature_element_idx]);
        label_frequency[first_feature[feature_element_idx].label]++; // This accumulates global class counts, must stay serial or atomic
        if (first_feature[feature_element_idx].unique_value_index != last_unique_index && last_unique_index != -1) {
            possible_split_indices[0].push_back(feature_element_idx);
        }
        last_unique_index = first_feature[feature_element_idx].unique_value_index;
    }

    // ... Gini calculation for Feature 0 (Keep Serial) ...
    float best_gini_0 = 1.0f;
    if (sort_by_gini_index) {
        // ... (Your existing code for feature 0 Gini) ...
         std::vector<int> left_label_frequency(class_number, 0);
         std::vector<int> right_label_frequency(label_frequency);
         for (int i = 0; i < first_feature.size() - 1; i++) {
            right_label_frequency[first_feature[i].label]--;
            left_label_frequency[first_feature[i].label]++;
            
            // ... (Calculation logic) ...
            float left_gini = 1.0f; float right_gini = 1.0f;
            int left_count = i + 1; int right_count = int(first_feature.size()) - left_count;
            for (int label = 0; label < class_number; label++) {
                float left_prob = (float)left_label_frequency[label] / left_count;
                left_gini -= left_prob * left_prob;
                float right_prob = (float)right_label_frequency[label] / right_count;
                right_gini -= right_prob * right_prob;
            }
            float gini = (left_gini * left_count + right_gini * right_count) / first_feature.size();
            if (gini < best_gini_0) best_gini_0 = gini;
         }
    }
    gini_values[0] = std::make_pair(best_gini_0, 0);


    // --- PARALLELIZATION START ---
    // Process Features 1 to N in parallel
    // Variables declared inside the loop are private to each thread
    #pragma omp parallel for schedule(static)
    for (int feature_idx = 1; feature_idx < num_features; feature_idx++) {
        const auto& current_feature = sorted_dataset->get_feature(feature_idx);
        feature_data[feature_idx].resize(current_feature.size());
        
        int local_last_unique = -1;
        
        // Private vectors for Gini calculation
        std::vector<int> local_left_freq(class_number, 0);
        std::vector<int> local_right_freq(label_frequency); // Copy the global frequencies
        float local_best_gini = 1.0f;

        for (int feature_element_idx = 0; feature_element_idx < current_feature.size(); feature_element_idx++) {
            feature_data[feature_idx][feature_element_idx] = std::move(current_feature[feature_element_idx]);

            if (current_feature[feature_element_idx].unique_value_index != local_last_unique && local_last_unique != -1) {
                possible_split_indices[feature_idx].push_back(feature_element_idx);
            }

            local_last_unique = current_feature[feature_element_idx].unique_value_index;

            if (sort_by_gini_index) {
                // Logic identical to original, but using thread-local variables
                local_right_freq[current_feature[feature_element_idx].label]--;
                local_left_freq[current_feature[feature_element_idx].label]++;

                float l_gini = 1.0f; float r_gini = 1.0f;
                int l_count = feature_element_idx + 1; 
                int r_count = int(current_feature.size()) - l_count;

                for (int label = 0; label < class_number; label++) {
                    if (l_count > 0) {
                        float p = (float)local_left_freq[label] / l_count;
                        l_gini -= p * p;
                    }
                    if (r_count > 0) {
                        float p = (float)local_right_freq[label] / r_count;
                        r_gini -= p * p;
                    }
                }

                float gini_idx = (l_gini * l_count + r_gini * r_count) / current_feature.size();
                if (gini_idx < local_best_gini) {
                    local_best_gini = gini_idx;
                }
            }
        }
        gini_values[feature_idx] = std::make_pair(local_best_gini, feature_idx);
    }
    // --- PARALLELIZATION END ---

    if (sort_by_gini_index) {
        std::sort(gini_values.begin(), gini_values.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });
    }
    this->bitset.set_hash(std::hash<DataviewBitset>()(this->bitset));
}
int Dataview::get_dataset_size() const {
    return int(feature_data[0].size());
}

int Dataview::get_feature_number() const {
    return int(feature_data.size());
}

const std::vector<Dataset::FeatureElement>& Dataview::get_sorted_dataset_feature(int feature_index) const {
    return feature_data[feature_index];
}

int Dataview::get_class_number() const {
    return class_number;
}

const std::vector<int>& Dataview::get_label_frequency() const {
    return label_frequency;
}

const std::vector<Dataset::FeatureElement>& Dataview::get_unsorted_dataset_feature(int feature_index) const {
    return unsorted_dataset->feature_data[feature_index];
}

const std::vector<int>& Dataview::get_possible_split_indices(int feature_index) const {
    return possible_split_indices[feature_index];
}

void Dataview::split_data_points(
    const Dataview& current_dataview,
    int feature_index,
    int split_point,
    int split_unique_value_index,
    Dataview& left_dataview,
    Dataview& right_dataview,
    int current_max_depth
) {
    const int num_features = current_dataview.get_feature_number();
    const int class_number = current_dataview.get_class_number();

    // Pre-allocate feature containers
    left_dataview.feature_data.resize(num_features);
    right_dataview.feature_data.resize(num_features);

    left_dataview.possible_split_indices.resize(num_features);
    right_dataview.possible_split_indices.resize(num_features);

    left_dataview.gini_values.resize(num_features);
    right_dataview.gini_values.resize(num_features);

    left_dataview.label_frequency.assign(class_number, 0);
    right_dataview.label_frequency.assign(class_number, 0);


    const auto& unsorted_split_feature =
        current_dataview.unsorted_dataset->feature_data[feature_index];

    // Serial: label frequencies for new views
    Dataview::initialize_split_parameters(
        current_dataview.get_sorted_dataset_feature(feature_index),
        class_number,
        current_dataview.label_frequency,
        split_point,
        left_dataview.label_frequency,
        right_dataview.label_frequency
    );

    const int left_size  = split_point;
    const int right_size = current_dataview.get_dataset_size() - split_point;

    // --- PARALLEL DATA SHUFFLING ---
    #pragma omp parallel for schedule(static)
    for (int feature_no = 0; feature_no < num_features; feature_no++) {
        const auto& src_data = current_dataview.feature_data[feature_no];

        auto& l_data = left_dataview.feature_data[feature_no];
        auto& r_data = right_dataview.feature_data[feature_no];

        l_data.resize(left_size);
        r_data.resize(right_size);

        int l_ptr = 0;
        int r_ptr = 0;
        int l_last_idx = -1;
        int r_last_idx = -1;

        std::vector<int> l_splits;
        std::vector<int> r_splits;
        l_splits.reserve(src_data.size() / 8);
        r_splits.reserve(src_data.size() / 8);

        for (const auto& element : src_data) {
            if (unsorted_split_feature[element.data_point_index].unique_value_index >= split_unique_value_index) {
                r_data[r_ptr] = element;

                if (element.unique_value_index != r_last_idx && r_last_idx != -1) {
                    r_splits.push_back(r_ptr);
                }
                r_last_idx = element.unique_value_index;
                r_ptr++;
            } else {
                l_data[l_ptr] = element;

                if (element.unique_value_index != l_last_idx && l_last_idx != -1) {
                    l_splits.push_back(l_ptr);
                }
                l_last_idx = element.unique_value_index;
                l_ptr++;
            }
        }

        // Optional sanity (debug builds)
        // RUNTIME_ASSERT(l_ptr == left_size, "Left size mismatch in split_data_points");
        // RUNTIME_ASSERT(r_ptr == right_size, "Right size mismatch in split_data_points");

        left_dataview.possible_split_indices[feature_no]  = std::move(l_splits);
        right_dataview.possible_split_indices[feature_no] = std::move(r_splits);

        left_dataview.gini_values[feature_no]  = {1.0f, feature_no};
        right_dataview.gini_values[feature_no] = {1.0f, feature_no};
    }

    left_dataview.unsorted_dataset  = current_dataview.unsorted_dataset;
    right_dataview.unsorted_dataset = current_dataview.unsorted_dataset;

    // If you depend on gini-based ordering as a heuristic, keep this (optional).
    if (current_dataview.sort_by_gini_index && current_max_depth > 3) {
        std::sort(left_dataview.gini_values.begin(), left_dataview.gini_values.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::sort(right_dataview.gini_values.begin(), right_dataview.gini_values.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
}


DataviewBitset::DataviewBitset(const Dataview& dataview) 
    : size(dataview.get_dataset_size()), 
      bitset(dataview.get_unsorted_dataset_feature(0).size()) {
    
    const auto& instances = dataview.get_sorted_dataset_feature(0);
    for (const auto& instance : instances) {
        bitset.set_bit(instance.data_point_index);
    }
}
void Dataview::initialize_split_parameters(const std::vector<Dataset::FeatureElement>& current_feature, int class_number, const std::vector<int> &current_label_frequency, int split_point, std::vector<int> &left_label_frequency, std::vector<int> &right_label_frequency) {
    // Optimization: Calculate the smaller side and derive the larger side
    if (split_point < current_feature.size() - split_point) {
        // Left side is smaller: Count Left, derive Right
        for (int left_counter = 0; left_counter < split_point; left_counter++) {
            left_label_frequency[current_feature[left_counter].label]++;
        }

        for (int label_instance = 0; label_instance < class_number; label_instance++) {
            right_label_frequency[label_instance] = current_label_frequency[label_instance] - left_label_frequency[label_instance];
        }
    } else {
        // Right side is smaller: Count Right, derive Left
        for (int right_counter = split_point; right_counter < current_feature.size(); right_counter++) {
            right_label_frequency[current_feature[right_counter].label]++;
        }

        for (int label_instance = 0; label_instance < class_number; label_instance++) {
            left_label_frequency[label_instance] = current_label_frequency[label_instance] - right_label_frequency[label_instance];
        }
    }
}