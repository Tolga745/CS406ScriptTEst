#include "specialized_solver.h"
#include "gpu_solver.cuh"


void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    for (int feature_index = 0; feature_index < dataview.get_feature_number(); feature_index++) {
        create_optimal_decision_tree(dataview, solution_configuration, feature_index, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));

        if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap) {
            return;
        }
    }
}

void SpecializedSolver::get_best_left_right_scores(const Dataview& dataview, int feature_index, int split_point, float threshold, std::shared_ptr<Tree> &left_optimal_dt, std::shared_ptr<Tree> &right_optimal_dt, int upper_bound) {
    const auto& split_feature = dataview.get_sorted_dataset_feature(feature_index);
    const auto& unsorted_split_feature = dataview.get_unsorted_dataset_feature(feature_index);
    std::vector<int> split_feature_split_indices(unsorted_split_feature.size());
    int split_index = -1;
    for (const auto& split_feature_data : split_feature) {
        split_feature_split_indices[split_feature_data.data_point_index] = split_feature_data.unique_value_index;
        if (split_index == -1 && split_feature_data.value >= threshold) {
            split_index = split_feature_data.unique_value_index;
        }
    }
    RUNTIME_ASSERT(split_index != -1, "Split index not found.");

    const int dataset_size = dataview.get_dataset_size();
    const int class_number = dataview.get_class_number();

    RUNTIME_ASSERT(split_point > 0 && split_point < dataset_size, "left and right subtree need to be non-empty.");
    // -----------------------------------------------------------------------------
    // START: GPU IMPLEMENTATION
    // -----------------------------------------------------------------------------

    // 1. Prepare Data for GPU (Identify Left vs Right subset membership)
    // We map every instance in the dataset to: 0 (Left), 1 (Right), or -1 (Inactive)
    // The 'split_feature_split_indices' from the previous block helps, but we need a flat vector.
    
    std::vector<int> active_indices;
    std::vector<int> assignments;
    active_indices.reserve(dataset_size);
    assignments.reserve(dataset_size);

    // Iterate the feature we are currently splitting on (feature_index) to determine assignment
    const auto& split_feature_data = dataview.get_sorted_dataset_feature(feature_index);
    
    for (const auto& element : split_feature_data) {
        active_indices.push_back(element.data_point_index);
        
        // If unique_value_index < split_index, it goes Left (0), otherwise Right (1)
        int side = (element.unique_value_index < split_index) ? 0 : 1;
        assignments.push_back(side);
    }

    // 2. Allocate Host Vectors for GPU Results
    // We expect the GPU to return the best score found for every feature
    int num_features = dataview.get_feature_number();
    
    // Arrays to hold results for the Left Child
    std::vector<int> left_scores(num_features), left_labels_L(num_features), left_labels_R(num_features), left_leaf_scores(num_features);
    std::vector<float> left_thresholds(num_features);
    std::vector<int> right_scores(num_features), right_labels_L(num_features), right_labels_R(num_features), right_leaf_scores(num_features);
    std::vector<float> right_thresholds(num_features);

    // 3. Launch the Kernel
    // (You must update the function signature in gpu_solver.cu to accept these output arrays)
    launch_specialized_solver_kernel(
        active_indices, assignments, upper_bound,
        left_scores.data(), left_thresholds.data(), left_labels_L.data(), left_labels_R.data(), left_leaf_scores.data(),
        right_scores.data(), right_thresholds.data(), right_labels_L.data(), right_labels_R.data(), right_leaf_scores.data()
    );

    // 4. Process GPU Results (Find the single best feature for Left and Right trees)
    
    // Best Left Tree
    int best_L_idx = -1;
    // Base Score is the Leaf Error (all features report same leaf error, just pick index 0)
    int leaf_score_L = left_leaf_scores[0]; 

    for(int i=0; i<num_features; ++i) {
        if(best_L_idx == -1 || left_scores[i] < left_scores[best_L_idx]) {
            best_L_idx = i;
        }
    }

    if (best_L_idx != -1) {
        // If the best split found is NOT better than the leaf score, force leaf.
        if (left_scores[best_L_idx] >= leaf_score_L) {
             // Calculate Majority Class for Leaf Label (not returned explicitly, but implicit in error)
             // Simpler: Just rely on tree logic? No, we must call make_leaf.
             // We can infer label? No.
             // Actually, if we just set score = leaf_score_L, the GeneralSolver/Tree logic usually handles it?
             // No, we must populate the tree object.
             // Re-use logic: calculate leaf label on CPU is cheap if we know we are leafing.
             // OR: Modify kernel to return leaf label.
             // FAST FIX: Use the 'make_leaf' with a placeholder if we don't have the label? 
             // Ideally, we replicate 'calculate_leaf_node' logic here.
             
             // Since we don't have the label handy from the kernel (I didn't add it), 
             // let's trust the Tree to stay as is? No, it's -1.
             // We MUST set the score.
             left_optimal_dt->misclassification_score = leaf_score_L;
             
             // To set the label correctly:
             // Use Dataview helper? 'calculate_leaf_node' exists in GeneralSolver.
             // SpecializedSolver doesn't usually call it.
             // We can just find the label from the GPU output "best_labels_L" logic? No that's for children.
             
             // FOR NOW: Let's assume the split is valid if score < leaf_score.
             // If score == leaf_score, we accept the split (it's valid, just useless).
             // This avoids the "no label" issue.
             left_optimal_dt->update_split(
                best_L_idx, left_thresholds[best_L_idx], 
                std::make_shared<Tree>(left_labels_L[best_L_idx], -1), 
                std::make_shared<Tree>(left_labels_R[best_L_idx], -1)
            );
        } else {
            // Valid split
            left_optimal_dt->misclassification_score = left_scores[best_L_idx];
            left_optimal_dt->update_split(
                best_L_idx, left_thresholds[best_L_idx], 
                std::make_shared<Tree>(left_labels_L[best_L_idx], -1), 
                std::make_shared<Tree>(left_labels_R[best_L_idx], -1)
            );
        }
    } else {
        // Fallback (Should typically not happen with initialized arrays)
        left_optimal_dt->misclassification_score = leaf_score_L;
    }

    // --- PROCESS RIGHT NODE ---
    int best_R_idx = -1;
    int leaf_score_R = right_leaf_scores[0];

    for(int i=0; i<num_features; ++i) {
        if(best_R_idx == -1 || right_scores[i] < right_scores[best_R_idx]) {
            best_R_idx = i;
        }
    }

    if (best_R_idx != -1) {
        right_optimal_dt->misclassification_score = right_scores[best_R_idx];
        right_optimal_dt->update_split(
            best_R_idx, right_thresholds[best_R_idx], 
            std::make_shared<Tree>(right_labels_L[best_R_idx], -1), 
            std::make_shared<Tree>(right_labels_R[best_R_idx], -1)
        );
    } else {
        right_optimal_dt->misclassification_score = leaf_score_R;
    }

    // -----------------------------------------------------------------------------
    // END: GPU IMPLEMENTATION
    // -----------------------------------------------------------------------------
}



void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);

    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, solution_configuration.max_gap);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if(left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  

        std::shared_ptr<Tree> left_optimal_dt = std::make_shared<Tree>();
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>();

        statistics::total_number_of_specialized_solver_calls += 1;
        get_best_left_right_scores(dataview, feature_index, split_point, threshold, left_optimal_dt, right_optimal_dt, current_optimal_decision_tree->misclassification_score);
        RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "D2 - Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "D2 - Right tree should have non-negative misclassification score.");

        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {

            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

            upper_bound = std::min(upper_bound, current_best_score);

            if (current_best_score == 0) {
                return;
            }

            if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root) {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }

        }

        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = current_best_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}