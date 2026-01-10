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
    
    // 1. Allocate Output Arrays
    int num_features = dataview.get_feature_number();
    std::vector<int> left_scores(num_features), left_labels_L(num_features), left_labels_R(num_features), left_child_scores_L(num_features), left_child_scores_R(num_features), left_leaf_scores(num_features), left_leaf_labels(num_features);
    std::vector<float> left_thresholds(num_features);

    std::vector<int> right_scores(num_features), right_labels_L(num_features), right_labels_R(num_features), right_child_scores_L(num_features), right_child_scores_R(num_features), right_leaf_scores(num_features), right_leaf_labels(num_features);
    std::vector<float> right_thresholds(num_features);

    // 2. Call Unified GPU Solver (works for Root and Children)
    // Ensure dataview.gpu_view is populated!
    run_specialized_solver_gpu(
        dataview.gpu_view,
        feature_index, 
        threshold, 
        upper_bound,
        left_scores.data(), left_thresholds.data(), left_labels_L.data(), left_labels_R.data(), left_child_scores_L.data(), left_child_scores_R.data(), left_leaf_scores.data(), left_leaf_labels.data(),
        right_scores.data(), right_thresholds.data(), right_labels_L.data(), right_labels_R.data(), right_child_scores_L.data(), right_child_scores_R.data(), right_leaf_scores.data(), right_leaf_labels.data()
    );

    // 3. Process Left Results (Unchanged)
    int best_L_idx = -1;
    for(int i=0; i<num_features; ++i) {
        if(best_L_idx == -1 || left_scores[i] < left_scores[best_L_idx]) {
            best_L_idx = i;
        }
    }
    
    int leaf_score_L = left_leaf_scores[0]; 
    if (best_L_idx != -1 && left_scores[best_L_idx] < leaf_score_L) {
        left_optimal_dt->misclassification_score = left_scores[best_L_idx];
        left_optimal_dt->update_split(
            best_L_idx, left_thresholds[best_L_idx], 
            std::make_shared<Tree>(left_labels_L[best_L_idx], left_child_scores_L[best_L_idx]), 
            std::make_shared<Tree>(left_labels_R[best_L_idx], left_child_scores_R[best_L_idx])
        );
    } else {
        left_optimal_dt->make_leaf(left_leaf_labels[0], leaf_score_L);
    }

    // 4. Process Right Results (Unchanged)
    int best_R_idx = -1;
    for(int i=0; i<num_features; ++i) {
        if(best_R_idx == -1 || right_scores[i] < right_scores[best_R_idx]) {
            best_R_idx = i;
        }
    }

    int leaf_score_R = right_leaf_scores[0];
    if (best_R_idx != -1 && right_scores[best_R_idx] < leaf_score_R) {
        right_optimal_dt->misclassification_score = right_scores[best_R_idx];
        right_optimal_dt->update_split(
            best_R_idx, right_thresholds[best_R_idx], 
            std::make_shared<Tree>(right_labels_L[best_R_idx], right_child_scores_L[best_R_idx]), 
            std::make_shared<Tree>(right_labels_R[best_R_idx], right_child_scores_R[best_R_idx])
        );
    } else {
        right_optimal_dt->make_leaf(right_leaf_labels[0], leaf_score_R);
    }
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