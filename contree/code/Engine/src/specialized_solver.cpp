#include "specialized_solver.h"
#include "gpu_solver.cuh"


void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    
    // --- JIT UPLOAD (ONCE) ---
    GPUDataview active_gpu_view = dataview.gpu_view;
    prepare_gpu_view(dataview, active_gpu_view); // Ensures data is on GPU (uploaded if needed)
    // -------------------------

    SolverBuffers buffers;
    buffers.resize(dataview.get_feature_number());
    
    for (int feature_index = 0; feature_index < dataview.get_feature_number(); feature_index++) {
        // Pass the prepared view down
        create_optimal_decision_tree(dataview, active_gpu_view, solution_configuration, feature_index, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score), buffers);

        if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap) {
            return;
        }
    }
}

void SpecializedSolver::get_best_left_right_scores(const Dataview& dataview, const GPUDataview& gpu_view, int feature_index, int split_point, float threshold, std::shared_ptr<Tree> &left_optimal_dt, std::shared_ptr<Tree> &right_optimal_dt, int upper_bound, SolverBuffers& buffers) {
    int num_features = dataview.get_feature_number();
    
    // NO LOCAL ALLOCATION HERE! We use 'buffers'.
    
    // Call Unified GPU Solver using pointers from the buffer
    run_specialized_solver_gpu(
        gpu_view, feature_index, threshold, upper_bound,
        buffers.left_scores.data(), buffers.left_thresholds.data(), buffers.left_labels_L.data(), buffers.left_labels_R.data(), buffers.left_child_scores_L.data(), buffers.left_child_scores_R.data(), buffers.left_leaf_scores.data(), buffers.left_leaf_labels.data(),
        buffers.right_scores.data(), buffers.right_thresholds.data(), buffers.right_labels_L.data(), buffers.right_labels_R.data(), buffers.right_child_scores_L.data(), buffers.right_child_scores_R.data(), buffers.right_leaf_scores.data(), buffers.right_leaf_labels.data(),
        false // FETCH FULL RESULTS = FALSE
    );

    bool potential_improvement = false;

    int best_L_idx = -1;
    for(int i=0; i<num_features; ++i) {
        if(best_L_idx == -1 || buffers.left_scores[i] < buffers.left_scores[best_L_idx]) best_L_idx = i;
    }
    int best_score_L = (best_L_idx != -1 && buffers.left_scores[best_L_idx] < buffers.left_leaf_scores[0]) 
                        ? buffers.left_scores[best_L_idx] : buffers.left_leaf_scores[0];

    // Check Right
    int best_R_idx = -1;
    for(int i=0; i<num_features; ++i) {
        if(best_R_idx == -1 || buffers.right_scores[i] < buffers.right_scores[best_R_idx]) best_R_idx = i;
    }
    int best_score_R = (best_R_idx != -1 && buffers.right_scores[best_R_idx] < buffers.right_leaf_scores[0])
                        ? buffers.right_scores[best_R_idx] : buffers.right_leaf_scores[0];

    // If combined score beats upper bound, we MUST fetch details to construct the tree
    if (best_score_L + best_score_R < upper_bound) {
        potential_improvement = true;
    }

    // 3. SLOW PASS: If promising, fetch everything
    if (potential_improvement) {
        run_specialized_solver_gpu(
            gpu_view, feature_index, threshold, upper_bound,
            buffers.left_scores.data(), buffers.left_thresholds.data(), buffers.left_labels_L.data(), buffers.left_labels_R.data(), buffers.left_child_scores_L.data(), buffers.left_child_scores_R.data(), buffers.left_leaf_scores.data(), buffers.left_leaf_labels.data(),
            buffers.right_scores.data(), buffers.right_thresholds.data(), buffers.right_labels_L.data(), buffers.right_labels_R.data(), buffers.right_child_scores_L.data(), buffers.right_child_scores_R.data(), buffers.right_leaf_scores.data(), buffers.right_leaf_labels.data(),
            true // FETCH FULL RESULTS = TRUE
        );
    }

    // 4. Construct Result (Existing Logic)
    // Left
    best_L_idx = -1; // Re-find to be safe (or reuse)
    for(int i=0; i<num_features; ++i) { if(best_L_idx == -1 || buffers.left_scores[i] < buffers.left_scores[best_L_idx]) best_L_idx = i; }
    int leaf_score_L = buffers.left_leaf_scores[0]; 

    if (potential_improvement && best_L_idx != -1 && buffers.left_scores[best_L_idx] < leaf_score_L) {
        left_optimal_dt->misclassification_score = buffers.left_scores[best_L_idx];
        left_optimal_dt->update_split(best_L_idx, buffers.left_thresholds[best_L_idx], std::make_shared<Tree>(buffers.left_labels_L[best_L_idx], buffers.left_child_scores_L[best_L_idx]), std::make_shared<Tree>(buffers.left_labels_R[best_L_idx], buffers.left_child_scores_R[best_L_idx]));
    } else {
        left_optimal_dt->make_leaf(buffers.left_leaf_labels[0], leaf_score_L);
    }

    // Right
    best_R_idx = -1;
    for(int i=0; i<num_features; ++i) { if(best_R_idx == -1 || buffers.right_scores[i] < buffers.right_scores[best_R_idx]) best_R_idx = i; }
    int leaf_score_R = buffers.right_leaf_scores[0];

    if (potential_improvement && best_R_idx != -1 && buffers.right_scores[best_R_idx] < leaf_score_R) {
        right_optimal_dt->misclassification_score = buffers.right_scores[best_R_idx];
        right_optimal_dt->update_split(best_R_idx, buffers.right_thresholds[best_R_idx], std::make_shared<Tree>(buffers.right_labels_L[best_R_idx], buffers.right_child_scores_L[best_R_idx]), std::make_shared<Tree>(buffers.right_labels_R[best_R_idx], buffers.right_child_scores_R[best_R_idx]));
    } else {
        right_optimal_dt->make_leaf(buffers.right_leaf_labels[0], leaf_score_R);
    }
}



void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const GPUDataview& gpu_view, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound, SolverBuffers& buffers) {
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
        
        // Pass gpu_view down
        get_best_left_right_scores(dataview, gpu_view, feature_index, split_point, threshold, left_optimal_dt, right_optimal_dt, current_optimal_decision_tree->misclassification_score, buffers);
        
        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {
            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

            upper_bound = std::min(upper_bound, current_best_score);

            if (current_best_score == 0) {
                return;
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