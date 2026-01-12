#include "general_solver.h"
#include <algorithm>
#include <mutex>
#include <atomic>
#include <queue>
#include <omp.h> // Make sure to enable OpenMP in your compiler settings

// Mutex to protect the update of 'current_optimal_decision_tree' when merging threads
std::mutex GeneralSolver::tree_mutex;

// top-level optimization
void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    // early stopping
    if (current_optimal_decision_tree->misclassification_score == 0 || dataview.get_dataset_size() == 0) {
        return;
    }

    // Cache check (Thread-safe due to internal Cache mutex)
    if (Cache::global_cache.is_cached(dataview, solution_configuration.max_depth)) {
        current_optimal_decision_tree = Cache::global_cache.retrieve(dataview, solution_configuration.max_depth);
        // Note: statistical updates might need to be atomic if precise tracking is required in parallel
        // statistics::total_number_cache_hits += 1; 
        return;
    }

    calculate_leaf_node(dataview.get_class_number(), dataview.get_dataset_size(), dataview.get_label_frequency(), current_optimal_decision_tree);
    
    // early stopping
    if (solution_configuration.max_depth == 0) {
        return;
    }

    if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap || dataview.get_dataset_size() == 1) {
        return;
    }

    // depth-2 special solver
    if (solution_configuration.max_depth == 2) {
        SpecializedSolver::create_optimal_decision_tree(dataview, solution_configuration, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));
        return;
    }

    // --- PARALLELIZATION START ---
    
    // 1. Shared Atomic Upper Bound: Allows threads to read the "record to beat" without locking.
    std::atomic<int> shared_ub(std::min(upper_bound, current_optimal_decision_tree->misclassification_score));

    // 2. Parallel Loop over Features
    // schedule(dynamic) is crucial because some features take much longer to solve than others.
    #pragma omp parallel for schedule(dynamic)
    for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
        
        // Check global time limit inside the parallel loop
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) continue;

        // Optimization: If another thread found a perfect solution (0 error), stop working.
        int current_global_ub = shared_ub.load();
        if (current_global_ub == 0) continue;

        int feature_index = dataview.gini_values[feature_nr].second;

        // 3. Thread-Local Tree Copy
        // We cannot modify 'current_optimal_decision_tree' directly.
        // Each thread works on its own copy initialized with the *current best known* score.
        std::shared_ptr<Tree> thread_local_tree = std::make_shared<Tree>(*current_optimal_decision_tree);
        
        // Pass the ATOMIC upper bound value to the solver
        create_optimal_decision_tree(dataview, solution_configuration, feature_index, thread_local_tree, current_global_ub);

        // 4. Critical Section: Merge Results
        // Only lock if we actually found something better than the *snapshot* we took earlier.
        if (thread_local_tree->misclassification_score < current_global_ub) {
            std::lock_guard<std::mutex> lock(tree_mutex);
            
            // Double-check: Has the global state improved while we were waiting for the lock?
            if (thread_local_tree->misclassification_score < current_optimal_decision_tree->misclassification_score) {
                
                // Update the master tree pointer
                current_optimal_decision_tree = thread_local_tree;
                
                // Update the atomic bound so other threads see it immediately
                shared_ub.store(current_optimal_decision_tree->misclassification_score);
            }
        }
    }
    // --- PARALLELIZATION END ---

    if (current_optimal_decision_tree->misclassification_score <= upper_bound) {
        Cache::global_cache.store(dataview, solution_configuration.max_depth, current_optimal_decision_tree);
    }
}

// per feature optimization
// branch step
void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {    
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);
    // interval 
    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) return;
        
        // Check if our current best score is already beaten by the global upper bound 
        // (This happens if `upper_bound` passed in was an atomic load from the parent loop)
        if (current_optimal_decision_tree->misclassification_score == 0) return;

        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if (left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const int interval_half_distance = std::max(split_point - possible_split_indices[left], possible_split_indices[right] - split_point);

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  
        const int split_unique_value_index = current_feature[split_point].unique_value_index;
        
        Dataview left_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview right_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        
        // This is computationally expensive and memory intensive. 
        // We keep it serial here to avoid over-subscribing threads (since we are already parallelizing features).
        Dataview::split_data_points(dataview, feature_index, split_point, split_unique_value_index, left_dataview, right_dataview, solution_configuration.max_depth);

        std::shared_ptr<Tree> left_optimal_dt  = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);

        // Here firstly compute the bigger dataset since it might make computing the smaller dataset obsolete
        auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size() ) ? left_dataview : right_dataview;
        auto& larger_data  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_dataview : left_dataview;

        auto& smaller_optimal_dt = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? left_optimal_dt : right_optimal_dt;
        auto& larger_optimal_dt  = (left_dataview.get_dataset_size()  < right_dataview.get_dataset_size() ) ? right_optimal_dt : left_optimal_dt;

        int larger_ub = solution_configuration.use_upper_bound ? std::min(upper_bound, current_optimal_decision_tree->misclassification_score)
                                       : current_optimal_decision_tree->misclassification_score;
                
        // statistics::total_number_of_general_solver_calls += 1; // Careful with stats in parallel environments
        
        // Recursive Call 1
        const Configuration left_solution_configuration = solution_configuration.GetLeftSubtreeConfig();
        GeneralSolver::create_optimal_decision_tree(larger_data, left_solution_configuration, larger_optimal_dt, larger_ub);

        int smaller_ub = solution_configuration.use_upper_bound ? std::max(std::min(current_optimal_decision_tree->misclassification_score, upper_bound) - larger_optimal_dt->misclassification_score, interval_half_distance) 
                                        : current_optimal_decision_tree->misclassification_score;

        if (smaller_ub > 0 || (smaller_ub == 0 && current_optimal_decision_tree->misclassification_score == larger_optimal_dt->misclassification_score)) {
            // statistics::total_number_of_general_solver_calls += 1;
            
            // Recursive Call 2
            const Configuration right_solution_configuration = solution_configuration.GetRightSubtreeConfig(left_solution_configuration.max_gap);
            GeneralSolver::create_optimal_decision_tree(smaller_data, right_solution_configuration, smaller_optimal_dt, smaller_ub);
            
            RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "Left tree should have non-negative misclassification score.");
            RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "Right tree should have non-negative misclassification score.");

            const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

            if (current_best_score < current_optimal_decision_tree->misclassification_score) {
                RUNTIME_ASSERT(left_optimal_dt->is_initialized(), "Left tree should be initialized.");
                RUNTIME_ASSERT(right_optimal_dt->is_initialized(), "Right tree should be initialized.");

                current_optimal_decision_tree->misclassification_score = current_best_score;
                current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

                if (current_best_score == 0) {
                    return;
                }

                if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root)  {
                    const auto stop = std::chrono::high_resolution_clock::now();
                    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                    // Use a lock for I/O if multiple roots are processed (unlikely here but good practice)
                    // static std::mutex io_mutex;
                    // std::lock_guard<std::mutex> io_lock(io_mutex);
                    std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
                }
            }
        } else {
            smaller_optimal_dt->misclassification_score = -1;
        }
        
        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}

void GeneralSolver::calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree) {
    int best_classification_score = -1;
    int best_classification_label = -1;

    for (int label = 0; label < class_number; label++) {
        if (label_frequency[label] > best_classification_score) {
            best_classification_score = label_frequency[label];
            best_classification_label = label;
        }
    }

    const int best_misclassification_score = instance_number - best_classification_score;

    if (best_misclassification_score < current_optimal_decision_tree->misclassification_score) {
        RUNTIME_ASSERT(best_classification_label != -1, "Cannot assign negative leaf label.");
        current_optimal_decision_tree->make_leaf(best_classification_label, best_misclassification_score);
    }
}